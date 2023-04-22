from torch import nn, optim
from model_SAGE import Model as Model_SAGE
from model_gcn import Model as Model_GCN
from model_gat import Model as Model_GAT
from model_gin import Model as Model_GIN
from model_attengcn import Model as Model_ATTENGCN
from prepareData import prepare_data
import numpy as np
import torch
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

today = datetime.date.today().strftime("%Y%m%d")[2:]
# Models = [Model_SAGE, Model_GCN, Model_GAT, Model_GIN, Model_ATTENGCN]
Models = [Model_GCN]

class Config(object):
    def __init__(self):
        self.data_path = '/mnt/yzy/NIMCGCN/datasets/data(MDA108)'
        self.validation = 10
        self.save_path = '/mnt/yzy/NIMCGCN/Prediction'

        self.lr = 0.0001            # learning rate
        self.weight_decay = 0       # weight decay
        self.epoch = 250            # epoch
        self.alpha = 0.2            # alpha for zero target in loss function


class Sizes(object):
    def __init__(self):
        self.m = 1043               # miRNA number
        self.d = 2166               # drug number
        # self.fg = 64                # x(miRNA) feature dimension
        # self.fd = 64                # y(Drug) feature dimension
        # self.k = 32                 # out feature channels
        self.fg = 64                  # x(miRNA) feature dimension
        self.fd = 64                  # y(Drug) feature dimension
        self.k = 64                 # out feature channels
        self.gcn_layers = 2         # gcn layers
        self.view = 1               # view number


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, input, target, one_index, zero_index,):

        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        # print("one ",one_index[0].shape, one_index[0].dtype)
        # print("zero ",zero_index[0].shape, zero_index[0].dtype)

        # loss_one = (1-opt.alpha)*loss_sum[one_index].sum()
        # loss_zero = opt.alpha*loss_sum[zero_index].sum()

        return (1-opt.alpha)*loss_sum[one_index].sum() + opt.alpha*loss_sum[zero_index].sum()
        # return loss_one + loss_zero

def train(model, train_data, optimizer, opt, train_one_index, train_zero_index):

    model.train()

    loss= Myloss()

    # x,y = torch.where(train_data["md_p"] == 1)
    # one_index = torch.stack((x,y),dim=0).cuda()
    one_index = train_one_index[0], train_one_index[1]

    # x,y = torch.where(train_data["md_p"] == 0)
    # zero_index = torch.stack((x,y),dim=0).cuda()
    zero_index = train_zero_index[0], train_zero_index[1]
    
    for epoch in tqdm(range(1, opt.epoch+1)):
        torch.cuda.empty_cache()
        model.zero_grad()

        score = model(train_data)
        losss = loss(score, train_data['md_true'].cuda(), one_index, zero_index)

        losss.backward()
        optimizer.step()

        tqdm.write("epoch {}'s loss = {}".format(
            epoch, losss.item()/(len(one_index[0])+len(zero_index[0]))))

    score = model(train_data)
    return score


dataset = torch.load("/mnt/yzy/NIMCGCN/new_dataset.pt")
opt = Config()
sizes = Sizes()

if __name__ == "__main__":
    torch.cuda.set_device(0)

    for Model in Models:

        print(str(Model))
        torch.cuda.empty_cache()

        final_score = torch.empty((sizes.m, sizes.d)).cuda()
        aucs = []

        for i in range(opt.validation):
            
            torch.cuda.empty_cache()

            val_one_index = dataset['fold_one_index'][i]
            val_zero_index = dataset['fold_zero_index'][i]
            val_one_index.cuda()
            val_zero_index.cuda()

            train_one_index = torch.cat([dataset['fold_one_index'][j] for j in range(opt.validation) if j != i], dim=1)
            train_zero_index = torch.cat([dataset['fold_zero_index'][j] for j in range(opt.validation) if j != i], dim=1)
            train_one_index.cuda()
            train_zero_index.cuda()

            t_dataset = {
                    'md_true':dataset['md_true'],
                    'mm':dataset['mm'],
                    'dd':dataset['dd'],
            }

            # print("train_one_index:", train_one_index.shape)
            # print("train_zero_index:", train_zero_index.shape)
            # print("val_one_index:", val_one_index.shape)
            # print("val_zero_index:", val_zero_index.shape)

            model = Model(sizes)
            model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                                weight_decay=opt.weight_decay)

            score = train(model, t_dataset, optimizer, opt, train_one_index, train_zero_index)
            
            with torch.no_grad():
                val_index = torch.cat((val_one_index,val_zero_index),dim=1)
                val_index = val_index[0], val_index[1]

                final_score[val_index] = score[val_index]

                score = score[val_index].detach().cpu().numpy()
                label = dataset['md_true'][val_index].detach().cpu().numpy()
                
                score = score.reshape(-1).tolist()
                label = label.reshape(-1).tolist()
                fpr, tpr, _ = metric.roc_curve(label,score)
                auc = metric.auc(fpr, tpr)
                aucs.append(auc)
                print("auc {} - {}".format(i+1, auc))

        print("Avarage Auc:", sum(aucs)/opt.validation)

        with torch.no_grad():
            # scores = scores.mean(axis=2)
            final_score = final_score.detach().cpu().numpy()
            final_target = dataset['md_true'].detach().cpu().numpy()
            np.save("{0}/{1}_{2}FoldCV_{3}_[lr]{4}_[wd]{5}_[ep]{6}_[cvMthd]elem_[miRDim]{7}_[drugDim]{8}_[kFdim]{9}_[alpha]{10}.npy"
                    .format(opt.save_path, model.name, opt.validation, today, opt.lr, opt.weight_decay, 
                            opt.epoch, sizes.fg, sizes.fd, sizes.k, opt.alpha), final_score)
            score  = final_score.reshape(-1).tolist()
            target = final_target.reshape(-1).tolist()
            fpr, tpr, _ = metric.roc_curve(target,score)
            auc = metric.auc(fpr, tpr)
            print("Total Auc:", auc)
