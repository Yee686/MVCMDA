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
Models = [Model_SAGE, Model_GCN, Model_GAT, Model_GIN, Model_ATTENGCN]
# Models = [Model_GCN, Model_GAT, Model_GIN, Model_ATTENGCN]

class Config(object):
    def __init__(self):
        self.data_path = '/mnt/yzy/NIMCGCN/datasets/data(MDA108)'
        self.validation = 10
        self.save_path = ' '

        self.lr = 0.0001            # learning rate
        self.weight_decay = 0       # weight decay
        self.epoch = 250            # epoch
        self.alpha = 0.2            # alpha for loss function


class Sizes(object):
    def __init__(self, dataset):
        self.m = 1043               # miRNA number
        self.d = 2166               # drug number
        # self.fg = 64                # x(miRNA) feature dimension
        # self.fd = 64                # y(Drug) feature dimension
        # self.k = 32                 # out feature channels
        self.fg = 96                # x(miRNA) feature dimension
        self.fd = 96                # y(Drug) feature dimension
        self.k = 64                 # out feature channels
        self.gcn_layers = 2         # gcn layers
        self.view = 1               # view number


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)

        return (1-opt.alpha)*loss_sum[one_index].sum()+opt.alpha*loss_sum[zero_index].sum()


def train(model, label, train_data, optimizer, opt, k):
    model.train()

    # regression_crit = Myloss()
    loss = torch.nn.MSELoss(reduction='mean')

    one_index = np.array(
        np.where(train_data["md_p"].clone().cpu() == 1)).T.tolist()
    zero_index = np.array(
        np.where(train_data["md_p"].clone().cpu() == 0)).T.tolist()
    
    for epoch in tqdm(range(1, opt.epoch+1)):
        torch.cuda.empty_cache()
        model.zero_grad()
        score = model(train_data)

        losss = loss(score, train_data['md_p'].cuda())
        losss.backward()
        optimizer.step()

        # tqdm.write("epoch {}'s loss = {}".format(
        #     epoch, losss.item()/(len(one_index[0])+len(zero_index[0]))))
    # 计算AUC
    score = score.detach().cpu().numpy()
    scores[:, :, k] = score

    score = score.reshape(-1).tolist()
    fpr, tpr, _ = metric.roc_curve(label, score)
    auc = metric.auc(fpr, tpr)
    aucs.append(auc)

    return auc


dataset = torch.load("/mnt/yzy/NIMCGCN/NIMCcode/dataset.pt")
opt = Config()
sizes = Sizes(dataset)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    label = dataset["md_true"].clone().cpu().numpy().reshape(-1).tolist()
    print(sum(label))

    for Model in Models:

        aucs = []
        scores = np.zeros((1043, 2166, opt.validation))
        print(str(Model))
        torch.cuda.empty_cache()

        for i in range(opt.validation):
            dataset["md_p"] = dataset["md_true"].clone()
            dataset["md_p"][tuple(np.array(dataset["fold_index"][i]).T)] = 0

            model = Model(sizes)
            model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                                weight_decay=opt.weight_decay)

            auc = train(model, label, dataset, optimizer, opt, i)
            print("auc {} - {}".format(i+1, auc))

        print("Avarage Auc:", sum(aucs)/opt.validation)

        with torch.no_grad():
            scores = scores.mean(axis=2)
            np.save("/mnt/yzy/NIMCGCN/Prediction/Compare/{}_{}FoldCV_{}_[lr]{}_[wd]{}_[ep]{}_[cvMthd]elem_[miRDim]{}_[drugDim]{}_[kFdim]{}.npy"
                    .format(model.name, opt.validation, today, opt.lr, opt.weight_decay, opt.epoch, sizes.fg, sizes.fd, sizes.k), scores)
            scores = scores.reshape(-1).tolist()
            fpr, tpr, _ = metric.roc_curve(label, scores)
            auc = metric.auc(fpr, tpr)
            print("Total Auc:", auc)
