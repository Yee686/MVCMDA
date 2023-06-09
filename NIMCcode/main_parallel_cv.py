from torch import nn, optim, distributed
# from model_SAGE import Model
# from model_gcn import Model
from model_gat import Model
# from model_gin import Model
from prepareData import prepare_data
import numpy as np
import torch
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

today = datetime.date.today().strftime("%Y%m%d")[2:]

class Config(object):
    def __init__(self):
        self.data_path = '/mnt/yzy/NIMCGCN/datasets/data(MDA108)'
        self.validation = 10
        #self.save_path = '../data'
        self.save_path = ' '
        self.lr = 0.0008
        self.weight_decay = 0
        self.epoch = 250
        self.alpha = 0.2  

class Sizes(object):
    def __init__(self, dataset):
        self.m = 1043
        self.d = 2166
        self.fg = 64
        self.fd = 64
        self.k = 32

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
    
    one_index = np.array(np.where(train_data["md_p"].clone().cpu()==1)).T.tolist()
    zero_index = np.array(np.where(train_data["md_p"].clone().cpu()==0)).T.tolist()
    
    for epoch in tqdm(range(1, opt.epoch+1)):
        torch.cuda.empty_cache()
        model.zero_grad()
        score = model(train_data)
    
        losss = loss(score, train_data['md_p'].cuda())
        losss.backward()
        optimizer.step()

        tqdm.write("loss:",losss.item()/(len(one_index[0])+len(zero_index[0])))

    # 计算AUC
    with torch.no_grad():
        score = score.detach().cpu().numpy()
        scores[:,:,k] = score
        
        score = score.reshape(-1).tolist()
        fpr,tpr,_ = metric.roc_curve(label,score)
        auc = metric.auc(fpr,tpr)
        aucs.append(auc)

    return auc


opt = Config()
aucs = []
scores = np.zeros((1043,2166,opt.validation))

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    distributed.init_process_group(backend='nccl', init_method='tcp://localhost:22996', world_size=2, rank=0)
    print('init')
    torch.cuda.set_device(1)

    # dataset = prepare_data(opt)
    dataset = torch.load("/mnt/yzy/NIMCGCN/NIMCcode/dataset.pt")

    sizes = Sizes(dataset)          # sizes为模型相关超参 
    label = dataset["md_true"].clone().cpu().numpy().reshape(-1).tolist()
    print(sum(label))

    for i in range(opt.validation):
        print('-'*50)

        dataset["md_p"] = dataset["md_true"].clone()
        #抹去验证集的标签
        dataset["md_p"][tuple(np.array(dataset["fold_index"][i]).T)] = 0

        model = Model(sizes)
        
        print('init')
        model = nn.parallel.DistributedDataParallel(model)
        print('dist')
        print(model)
        model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay = opt.weight_decay)
        
        auc = train(model, label, dataset, optimizer, opt, i)
        print("auc {} - {}".format(i+1, auc))
    
    print("Avarage Auc:",sum(aucs)/opt.validation)

    with torch.no_grad():
        scores = scores.mean(axis=2)
        np.save("/mnt/yzy/NIMCGCN/Prediction/Nimc/{}_{}FoldCV_{}_[lr]{}_[wd]{}_[ep]{}_[cvmhod]elem" \
                .format(model.name, opt.validation, today, opt.lr, opt.weight_decay, opt.epoch), scores)
        scores = scores.reshape(-1).tolist()
        fpr,tpr,_ = metric.roc_curve(label,scores)
        auc = metric.auc(fpr,tpr)
        print("Total Auc:",auc)