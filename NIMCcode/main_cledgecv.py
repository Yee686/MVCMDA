from torch import nn, optim
from model_cledgecv import Model
from prepareData import prepare_data
import numpy as np
import torch
import sklearn.metrics as metric
import matplotlib.pyplot as plt

class Config(object):
    def __init__(self):
        #self.data_path = '../data'
        self.data_path = '/mnt/yzy/NIMCGCN/datasets/data(MDA108)'
        self.validation = 10
        #self.save_path = '../data'
        self.save_path = ' '
        self.epoch = 250
        # self.epoch = 2
        self.alpha = 0.2


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        # print(one_index)
        return (1-opt.alpha)*loss_sum[one_index].sum()+opt.alpha*loss_sum[zero_index].sum()
        return loss_sum


class Sizes(object):
    def __init__(self, dataset):
        self.m = 1043
        self.d = 2166
        self.fg = 64
        self.fd = 64
        self.k = 32

cnt = 1
def train(model, label, train_data, optimizer, opt):
    model.train()
    # regression_crit = Myloss()
    loss = torch.nn.MSELoss(reduction='mean')
    # print("md_p",train_data["md_p"].cuda().shape)
    # one_index = torch.LongTensor(np.array(np.where(train_data["md_p"].clone().cpu()==1)).T.tolist())
    one_index = np.array(np.where(train_data["md_p"].clone().cpu()==1)).T.tolist()

    # print(one_index)
    # zero_index = torch.LongTensor(np.array(np.where(train_data["md_p"].clone().cpu()==0)).T.tolist())
    zero_index = np.array(np.where(train_data["md_p"].clone().cpu()==0)).T.tolist()
    # print(torch.tensor(one_index).shape,torch.tensor(zero_index).shape)
    # def train_epoch():
    #     model.zero_grad()
    #     score = model(train_data)
    #     loss = regression_crit(one_index, zero_index, train_data["md_p"].cuda(), score)
    #     loss.backward()
    #     optimizer.step()
    #     return loss
    for epoch in range(1, opt.epoch+1):
        model.zero_grad()
        score = model(train_data)
        # print(score)
        # losss = regression_crit(one_index, zero_index, train_data["md_p"].cuda(), score)
        losss = loss(score, train_data['md_p'].cuda())
        losss.backward()
        optimizer.step()

        print(epoch,":",losss.item()/(len(one_index[0])+len(zero_index[0])))
    # 计算AUC
    score = score.detach().cpu().numpy()
    global cnt
    np.save("/mnt/yzy/NIMCGCN/Prediction/Nimc/0324cv{}".format(cnt),score)
    cnt += 1
    score = score.reshape(-1).tolist()
    # print(score)
    fpr,tpr,_ = metric.roc_curve(label,score)
    auc = metric.auc(fpr,tpr)
    aucs.append(auc)
    return auc


opt = Config()
aucs = []

if __name__ == "__main__":
    # 预处理数据集
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    # dataset = prepare_data(opt)
    dataset = torch.load("/mnt/yzy/NIMCGCN/NIMCcode/dataset.pt")
    # print(dataset)
    sizes = Sizes(dataset)          # sizes为关联网络相关超参 
    label = dataset["md_true"].clone().cpu().numpy().reshape(-1).tolist()
    print(sum(label))
    for i in range(opt.validation):
        print('-'*50)
        dataset["md_p"] = dataset["md_true"].clone()
        #抹去验证集的标签
        dataset["md_p"][tuple(np.array(dataset["fold_index"][i]).T)] = 0
        model = Model(sizes)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        auc = train(model, label, dataset, optimizer, opt)
        print("auc {} - {}".format(i,auc))
    
    print("Avarage Auc:",sum(aucs)/opt.validation)
