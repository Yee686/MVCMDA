import numpy as np
from torch import nn, optim
import torch
from torch import tensor
from prepareData import prepare_data
from model import Model

import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import KFold


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation
        print("-One--",self.data_set['md']['train'][0].shape)
        print("-Zero--",self.data_set['md']['train'][1].shape)

        one_index = self.data_set['md']['train'][0].cpu().tolist()
        zero_index = self.data_set['md']['train'][1].cpu().tolist()
        print(torch.tensor(one_index).shape,torch.tensor(zero_index).shape)
        train_set = {}
        valid_set = {}

        # 前放one 后放zero
        trains = [[],[]]
        valids = [[],[]]
        # print("***")
        kf = KFold(n_splits=self.nums)
        # 划分数据集
        for train,valid in kf.split(one_index):
            # print(torch.tensor(train).shape,torch.tensor(valid).shape)
            # trains[0].append(train)
            # valids[0].append(valid)
            print(torch.tensor([one_index[i] for i in train]).shape)
            trains[0].append([one_index[i] for i in train])
            valids[0].append([one_index[i] for i in valid])
        # print("***")
        for train,valid in kf.split(zero_index):
            # print(np.array(train).shape,np.array(train).shape)
            trains[1].append([zero_index[i] for i in train])
            valids[1].append([zero_index[i] for i in valid])
        # print(torch.tensor(one_index).shape,torch.tensor(zero_index).shape)
        for i in range(0,self.nums):
            # print(trains[0][i],trains[1][i])
            # print(valids[0][i],trains[1][i])
            train_set[i] = [torch.LongTensor(trains[0][i]),torch.LongTensor(trains[1][i])]
            valid_set[i] = [torch.LongTensor(valids[0][i]),torch.LongTensor(valids[1][i])]

        self.data_set['md']['train'] = train_set
        self.data_set['md']['valid'] = valid_set
        # print(self.data_set['md']['train'])

    # def __getitem__(self, index):
    #     return (self.data_set['dd'], self.data_set['mm'],
    #             self.data_set['md']['train'], None,
    #             self.data_set['md_p'], self.data_set['md_true'])

    def __getitem__(self, index):
        return (self.data_set['dd'], self.data_set['mm'],
                self.data_set['md']['train'][index], self.data_set['md']['valid'][index],
                self.data_set['md_p'], self.data_set['md_true'])

    def __len__(self):
        return self.nums



class Config(object):
    def __init__(self):
        
        self.data_path = '/mnt/yzy/NIMCGCN/datasets/data(MDA108)'
        self.validation = 10
        # self.save_path = 'C:/Users/86136/Desktop/NIMCGCN-master/datasets/data(new)'
        self.epoch = 2
        self.alpha = 0.2



class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-opt.alpha)*loss_sum[one_index].sum()+opt.alpha*loss_sum[zero_index].sum()

def MyAUC(predict,train_data):
    # 需将所有Tensor转换为list() 否则调用numpy和sklearn时会报错
    # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    predict = predict.cpu().tolist()
    one_index = train_data[3][0].cpu().tolist()
    zero_index = train_data[3][1].cpu().tolist()
    # print(torch.tensor(one_index).shape,torch.tensor(zero_index).shape)
    label = list([1]*len(one_index) + [0]*len(zero_index))
    # print("num of label:",len(label))
    pred = []
    for pos in one_index + zero_index:
        pred.append(predict[pos[0]][pos[1]])
    # print(type(label),type(pred)) 
    auc = metrics.roc_auc_score(label,pred)
    # precision,recall,_ = metrics.precision_recall_curve(label,pred)
    # aupr = metrics.auc(sorted(precision),sorted(recall))

    return auc,aupr

class Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)
        self.d = dataset['dd']['data'].size(0)
        self.fg = 128
        self.fd = 128
        self.k = 32


def train(model, train_data, optimizer, opt):
    model.train()
    regression_crit = Myloss()
    # print(train_data[2][0].shape,train_data[2][1].shape)
    one_index = train_data[2][0].cuda().t().tolist()
    zero_index = train_data[2][1].cuda().t().tolist()
    # print("In Train")
    # print("-One--",torch.tensor(one_index).shape)
    # print("-Zero-",torch.tensor(zero_index).shape)
    def train_epoch():
        model.zero_grad()
        score = model(train_data)

        print(score.shape)
        loss = regression_crit(one_index, zero_index, train_data[4].cuda(), score)

        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(1, opt.epoch+1):
        train_reg_loss = train_epoch()
        print(epoch,":",train_reg_loss.item()/(len(one_index[0])+len(zero_index[0])))
        # print(train_reg_loss)
    

def device_setting():
    torch.cuda.set_device(0)

opt = Config()
if __name__ == "__main__":
    device_setting()
    if hasattr(torch.cuda, 'empty_cache'):
        print("缓存已清空 准备开始")
        torch.cuda.empty_cache()

    dataset = prepare_data(opt)

    sizes = Sizes(dataset)
    # sizes为关联网络相关超参

    train_data = Dataset(opt, dataset)
    # 构造训练集

    aucs = []
    auprs = []
    for i in range(opt.validation):
        print('-'*50)
        model = Model(sizes)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, train_data[i], optimizer, opt)
        predict = model(train_data[i])
        print("predict shape",predict.shape)
        auc,aupr = MyAUC(predict,train_data[i])
        aucs.append(auc)
        auprs.append(aupr)
        print("validation {} auc is {}".format(i+1,auc))
        print("validation {} aupr is {}\n".format(i+1,aupr))
    print(aucs,"\n",auprs,sep='')
    print("Average auc is {}".format(sum(aucs)/opt.validation))
    print("Average aupr is {}".format(sum(auprs)/opt.validation))

    torch.cuda.empty_cache()
        
