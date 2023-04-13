from torch import nn, optim
from model import Model
from prepareData import prepare_data
import numpy as np
import torch


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation      
        print("-One--",self.data_set['md']['train'][0].shape)
        print("-Zero-",self.data_set['md']['train'][1].shape)
    # def __getitem__(self, index):
    #     return (self.data_set['dd'], self.data_set['mm'],
    #             self.data_set['md']['train'], None,
    #             self.data_set['md_p'], self.data_set['md_true'])

    def __getitem__(self, index):

        return (self.data_set['dd'], self.data_set['mm'],
                self.data_set['md']['train'],None,
                self.data_set['md_p'], self.data_set['md_true'])

    def __len__(self):
        return self.nums

class Config(object):
    def __init__(self):
        #self.data_path = '../data'
        self.data_path = '/mnt/yzy/NIMCGCN/datasets/data(MDA108)'
        self.validation = 5
        #self.save_path = '../data'
        self.save_path = ' '
        self.epoch = 2
        self.alpha = 0.2


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-opt.alpha)*loss_sum[one_index].sum()+opt.alpha*loss_sum[zero_index].sum()


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
    one_index = train_data[2][0].cuda().t().tolist()
    zero_index = train_data[2][1].cuda().t().tolist()
    # .t() 是 .transpose函数的简写版本，但两者都只能对2维以下的tensor进行转置
    print(torch.tensor(one_index).shape,torch.tensor(zero_index).shape)
    def train_epoch():
        model.zero_grad()
        score = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[4].cuda(), score)
        loss.backward()
        optimizer.step()
        return loss
    for epoch in range(1, opt.epoch+1):
        train_reg_loss = train_epoch()
        print(epoch,":",train_reg_loss.item()/(len(one_index[0])+len(zero_index[0])))


opt = Config()


def main():
    dataset = prepare_data(opt)

    # dataset = {'md':{'train':[[one_tensor],[zero_tensor]]},
    #            'md_p':m-d,
    #            'md_true':m-d,
    #            'dd':{'data':d-d,'edge_index':边},
    #            'mm':{'data':m-m,'edge_index':边}}

    sizes = Sizes(dataset)
    # sizes为关联网络相关超参

    train_data = Dataset(opt, dataset)
    # 构造训练集

    for i in range(opt.validation):
        print('-'*50)
        model = Model(sizes)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, train_data[i], optimizer, opt)


if __name__ == "__main__":
    main()