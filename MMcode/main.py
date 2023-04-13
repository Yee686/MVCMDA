from ast import arg
import random
import numpy as np
from param import parameter_parser
from MMGCN import MMGCN
from SSIDDI import SSIGCN
from dataprocessing import data_pro
import torch

import sklearn.metrics as metric
import matplotlib.pyplot as plt

def train(model, train_data, optimizer, opt):
    model.train()
    for epoch in range(0, opt.epoch):
        model.zero_grad()
        score = model(train_data)
        loss = torch.nn.MSELoss(reduction='mean')
        loss = loss(score, train_data['md_p'].cuda())
        loss.backward()
        optimizer.step()
        print("epoch",epoch,": ",loss.item())
    # scoremin, scoremax = score.min(), score.max()
    # score = (score - scoremin) / (scoremax - scoremin)
    return score

class MDargs():
    def __init__(self):
        self.dataset_path = "/mnt/yzy/NIMCGCN/datasets/data(MDA108)"
        self.epoch = 250
        # self.epoch = 2
        self.gcn_layers = 2
        self.out_channels = 48
        self.fm = 128
        self.miRNA_number = 1043
        self.fd = 128
        self.disease_number = 2166
        self.view = 1
        self.validation = 10        

def main():
    # args = parameter_parser()
    aucs = []
    torch.cuda.set_device(1)
    args = MDargs()
    # data = data_pro(args)
    data = torch.load("/mnt/yzy/NIMCGCN/MMcode/dataset.pt")
    # dataset = Dataset(datapro,args.validation)
    print("数据读取成功")
    label = data["md_true"].clone().cpu().numpy().reshape(-1).tolist()
    for i in range(0,args.validation):
        model = MMGCN(args)
        data["md_p"] = data["md_true"].clone()
        data["md_p"][tuple(np.array(data["fold_index"][i]).T)] = 0
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        score = train(model, data, optimizer, args)
        score = score.detach().cpu().numpy()
        np.save("/mnt/yzy/NIMCGCN/Prediction/MMnimc/0323cv{}.npy".format(i+1),score)
        score = score.reshape(-1).tolist()
        fpr,tpr,_ = metric.roc_curve(label,score)
        auc = metric.auc(fpr,tpr)
        aucs.append(auc)
        print("AUC",i,":",auc)
    print("Avarage Auc:",sum(aucs)/args.validation)
    
if __name__ == "__main__":
    main()