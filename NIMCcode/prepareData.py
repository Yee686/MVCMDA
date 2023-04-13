import csv
import torch as t
import random
import numpy as np

def read_csv(path):
    with open(path, 'r', newline='',encoding="UTF-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='',encoding="UTF-8-sig") as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def prepare_data(opt):
    #读m-d关联矩阵
    dataset = dict()
    dataset['md_p'] = read_csv(opt.data_path + '/m-d.csv')
    dataset['md_true'] = read_csv(opt.data_path + '/m-d.csv')

    zero_index = []
    one_index = []
    #分别记录值为0与1的索引
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    # 打乱
    random.seed(0)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = t.LongTensor(zero_index)
    one_tensor = t.LongTensor(one_index)

    # """将验证1边抹去做法"""
    # onei = np.array(one_index)
    # kfold = opt.validation
    # total = onei.shape[0]
    # cvsize = int(total/kfold)
    # print(total,cvsize)
    # temp = np.array(onei[:total-total%kfold]).reshape(kfold,cvsize,-1).tolist()
    
    # # temp[kfold-1] += onei[total-total%kfold]
    # temp = np.array(temp)
    # print(temp.shape)
    # print(kfold,total)
    # dataset["fold_index"] = temp
    # """将验证1边抹去做法"""

    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]

    dd_matrix = read_csv(opt.data_path + '/d-d.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

    mm_matrix = read_csv(opt.data_path + '/m-m.csv')
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    return dataset

class Opt():
    def __init__(self):        
        self.data_path = '/mnt/yzy/NIMCGCN/datasets/data(MDA108)'
        self.validation = 10
        #self.save_path = '../data'
        self.save_path = ' '
        # self.epoch = 200
        self.epoch = 200
        self.alpha = 0.2

if __name__ == "__main__":
    opt = Opt()
    dataset = prepare_data(opt)
    t.save(dataset,"NIMCcode/dataset.pt")