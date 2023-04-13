import csv
from os import sep
from scipy.sparse.construct import rand
import torch
import random
import numpy as np

from sklearn.model_selection import KFold

def read_csv(path):
    with open(path, 'r', newline='',encoding="UTF-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def data_pro(args):
    dataset = dict()

    dataset['md_p'] = read_csv(args.dataset_path + '/m-d.csv')
    dataset['md_true'] = read_csv(args.dataset_path + '/m-d.csv')



    zero_index = []
    one_index = []
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])

    random.seed(0)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    print("One_index_shape:{}\nzero_index_shape:{}".format(torch.LongTensor(one_index).shape,torch.LongTensor(zero_index).shape))
    dataset['md'] = [torch.LongTensor(one_index),torch.LongTensor(zero_index)]

    onei = np.array(one_index)
    kfold = args.validation
    total = onei.shape[0]
    cvsize = int(total/kfold)
    print(total,cvsize)
    temp = np.array(onei[:total-total%kfold]).reshape(kfold,cvsize,-1).tolist()
    
    # temp[kfold-1] += onei[total-total%kfold]
    temp = np.array(temp)
    print(temp.shape)
    print(kfold,total)
    dataset["fold_index"] = temp
    # "disease functional sim"
    # dd_f_matrix = read_csv(args.dataset_path + '/d_d_f.csv')
    # dd_f_edge_index = get_edge_index(dd_f_matrix)
    # dataset['dd_f'] = {'data_matrix': dd_f_matrix, 'edges': dd_f_edge_index}

    "disease semantic sim"
    # 药物功能结构相似度
    dd_s_matrix = read_csv(args.dataset_path + '/d-d.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    print("drug-drug-edge:\n",dd_s_edge_index.shape,sep="")
    dataset['dd_s'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    # "miRNA functional sim"
    # mm_f_matrix = read_csv(args.dataset_path + '/m_m_f.csv')
    # mm_f_edge_index = get_edge_index(mm_f_matrix)
    # dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}

    "miRNA sequence sim"
    # miRNA语言相似度
    mm_s_matrix = read_csv(args.dataset_path + '/m-m.csv')
    mm_s_edge_index = get_edge_index(mm_s_matrix)
    print("miRNA-miRNA-edge:\n",mm_s_edge_index.shape,sep="")
    dataset['mm_s'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    return dataset

class MDargs():
    def __init__(self):
        self.dataset_path = "/mnt/yzy/NIMCGCN/datasets/data(MDA108)"
        # self.epoch = 200
        self.epoch = 2
        self.gcn_layers = 2
        self.out_channels = 48
        self.fm = 128
        self.miRNA_number = 1043
        self.fd = 128
        self.disease_number = 2166
        self.view = 1
        self.validation = 10
        
    
if __name__ == "__main__":
    args = MDargs()
    datapro = data_pro(args)
    # torch.save(datapro,"MMcode/dataset.pt")
    # dataset = Dataset(datapro,args.validation)
