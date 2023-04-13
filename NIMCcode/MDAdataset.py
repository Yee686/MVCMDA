import torch 
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import csv

class MDADataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_file_name = ["m-d.csv","d-d.csv","m-m.csv"]
        data_root_path = "/mnt/yzy/NIMCGCN/datasets/data(MDA108)/" 
        return [ data_root_path + name for name in data_file_name]

    @property
    def processed_file_names(self):
        return ['MDAdata.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

### 读取CSV文件
def read_csv(path):
    with open(path, 'r', newline='',encoding="UTF-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.FloatTensor(md_data)
### 获取边信息
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

if __name__ == "__main__":
