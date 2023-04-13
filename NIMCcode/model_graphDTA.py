import torch as t
from torch import nn
from torch_geometric.nn import conv


class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.m = sizes.m
        self.d = sizes.d

        self.gcn_x1 = conv.GCNConv(self.fg, self.fg*2)
        self.gcn_x2 = conv.GCNConv(self.fg*2, self.fg*4)
        
        self.gcn_y1 = conv.GCNConv(self.fd, self.fd*2)
        self.gcn_y2 = conv.GCNConv(self.fd*2, self.fd*4)

        self.linear_x_1 = nn.Linear(self.fg*4, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, self.k)

        self.linear_y_1 = nn.Linear(self.fd*4, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, self.k)

    def forward(self, input):
        t.manual_seed(1)
        x_m = t.randn(self.m, self.fg)
        x_d = t.randn(self.d, self.fd)

        '''
        elf.data_set['dd'],             self.data_set['mm'],
        self.data_set['md']['train'],   self.data_set['md']['valid'],
        self.data_set['md_p'],          self.data_set['md_true']
        '''
        
        # x_m   miRNA-miRNA-edge    miRNA-miRNA-matrix-value
        X1 = t.relu(self.gcn_x1(x_m.cuda(), input[1]['edge_index'].cuda(), input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
        X  = t.relu(self.gcn_x2(X1, input[1]['edge_index'].cuda(), input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))

        # x_d   drug-drug-edge      drug-drug-matrix-value
        Y1 = t.relu(self.gcn_y1(x_d.cuda(), input[0]['edge_index'].cuda(), input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))
        Y  = t.relu(self.gcn_y2(Y1, input[0]['edge_index'].cuda(), input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))
    

        x1 = t.relu(self.linear_x_1(X))
        x2 = t.relu(self.linear_x_2(x1))
        x  = t.relu(self.linear_x_3(x2))

        y1 = t.relu(self.linear_y_1(Y))
        y2 = t.relu(self.linear_y_2(y1))
        y  = t.relu(self.linear_y_3(y2))

        return x.mm(y.t())
