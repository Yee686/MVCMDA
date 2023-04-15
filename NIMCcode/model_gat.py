import torch as t
from torch import nn
from torch_geometric.nn import conv


class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.name = 'GAT'

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.m = sizes.m
        self.d = sizes.d

        self.gat_x1 = conv.GATConv(self.fg, self.fg, heads = 2, add_self_loops = False)
        self.dropout_x1 = nn.Dropout(0.2)
        self.gat_x2 = conv.GATConv(self.fg*2, self.fg, heads = 1, add_self_loops = False)
        self.dropout_x2 = nn.Dropout(0.2)
        
        self.gat_y1 = conv.GATConv(self.fd, self.fd, heads = 2, add_self_loops = False)
        self.dropout_y1 = nn.Dropout(0.2)
        self.gat_y2 = conv.GATConv(self.fd*2, self.fd, heads = 1, add_self_loops = False)
        self.dropout_y2 = nn.Dropout(0.2)

        self.linear_x_1 = nn.Linear(self.fg, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, self.k)

        self.linear_y_1 = nn.Linear(self.fd, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, self.k)

    def forward(self, input):
        t.manual_seed(1)
        x_m = t.randn(self.m, self.fg).cuda()
        x_d = t.randn(self.d, self.fd).cuda()

        '''
        elf.data_set['dd'],             self.data_set['mm'],
        self.data_set['md']['train'],   self.data_set['md']['valid'],
        self.data_set['md_p'],          self.data_set['md_true']
        '''
        
        # x_m   miRNA-miRNA-edge    miRNA-miRNA-matrix-value
        edge_index_x = input['mm']['edge_index'].cuda()
        # edge_attr = input['mm']['data'][input['mm']['edge_index'][0],input['mm']['edge_index'][1]].cuda()
        # edge_attr = edge_attr.unsqueeze(dim = 1)
        # print(edge_index.shape,edge_attr.shape)
        # print(edge_index_x.shape) ########
        X1 = t.relu(self.gat_x1(x_m, edge_index_x))
        X1 = self.dropout_x1(X1)
        X  = t.relu(self.gat_x2(X1, edge_index_x))
        X  = self.dropout_x2(X)

        # x_d   drug-drug-edge      drug-drug-matrix-value
        edge_index_y = input['dd']['edge_index'].cuda()
        # edge_attr = input['dd']['data'][input['dd']['edge_index'][0],input['dd']['edge_index'][1]].cuda()
        # edge_attr = edge_attr.unsqueeze(dim = 1)
        # print(x_d.device, edge_index_y.device)
        Y1 = t.relu(self.gat_y1(x_d, edge_index_y))
        Y1 = self.dropout_y1(Y1)
        Y  = t.relu(self.gat_y2(Y1, edge_index_y ))
        Y  = self.dropout_y2(Y)
    

        x1 = t.relu(self.linear_x_1(X))
        x2 = t.relu(self.linear_x_2(x1))
        x  = t.relu(self.linear_x_3(x2))

        y1 = t.relu(self.linear_y_1(Y))
        y2 = t.relu(self.linear_y_2(y1))
        y  = t.relu(self.linear_y_3(y2))

        return x.mm(y.t())
