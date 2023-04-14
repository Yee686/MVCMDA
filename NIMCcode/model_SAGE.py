import torch as t
from torch import nn
from torch_geometric.nn import conv


class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.name = 'SAGE'

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.m = sizes.m
        self.d = sizes.d

        self.sage_x1 = conv.SAGEConv(self.fg, self.fg*2)
        self.sage_y1 = conv.SAGEConv(self.fd, self.fd*2)
        self.sage_x2 = conv.SAGEConv(self.fg*2, self.fg)
        self.sage_y2 = conv.SAGEConv(self.fd*2, self.fd)

        self.linear_x_1 = nn.Linear(self.fg, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, self.k)

        self.linear_y_1 = nn.Linear(self.fd, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, self.k)

    def forward(self, input):
        t.manual_seed(1)
        x_m = t.randn(self.m, self.fg)
        x_d = t.randn(self.d, self.fd)

        
        X1 = t.relu(self.sage_x1(x_m.cuda(), input['mm']['edge_index'].cuda()))
        X  = t.relu(self.sage_x2(X1, input['mm']['edge_index'].cuda()))

        Y1 = t.relu(self.sage_y1(x_d.cuda(), input['dd']['edge_index'].cuda()))
        Y  = t.relu(self.sage_y2(Y1, input['dd']['edge_index'].cuda())) 

        x1 = t.relu(self.linear_x_1(X))
        x2 = t.relu(self.linear_x_2(x1))
        x  = t.relu(self.linear_x_3(x2))

        y1 = t.relu(self.linear_y_1(Y))
        y2 = t.relu(self.linear_y_2(y1))
        y  = t.relu(self.linear_y_3(y2))

        return x.mm(y.t())
