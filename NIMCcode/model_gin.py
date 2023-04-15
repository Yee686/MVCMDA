import torch as t
from torch import nn
from torch_geometric.nn import conv


class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.name = 'GIN'

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.m = sizes.m
        self.d = sizes.d

        self.gin_x1 = conv.GINConv(nn.Sequential(nn.Linear(self.fg, self.fg*2),nn.Linear(self.fg*2, self.fg*4)),train_eps=True)
        self.dropout_x1 = nn.Dropout(0.2)
        self.gin_x2 = conv.GINConv(nn.Sequential(nn.Linear(self.fg*4, self.fg*2),nn.Linear(self.fg*2, self.fg)),train_eps=True)
        self.dropout_x2 = nn.Dropout(0.2)

        self.gin_y1 = conv.GINConv(nn.Sequential(nn.Linear(self.fg, self.fg*2),nn.Linear(self.fg*2, self.fg*4)),train_eps=True)
        self.dropout_y1 = nn.Dropout(0.2)
        self.gin_y2 = conv.GINConv(nn.Sequential(nn.Linear(self.fg*4, self.fg*2),nn.Linear(self.fg*2, self.fg)),train_eps=True)
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

         # x_m   miRNA-miRNA-edge    miRNA-miRNA-matrix-value
        edge_index = input['mm']['edge_index'].cuda()
        # edge_attr = input['mm']['data'][input['mm']['edge_index'][0],input['mm']['edge_index'][1]].cuda()
        X1 = t.relu(self.gin_x1(x_m, edge_index))
        x1 = self.dropout_x1(X1)
        X  = t.relu(self.gin_x2(X1, edge_index))
        X  = self.dropout_x2(X)

        # x_d   drug-drug-edge      drug-drug-matrix-value
        edge_index = input['dd']['edge_index'].cuda()
        # edge_attr = input['dd']['data'][input['dd']['edge_index'][0],input['dd']['edge_index'][1]].cuda()
        Y1 = t.relu(self.gin_y1(x_d, edge_index ))
        Y1 = self.dropout_y1(Y1)
        Y  = t.relu(self.gin_y2(Y1, edge_index ))
        Y  = self.dropout_y2(Y)

        x1 = t.relu(self.linear_x_1(X))
        x2 = t.relu(self.linear_x_2(x1))
        x  = t.relu(self.linear_x_3(x2))

        y1 = t.relu(self.linear_y_1(Y))
        y2 = t.relu(self.linear_y_2(y1))
        y  = t.relu(self.linear_y_3(y2))

        return x.mm(y.t())
        # return x2.mm(y2.t())

