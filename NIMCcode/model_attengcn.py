import torch as t
from torch import nn
from torch_geometric.nn import GCNConv

t.backends.cudnn.enabled = False

class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()
        self.name = 'attenGCN'

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.m = sizes.m
        self.d = sizes.d
        self.gcn_layers = sizes.gcn_layers
        self.view = sizes.view
        # self.out_channels = sizes.out_channels
        # self.miRNA_number = sizes.miRNA_number
        # self.drug_number = sizes.drug_number
        
        self.gcn_x1_s = GCNConv(self.fg, self.fg)
        self.gcn_x2_s = GCNConv(self.fg, self.fg)

        self.gcn_y1_s = GCNConv(self.fd, self.fd)
        self.gcn_y2_s = GCNConv(self.fd, self.fd)

        self.globalAvgPool_x = nn.AvgPool2d((self.fg, self.m), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.fd, self.d), (1, 1))
        self.fc1_x = nn.Linear(in_features=self.view*self.gcn_layers,
                             out_features=5*self.view*self.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5*self.view*self.gcn_layers,
                             out_features=self.view*self.gcn_layers)

        self.fc1_y = nn.Linear(in_features=self.view * self.gcn_layers,
                             out_features=5 * self.view * self.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.view * self.gcn_layers,
                             out_features=self.view * self.gcn_layers)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()

        self.cnn_x = nn.Conv1d(in_channels=self.view*self.gcn_layers,
                               out_channels=self.k,
                               kernel_size=(self.fg, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=self.view*self.gcn_layers,
                               out_channels=self.k,
                               kernel_size=(self.fd, 1),
                               stride=1,
                               bias=True)

    def forward(self, input):
        t.manual_seed(1)
        x_m = t.randn(self.m, self.fg).cuda()
        x_d = t.randn(self.d, self.fd).cuda()

        edge_index = input['mm']['edge_index'].cuda()
        edge_attr = input['mm']['data'][input['mm']['edge_index'][0],input['mm']['edge_index'][1]].cuda()
        x_m_s1 = t.relu(self.gcn_x1_s(x_m, edge_index, edge_attr))
        x_m_s2 = t.relu(self.gcn_x2_s(x_m_s1, edge_index, edge_attr))

        edge_index = input['dd']['edge_index'].cuda()
        edge_attr = input['dd']['data'][input['dd']['edge_index'][0],input['dd']['edge_index'][1]].cuda()
        y_d_s1 = t.relu(self.gcn_y1_s(x_d, edge_index, edge_attr))
        y_d_s2 = t.relu(self.gcn_y2_s(y_d_s1, edge_index, edge_attr))

        XM = t.cat((x_m_s1, x_m_s2), 1).t()

        XM = XM.view(1, self.view*self.gcn_layers, self.fg, -1)

        x_channel_attenttion = self.globalAvgPool_x(XM)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        x_channel_attenttion = t.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        XM_channel_attention = x_channel_attenttion * XM

        XM_channel_attention = t.relu(XM_channel_attention)

        YD = t.cat(( y_d_s1, y_d_s2), 1).t()

        YD = YD.view(1, self.view*self.gcn_layers, self.fd, -1)

        y_channel_attenttion = self.globalAvgPool_y(YD)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = t.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,1)
        YD_channel_attention = y_channel_attenttion * YD
        YD_channel_attention = t.relu(YD_channel_attention)



        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.k, self.m).t()



        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.k, self.d).t()


        return x.mm(y.t())
