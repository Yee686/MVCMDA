import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv
torch.backends.cudnn.enabled = False

class LayerViewAttention(nn.Module):
    def __init__(self,embedding_dim, node_number):
        super(LayerViewAttention,self).__init__()

        self.globalAvgPool = nn.AvgPool2d((embedding_dim,node_number),(1,1))

        self.fc1 = nn.Linear(4, 4)  #2层特征 2个视图
        self.fc2 = nn.Linear(4, 4)

    def forward(self,x):
        channel_att = self.globalAvgPool(x)
        channel_att = channel_att.view(channel_att.size(0),-1)
        channel_att = torch.relu(self.fc1(channel_att))
        channel_att = torch.sigmoid(self.fc2(channel_att))
        channel_att = channel_att.view(channel_att.size(0),channel_att.size(1),1,1)
        xx = torch.relu(channel_att)
        return xx


class GCNEncoder(nn.Module):
    def __init__(self,embedding_dim,hidden_channels):
        super(GCNEncoder,self).__init__()
        self.gcn1 = GCNConv(embedding_dim,hidden_channels)
        self.gcn2 = GCNConv(hidden_channels,hidden_channels)

    def forward(self,x,edge_index,edge_weight):
        x = torch.relu(self.gcn1(x,edge_index,edge_weight))
        x = torch.relu(self.gcn2(x,edge_index,edge_weight))
        return x

class SAGEEncoder(nn.Module):
    def __init__(self,embedding_dim,hidden_channels):
        super(SAGEEncoder,self).__init__()
        self.sage1 = SAGEConv(embedding_dim,hidden_channels)
        self.sage2 = SAGEConv(hidden_channels,hidden_channels)
    
    def forward(self,x,edge_index,edge_weight):
        x = torch.relu(self.sage1(x,edge_index,edge_weight))
        x = torch.relu(self.sage2(x,edge_index,edge_weight))
        return x

class ChannelFusion(nn.Module):
    def __init__(self,embedding_dim,out_channels):
        super(ChannelFusion,self).__init__()
        self.out_channels = out_channels
        self.cnn = nn.Conv1d(in_channels=embedding_dim,
                             out_channels=out_channels,
                             kernel_size=(embedding_dim,1),
                             stride=1,
                             bias=True)

    def forward(self,x):
        x = self.cnn(x)
        x = x.veiw(self.out_channels,-1).t()
        return x

class MultiViewGNN(nn.Module):
    def __init__(self, args):
        super(MultiViewGNN,self).__init__()
        self.view = args.view
        self.gnn_layers = args.gcn_layers
        self.embedding_dim = args.embedding_dim
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels
        self.miRNA_number = args.miRNA_number
        self.drug_number = args.drug_number
        self.encoder_type = args.encoder_type
        
        if self.encoder_type == 'GCN':
            Encoder = GCNEncoder
        elif self.encoder_type == 'SAGE':
            Encoder = SAGEEncoder
        self.mirna_view1_encoder = Encoder(self.embedding_dim, self.hidden_channels)
        self.mirna_view2_encoder = Encoder(self.embedding_dim, self.hidden_channels)
        self.drug_view1_encoder = Encoder(self.embedding_dim, self.hidden_channels)
        self.drug_view2_encoder = Encoder(self.embedding_dim, self.hidden_channels)

        self.mirna_attention = LayerViewAttention(self.embedding_dim, self.miRNA_number)
        self.drug_attention = LayerViewAttention(self.embedding_dim, self.drug_number)

        self.mirna_fusion = ChannelFusion(self.embedding_dim, self.out_channels)
        self.drug_fusion = ChannelFusion(self.embedding_dim, self.out_channels)

    def forward(self, data):
        torch.manual_seed(0)
        mirna_embedding = torch.randn(self.miRNA_number, self.embedding_dim)
        drug_embedding = torch.randn(self.drug_number, self.embedding_dim)

        mirna_view1_edge = data['mm_seq']['edge'][0],data['mm_seq']['edge'][1]
        mirna_view1_attr = data['mm_seq']['edge'][0

class MMGCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(MMGCN, self).__init__()
        self.args = args
        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)

        self.gcn_y1_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)

        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))
        self.fc1_x = nn.Linear(in_features=self.args.view*self.args.gcn_layers,
                             out_features=5*self.args.view*self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5*self.args.view*self.args.gcn_layers,
                             out_features=self.args.view*self.args.gcn_layers)

        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                             out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
                             out_features=self.args.view * self.args.gcn_layers)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()

        self.cnn_x = nn.Conv1d(in_channels=self.args.view*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=self.args.view*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fd, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        x_m = torch.randn(self.args.miRNA_number, self.args.fm)
        x_d = torch.randn(self.args.disease_number, self.args.fd)


        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        # 节点特征 边 边特征
        x_m_s1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))
        x_m_s2 = torch.relu(self.gcn_x2_s(x_m_s1, data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))

        y_d_f1 = torch.relu(self.gcn_y1_f(x_d.cuda(), data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))

        y_d_s1 = torch.relu(self.gcn_y1_s(x_d.cuda(), data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))

        XM = torch.cat((x_m_f1, x_m_f2, x_m_s1, x_m_s2), 1).t()
        # XM = torch.cat((x_m_s1, x_m_s2), 1).t()

        XM = XM.view(1, self.args.view*self.args.gcn_layers, self.args.fm, -1)

        x_channel_attenttion = self.globalAvgPool_x(XM)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        XM_channel_attention = x_channel_attenttion * XM

        XM_channel_attention = torch.relu(XM_channel_attention)

        YD = torch.cat((y_d_f1, y_d_f2, y_d_s1, y_d_s2), 1).t()
        # YD = torch.cat(( y_d_s1, y_d_s2), 1).t()

        YD = YD.view(1, self.args.view*self.args.gcn_layers, self.args.fd, -1)

        y_channel_attenttion = self.globalAvgPool_y(YD)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = torch.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,1)
        YD_channel_attention = y_channel_attenttion * YD
        YD_channel_attention = torch.relu(YD_channel_attention)



        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.out_channels, self.args.miRNA_number).t()



        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.out_channels, self.args.disease_number).t()


        return x.mm(y.t())