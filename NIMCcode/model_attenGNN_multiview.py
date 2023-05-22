import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
torch.backends.cudnn.enabled = False

class LayerViewAttention(nn.Module):
    def __init__(self,embedding_dim, node_number, channel):
        super(LayerViewAttention,self).__init__()
        self.globalAvgPool = nn.AvgPool2d((embedding_dim,node_number),(1,1))
        self.fc1 = nn.Linear(channel, channel)  #2层特征*2个视图
        self.fc2 = nn.Linear(channel, channel)

    def forward(self,x):
        channel_att = self.globalAvgPool(x)
        # print("glb:",channel_att.shape)

        channel_att = channel_att.view(channel_att.size(0),-1)
        # print("view:",channel_att.shape)

        channel_att = torch.relu(self.fc1(channel_att))
        channel_att = torch.sigmoid(self.fc2(channel_att))
        channel_att = channel_att.view(channel_att.size(0),channel_att.size(1),1,1)
        # print("sigmoid:",channel_att.shape)
        channel_att = torch.relu(channel_att)   # 求视图和层级注意力
        xx = torch.relu(x * channel_att)        # 对视图和层级加权
        return xx
    
class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.query = nn.Linear(in_channels, out_channels, bias=False)
        self.key = nn.Linear(in_channels, out_channels, bias=False)
        self.value = nn.Linear(in_channels, out_channels, bias=False)
        self.combine_heads = nn.Linear(out_channels, out_channels, bias=False)  

    def forward(self, x):
        x.transpose_(-1, -2)
        _, views, num_features, nodes = x.size()

        # 计算QKV
        query = self.query(x).view(views, num_features, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(views, num_features, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(views, num_features, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = F.softmax(scores, dim=-1)

        # 加权融合
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(views, num_features, -1)
        output = self.combine_heads(attention_output)

        return output

class GCNEncoder(nn.Module):
    def __init__(self,embedding_dim,hidden_channels):
        super(GCNEncoder,self).__init__()
        self.gcn1 = GCNConv(embedding_dim,hidden_channels)
        # self.dropout1 = nn.Dropout(0.2)
        self.gcn2 = GCNConv(hidden_channels,hidden_channels)
        # self.dropout2 = nn.Dropout(0.2)

    def forward(self,x,edge_index,edge_weight):
        x1 = torch.relu(self.gcn1(x,edge_index,edge_weight))
        # x1 = self.dropout1(x1)
        x2 = torch.relu(self.gcn2(x1,edge_index,edge_weight))
        # x2 = self.dropout2(x2)
        return torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)),dim=2)

class SAGEEncoder(nn.Module):
    def __init__(self,embedding_dim,hidden_channels):
        super(SAGEEncoder,self).__init__()
        self.sage1 = SAGEConv(embedding_dim,hidden_channels)
        self.sage2 = SAGEConv(hidden_channels,hidden_channels)
    
    def forward(self,x,edge_index):
        x1 = torch.relu(self.sage1(x,edge_index))
        x2 = torch.relu(self.sage2(x1,edge_index))
        return torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)),dim=2)

class GATEncoder(nn.Module): 
    def __init__(self,embedding_dim,hidden_channels, ha, hb):
        super(GATEncoder,self).__init__()
        self.sage1 = GATConv(embedding_dim,hidden_channels, heads=ha, self_loop=False)
        self.sage2 = GATConv(hidden_channels*ha,hidden_channels, heads=hb, self_loop=False)
    
    def forward(self,x,edge_index):
        x1 = torch.relu(self.sage1(x,edge_index))
        # print(x1.shape)
        x2 = torch.relu(self.sage2(x1,edge_index))
        # print(x2.shape)
        return torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)),dim=2)
    
class GINEncoder(nn.Module):
    def __init__(self,embedding_dim,hidden_channels):
        super(GINEncoder,self).__init__()
        self.sage1 = GINConv(nn.Sequential(nn.Linear(embedding_dim, embedding_dim*2),
                                           nn.Linear(embedding_dim*2, embedding_dim))
                                           , train_eps=True)
        self.sage2 = GINConv(nn.Sequential(nn.Linear(embedding_dim, embedding_dim*2),
                                           nn.Linear(embedding_dim*2, embedding_dim))
                                           , train_eps=True)
    
    def forward(self,x,edge_index):
        x1 = torch.relu(self.sage1(x,edge_index))
        x2 = torch.relu(self.sage2(x1,edge_index))
        return torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)),dim=2)

class ChannelFusion(nn.Module):
    def __init__(self,embedding_dim,out_channels):
        super(ChannelFusion,self).__init__()
        self.out_channels = out_channels

        self.cnn = nn.Conv2d(
            in_channels=4,
            out_channels= out_channels,
            kernel_size=(embedding_dim,1),
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self,x):
        # print(x.shape)
        x = self.cnn(x)
        # print(x.shape)
        x = x.view(self.out_channels,-1).t()
        return x

class MultiViewGNN(nn.Module):
    def __init__(self, args):
        super(MultiViewGNN,self).__init__()
        self.view = 2
        self.layers = 2
        self.embedding_dim = args.embedding_dim
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.out_channels
        self.miRNA_number = 1043
        self.drug_number = 2166
        self.encoder_type = args.encoder_type
        self.attention_type = args.attention_type
        self.fusion_type = args.fusion_type
        self.name = self.attention_type+self.encoder_type+self.fusion_type
        
        if self.encoder_type == 'GCN':
            self.mirna_view1_encoder = GCNEncoder(self.embedding_dim, self.hidden_channels)
            self.mirna_view2_encoder = GCNEncoder(self.embedding_dim, self.hidden_channels)
            self.drug_view1_encoder = GCNEncoder(self.embedding_dim, self.hidden_channels)
            self.drug_view2_encoder = GCNEncoder(self.embedding_dim, self.hidden_channels)
        elif self.encoder_type == 'SAGE':
            self.mirna_view1_encoder = SAGEEncoder(self.embedding_dim, self.hidden_channels)
            self.mirna_view2_encoder = SAGEEncoder(self.embedding_dim, self.hidden_channels)
            self.drug_view1_encoder = SAGEEncoder(self.embedding_dim, self.hidden_channels)
            self.drug_view2_encoder = SAGEEncoder(self.embedding_dim, self.hidden_channels)
        elif self.encoder_type == 'GAT':
            self.mirna_view1_encoder = GATEncoder(self.embedding_dim, self.hidden_channels, 1, 1)
            self.mirna_view2_encoder = GATEncoder(self.embedding_dim, self.hidden_channels, 1, 1)
            self.drug_view1_encoder = GATEncoder(self.embedding_dim, self.hidden_channels, 1, 1)
            self.drug_view2_encoder = GATEncoder(self.embedding_dim, self.hidden_channels, 1, 1)
        elif self.encoder_type == 'GIN':
            self.mirna_view1_encoder = GINEncoder(self.embedding_dim, self.hidden_channels)
            self.mirna_view2_encoder = GINEncoder(self.embedding_dim, self.hidden_channels)
            self.drug_view1_encoder = GINEncoder(self.embedding_dim, self.hidden_channels)
            self.drug_view2_encoder = GINEncoder(self.embedding_dim, self.hidden_channels)

        if self.attention_type == 'LayerView':
            self.mirna_attention = LayerViewAttention(self.embedding_dim, self.miRNA_number,4)
            self.drug_attention = LayerViewAttention(self.embedding_dim, self.drug_number,2)

        elif self.attention_type == 'MultiHead':
            self.mirna_attention = MultiHeadAttention(self.embedding_dim, self.embedding_dim, 2)
            self.drug_attention = MultiHeadAttention(self.embedding_dim, self.embedding_dim, 2)

        if self.fusion_type == 'Cnn':
            self.mirna_fusion = ChannelFusion(self.embedding_dim, self.out_channels)
            self.drug_fusion = ChannelFusion(self.embedding_dim, self.out_channels)

            self.mirna_trans = nn.Sequential(nn.Linear(self.out_channels, self.out_channels*2),
                                                nn.Linear(self.out_channels*2, self.embedding_dim))
            self.drug_trans = nn.Sequential(nn.Linear(self.out_channels, self.out_channels*2),
                                                nn.Linear(self.out_channels*2, self.out_channels))

        elif self.fusion_type == 'Fcn':
            self.mirna_fusion = nn.Sequential(nn.Linear(self.embedding_dim*4, self.embedding_dim*2),nn.ReLU(),
                                                nn.Linear(self.embedding_dim*2, self.embedding_dim),nn.ReLU())
            self.drug_fusion = nn.Sequential(nn.Linear(self.embedding_dim*2, self.embedding_dim*2),nn.ReLU(),
                                                nn.Linear(self.embedding_dim*2, self.embedding_dim),nn.ReLU())

    
    def forward(self, data):
        torch.manual_seed(0)
        mirna_embedding = torch.randn(self.miRNA_number, self.embedding_dim).cuda()
        drug_embedding = torch.randn(self.drug_number, self.embedding_dim).cuda()

        mirna_view1_edge = data['mm_seq']['edge'].cuda()
        # mirna_view1_attr = data['mm_seq']['attr'][data['mm_seq']['edge'][0],data['mm_seq']['edge'][1]].cuda()

        mirna_view2_edge = data['mm_func']['edge'].cuda()
        # mirna_view2_attr = data['mm_func']['attr'][data['mm_func']['edge'][0],data['mm_func']['edge'][1]].cuda()       

        drug_view1_edge = data['dd_seq']['edge'].cuda()
        # drug_view1_attr = data['dd_seq']['attr'][data['dd_seq']['edge'][0],data['dd_seq']['edge'][1]].cuda()

        drug_view2_edge = data['dd_mol']['edge'].cuda()
        # drug_view2_attr = data['dd_mol']['attr'][data['dd_mol']['edge'][0],data['dd_mol']['edge'][1]].cuda()

        mirna_view1_embedding,mirna_view2_embedding = None, None

        if self.encoder_type == 'GCN':
            mirna_view1_attr = data['mm_seq']['attr'][data['mm_seq']['edge'][0],data['mm_seq']['edge'][1]].cuda()
            mirna_view2_attr = data['mm_func']['attr'][data['mm_func']['edge'][0],data['mm_func']['edge'][1]].cuda()       
            drug_view1_attr = data['dd_seq']['attr'][data['dd_seq']['edge'][0],data['dd_seq']['edge'][1]].cuda()
            drug_view2_attr = data['dd_mol']['attr'][data['dd_mol']['edge'][0],data['dd_mol']['edge'][1]].cuda()

            mirna_view1_embedding = self.mirna_view1_encoder(mirna_embedding,mirna_view1_edge,mirna_view1_attr)
            mirna_view2_embedding = self.mirna_view2_encoder(mirna_embedding,mirna_view2_edge,mirna_view2_attr)
        
            drug_view1_embedding = self.drug_view1_encoder(drug_embedding,drug_view1_edge,drug_view1_attr)
            drug_view2_embedding = self.drug_view2_encoder(drug_embedding,drug_view2_edge,drug_view2_attr)

        elif self.encoder_type == 'SAGE':
            mirna_view1_embedding = self.mirna_view1_encoder(mirna_embedding,mirna_view1_edge)
            mirna_view2_embedding = self.mirna_view2_encoder(mirna_embedding,mirna_view2_edge)

            drug_view1_embedding = self.drug_view1_encoder(drug_embedding,drug_view1_edge)
            drug_view2_embedding = self.drug_view2_encoder(drug_embedding,drug_view2_edge)
        
        elif self.encoder_type == 'GAT':
            mirna_view1_embedding = self.mirna_view1_encoder(mirna_embedding,mirna_view1_edge)
            mirna_view2_embedding = self.mirna_view2_encoder(mirna_embedding,mirna_view2_edge)

            drug_view1_embedding = self.drug_view1_encoder(drug_embedding,drug_view1_edge)
            drug_view2_embedding = self.drug_view2_encoder(drug_embedding,drug_view2_edge)
        
        elif self.encoder_type == 'GIN':
            mirna_view1_embedding = self.mirna_view1_encoder(mirna_embedding,mirna_view1_edge)
            mirna_view2_embedding = self.mirna_view2_encoder(mirna_embedding,mirna_view2_edge)

            drug_view1_embedding = self.drug_view1_encoder(drug_embedding,drug_view1_edge)
            drug_view2_embedding = self.drug_view2_encoder(drug_embedding,drug_view2_edge)

        mirna_embedding = torch.cat((mirna_view1_embedding,mirna_view2_embedding),dim=2).unsqueeze(0).transpose(1,3)
        # print(mirna_view1_embedding.shape)
        # mirna_embedding = mirna_view2_embedding.unsqueeze(0).transpose(1,3)
        
        # print(mirna_embedding.shape)
      
        
        # drug_embedding = torch.cat((drug_view1_embedding,drug_view2_embedding),dim=2).unsqueeze(0).transpose(1,3)
        drug_embedding = drug_view2_embedding.unsqueeze(0).transpose(1,3)

        # print(drug_embedding.shape)

        mirna_embedding = self.mirna_attention(mirna_embedding)
        drug_embedding = self.drug_attention(drug_embedding)

        if self.fusion_type == 'Cnn':
            '''with cnn fusion'''
            mirna_embedding = mirna_embedding.squeeze(0)
            drug_embedding = drug_embedding.squeeze(0)

            mirna_embedding = self.mirna_fusion(mirna_embedding)
            drug_embedding = self.drug_fusion(drug_embedding)

            mirna_embedding = self.mirna_trans(mirna_embedding)
            drug_embedding = self.drug_trans(drug_embedding)

        elif self.fusion_type == 'Fcn':
            '''with fcn fusion'''
            mirna_embedding = mirna_embedding.reshape(self.miRNA_number,-1)
            drug_embedding = drug_embedding.reshape(self.drug_number,-1)

            mirna_embedding = self.mirna_fusion(mirna_embedding)
            drug_embedding = self.drug_fusion(drug_embedding)

        elif self.fusion_type == 'Mean':
            '''with mean fusion'''
            drug_embedding = drug_embedding.squeeze(0)
            mirna_embedding = mirna_embedding.squeeze(0)

            drug_embedding = drug_embedding.mean(dim=0)
            drug_embedding = drug_embedding.squeeze().T
            mirna_embedding = mirna_embedding.mean(dim=0)
            mirna_embedding = mirna_embedding.squeeze().T

        # # print("mirna_embedding",mirna_embedding.shape)
        # # print("drug_embedding",drug_embedding.shape)

        return mirna_embedding@drug_embedding.t()