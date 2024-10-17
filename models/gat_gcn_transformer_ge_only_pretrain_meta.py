import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class FeatureEncoder(torch.nn.Module):
    def __init__(self, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.1):

        super(FeatureEncoder, self).__init__()

        # self.n_output = n_output
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=num_features_xd, nhead=1, dropout=0.5)
        self.ugformer_layer_1 = nn.TransformerEncoder(self.encoder_layer_1, 1)
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=num_features_xd*10, nhead=1, dropout=0.5)
        self.ugformer_layer_2 = nn.TransformerEncoder(self.encoder_layer_2, 1)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line ge feature
        self.conv_xt_ge_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_ge_1 = nn.MaxPool1d(3)
        self.conv_xt_ge_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_ge_2 = nn.MaxPool1d(3)
        self.conv_xt_ge_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt_ge_3 = nn.MaxPool1d(3)
        # self.fc1_xt_ge = nn.Linear(4224, output_dim)
        self.fc1_xt_ge = nn.Linear(79616, 1024)
        self.fc2_xt_ge = nn.Linear(1024, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch       # *78, N*2, B
        x = torch.unsqueeze(x, 1)            # torch.Size([14848, 1, 78])
        x = self.ugformer_layer_1(x)         # torch.Size([14848, 1, 78])
        x = torch.squeeze(x,1)               # torch.Size([14848, 78])
        x = self.conv1(x, edge_index)        # 14848, 780
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.ugformer_layer_2(x)         # 14848,1,780
        x = torch.squeeze(x,1)
        x = self.conv2(x, edge_index)        # 14848,780
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)      # B, 1560
        x = self.relu(self.fc_g1(x))         # B, 1500
        x = self.dropout(x)                
        x = self.fc_g2(x)                    # B， 128

        # target_mut input feed-forward:
        target_ge = data.target_ge           # torch.Size([512, 16906])
        target_ge = target_ge[:,None,:]
        conv_xt_ge = self.conv_xt_ge_1(target_ge)    # torch.Size([B, 32, 16899])
        # cell_node = conv_xt_ge
        # cell_node.retain_grad()
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_1(conv_xt_ge)   # torch.Size([512, 32, 5633])
        conv_xt_ge = self.conv_xt_ge_2(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_2(conv_xt_ge)
        conv_xt_ge = self.conv_xt_ge_3(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_3(conv_xt_ge)    # torch.Size([B, 128, 33])
        xt_ge = conv_xt_ge.view(-1, conv_xt_ge.shape[1] * conv_xt_ge.shape[2])      # torch.Size([B, 4224])
        xt_ge = self.fc1_xt_ge(xt_ge)                 # B, 128
        ########################
        xt_ge = self.relu(xt_ge)
        xt_ge = self.dropout(xt_ge)
        xt_ge = self.fc2_xt_ge(xt_ge)
        ######################################################
        # concat
        xc = torch.cat((x, xt_ge), 1)
        # add some dense layers

        return xc
        # return out
class RelationNetwork(torch.nn.Module):
    def __init__(self, n_output=2, output_dim=128, dropout=0.1):

        super(RelationNetwork, self).__init__()
        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)     
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, xc, param = None):
        if param is not None:
            fc1_w = param[0]
            fc1_b = param[1]
            xc = F.linear(xc, fc1_w, fc1_b) 
            xc = self.relu(xc)
            xc = self.dropout(xc)
            fc2_w = param[2]
            fc2_b = param[3]
            xc = F.linear(xc, fc2_w, fc2_b)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            out_w = param[4]
            out_b = param[5]
            out = F.linear(xc, out_w, out_b)
            out = nn.Sigmoid()(out)     # B, 1
        else:
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            out = self.out(xc)          # B, 1

            out = nn.Sigmoid()(out)     # B, 1
        return out
    
class OneClassifier(torch.nn.Module):
    def __init__(self, n_output=2, output_dim=128):

        super(OneClassifier, self).__init__()
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([2, 256]))  
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(2))
        self.vars.append(self.fc1_w)
        self.vars.append(self.fc1_b)
    def forward(self, xc, param = None):
        if param is not None:
            fc1_w = param[0]
            fc1_b = param[1]
            out = F.linear(xc, fc1_w, fc1_b) 
        else:
            param = self.vars
            out = F.linear(xc, param[0], param[1])
        return out
    
    def parameters(self):
        return self.vars

class FeatureRelationNetwork(torch.nn.Module):
    def __init__(self, n_output=2, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.1):

        super(FeatureRelationNetwork, self).__init__()

        self.FeatureEncoder = FeatureEncoder(num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2)
        self.RelationNetwork = RelationNetwork(n_output=n_output, output_dim=output_dim, dropout=0.2)    
        self.OneClassifier = OneClassifier(n_output=2, output_dim=output_dim)
    def forward(self, data):
        xc = self.FeatureEncoder(data)
        out = self.RelationNetwork(xc)
        return out, xc

if __name__ == '__main__':
    from captum.attr import IntegratedGradients

    
