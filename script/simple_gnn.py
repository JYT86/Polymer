import torch
from torch_geometric.nn import GCNConv, global_mean_pool, GINEConv, GATConv
from torch import nn
from torch.nn import functional as F
import pandas as pd
from torch_geometric.loader import DataLoader
from utils import SMILESDataset

class GNNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.out = torch.nn.Linear(hidden_dim, out_dim)
        self.out = torch.nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/2), out_dim)
        )

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.out(x)
    
class GNNFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, graph_feature_dim, smiles_embedding_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.gnn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.graph_feature_proj = nn.Linear(graph_feature_dim, hidden_dim)
        self.smiles_proj = nn.Linear(smiles_embedding_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.alpha_param = nn.Parameter(torch.tensor(0.5))
        self.beta_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, batch, graph_feature, smiles_embedding):

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)

        alpha = torch.sigmoid(self.alpha_param)
        beta = torch.sigmoid(self.beta_param)

        gnn_feat = self.gnn_proj(x)
        graph_feat = self.graph_feature_proj(graph_feature)
        smiles_feat = self.smiles_proj(smiles_embedding)

        gnn_feat = F.layer_norm(gnn_feat, gnn_feat.shape[1:])
        graph_feat = F.layer_norm(graph_feat, graph_feat.shape[1:])
        smiles_feat = F.layer_norm(smiles_feat, smiles_feat.shape[1:])

        fused = torch.cat([gnn_feat * alpha, smiles_feat * beta, graph_feat * (1 - alpha - beta)], dim=1)
        return self.fusion(fused)

if __name__ == "__main__":
    
    df = pd.read_csv('../data/train.csv')
    dataset = SMILESDataset(df, 'FFV', False, True)
    dataloader = DataLoader(dataset, 32, True, )

    # model = GNNRegressor(7, 64, 1).cuda()
    model = GNNFusion(7, 1024, 210-13, 768, 1).cuda()
    for batch, graph_feat, smiles_embedding in dataloader:
        batch = batch.cuda()
        graph_feat = graph_feat.float().cuda()
        smiles_embedding = smiles_embedding.float().cuda()

        output = model(batch.x, batch.edge_index, batch.batch, graph_feat, smiles_embedding)
        print('nan', torch.isnan(output).any().item())
        print('inf', torch.isinf(output).any().item())