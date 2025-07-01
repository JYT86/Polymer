

import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch import nn
from torch.nn import functional as F

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
    def __init__(self, in_dim, hidden_dim, graph_feature_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.out = torch.nn.Linear(hidden_dim, out_dim)
        
        self.gnn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.graph_feature_proj = nn.Linear(graph_feature_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch, graph_feature):

        # print("Graph feature NaN:", torch.isnan(graph_feature).any().item())
        # print("Graph feature Inf:", torch.isinf(graph_feature).any().item())

        # print()

        # print("Node feature NaN:", torch.isnan(x).any().item())
        # print("Node feature Inf:", torch.isinf(x).any().item())
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        gnn_feat = self.gnn_proj(x)
        graph_feat = self.graph_feature_proj(graph_feature)
    
        fused = torch.cat([gnn_feat, graph_feat], dim=1)
        return self.fusion(fused)

if __name__ == "__main__":
    import pandas as pd
    from torch_geometric.loader import DataLoader
    

    df = pd.read_csv('../data/train.csv')
    dataset = SMILESDataset(df, 'FFV', False, True)
    dataloader = DataLoader(dataset, 32, True, )

    # model = GNNRegressor(7, 64, 1).cuda()
    model = GNNFusion(7, 1024, 210-13, 1).cuda()
    for batch, graph_feat in dataloader:

        batch = batch.cuda()
        graph_feat = graph_feat.float().cuda()
        output = model(batch.x, batch.edge_index, batch.batch, graph_feat)
        print('nan', torch.isnan(output).any().item())
        print('inf', torch.isinf(output).any().item())
        


        

