

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
        self.out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.out(x)
    

if __name__ == "__main__":
    import pandas as pd
    from torch_geometric.loader import DataLoader
    

    df = pd.read_csv('../data/train.csv')
    dataset = SMILESDataset(df, 'FFV')
    dataloader = DataLoader(dataset, 32, True, )

    model = GNNRegressor(7, 64, 1).cuda()
    for batch in dataloader:
        batch = batch.cuda()
        output = model(batch.x, batch.edge_index, batch.batch)
        print(output)
        break


        

