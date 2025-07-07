


import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader

from utils import SMILESDataset, scaling_error, EarlyStop
from simple_gnn import GNNRegressor, GNNFusion

from sklearn.model_selection import train_test_split

import pandas as pd
from typing import Optional, Literal 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def prediction(pred_df:pd.DataFrame, model_to_use: Optional[Literal["GNNFusion", "GNNRegressor"]]="GNNFusion"):
    dataset = SMILESDataset(pred_df, None)
    dataloader = DataLoader(dataset, 32, False, )
    if model_to_use == 'GNNFusion':
        model = GNNFusion(7, 1024, 210-13, 1).to(device)
    else:
        model = GNNRegressor(7, 1024, 1).to(device)
    properties = ['Tg', 'Density', 'FFV', 'Tc', 'Rg']
    for property in properties:
        if model_to_use == 'GNNFusion':
            model.load_state_dict(torch.load(f'checkpoints/best_model_GCN3_alpha_layernorm_{property}.pt'))
        else:
            model.load_state_dict(torch.load(f'checkpoints/best_model_{property}.pt'))
        model.eval()
        with torch.no_grad():
            preds = []
            for batch in dataloader:
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.batch)
                if dataset.normalize:
                    preds.extend(dataset.unscale(output.detach().cpu().view(-1)).tolist())
                else:
                    preds.extend(output.detach().cpu().view(-1).tolist())

        pred_df[property] = preds

    return pred_df
if __name__ == "__main__":
    test_df = prediction('../data/test.csv')