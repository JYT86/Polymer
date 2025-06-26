import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader

from utils import SMILESDataset, scaling_error
from simple_gnn import GNNRegressor

from sklearn.model_selection import train_test_split

import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('../data/train.csv')
    dataset = SMILESDataset(df, 'FFV')
    train_ids, test_ids = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_set = Subset(dataset, train_ids)
    val_set = Subset(dataset, test_ids)
    train_loader = DataLoader(train_set, 32, True, )
    val_loader = DataLoader(val_set, 32, True, )

    model = GNNRegressor(7, 64, 1).to(device)
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for i in range(10):
        total_loss = 0
        for batch in train_loader:
            batch = batch.cuda()
            output = model(batch.x, batch.edge_index, batch.batch)

            optimizer.zero_grad()
            l = loss_fn(output.reshape(-1), batch.y)
            l.backward()
            optimizer.step()
            total_loss += l.item()
        print(f'epoch {i}: loss: {total_loss/ len(train_loader): .4f}')

        with torch.no_grad():
            preds, trues = [], []
            for batch in val_loader:
                batch = batch.cuda()
                output = model(batch.x, batch.edge_index, batch.batch)
                
                preds.extend(output.detach().cpu().view(-1).tolist())
                trues.extend(batch.y.detach().cpu().view(-1).tolist())
            
            plt.figure()
            plt.plot(trues, preds, 'o', alpha=0.6)
            plt.plot([min(trues), max(trues)],[min(trues), max(trues)], 'r--')
            error_value = scaling_error(pd.DataFrame({'id': range(len(trues)), 'FFV': trues}), pd.DataFrame({'id': range(len(preds)), 'FFV': preds}), 'FFV')
            plt.title(f"Epoch {i+1}, Score:{error_value:.4f}")
            plt.xlabel('True')
            plt.xlabel('Pred')
            plt.savefig(f'../graphs/gnn/epoch_{i}_result.png')
            plt.close()


        


