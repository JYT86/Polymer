import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader

from utils import SMILESDataset, scaling_error, EarlyStop
from simple_gnn import GNNRegressor

from sklearn.model_selection import train_test_split

import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    properties = ['Tg', 'Density', 'FFV', 'Tc', 'Rg']
    tolerance_dict = {
        'Tg':50,
        'Density':10,
        'FFV':10,
        'Tc':10,
        'Rg':50, 
    }
    df = pd.read_csv('../data/train.csv')
    for property in properties:
        print(f'Now training {property}')
        dataset = SMILESDataset(df, property, False, False)
        train_ids, test_ids = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        
        train_set = Subset(dataset, train_ids)
        val_set = Subset(dataset, test_ids)
        train_loader = DataLoader(train_set, 32, True, )
        val_loader = DataLoader(val_set, 32, True, )
        
        model = GNNRegressor(7, 1024, 1).to(device)
        # model = GNNFusion(7, 1024, 210-13, 1).to(device)
        loss_fn = MSELoss()
        optimizer = Adam(model.parameters(), lr=1e-4)
        earlystop = EarlyStop(50, float(1e-4))
        best = float('inf')

        for i in range(1000):
            total_loss = 0
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.batch)
        
                optimizer.zero_grad()
                if property == 'Tg':
                    l = loss_fn(output.reshape(-1), batch.y/1000)
                else:
                    l = loss_fn(output.reshape(-1), batch.y)
                l.backward()
                optimizer.step()
                total_loss += l.item()
            # print(f'epoch {i}: loss: {total_loss/ len(train_loader): .4f}')
        
            model.eval()
            with torch.no_grad():
                preds, trues = [], []
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch.x, batch.edge_index, batch.batch)
                    if dataset.normalize:
                        preds.extend(dataset.unscale(output.detach().cpu().view(-1)).tolist())
                        trues.extend(dataset.unscale(batch.y.detach().cpu().view(-1)).tolist())
                    else:
                        preds.extend(output.detach().cpu().view(-1).tolist())
                        trues.extend(batch.y.detach().cpu().view(-1).tolist())

                if property == 'Tg':
                    preds = [i*1000 for i in preds]
        
                error_value = scaling_error(pd.DataFrame({'id': range(len(trues)), property: trues}), pd.DataFrame({'id': range(len(preds)), property: preds}), property)
                if error_value < best:
                    best = error_value
                    torch.save(model.state_dict(), f'../checkpoints/best_model_gnn_{property}.pt')
                    print(f"Saved new best model at epoch {i+1} with score {error_value:.4f}")
                    plt.figure()
                    plt.plot(trues, preds, 'o', alpha=0.6)
                    plt.plot([min(trues), max(trues)],[min(trues), max(trues)], 'r--')
                    plt.title(f"Epoch {i+1}, Score:{error_value:.4f}")
                    plt.xlabel('True')
                    plt.ylabel('Pred')
                    plt.savefig(f'../graphs/gnnf/epoch_{property}_result.png')
                    plt.close()
        
                
                if earlystop.check(error_value):
                    print(f'Training early stop due to no improvement for {earlystop.patience} epochs')
                    break


