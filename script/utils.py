


import torch
from torch_geometric.data import Data, Dataset
from pandas import DataFrame

import rdkit
from rdkit import Chem


def mol_to_graph(smiles: str, y: float) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    
    atom_features = []
    for atom in mol.GetAtoms():
            atom_features.append([
            atom.GetAtomicNum(),                      # 原子序号（C=6, O=8...）
            atom.GetTotalDegree(),                    # 键连数
            atom.GetFormalCharge(),                   # 形式电荷
            atom.GetTotalNumHs(),                     # 氢原子数（包括显式/隐式）
            int(atom.GetIsAromatic()),                # 是否为芳香性
            int(atom.GetHybridization()),             # 杂化类型（SP=0, SP2=1, ...）
            int(atom.IsInRing()),                     # 是否在环中    
        ])

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble() # 键型

        edge_index += [[i, j], [j, i]]
        edge_attr += [[bond_type], [bond_type]]

    data = Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float)
    )

    return data


class SMILESDataset(Dataset):
    def __init__(self, df: DataFrame, property: str = 'FFV'):
        super().__init__()
        self.smiles = df[df[property].notna()]['SMILES'].tolist()
        self.y = df[df[property].notna()][property].tolist()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        graph_data = mol_to_graph(self.smiles[idx], self.y[idx])
        return graph_data
        

if __name__ == '__main__':
    import pandas as pd
    from torch_geometric.loader import DataLoader

    df = pd.read_csv('../data/train.csv')
    dataset = SMILESDataset(df, 'FFV')
    dataloader = DataLoader(dataset, 32, True, )
    for batch in dataloader:
        print(batch.x.shape)
        print(batch.edge_attr)
