


import torch
from torch_geometric.data import Data, Dataset
from pandas import DataFrame

import rdkit
from rdkit import Chem

import numpy as np
import pandas as pd


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
    def __init__(self, df: DataFrame, property: str = 'FFV', normalize=False):
        super().__init__()
        self.smiles = df[df[property].notna()]['SMILES'].tolist()
        self.y = df[df[property].notna()][property].tolist()
        self.normalize = normalize

        # 归一化处理
        if normalize:
            self.y_min = min(self.y)
            self.y_max = max(self.y)
            self.y = [(v - self.y_min) / (self.y_max - self.y_min) for v in self.y]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        graph_data = mol_to_graph(self.smiles[idx], self.y[idx])
        return graph_data
    
    def unscale(self, y_scaled_tensor):
        return y_scaled_tensor * (self.y_max - self.y_min) + self.y_min

class EarlyStop:
    def __init__(self, patience: int, tolerance: float = 1e-5):
        self.patience = patience
        self.tolerance = tolerance
        self.best_score = None       # 保存目前最好的 score
        self.counter = 0             # 记录连续“没进步”的次数

    def check(self, score: float):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.best_score - score > self.tolerance:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
        

#################################################################
class ParticipantVisibleError(Exception):
    pass


# These values are from the train data.
MINMAX_DICT =  {
        'Tg': [-148.0297376, 472.25],
        'FFV': [0.2269924, 0.77709707],
        'Tc': [0.0465, 0.524],
        'Density': [0.748691234, 1.840998909],
        'Rg': [9.7283551, 34.672905605],
    }
NULL_FOR_SUBMISSION = -9999


def scaling_error(labels, preds, property):
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)


def get_property_weights(labels):
    property_weight = []
    for property in MINMAX_DICT.keys():
        valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION)
        property_weight.append(valid_num)
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / property_weight)
    return (property_weight / np.sum(property_weight)) * len(property_weight)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute weighted Mean Absolute Error (wMAE) for the Open Polymer challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'id': unique identifier for each sequence
      - Columns 'Tg', 'FFV', 'Tc', 'Density', 'Rg' as the predicted targets

    Examples
    --------
     import pandas as pd
     row_id_column_name = "id"
     solution = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4})
     submission = pd.DataFrame({'id': range(4), 'Tg': [0.5]*4, 'FFV': [0.5]*4, 'Tc': [0.5]*4, 'Density': [0.5]*4, 'Rg': [0.5]*4})
     round(score(solution, submission, row_id_column_name=row_id_column_name), 4)
    0.2922
     submission = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4} )
     score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    """
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []
    property_weights = get_property_weights(solution[chemical_properties])
    for property in chemical_properties:
        is_labeled = solution[property] != NULL_FOR_SUBMISSION
        property_maes.append(scaling_error(solution.loc[is_labeled, property], submission.loc[is_labeled, property], property))

    if len(property_maes) == 0:
        raise RuntimeError('No labels')
    return float(np.average(property_maes, weights=property_weights))

#################################################################


if __name__ == '__main__':
    import pandas as pd
    from torch_geometric.loader import DataLoader

    df = pd.read_csv('../data/train.csv')
    dataset = SMILESDataset(df, 'FFV')
    dataloader = DataLoader(dataset, 32, True, )
    for batch in dataloader:
        print(batch.x.shape)
        print(batch.edge_attr)
        print(batch.batch)
