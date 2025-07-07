# NeurIPS - Open Polymer Prediction 2025

## Structure of the Dataset

- `data/` — Folder for storing the training and test data from the competition.
- `script/` — Folder containing all source code.
- `graphs/` — Folder for saving graphical results and visualizations.

## Getting Started

This project implements two types of models: `GNNRegressor` and `GNNFusion`, both located in `script/simple_gnn.py`.

- `GNNRegressor` uses only atom-level features extracted via RDKit.
- `GNNFusion` incorporates both atom-level features and graph-level features to enhance prediction performance.

### To Predict

Run the following commands:

```bash
cd script
python -u pred.py
```

### To train 
To train a model from scratch, use one of the following commands: 

**Train** `GNNRegressor`
```bash
cd script
python -u run_train_gnn.py
```
**Train** `GNNFusion`
```bash
cd script
python -u run_train_gnnf.py
```