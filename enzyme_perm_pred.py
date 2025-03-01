# python analyses
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import time

# machine learning analysis and pre-processing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, ConstantKernel, RBF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc
import umap

# deep learning
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import   Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from chemprop.features import load_features
from chemprop.models import MoleculeModel
from chemprop.train import run_training
from chemprop.args import TrainArgs
import chemprop
from chemprop import data#, featurizers, models

# for cheminformatics handling
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw

# for interactive plots
import plotly.express as px
import plotly.graph_objects as go

import shap

import seaborn as sns
from collections import defaultdict


def train_mlp_with_early_stopping(X_train, X_val, y_train, y_val, epochs=100, patience=10):
    mlp = MLPRegressor(hidden_layer_sizes=(300, 200, 32,16), # 400, 200, 50,
                        max_iter=1,  # We will manually handle epochs
                        warm_start=True,  # Keep existing model when calling fit multiple times
                        random_state=42,
                          alpha=0.01,
                        learning_rate='adaptive',
                      learning_rate_init = 0.001)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        mlp.fit(X_train, y_train)

        y_train_pred = mlp.predict(X_train)
        y_val_pred = mlp.predict(X_val)

        train_loss = mean_squared_error(y_train, y_train_pred)
        val_loss = mean_squared_error(y_val, y_val_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return mlp, train_losses, val_losses

new_dataset = pd.read_excel('/work/pi_annagreen_umass_edu/nelson/datasets/Screening1 in vitro enzyme Mtb_ALL.xlsx')
new_dataset

enzyme_smiles = new_dataset.SMILES.iloc[4::]
enzyme_activity = new_dataset['Inhibition'].iloc[4::]
enzyme_activity

"""TRAINING THE MODEL"""
siegrist = pd.read_excel('/work/pi_annagreen_umass_edu/nelson/datasets/20240813_comprehensive_data(2).xlsx',
                         sheet_name = 'smiles_react_perm_380')
siegrist.head()

df_mtb_380 = pd.DataFrame({
    'Smiles': siegrist['Smile'],
    'MTB Standardized Residuals': siegrist['mtb_resid_std']
})

df_msm_380 = pd.DataFrame({
    'Smiles': siegrist['Smile'],
    'MSM Standardized Residuals': siegrist['msm_resid_std']
})
siegrist

df_mtb_380.dropna(axis=0, inplace=True)

# Filter invalid SMILES strings
valid_smiles = []
invalid_indices = []
for i, smile in enumerate(df_mtb_380['Smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        valid_smiles.append(smile)
    else:
        invalid_indices.append(i)

# Remove invalid entries from df
df_mtb_380 = df_mtb_380.drop(invalid_indices).reset_index(drop=True)

# Display the last 300 rows
print(df_mtb_380.shape)


y_mtb380 = df_mtb_380['MTB Standardized Residuals']

# 1200
siegrist = pd.read_excel('/work/pi_annagreen_umass_edu/nelson/datasets/20240813_comprehensive_data(2).xlsx',
                         sheet_name = 'smiles_react_perm_1200')
siegrist.head()

df_mtb_1200 = pd.DataFrame({
    'Smiles': siegrist['Smile'],
    'MTB Standardized Residuals': siegrist['mtb_resid_std']
})

df_msm_1200 = pd.DataFrame({
    'Smiles': siegrist['Smile'],
    'MSM Standardized Residuals': siegrist['msm_resid_std']
})
siegrist

df_mtb_1200.dropna(axis=0, inplace=True)

# Filter invalid SMILES strings
valid_smiles = []
invalid_indices = []
for i, smile in enumerate(df_mtb_1200['Smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        valid_smiles.append(smile)
    else:
        invalid_indices.append(i)

# Remove invalid entries from df
df_mtb_1200 = df_mtb_1200.drop(invalid_indices).reset_index(drop=True)

# Display the last 300 rows
print(df_mtb_1200.shape)


y_mtb1200 = df_mtb_1200['MTB Standardized Residuals']


# 1519
df_mtb_1200 = pd.DataFrame({
    'Smiles': siegrist['Smile'],
    'MTB Standardized Residuals': siegrist['mtb_resid_std']
})

df_mtb_1200, df_mtb_380

df_mtb_1519 = pd.concat([df_mtb_1200, df_mtb_380], axis=0)
df_mtb_1519



df_mtb_1519.dropna(axis=0, inplace=True)

# Filter invalid SMILES strings
valid_smiles = []
invalid_indices = []
for i, smile in enumerate(df_mtb_1519['Smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        valid_smiles.append(smile)
    else:
        invalid_indices.append(i)

# Remove invalid entries from df
df_mtb_1519 = df_mtb_1519.drop(invalid_indices).reset_index(drop=True)

# Display the last 300 rows
print(df_mtb_1519.shape)


y_mtb1519 = df_mtb_1519['MTB Standardized Residuals']

sieg_40 = pd.read_excel('/work/pi_annagreen_umass_edu/nelson/datasets/20240813_comprehensive_data(2).xlsx',
                         sheet_name = 'smiles_react_perm_40')

sieg_40

df_mtb_40 = pd.DataFrame({
    'Smiles': sieg_40['Smile'],
    'MTB Standardized Residuals': sieg_40['mtb_resid_std']#sieg_40['Standard Residuals']
})

df_mtb_1600  = pd.concat([df_mtb_1519, df_mtb_40], axis=0)
df_mtb_1600

df_mtb_1600.dropna(axis=0, inplace=True)



# Display the last 300 rows
print(df_mtb_1600.shape)

df_mtb_1600.dropna(axis=0, inplace=True)

# Filter invalid SMILES strings
valid_smiles = []
invalid_indices = []
for i, smile in enumerate(df_mtb_1600['Smiles']):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        valid_smiles.append(smile)
    else:
        invalid_indices.append(i)

# Remove invalid entries from df
df_mtb_1600 = df_mtb_1600.drop(invalid_indices).reset_index(drop=True)

# Display the last 300 rows
print(df_mtb_1600.shape)


y_mtb1600 = df_mtb_1600['MTB Standardized Residuals']

# Display the dataframe
print(df_mtb_1600)

df_mtb_1600.to_csv('siegrist_clean_mtb1600.csv')
arguments = [
    '--data_path', 'siegrist_clean_mtb1600.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'siegrist_mtb1600_test_checkpoints_reg',
    '--ffn_hidden_size', '300',
    '--epochs', '30',
    '--save_smiles_splits',
    '--smiles_columns', 'Smiles',
    '--target_columns', 'MTB Standardized Residuals',
    '--split_type', 'scaffold_balanced',
    '--hidden_size', '300',
    '--metric', 'mae',
    '--num_folds', '1'
    #'--features_generator', 'morgan', #'rdkit_2d',
    #'--ensemble_size', '3'

]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

arguments = [
    '--test_path', 'siegrist_clean_mtb1600.csv',
    '--preds_path', 'content/siegrist_excel_fingerprint_mtb1600.csv',
    '--checkpoint_dir', 'siegrist_mtb1600_test_checkpoints_reg',
    '--smiles_columns', 'Smiles',
    '--fingerprint_type', 'MPN'
]

args = chemprop.args.FingerprintArgs().parse_args(arguments)
preds = chemprop.train.molecule_fingerprint.molecule_fingerprint(args=args)

fingerprint = pd.read_csv('content/siegrist_excel_fingerprint_mtb1600.csv')
print(fingerprint.head())
print(fingerprint.shape)

fingerprint_ = fingerprint.iloc[:,1::]

index_bad = []
for idx, val in enumerate(fingerprint_.iloc[:, 1]):
    if val == 'Invalid SMILES':
        index_bad.append(idx)


y_mtb1600 = y_mtb1600.drop(index_bad).reset_index(drop=True)


X_mtb1600 = fingerprint_.drop(index_bad).reset_index(drop=True)
print(X_mtb1600)

X_int, X_test, y_int, y_test = train_test_split(X_mtb1600, y_mtb1600, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_int, y_int, test_size=0.10, random_state=42)

# Train the model with early stopping
_, train_losses, val_losses = train_mlp_with_early_stopping(X_train, X_val, y_train, y_val, epochs=400, patience=10)

optimal_epoch = np.argmin(val_losses)
optimal_val_loss = val_losses[optimal_epoch]

print(f"Optimal Epoch: {optimal_epoch+1}")
print(f"Validation Loss at Optimal Epoch: {optimal_val_loss:.4f}")

mlp_optimal = MLPRegressor(hidden_layer_sizes=(300, 200, 32,16), #
                              max_iter=optimal_epoch+1,
                              random_state=42,
                          alpha=0.01,
                        learning_rate='adaptive',
                      learning_rate_init = 0.001)

mlp_optimal.fit(X_train, y_train)
y_val_pred_optimal = mlp_optimal.predict(X_val)
y_test_pred_optimal = mlp_optimal.predict(X_test)

print("Performance with Early Stopping:")
print('Train Set MSE:', mean_squared_error(y_train, mlp_optimal.predict(X_train)))
print('Validation Set MSE:', mean_squared_error(y_val, y_val_pred_optimal))
print('Test Set MSE:', mean_squared_error(y_test, y_test_pred_optimal))

print('Train Set R2:', r2_score(y_train, mlp_optimal.predict(X_train)))
print('Validation Set R2:', r2_score(y_val, y_val_pred_optimal))
print('Test Set R2:', r2_score(y_test, y_test_pred_optimal))

# Plot the trend of training and validation sets
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.ylim(0,1.5)
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()


# parity plot
train_mse = mean_squared_error(y_train, mlp_optimal.predict(X_train))
val_mse = mean_squared_error(y_val, y_val_pred_optimal)
test_mse = mean_squared_error(y_test, y_test_pred_optimal)

y_train_pred_optimal = mlp_optimal.predict(X_train)

train_r2 = r2_score(y_train, mlp_optimal.predict(X_train))
val_r2 = r2_score(y_val, y_val_pred_optimal)
test_r2 = r2_score(y_test, y_test_pred_optimal)

train_mae = mean_absolute_error(y_train, mlp_optimal.predict(X_train))
val_mae = mean_absolute_error(y_val, y_val_pred_optimal)
test_mae = mean_absolute_error(y_test, y_test_pred_optimal)

# Determine common scale limits
y_min = min(y_train.min(), y_test.min())
y_max = max(y_train.max(), y_test.max())

# Define padding for text
padding = (y_max - y_min) * 0.05

# Create a parity plot (Test)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred_optimal, alpha=0.5)
plt.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.xlim(y_min, y_max)
plt.ylim(y_min, y_max)
plt.title('Parity Plot (Test)')
plt.text(y_min + padding, y_max - padding, f'$R^2$: {test_r2:.2f}\nMSE: {test_mse:.2f}\nMAE: {test_mae:.2f}',
         fontsize=12, verticalalignment='top')
plt.legend()
plt.show()

# Create a parity plot (Train)
plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_train_pred_optimal, alpha=0.5)
plt.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.xlim(y_min, y_max)
plt.ylim(y_min, y_max)
plt.title('Parity Plot (Train)')
plt.text(y_min + padding, y_max - padding, f'$R^2$: {train_r2:.2f}\nMSE: {train_mse:.2f}\nMAE: {train_mae:.2f}',
         fontsize=12, verticalalignment='top')
plt.legend()
plt.show()

## Trying to make predictions for enzyme
enzyme_smiles.to_csv('enzymes.csv')
arguments = [
    '--test_path', 'enzymes.csv',
    '--preds_path', 'content/enzyme_f.csv',
    '--checkpoint_dir', 'siegrist_mtb1600_test_checkpoints_reg',
    '--smiles_columns', 'SMILES',
    '--fingerprint_type', 'MPN'
]

args = chemprop.args.FingerprintArgs().parse_args(arguments)
preds = chemprop.train.molecule_fingerprint.molecule_fingerprint(args=args)

fingerprint = pd.read_csv('content/enzyme_f.csv')
print(fingerprint.head())
print(fingerprint.shape)

fingerprint_ = fingerprint.iloc[:,1::]
fingerprint_
enzyme_perm1 = mlp_optimal.predict(fingerprint_)
np.savetxt("enzyme_perm_array.csv", enzyme_perm1, delimiter=",")
enzyme_perm2 = pd.Series(enzyme_perm1)
enzyme_perm = pd.concat([enzyme_smiles, enzyme_activity, enzyme_perm2], axis=1)
enzyme_perm.to_csv('/work/pi_annagreen_umass_edu/nelson/area_42/area_42_workflows/enzyme_perm.csv')