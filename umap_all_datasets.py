from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to generate MACCS keys
def smiles_to_maccs(smiles_list):
    maccs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            maccs.append(list(MACCSkeys.GenMACCSKeys(mol)))
        else:
            maccs.append([0] * 167)  # MACCS keys have 167 bits
    return maccs

# Function to perform UMAP embedding and plot results
# Function to perform UMAP embedding and plot results
def plot_umap_2d(smiles_list, labels, title):
    # Generate MACCS keys
    maccs_features = smiles_to_maccs(smiles_list)

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(maccs_features)

    # Apply UMAP for 2D embedding
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = umap_2d.fit_transform(scaled_features)

    # Define colors and sizes for different datasets
    color_map = {'MLSMR': 'blue', 'TAACF': 'green', 'Test azides libraries': 'red'}
    size_map = {'MLSMR': 20, 'TAACF': 20, 'Test azides libraries': 80}  # In-house points are larger

    # Create a 2D plot
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        subset = embedding_2d[[i for i, lbl in enumerate(labels) if lbl == label]]
        plt.scatter(
            subset[:, 0], subset[:, 1],
            label=label, alpha=0.7, s=size_map[label], color=color_map[label]
        )

    plt.title(title, fontsize=14)
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend()

    plt.tight_layout()
    plt.savefig("/work/pi_annagreen_umass_edu/nelson/area_42/area_42_workflows/umap_all_datasets.pdf")
    plt.savefig("/work/pi_annagreen_umass_edu/nelson/area_42/area_42_workflows/umap_all_datasets.png")
    plt.show()

taacf = pd.read_excel('/work/pi_annagreen_umass_edu/nelson/datasets/TAACF from low-to-high activity.xlsx')
mlsmr = pd.read_excel('/work/pi_annagreen_umass_edu/nelson/datasets/Prep notebook (Seigrist).xlsx', sheet_name = 'mlsmr')
df_mtb_1600 = pd.read_csv('/work/pi_annagreen_umass_edu/nelson/datasets/PhD paper data with SMILES.csv')

# Dataset 1: MLSMR, TAACF, and in-house SMILES
mlsmr_smiles, taacf_smiles, inhouse_smiles = mlsmr['SMILES'], taacf['SMILES'], df_mtb_1600['Smiles']
all_smiles = list(mlsmr_smiles) + list(taacf_smiles) + list(inhouse_smiles)
all_labels = ['MLSMR'] * len(mlsmr_smiles) + ['TAACF'] * len(taacf_smiles) + ['Test azides libraries'] * len(inhouse_smiles)

plot_umap_2d(all_smiles, all_labels, "2D UMAP for MLSMR, TAACF, and test azide libraries SMILES")


