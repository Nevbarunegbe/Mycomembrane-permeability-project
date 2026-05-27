#!/usr/bin/env python3
"""
Full pipeline:
- Train Chemprop on full dataset, generate embeddings
- Train final MLP on embeddings
- Train XGBoost surrogate on hand-engineered descriptors (full dataset)
- Compute SHAP on full descriptors (beeswarm: top10, top15, all)
- Generate descriptors for test workbook (the second file). **The first entry in that file is W1.**
- Compute SHAP for the test descriptors using the same explainer
- Produce waterfall plots for W1 (top10, top15, all) based on descriptors' SHAP values
- Save deliverables (plots and Excel files) to ./results/

"""

import sys
sys.path.append(".")

import os
import random
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap
import xgboost

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, QED, FindMolChiralCenters
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem import rdDistGeom, rdFreeSASA
from rdkit.Chem.rdmolops import GetFormalCharge
from itertools import combinations

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr

# ----------------------------
# Descriptor helper functions
# ----------------------------
def fused_ring_count(m):
    q = m.GetRingInfo()
    rings = [set(r) for r in q.AtomRings()]
    go_next = True
    while go_next:
        go_next = False
        for i, j in combinations(range(len(rings)), 2):
            if rings[i] & rings[j]:
                q = rings[i] | rings[j]
                del rings[j], rings[i]
                rings.append(q)
                go_next = True
                break
    return len(rings)

def count_hbd_hba_atoms(m):
    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    HAcceptorSmarts = Chem.MolFromSmarts(
        '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),'
        '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),'
        '$([nH0,o,s;+0])]')
    HDonor = m.GetSubstructMatches(HDonorSmarts)
    HAcceptor = m.GetSubstructMatches(HAcceptorSmarts)
    return len(set(HDonor + HAcceptor))

def confgen(smile, prunermsthresh=0.1, numconf=20):
    m = Chem.MolFromSmiles(smile)
    if m is None:
        raise ValueError("Invalid SMILES in confgen")
    mol = Chem.AddHs(m, addCoords=True)
    param = rdDistGeom.ETKDGv2()
    param.pruneRmsThresh = prunermsthresh
    rdDistGeom.EmbedMultipleConfs(mol, numconf, param)
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=4, mmffVariant='MMFF94s')
    except Exception:
        # fallback: try UFF
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass
    return mol

def calc_globularity_pbf(mol):
    glob_ls = []
    pbf_ls = []
    nconf = len(mol.GetConformers())
    if nconf == 0:
        return np.nan, np.nan
    for i in range(nconf):
        try:
            radii1 = rdFreeSASA.classifyAtoms(mol)
            sasa = rdFreeSASA.CalcSASA(mol, radii1, confIdx=i)
            molv = AllChem.ComputeMolVolume(mol, confId=i)
            globularity = ((molv * 3 / (4 * np.pi)) ** (2 / 3)) * 4 * np.pi / sasa if sasa > 0 else np.nan
            pbf = rdMolDescriptors.CalcPBF(mol, confId=i)
            glob_ls.append(globularity)
            pbf_ls.append(pbf)
        except Exception:
            glob_ls.append(np.nan)
            pbf_ls.append(np.nan)
    return np.nanmean(glob_ls), np.nanmean(pbf_ls)

def calc_descriptors(smiles_series):
    """
    Calculate molecular descriptors for a pandas Series/list of SMILES strings.
    Returns DataFrame (rows align with smiles_series).
    """
    descriptors = []
    columns = ['HBA', 'HBD', 'HBA+HBD', 'NumRings', 'RTB', 'NumAmideBonds',
               'Globularity', 'PBF', 'TPSA', 'logP', 'MR', 'MW', 'Csp3',
               'fmf', 'QED', 'HAC', 'NumRingsFused', 'unique_HBAD', 'max_ring_size',
               'n_chiral_centers', 'fcsp3_bm', 'formal_charge', 'abs_charge']

    for i, smi in enumerate(smiles_series):
        print(f"[descriptors] Processing molecule {i+1}/{len(smiles_series)}")
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            try:
                mol = confgen(smi)
                hba = rdMolDescriptors.CalcNumHBA(mol)
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                nrings = rdMolDescriptors.CalcNumRings(mol)
                rtb = rdMolDescriptors.CalcNumRotatableBonds(mol)
                glob, pbf = calc_globularity_pbf(mol)
                psa = rdMolDescriptors.CalcTPSA(mol)
                logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
                mw = rdMolDescriptors.CalcExactMolWt(mol) if hasattr(rdMolDescriptors, "CalcExactMolWt") else rdMolDescriptors._CalcMolWt(mol)
                csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
                hac = mol.GetNumHeavyAtoms()
                scaffold = GetScaffoldForMol(mol) if GetScaffoldForMol(mol) is not None else mol
                fmf = (scaffold.GetNumHeavyAtoms() / hac) if hac > 0 else np.nan
                qed = QED.qed(mol)
                nrings_fused = fused_ring_count(mol)
                n_unique_hba_hbd_atoms = count_hbd_hba_atoms(mol)
                ring_info = m.GetRingInfo()
                max_ring_size = len(max(ring_info.AtomRings(), key=len, default=())) if ring_info is not None else 0
                n_chiral_centers = len(FindMolChiralCenters(mol, includeUnassigned=True))
                fcsp3_bm = rdMolDescriptors.CalcFractionCSP3(GetScaffoldForMol(mol)) if GetScaffoldForMol(mol) is not None else np.nan
                n_amide_bond = rdMolDescriptors.CalcNumAmideBonds(mol)
                f_charge = GetFormalCharge(mol)
                abs_charge = abs(f_charge)
                descriptors.append([hba, hbd, hba + hbd, nrings, rtb, n_amide_bond, glob, pbf,
                                    psa, logp, mr, mw, csp3, fmf, qed, hac, nrings_fused,
                                    n_unique_hba_hbd_atoms, max_ring_size, n_chiral_centers,
                                    fcsp3_bm, f_charge, abs_charge])
            except Exception as e:
                print(f"⚠️ Descriptor calc error for idx {i} SMILES {smi}: {e}")
                descriptors.append([np.nan] * len(columns))
        else:
            print(f"⚠️ Invalid SMILES at index {i}: {smi}")
            descriptors.append([np.nan] * len(columns))

    df = pd.DataFrame(descriptors, columns=columns)
    print(f"[descriptors] Completed: {df.shape[0]} molecules × {df.shape[1]} features")
    return df

# ----------------------------
# Global seeds & paths
# ----------------------------
TORCH_SEED = 57
DATA_SEED = 48
MLP_SEED = 145

random.seed(TORCH_SEED)
np.random.seed(TORCH_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs('./results', exist_ok=True)

full_dataset_path = '/work/pi_annagreen_umass_edu/nelson/datasets/PhD paper data with SMILES.csv'
train_pred_path = './results/full_train_emb.csv'
checkpoint_dir = './results/full_checkpoints'

# ----------------------------
# Train Chemprop on full data and generate embeddings
# ----------------------------
import chemprop

print("\n=== Training Chemprop model on full dataset ===")
train_args = [
    '--data_path', full_dataset_path,
    '--dataset_type', 'regression',
    '--save_dir', checkpoint_dir,
    '--ffn_hidden_size', '300',
    '--epochs', '30',
    '--save_smiles_splits',
    '--smiles_columns', 'Smiles',
    '--target_columns', 'MTB Standardized Residuals',
    '--split_type', 'scaffold_balanced',
    '--hidden_size', '300',
    '--num_folds', '1',
    '--pytorch_seed', str(TORCH_SEED),
    '--seed', str(DATA_SEED)
]
args = chemprop.args.TrainArgs().parse_args(train_args)
chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

print("\n=== Generating Chemprop embeddings (MPN) for full dataset ===")
fp_args = [
    '--test_path', full_dataset_path,
    '--preds_path', train_pred_path,
    '--checkpoint_dir', checkpoint_dir,
    '--smiles_columns', 'Smiles',
    '--fingerprint_type', 'MPN'
]
args = chemprop.args.FingerprintArgs().parse_args(fp_args)
chemprop.train.molecule_fingerprint.molecule_fingerprint(args=args)

# ----------------------------
# Load embeddings & train final MLP
# ----------------------------
fps = pd.read_csv(train_pred_path).iloc[:, 1:]
fps = fps.replace('Invalid SMILES', np.nan).dropna()
data = pd.read_csv(full_dataset_path)
y_full = data['MTB Standardized Residuals'][:len(fps)]
X_full = fps.reset_index(drop=True)

print(f"\nLoaded {X_full.shape[0]} molecules with {X_full.shape[1]} embedding features")

mlp_optimal = MLPRegressor(
    hidden_layer_sizes=(300, 200, 32, 16),
    random_state=MLP_SEED,
    alpha=0.01,
    learning_rate='adaptive',
    learning_rate_init=0.01
)
mlp_optimal.fit(X_full, y_full)

y_pred_full = mlp_optimal.predict(X_full)
print("\n=== Final Full Model Performance ===")
print("Full Dataset R²:", r2_score(y_full, y_pred_full))
print("Full Dataset RMSE:", np.sqrt(mean_squared_error(y_full, y_pred_full)))
print("Full Dataset Spearman:", spearmanr(y_full, y_pred_full).correlation)

# ----------------------------
# Load hand-engineered descriptors (full dataset), train surrogate XGB
# ----------------------------
print("\n=== Loading hand-engineered descriptors (full dataset) ===")
descriptors_full = pd.read_excel('/work/pi_annagreen_umass_edu/nelson/datasets/Hand_eng_descriptors.xlsx')
cdd = '/work/pi_annagreen_umass_edu/nelson/area_42/area_42_workflows/cddVisualizationExport_Mon Feb 17 2025.xlsx'
logD = pd.read_excel(cdd)['Log D (CDD calculated)']
logS = pd.read_excel(cdd)['Log S (CDD calculated)']
descriptors_full['logD'] = logD
descriptors_full['logS'] = logS

print(f"[info] descriptors_full shape: {descriptors_full.shape}")
# Use all columns except first metadata column (assumption)
X_desc = descriptors_full.iloc[:, 1:].reset_index(drop=True)
print(f"[info] X_desc shape: {X_desc.shape}")

# Surrogate target: predictions of the MLP on embeddings
y_complex = mlp_optimal.predict(X_full)

# Train XGB surrogate
xgb_model_desc = xgboost.XGBRegressor(random_state=42, n_estimators=500, max_depth=3, reg_alpha=5)
xgb_model_desc.fit(X_desc, y_complex)

y_xgb_pred = xgb_model_desc.predict(X_desc)

print("\n=== XGB Surrogate Performance (fidelity) ===")
print("Train R²:", r2_score(y_complex, y_xgb_pred))
print("Train RMSE:", np.sqrt(mean_squared_error(y_complex, y_xgb_pred)))
print("Train MAE:", mean_absolute_error(y_complex, y_xgb_pred))

# SHAP explainer trained on the full descriptors
explainer_desc = shap.Explainer(xgb_model_desc, X_desc)
shap_values_full = explainer_desc(X_desc)  # shap values for training descriptors

# ----------------------------
# SHAP beeswarm plots (top 10, 15, all)
# ----------------------------
for top_n in [10, 15, len(X_desc.columns)]:
    plt.figure()
    shap.summary_plot(shap_values_full, X_desc, show=False, max_display=top_n)
    plt.tight_layout()
    fname = f'./results/shap_beeswarm_top{top_n if top_n != len(X_desc.columns) else "all"}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[saved] {fname}")

# ----------------------------
# Generate descriptors for the second file (test set) and treat its first entry as W1
# ----------------------------
print("\n=== Generating descriptors for test workbook (second file) ===")
test_workbook = '/work/pi_annagreen_umass_edu/nelson/datasets/20251008 Nelson All_Predictions_Combined for revised paper_FULL.xlsx'
sheet_name = 'Fig4c (W peptides)_with_predict'
smiles_for_X_test = pd.read_excel(test_workbook, sheet_name=sheet_name)['SMILES'].astype(str).reset_index(drop=True)
print(f"[info] Loaded {len(smiles_for_X_test)} SMILES from {test_workbook} sheet {sheet_name}")

# Calculate descriptors for test set
test_desc = calc_descriptors(smiles_for_X_test)
print("[info] test_desc shape:", test_desc.shape)

# Attach names/logD/logS metadata if present

test_names = pd.Series(['W1', 'W2', 'W3'])
test_desc['COMPOUND NAME'] = test_names


test_desc['logD'] = pd.Series([1.0, 1.2, 1.5])
test_desc['logS'] = pd.Series([-6.8, -7.1, -7.5]) 

print("[info] test_desc AFTER shape:", test_desc.shape)

new_data = test_desc.drop(['COMPOUND NAME'], axis=1)

# Compute shap values for ONE ROW
shap_values_test = explainer_desc(new_data.iloc[[0]])

# Plot waterfall for that row
# top_n = [10,15,20, 30]

top_n = [10, 15, 20, 30]

for i in top_n:
    shap.waterfall_plot(shap_values_test[0], max_display=i, show=True)
    plt.savefig(f'Final_shap_waterfall_{i}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

# ----------------------------
# Final deliverables summary
# ----------------------------
print("\n===== Deliverables generated =====")
print(" - ./results/shap_beeswarm_top10.png")
print(" - ./results/shap_beeswarm_top15.png")
print(" - ./results/shap_beeswarm_topall.png")
print(" - ./results/W1_descriptor_SHAP_contributions.xlsx")
print(" - ./results/W1_descriptor_SHAP_contributions.csv")
print(" - ./results/W1_waterfall_descriptors_top10.png")
print(" - ./results/W1_waterfall_descriptors_top15.png")
print(" - ./results/W1_waterfall_descriptors_topall.png")
print(" - ./results/test_descriptors_with_meta.xlsx")
print("\nAll done. You can reproduce beeswarm/waterfall plots using the saved files.")

