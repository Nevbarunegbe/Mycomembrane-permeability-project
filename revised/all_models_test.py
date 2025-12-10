# compare_pipelines_full.py
import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import random
import itertools
import torch
import chemprop
import xgboost
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import rdkit

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import root_mean_squared_error as rmse_sklearn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

# -------------------------
# Config / paths / seeds
# -------------------------
TRAIN_PATH = '/work/pi_annagreen_umass_edu/nelson/area_42/area_84/train_scaffold_split.csv'
TEST_PATH = '/work/pi_annagreen_umass_edu/nelson/area_42/area_84/test_scaffold_split.csv'
TRAIN_PRED_PATH = './results/baseline_train_emb.csv'
TEST_PRED_PATH = './results/baseline_test_emb.csv'
CHECKPOINTS_PATH = './results/baseline_checkpoints'

RESULTS_CSV = './results/final_model_full_comparison_performance.csv'
PLOT_PNG = './results/final_models_train_val_r2_barplot.png'

GLOBAL_SEED = 42
N_MLP_EPOCHS = 400
MLP_PATIENCE = 10

# -------------------------
# Utility functions
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def calculate_rdkit_descriptors_from_smiles_list(smiles_list):
    """Return DataFrame: first column 'Smiles', remaining columns descriptors from RDKit.descList"""
    desc_names = [name for name, _ in Descriptors.descList]
    rows = []
    for smi in smiles_list:
        mol = None
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            rows.append([np.nan]*len(desc_names))
            continue
        vals = []
        for name, fn in Descriptors.descList:
            try:
                vals.append(fn(mol))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    df = pd.DataFrame(rows, columns=desc_names)
    df.insert(0, 'Smiles', list(smiles_list))
    return df

def calculate_ecfp_from_smiles_list(smiles_list, radius=2, n_bits=2048):
    """Return DataFrame: first column 'Smiles', remaining are ECFP bits as ints"""
    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append([np.nan]*n_bits)
                continue
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=int)
            rdkit.DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr.tolist())
        except Exception:
            rows.append([np.nan]*n_bits)
    col_names = [f'ECFP_{i}' for i in range(n_bits)]
    df = pd.DataFrame(rows, columns=col_names)
    df.insert(0, 'Smiles', list(smiles_list))
    return df

# -------------------------
# Chemprop embedding extraction
# -------------------------
def get_fingerprint(train_path, train_pred_path,
                    test_path, test_pred_path,
                    checkpoints_path,
                    data_seed=57, torch_seed=48):
    """
    Train chemprop with given seeds (uses scaffold_balanced as in original),
    then generate train and test MPN embeddings and return:
      train_fp_df, y_train_series, test_fp_df, y_test_series, index_bad_train, index_bad_test
    index_bad lists are indices removed due to 'Invalid SMILES' in chemprop output.
    """
    arguments = [
        '--data_path', train_path,
        '--dataset_type', 'regression',
        '--save_dir', checkpoints_path,
        '--ffn_hidden_size', '300',
        '--epochs', '30',
        '--save_smiles_splits',
        '--smiles_columns', 'Smiles',
        '--target_columns', 'MTB Standardized Residuals',
        '--split_type', 'scaffold_balanced',
        '--hidden_size', '300',
        '--num_folds', '1',
        '--pytorch_seed', str(torch_seed),
        '--seed', str(data_seed),
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

    # Train set embeddings
    arguments = [
        '--test_path', train_path,
        '--preds_path', train_pred_path,
        '--checkpoint_dir', checkpoints_path,
        '--smiles_columns', 'Smiles',
        '--fingerprint_type', 'MPN'
    ]
    args = chemprop.args.FingerprintArgs().parse_args(arguments)
    chemprop.train.molecule_fingerprint.molecule_fingerprint(args=args)

    # Test set embeddings
    arguments = [
        '--test_path', test_path,
        '--preds_path', test_pred_path,
        '--checkpoint_dir', checkpoints_path,
        '--smiles_columns', 'Smiles',
        '--fingerprint_type', 'MPN'
    ]
    args = chemprop.args.FingerprintArgs().parse_args(arguments)
    chemprop.train.molecule_fingerprint.molecule_fingerprint(args=args)

    # Load embeddings (drop the SMILES first column)
    train_fp = pd.read_csv(train_pred_path).iloc[:, 1:].copy()
    # detect Invalid SMILES rows
    index_bad_train = [i for i, v in enumerate(train_fp.iloc[:, 1]) if str(v) == 'Invalid SMILES']
    if len(index_bad_train) > 0:
        train_fp = train_fp.drop(index_bad_train).reset_index(drop=True)
    y_train = pd.read_csv(train_path)['MTB Standardized Residuals'].drop(index_bad_train).reset_index(drop=True)

    test_fp = pd.read_csv(test_pred_path).iloc[:, 1:].copy()
    index_bad_test = [i for i, v in enumerate(test_fp.iloc[:, 1]) if str(v) == 'Invalid SMILES']
    if len(index_bad_test) > 0:
        test_fp = test_fp.drop(index_bad_test).reset_index(drop=True)
    y_test = pd.read_csv(test_path)['MTB Standardized Residuals'].drop(index_bad_test).reset_index(drop=True)

    return train_fp, y_train, test_fp, y_test, index_bad_train, index_bad_test

# -------------------------
# MLP training with early stopping (original hyperparams)
# -------------------------
def train_mlp_with_early_stopping_original(X_train, X_val, y_train, y_val,
                                  epochs=N_MLP_EPOCHS, patience=MLP_PATIENCE, seed=145,
                                  hidden_layer_sizes=(300, 200, 32, 16),
                                  batch_size=64, alpha=0.01,
                                  learning_rate_init=0.01, verbose=False):
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        batch_size=batch_size,
        alpha=alpha,
        learning_rate='adaptive',
        learning_rate_init=learning_rate_init,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(epochs):
        mlp.fit(X_train, y_train)
        y_val_pred = mlp.predict(X_val)
        val_loss = mean_squared_error(y_val, y_val_pred)

        if verbose:
            y_train_pred = mlp.predict(X_train)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {mean_squared_error(y_train, y_train_pred):.6f} - Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            #store a copy by creating a new MLP with same random_state and fitting to full current epochs
            best_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                     max_iter=epoch + 1,
                                     random_state=seed,
                                     alpha=alpha,
                                     learning_rate='adaptive',
                                     learning_rate_init=learning_rate_init)
            best_model.fit(X_train, y_train)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # If best_model didn't get assigned (never improved), fallback to last mlp
    if best_model is None:
        best_model = mlp

    return best_model

# -------------------------
# Main loop: 7x7 seeds = 49 runs
# -------------------------
def run_all_and_plot():
    set_seed(GLOBAL_SEED)

    all_seeds = list(range(42, 92))
    torch_seeds = random.sample(all_seeds, 7)
    data_seeds = random.sample(all_seeds, 7)
    mlp_seeds = random.sample(range(100, 200), 49)
    seed_combinations = list(itertools.product(torch_seeds, data_seeds))

    # storage for run-level metrics: we'll store per-run per-pipeline train/val/test R2
    run_records = []

    for i, (torch_seed, data_seed) in enumerate(seed_combinations):
        mlp_seed = mlp_seeds[i]
        print(f"\n=== Run {i+1}/{len(seed_combinations)}: torch={torch_seed}, data={data_seed}, mlp={mlp_seed} ===")

        # 1) get chemprop embeddings for this run (train/test)
        X_emb_train_all, y_emb_train_all, X_emb_test_all, y_emb_test_all, idx_bad_train, idx_bad_test = \
            get_fingerprint(TRAIN_PATH, TRAIN_PRED_PATH, TEST_PATH, TEST_PRED_PATH, CHECKPOINTS_PATH,
                            data_seed=data_seed, torch_seed=torch_seed)

        # Build deterministic train/val split indices for the training embeddings
        n_train = len(X_emb_train_all)
        indices = np.arange(n_train)
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=mlp_seed)

        # Embedding splits
        X_train_emb = X_emb_train_all.iloc[train_idx].reset_index(drop=True)
        X_val_emb = X_emb_train_all.iloc[val_idx].reset_index(drop=True)
        X_test_emb = X_emb_test_all.reset_index(drop=True)

        y_train_emb = y_emb_train_all.iloc[train_idx].reset_index(drop=True)
        y_val_emb = y_emb_train_all.iloc[val_idx].reset_index(drop=True)
        y_test_emb = y_emb_test_all.reset_index(drop=True)

        # === Embedding-based pipelines ===
        # MLP (original) on embeddings (train with early stopping as original)
        mlp_emb = train_mlp_with_early_stopping_original(X_train_emb, X_val_emb, y_train_emb, y_val_emb,
                                                        epochs=N_MLP_EPOCHS, patience=MLP_PATIENCE, seed=mlp_seed,
                                                        hidden_layer_sizes=(300,200,32,16), batch_size=64,
                                                        alpha=0.01, learning_rate_init=0.01, verbose=False)
        y_train_pred_emb_mlp = mlp_emb.predict(X_train_emb)
        y_val_pred_emb_mlp = mlp_emb.predict(X_val_emb)
        y_test_pred_emb_mlp = mlp_emb.predict(X_test_emb)

        run_records.append({
            'Pipeline': 'MLP_emb',
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_emb, y_train_pred_emb_mlp),
            'Val R2': r2_score(y_val_emb, y_val_pred_emb_mlp),
            'Test R2': r2_score(y_test_emb, y_test_pred_emb_mlp)
        })

        # RF on embeddings
        rf_emb = RandomForestRegressor(max_depth=4, min_samples_split=5, random_state=mlp_seed)
        rf_emb.fit(X_train_emb, y_train_emb)
        run_records.append({
            'Pipeline': 'RF_emb',
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_emb, rf_emb.predict(X_train_emb)),
            'Val R2': r2_score(y_val_emb, rf_emb.predict(X_val_emb)),
            'Test R2': r2_score(y_test_emb, rf_emb.predict(X_test_emb))
        })

        # XGB on embeddings
        xgb_emb = xgboost.XGBRegressor(max_depth=2, learning_rate=0.1, reg_lambda=0.1, random_state=mlp_seed, verbosity=0)
        xgb_emb.fit(X_train_emb, y_train_emb)
        run_records.append({
            'Pipeline': 'XGB_emb',
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_emb, xgb_emb.predict(X_train_emb)),
            'Val R2': r2_score(y_val_emb, xgb_emb.predict(X_val_emb)),
            'Test R2': r2_score(y_test_emb, xgb_emb.predict(X_test_emb))
        })

        # === Descriptor-based (RDKit) pipelines ===
        # compute RDKit descriptors for the train CSV and test CSV in same order chemprop used
        df_train_csv = pd.read_csv(TRAIN_PATH)
        df_test_csv = pd.read_csv(TEST_PATH)

        rdkit_train_full = calculate_rdkit_descriptors_from_smiles_list(df_train_csv['Smiles'].fillna('').astype(str))
        rdkit_test_full = calculate_rdkit_descriptors_from_smiles_list(df_test_csv['Smiles'].fillna('').astype(str))

        # drop same bad indices (chemprop 'Invalid SMILES') so alignment matches embeddings
        if len(idx_bad_train) > 0:
            rdkit_train_full = rdkit_train_full.drop(idx_bad_train).reset_index(drop=True)
        else:
            rdkit_train_full = rdkit_train_full.reset_index(drop=True)
        if len(idx_bad_test) > 0:
            rdkit_test_full = rdkit_test_full.drop(idx_bad_test).reset_index(drop=True)
        else:
            rdkit_test_full = rdkit_test_full.reset_index(drop=True)

        # Use descriptor columns (drop 'Smiles')
        X_rdkit_all = rdkit_train_full.iloc[:, 1:].astype(float).reset_index(drop=True)
        X_rdkit_test_all = rdkit_test_full.iloc[:, 1:].astype(float).reset_index(drop=True)

        X_train_rdkit = X_rdkit_all.iloc[train_idx].reset_index(drop=True)
        X_val_rdkit = X_rdkit_all.iloc[val_idx].reset_index(drop=True)
        X_test_rdkit = X_rdkit_test_all.reset_index(drop=True)

        y_train_rdkit = y_train_emb.copy()
        y_val_rdkit = y_val_emb.copy()
        y_test_rdkit = y_test_emb.copy()

        # XGB on RDKit
        xgb_rdkit = xgboost.XGBRegressor(random_state=mlp_seed, max_depth=3, reg_alpha=50, verbosity=0)
        xgb_rdkit.fit(X_train_rdkit, y_train_rdkit)
        run_records.append({
            'Pipeline': 'XGB_rdkit',
            'Torch Seed': torch_seed, 'Data Seed': data_seed, 'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_rdkit, xgb_rdkit.predict(X_train_rdkit)),
            'Val R2': r2_score(y_val_rdkit, xgb_rdkit.predict(X_val_rdkit)),
            'Test R2': r2_score(y_test_rdkit, xgb_rdkit.predict(X_test_rdkit))
        })

        # RF on RDKit
        rf_rdkit = RandomForestRegressor(max_depth=4, min_samples_split=5, random_state=mlp_seed)
        rf_rdkit.fit(X_train_rdkit, y_train_rdkit)
        run_records.append({
            'Pipeline': 'RF_rdkit',
            'Torch Seed': torch_seed, 'Data Seed': data_seed, 'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_rdkit, rf_rdkit.predict(X_train_rdkit)),
            'Val R2': r2_score(y_val_rdkit, rf_rdkit.predict(X_val_rdkit)),
            'Test R2': r2_score(y_test_rdkit, rf_rdkit.predict(X_test_rdkit))
        })

        # MLP on RDKit (scale)
        scaler_rdkit = StandardScaler()
        X_train_rdkit_scaled = scaler_rdkit.fit_transform(X_train_rdkit)
        X_val_rdkit_scaled = scaler_rdkit.transform(X_val_rdkit)
        X_test_rdkit_scaled = scaler_rdkit.transform(X_test_rdkit)

        mlp_rdkit = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=200, random_state=mlp_seed,
                                 alpha=0.1, learning_rate_init=0.01, warm_start=False)
        mlp_rdkit.fit(X_train_rdkit_scaled, y_train_rdkit)
        run_records.append({
            'Pipeline': 'MLP_rdkit',
            'Torch Seed': torch_seed, 'Data Seed': data_seed, 'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_rdkit, mlp_rdkit.predict(X_train_rdkit_scaled)),
            'Val R2': r2_score(y_val_rdkit, mlp_rdkit.predict(X_val_rdkit_scaled)),
            'Test R2': r2_score(y_test_rdkit, mlp_rdkit.predict(X_test_rdkit_scaled))
        })

        # === ECFP-based pipelines ===
        df_train_csv = pd.read_csv(TRAIN_PATH)
        df_test_csv = pd.read_csv(TEST_PATH)

        ecfp_train_full = calculate_ecfp_from_smiles_list(df_train_csv['Smiles'].fillna('').astype(str))
        ecfp_test_full = calculate_ecfp_from_smiles_list(df_test_csv['Smiles'].fillna('').astype(str))

        if len(idx_bad_train) > 0:
            ecfp_train_full = ecfp_train_full.drop(idx_bad_train).reset_index(drop=True)
        else:
            ecfp_train_full = ecfp_train_full.reset_index(drop=True)

        if len(idx_bad_test) > 0:
            ecfp_test_full = ecfp_test_full.drop(idx_bad_test).reset_index(drop=True)
        else:
            ecfp_test_full = ecfp_test_full.reset_index(drop=True)

        X_ecfp_all = ecfp_train_full.iloc[:, 1:].astype(float).reset_index(drop=True)
        X_ecfp_test_all = ecfp_test_full.iloc[:, 1:].astype(float).reset_index(drop=True)

        X_train_ecfp = X_ecfp_all.iloc[train_idx].reset_index(drop=True)
        X_val_ecfp = X_ecfp_all.iloc[val_idx].reset_index(drop=True)
        X_test_ecfp = X_ecfp_test_all.reset_index(drop=True)

        y_train_ecfp = y_train_emb.copy()
        y_val_ecfp = y_val_emb.copy()
        y_test_ecfp = y_test_emb.copy()

        # XGB on ECFP
        xgb_ecfp = xgboost.XGBRegressor(random_state=mlp_seed, max_depth=3, reg_alpha=50, verbosity=0)
        xgb_ecfp.fit(X_train_ecfp, y_train_ecfp)
        run_records.append({
            'Pipeline': 'XGB_ecfp',
            'Torch Seed': torch_seed, 'Data Seed': data_seed, 'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_ecfp, xgb_ecfp.predict(X_train_ecfp)),
            'Val R2': r2_score(y_val_ecfp, xgb_ecfp.predict(X_val_ecfp)),
            'Test R2': r2_score(y_test_ecfp, xgb_ecfp.predict(X_test_ecfp))
        })

        # RF on ECFP
        rf_ecfp = RandomForestRegressor(max_depth=4, min_samples_split=5, random_state=mlp_seed)
        rf_ecfp.fit(X_train_ecfp, y_train_ecfp)
        run_records.append({
            'Pipeline': 'RF_ecfp',
            'Torch Seed': torch_seed, 'Data Seed': data_seed, 'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_ecfp, rf_ecfp.predict(X_train_ecfp)),
            'Val R2': r2_score(y_val_ecfp, rf_ecfp.predict(X_val_ecfp)),
            'Test R2': r2_score(y_test_ecfp, rf_ecfp.predict(X_test_ecfp))
        })

        # MLP on ECFP (scaled)
        scaler_ecfp = StandardScaler()
        X_train_ecfp_scaled = scaler_ecfp.fit_transform(X_train_ecfp)
        X_val_ecfp_scaled = scaler_ecfp.transform(X_val_ecfp)
        X_test_ecfp_scaled = scaler_ecfp.transform(X_test_ecfp)

        mlp_ecfp = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=200, random_state=mlp_seed,
                                alpha=10, learning_rate_init=0.01, warm_start=False)
        mlp_ecfp.fit(X_train_ecfp_scaled, y_train_ecfp)
        run_records.append({
            'Pipeline': 'MLP_ecfp',
            'Torch Seed': torch_seed, 'Data Seed': data_seed, 'MLP Seed': mlp_seed,
            'Train R2': r2_score(y_train_ecfp, mlp_ecfp.predict(X_train_ecfp_scaled)),
            'Val R2': r2_score(y_val_ecfp, mlp_ecfp.predict(X_val_ecfp_scaled)),
            'Test R2': r2_score(y_test_ecfp, mlp_ecfp.predict(X_test_ecfp_scaled))
        })

    # End runs loop

    # -------------------------
    # Aggregate results, save CSV
    # -------------------------
    results_df = pd.DataFrame(run_records)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved per-run results to: {RESULTS_CSV}")

    # compute per-pipeline mean & std for Train and Val (R2)
    agg_train = results_df.groupby('Pipeline')['Train R2'].agg(['mean', 'std']).rename(columns={'mean':'Train_Mean_R2','std':'Train_Std_R2'})
    agg_val = results_df.groupby('Pipeline')['Val R2'].agg(['mean', 'std']).rename(columns={'mean':'Val_Mean_R2','std':'Val_Std_R2'})
    agg_test = results_df.groupby('Pipeline')['Test R2'].agg(['mean', 'std']).rename(columns={'mean':'Test_Mean_R2','std':'Test_Std_R2'})

    summary = pd.concat([agg_train, agg_val, agg_test], axis=1).reset_index()
    summary = summary.fillna(0)
    summary.to_csv('./results/final_models_summary_r2.csv', index=False)
    print("Saved summary to ./results/model_summary_r2.csv")

    # -------------------------
    # Bar chart: Train & Val side-by-side with error bars (std)
    # -------------------------
    # order pipelines
    pipeline_order = ['RF_emb', 'XGB_emb', 'MLP_emb',
                      'RF_rdkit', 'XGB_rdkit', 'MLP_rdkit',
                      'RF_ecfp', 'XGB_ecfp', 'MLP_ecfp']

    # Use summary rows in that order
    plot_df = summary.set_index('Pipeline').reindex(pipeline_order).reset_index()

    # Data for plotting
    train_means = plot_df['Train_Mean_R2'].values
    train_stds = plot_df['Train_Std_R2'].values
    val_means = plot_df['Val_Mean_R2'].values
    val_stds = plot_df['Val_Std_R2'].values
    labels = plot_df['Pipeline'].values

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14,6))
    plt.bar(x - width/2, train_means, width, yerr=train_stds, capsize=5, label='Train R2')
    plt.bar(x + width/2, val_means, width, yerr=val_stds, capsize=5, label='Val R2')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('R²')
    plt.title('Train vs Val R² (mean ± std) across pipelines (49 runs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=300)
    plt.show()
    print(f"Saved plot to: {PLOT_PNG}")

    # -------------------------
    # Print Test summary (mean ± std) for quick comparison
    # -------------------------
    print("\n=== Test set summary (mean ± std R2) ===")
    for _, row in summary.iterrows():
        print(f"{row['Pipeline']}: Test R2 = {row['Test_Mean_R2']:.4f} ± {row['Test_Std_R2']:.4f}")

if __name__ == "__main__":
    run_all_and_plot()
