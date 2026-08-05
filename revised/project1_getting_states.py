import sys
sys.path.append(".")

import pandas as pd
import chemprop

import numpy as np
import random
import itertools
import torch

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_fingerprint(train_path, train_pred_path,
                    test_path, test_pred_path,
                    checkpoints_path,
                    data_seed=42, torch_seed=42):
    """Train Chemprop model and extract embeddings for train/test splits."""
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

    # Load train embeddings
    fingerprint = pd.read_csv(train_pred_path)
    fingerprint_ = fingerprint.iloc[:, 1:]
    index_bad = [i for i, val in enumerate(fingerprint_.iloc[:, 1]) if val == 'Invalid SMILES']

    data = pd.read_csv(train_path)
    y_train = data['MTB Standardized Residuals'].drop(index_bad).reset_index(drop=True)
    X_train = fingerprint_.drop(index_bad).reset_index(drop=True)

    # Load test embeddings
    fingerprint = pd.read_csv(test_pred_path)
    fingerprint_ = fingerprint.iloc[:, 1:]
    index_bad = [i for i, val in enumerate(fingerprint_.iloc[:, 1]) if val == 'Invalid SMILES']

    data = pd.read_csv(test_path)
    y_test = data['MTB Standardized Residuals'].drop(index_bad).reset_index(drop=True)
    X_test = fingerprint_.drop(index_bad).reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def train_mlp_with_early_stopping(X_train, X_val, y_train, y_val,
                                  epochs=100, patience=10, seed=42,
                                  hidden_layer_sizes=(300, 200, 32, 16),
                                  batch_size=64, alpha=0.01,
                                  learning_rate_init=0.01):
    """Train MLP with early stopping."""
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        batch_size=batch_size,
        alpha=alpha,
        learning_rate='adaptive',
        learning_rate_init=learning_rate_init
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        mlp.fit(X_train, y_train)
        val_loss = mean_squared_error(y_val, mlp.predict(X_val))
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return mlp


# ----------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------
if __name__ == '__main__':
    train_path = '/work/pi_annagreen_umass_edu/nelson/area_42/area_84/train_scaffold_split.csv'
    test_path = '/work/pi_annagreen_umass_edu/nelson/area_42/area_84/test_scaffold_split.csv'
    train_pred_path = './results/baseline_train_emb.csv'
    test_pred_path = './results/baseline_test_emb.csv'
    checkpoints_path = './results/baseline_checkpoints'

    set_seed(42)

    all_seeds = list(range(42, 92))
    torch_seeds = random.sample(all_seeds, 7)
    data_seeds = random.sample(all_seeds, 7)
    mlp_seeds = random.sample(range(100, 200), 49)
    seed_combinations = list(itertools.product(torch_seeds, data_seeds))

    results = []

    for i, (torch_seed, data_seed) in enumerate(seed_combinations):
        mlp_seed = mlp_seeds[i]
        print(f"\n>>> Experiment {i+1}/{len(seed_combinations)} | Torch={torch_seed}, Data={data_seed}, MLP={mlp_seed}")

        X_train_scaffold, y_train_scaffold, X_test_scaffold, y_test_scaffold = get_fingerprint(
            train_path, train_pred_path, test_path, test_pred_path,
            checkpoints_path, data_seed=data_seed, torch_seed=torch_seed
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_scaffold, y_train_scaffold, test_size=0.2, random_state=mlp_seed
        )

        mlp_model = train_mlp_with_early_stopping(
            X_train, X_val, y_train, y_val, epochs=400, patience=10, seed=mlp_seed
        )

        y_val_pred = mlp_model.predict(X_val)
        y_test_pred = mlp_model.predict(X_test_scaffold)
        y_train_pred = mlp_model.predict(X_train)

        result = {
            'Torch Seed': torch_seed,
            'Data Seed': data_seed,
            'MLP Seed': mlp_seed,

            'Validation R2': r2_score(y_val, y_val_pred),
            'Validation RMSE': root_mean_squared_error(y_val, y_val_pred),
            'Validation MAE': mean_absolute_error(y_val, y_val_pred),
            'Validation Spearman': spearmanr(y_val, y_val_pred).correlation,

            'Test R2': r2_score(y_test_scaffold, y_test_pred),
            'Test RMSE': root_mean_squared_error(y_test_scaffold, y_test_pred),
            'Test MAE': mean_absolute_error(y_test_scaffold, y_test_pred),
            'Test Spearman': spearmanr(y_test_scaffold, y_test_pred).correlation,

            'Train R2': r2_score(y_train, y_train_pred),
            'Train RMSE': root_mean_squared_error(y_train, y_train_pred),
            'Train MAE': mean_absolute_error(y_train, y_train_pred),
            'Train Spearman': spearmanr(y_train, y_train_pred).correlation
        }
        
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('./results/baseline_performance.csv', index=False)

    print("\n========== Summary Across All Experiments ==========")
    print("Train R²: {:.4f} ± {:.4f}".format(results_df['Train R2'].mean(), results_df['Train R2'].std()))
    print("Train RMSE: {:.4f} ± {:.4f}".format(results_df['Train RMSE'].mean(), results_df['Train RMSE'].std()))
    print("Train MAE: {:.4f} ± {:.4f}".format(results_df['Train MAE'].mean(), results_df['Train MAE'].std()))
    print("Train Spearman: {:.4f} ± {:.4f}".format(results_df['Train Spearman'].mean(), results_df['Train Spearman'].std()))

    print("Validation R²: {:.4f} ± {:.4f}".format(results_df['Validation R2'].mean(), results_df['Validation R2'].std()))
    print("Validation RMSE: {:.4f} ± {:.4f}".format(results_df['Validation RMSE'].mean(), results_df['Validation RMSE'].std()))
    print("Validation MAE: {:.4f} ± {:.4f}".format(results_df['Validation MAE'].mean(), results_df['Validation MAE'].std()))
    print("Validation Spearman: {:.4f} ± {:.4f}".format(results_df['Validation Spearman'].mean(), results_df['Validation Spearman'].std()))

    print("Test R²: {:.4f} ± {:.4f}".format(results_df['Test R2'].mean(), results_df['Test R2'].std()))
    print("Test RMSE: {:.4f} ± {:.4f}".format(results_df['Test RMSE'].mean(), results_df['Test RMSE'].std()))
    print("Test MAE: {:.4f} ± {:.4f}".format(results_df['Test MAE'].mean(), results_df['Test MAE'].std()))
    print("Test Spearman: {:.4f} ± {:.4f}".format(results_df['Test Spearman'].mean(), results_df['Test Spearman'].std()))

    # ⭐ Select median-performing random state combination (based on Validation R²)
    sorted_df = results_df.sort_values(by='Validation R2').reset_index(drop=True)
    median_idx = len(sorted_df) // 2
    median_run = sorted_df.loc[median_idx]

    print("\n========== ⚖️ Median Random State Combination ==========")
    print(median_run)
    median_run.to_frame().T.to_csv('./results/median_random_state.csv', index=False)
    print("\n✅ Median random states saved to ./results/median_random_state.csv")
