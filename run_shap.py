"""
SHAP explainability analysis for the trained MoCA/Minibest regression models.

WHAT THIS SCRIPT DOES
    1. For each assessment (MoCA, Minibest) and each target score, reads the
       performance JSON written by train_models.py and picks the model with
       the lowest nested-CV RMSE.
    2. Loads that saved model (.joblib) together with the normalized training
       data, then rebuilds a subject-aware train/test split from the combined
       feature file so SHAP values are computed on held-out subjects.
    3. Runs SHAP analysis (via utils.run_shap_analysis) and aggregates
       feature importance subject-fairly: mean |SHAP| per subject first,
       then mean across subjects, so no subject dominates the ranking.
    4. Saves per-target importance scores to
       results/shap/results/shap_feature_importances_{Assessment}_{Target}_{Model}.json
       (plots are written by the utils plotting helpers).

    Run this after train_models.py has completed; all paths come from
    config.yaml.
"""

import os
import sys
import yaml
import joblib
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from utils import run_shap_analysis, subject_aware_train_test_split, normalize_features


# Path configuration
ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH =  os.path.join(ROOT_PATH, 'config.yaml')


def main_shap():
    """
    Runs SHAP analysis on the best-performing models that were previously trained
    and saved by the main training pipeline.
    """
    # 1. Load the same configuration file to get all the paths
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
 
    # Define necessary paths from config
    models_path         = os.path.join(ROOT_PATH, config['result_paths']['models_path'])
    plot_path           = os.path.join(ROOT_PATH, config['result_paths']['figure_path'])
    output_path         = os.path.join(ROOT_PATH, config['result_paths']['output_path'])
    shap_path           = os.path.join(ROOT_PATH, config['result_paths']['shap_path'])
    data_path           = os.path.join(ROOT_PATH, config['data_paths']['feature_path'])
 
    exclude_cols = config['exclude_cols']
    test_size = config['test_size']
    random_seed = config['random_seed']
    
    os.makedirs(shap_path, exist_ok=True)
    
    # 2. Replicate the loops from the main script to process each target
    for name in ["MoCA", "Minibest"]:
        if name == "MoCA":
            targets = config['moca_targets']
            source_file = "ALL_COHORTS_features_full_MoCA.csv"                      
            exclude_targets = config['moca_exclude']                                
        elif name == "Minibest":
            targets = config['minibest_targets']
            source_file = "ALL_COHORTS_features_full_Minibest.csv"                  
            exclude_targets = config['minibest_exclude']
 
        for target in tqdm(targets, desc=f"Processing Targets for {name}"):
            print(f"\n{'='*25} Running SHAP for Target: {target} ({name}) {'='*25}")
 
            # 3. Load the performance JSON to find the best model for this target
            model_performance_path = os.path.join(
                output_path,
                f"{name}_{target}_all_models_performance_full.json"                 
            )
            try:
                with open(model_performance_path, 'r') as f:
                    model_performance = json.load(f)
            except FileNotFoundError:
                print(f"ERROR: Performance file not found for {target}. Skipping SHAP analysis.")
                print(f"Expected file at: {model_performance_path}")
                continue
            
            # Determine the best model by finding the one with the minimum RMSE
            best_model_dict = min(model_performance, key=lambda x: list(x.values())[0]['rmse'])
            best_model_name = list(best_model_dict.keys())[0]
            print(f"Found best model: {best_model_name}")
 
            # 4. Load the best model and the corresponding training data
            # Load the trained model
            trained_model_path = os.path.join(
                models_path,
                f"{best_model_name}_{name}_{target}_full.joblib"                    
            )
            trained_model = joblib.load(trained_model_path)
            print(f"Loaded model from: {trained_model_path}")
 
            # Load the normalized training data that was used to train the model
            xtrain_path = os.path.join(
                data_path,
                f"X_train_norm_{name}_{target}_full.joblib"                         
            )
            X_train_norm = joblib.load(xtrain_path)
            print(f"Loaded training data from: {xtrain_path}")
 
            if isinstance(X_train_norm, np.ndarray):
                source_data_tmp = pd.read_csv(os.path.join(data_path, source_file))
                cols_to_drop_tmp = exclude_cols + exclude_targets
                feature_cols = [c for c in source_data_tmp.columns if c not in cols_to_drop_tmp and c != target]
                if X_train_norm.shape[1] == len(feature_cols):
                    X_train_norm = pd.DataFrame(X_train_norm, columns=feature_cols)
                    print("WARNING: X_train_norm was numpy array — reconstructed DataFrame from source columns.")
                else:
                    print(f"ERROR: Column count mismatch — X_train_norm has {X_train_norm.shape[1]} cols, "
                          f"expected {len(feature_cols)}. Cannot reconstruct DataFrame. Skipping.")
                    continue
            
            X_train_norm = X_train_norm.drop(columns=exclude_targets, errors='ignore')
 
            # Load the test data
            source_data = pd.read_csv(os.path.join(data_path, source_file))
            y_target = source_data[target]
            
            cols_to_drop = exclude_cols + exclude_targets 
            X_all_feats = source_data.drop(columns=cols_to_drop)
 
            X_cleaned = X_all_feats.dropna()
            y_aligned = y_target.loc[X_cleaned.index]
            subjects_aligned = source_data.loc[X_cleaned.index, 'Subject']
            
            valid_target_idx = y_aligned.notna()
            X = X_cleaned[valid_target_idx]
            y = y_aligned[valid_target_idx]
            subjects = subjects_aligned[valid_target_idx]
 
            # Split the dataset into train and test
            X_train, X_test, _, y_test = subject_aware_train_test_split(
                X=X,
                y=y,
                groups=subjects,
                test_size=test_size,
                random_state=random_seed
            )
 
            # Extract test subjects for subject-fair SHAP aggregation
            subjects_test = subjects.loc[X_test.index]
 
            # Normalize test data
            _, X_test_norm, _ = normalize_features(
                train_data=X_train,
                test_data=X_test,
                feature_cols=X.columns.tolist()
            )
 
            # 5. Run the SHAP analysis with subjects for subject-fair plots
            feature_importances = run_shap_analysis(
                model_instance=trained_model,
                model_name=best_model_name,
                X_train=X_train_norm,
                X_test=X_test_norm,
                target_name=target,
                assess_name=name,
                config=config,
                subjects=subjects_test
            )
 
            # 6. Save the SHAP feature importance results (subject-fair)
            shap_results_path = os.path.join(shap_path, "results")
            os.makedirs(shap_results_path, exist_ok=True)
            shap_file_path = os.path.join(
                shap_results_path,
                f'shap_feature_importances_{name}_{target}_{best_model_name}.json'
            )
 
            if isinstance(feature_importances.values, np.ndarray):
                shap_vals = feature_importances.values
            else:
                shap_vals = np.array(feature_importances.values)
 
            # feature_names = X_train_norm.columns
            feature_names = feature_importances.feature_names
 
            # Subject-fair: mean |SHAP| per subject, then mean across subjects
            shap_abs_df = pd.DataFrame(np.abs(shap_vals), columns=feature_names)
            shap_abs_df['subject'] = subjects_test.values
            subject_mean_shap = shap_abs_df.groupby('subject').mean()
            mean_abs_shap = subject_mean_shap.mean(axis=0).values
 
            feature_importance_scores = dict(zip(feature_names, mean_abs_shap.tolist()))
            
            shap_values_dict = {
                target: feature_importance_scores
            }
            
            with open(shap_file_path, 'w') as f:
                json.dump(shap_values_dict, f, indent=4)
            
            print(f"SHAP feature importances saved to: {shap_file_path}")

if __name__ == "__main__":
    main_shap()