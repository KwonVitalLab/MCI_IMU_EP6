"""
Model training and evaluation pipeline for MoCA/Minibest score regression.

WHAT THIS SCRIPT DOES
    1. Loads the labeled window-level feature files
       (ALL_COHORTS_features_full_{MoCA,Minibest}.csv) produced by
       extract_features.py + assign_labels.py.
    2. For every (assessment, target) pair, runs repeated nested
       cross-validation with RandomForest (cuML), CatBoost, and XGBoost:
         - outer loop: GroupKFold by Subject (no subject leakage),
         - inner loop: Optuna hyperparameter tuning with early stopping,
         - per-fold z-score normalization; window predictions aggregated to
           the SubjectTime (subject x early/late) level before RMSE.
       Each outer fold runs in a fresh spawned process (CUDA-safe), and
       finished models are checkpointed so interrupted runs can resume.
    3. Reports mean RMSE with confidence intervals, plots Bland-Altman and
       Pearson-correlation figures, retrains the best-parameter model on all
       data, and saves:
         - trained models (.joblib) and normalized data for SHAP,
         - per-target performance JSONs used by run_shap.py.

    Supports SLURM job arrays (SLURM_ARRAY_TASK_ID selects one
    (dataset, target) unit) and a reduced smoke_test mode. All paths, targets,
    CV settings, and search spaces come from config.yaml.
"""

import os
import sys
import yaml
import json
import joblib
import optuna

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm

from cuml.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from scipy import stats
from pathlib import Path


from utils import plot_pearson_correlation, plot_bland_altman, normalize_features


# Path configuration
ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH =  os.path.join(ROOT_PATH, 'config.yaml')

# CUDA contexts do not survive fork(); worker processes must be spawned fresh.
_MP_CTX = mp.get_context("spawn")


NON_FEATURE_COLS = [
    # identifiers / grouping keys
    "Subject", "Cohort", "Time", "SubjectTime",
    # de-identified window metadata 
    "VisitIndex", "CohortProgramDay",
    "Window Start TimeOfDay", "Window Start Elapsed (s)",
    # legacy absolute-datetime column (harmless no-op if absent)
    "Window Start Datetime",
]


def safe_clone(model):
    """Return an unfitted clone of `model`.

    Falls back to re-instantiation from get_params() if sklearn's clone returns
    something that looks already-fitted (observed with some GPU-backed
    estimators).
    """
    try:
        cloned = clone(model)
        if hasattr(cloned, 'n_features_in_'):
            raise RuntimeError("clone() returned a fitted model")
        return cloned
    except Exception:
        return type(model)(**model.get_params())


# =============================================================================
# CROSS VALIDATION
# =============================================================================
def run_cross_validation(X, y, subjects, subject_time, reps, folds, model_instance, suggested_params):
    """
    Repeated GroupKFold cross-validation with per-fold normalization.
 
    GroupKFold splits by `subjects` (all windows from one subject stay together).
    Prediction aggregation uses `subject_time` (separates early vs late).
 
    Returns
    -------
    rmses : list of float
        Per-fold subject-timepoint-level RMSE across all reps.
    all_subject_time_true : list
        Subject-timepoint-averaged ground truth values.
    all_subject_time_pred : list
        Subject-timepoint-averaged predicted values.
    all_subject_time_ids : list
        SubjectTime IDs corresponding to predictions.
    """
    rmses = []
    all_subject_time_true = []
    all_subject_time_pred = []
    all_subject_time_ids = []
 
    model = safe_clone(model_instance)
    model.set_params(**suggested_params)
 
    for rep in tqdm(range(reps), desc="CV Reps", leave=False):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X.iloc[indices]
        y_shuffled = y.iloc[indices]
        subjects_shuffled = subjects.iloc[indices]
        subject_time_shuffled = subject_time.iloc[indices]
 
        # GroupKFold splits by Subject
        gkf = GroupKFold(n_splits=folds)
        for train_index, val_index in gkf.split(X_shuffled, y_shuffled, subjects_shuffled):
            X_fold_train = X_shuffled.iloc[train_index]
            X_fold_val   = X_shuffled.iloc[val_index]
            y_fold_train = y_shuffled.iloc[train_index]
            y_fold_val   = y_shuffled.iloc[val_index]
            subject_time_val = subject_time_shuffled.iloc[val_index]
 
            X_fold_train_norm, X_fold_val_norm, _ = normalize_features(
                train_data=X_fold_train,
                test_data=X_fold_val,
                feature_cols=X.columns.tolist()
            )
 
            model_fold = safe_clone(model)
            model_fold.fit(X_fold_train_norm, y_fold_train)
 
            window_preds = model_fold.predict(X_fold_val_norm)
 
            # Aggregate by SubjectTime (separates early vs late)
            eval_df = pd.DataFrame({
                'subject_time': subject_time_val.values,
                'true':         y_fold_val.values,
                'pred':         window_preds
            })
 
            subject_level_eval = eval_df.groupby('subject_time').mean()
            subject_true = subject_level_eval['true']
            subject_pred = subject_level_eval['pred']
 
            all_subject_time_true.extend(subject_true.values)
            all_subject_time_pred.extend(subject_pred.values)
            all_subject_time_ids.extend(subject_level_eval.index.tolist())
 
            rmse = np.sqrt(mean_squared_error(subject_true, subject_pred))
            rmses.append(rmse)
 
    return rmses, all_subject_time_true, all_subject_time_pred, all_subject_time_ids


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def run_hyperparam_tuning(X, y, subjects, subject_time, model_instance, model_name,
                          params_search_space, config, folds=5, reps=5, n_trials=30,
                          study_suffix="", run_final_eval=False):
    log_dir = os.path.join(ROOT_PATH, 'log')
    os.makedirs(log_dir, exist_ok=True)
    # Unique study name per outer fold + rep
    study_name = f"{model_name}_{y.name}_{study_suffix}_study"
 
    def objective(trial):
        params_suggested_by_trial = {}
        for param_name, properties in params_search_space.items():
            if properties['type'] == 'float':
                params_suggested_by_trial[param_name] = trial.suggest_float(
                    param_name,
                    properties['range'][0],
                    properties['range'][1],
                    log=properties.get('log', False)
                )
            elif properties['type'] == 'int':
                params_suggested_by_trial[param_name] = trial.suggest_int(
                    param_name,
                    properties['range'][0],
                    properties['range'][1]
                )
            elif properties['type'] == 'categorical':
                params_suggested_by_trial[param_name] = trial.suggest_categorical(
                    param_name,
                    properties['range']
                )
 
        final_rmse_scores, _, _, _ = run_cross_validation(
            X=X,
            y=y,
            subjects=subjects,
            subject_time=subject_time,
            reps=reps,
            folds=folds,
            model_instance=model_instance,
            suggested_params=params_suggested_by_trial
        )
 
        return np.mean(final_rmse_scores) if final_rmse_scores else float('inf')
 
    patience = config.get('optuna_patience', 10)
 
    def early_stopping_callback(study, trial):
        best_trial_number = study.best_trial.number
        current_trial_number = trial.number
        if current_trial_number - best_trial_number >= patience:
            study.stop()
            print(f"\n[EARLY STOP] No improvement in last {patience} trials. "
                  f"Best RMSE: {study.best_value:.4f} at trial {best_trial_number}. "
                  f"Stopping {model_name} on '{y.name}'.")
 
    study = optuna.create_study(
        direction='minimize'
    )
 
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[early_stopping_callback]
    )
 
    best_trial = study.best_trial
    best_params = best_trial.params
 
    print("\n============ Hyperparameter Tuning Results ============")
    print(f"Best Trial #{best_trial.number} for target '{y.name}' with RMSE: {best_trial.value:.4f}")
    print(f"Best hyperparameters: {best_params}")

    if not run_final_eval:
        return best_params, None, None, None, None

    # Re-run final repeated CV with best params for the performance report
    final_rmse_scores, final_true, final_pred, final_ids = run_cross_validation(
        X=X,
        y=y,
        subjects=subjects,
        subject_time=subject_time,
        reps=reps,
        folds=folds,
        model_instance=model_instance,
        suggested_params=best_params
    )
 
    mean_rmse    = np.mean(final_rmse_scores)
    std_err_mean = stats.sem(final_rmse_scores)
    ci = stats.t.interval(
        confidence=config["confidence_level"],
        df=len(final_rmse_scores) - 1,
        loc=mean_rmse,
        scale=std_err_mean
    )
    final_performance_metrics = {
        "mean_rmse": mean_rmse,
        "confidence_interval": ci,
        "std_error": std_err_mean
    }
 
    return best_params, final_performance_metrics, final_true, final_pred, final_ids


# =============================================================================
# PROCESS-ISOLATED OUTER FOLD
# =============================================================================

def build_model(model_name, random_seed):
    """Construct an unfitted model. Must be callable inside spawned workers,
    so models are built from name rather than pickled across the boundary."""
    if model_name == "RandomForest":
        return RandomForestRegressor(
            random_state=random_seed,
        )
    if model_name == "CatBoost":
        return CatBoostRegressor(
            random_state=random_seed,
            bootstrap_type='Bernoulli',
            task_type="GPU",
            verbose=0
        )
    if model_name == "XGBoost":
        return XGBRegressor(
            random_state=random_seed,
            tree_method="hist",
            device="cuda",
            n_jobs=1
        )
    raise ValueError(f"Unknown model name: {model_name}")


def _init_rmm_pool():
    """Enable RAPIDS pool allocator so cuML's per-fit device allocations reuse
    a pool instead of fragmenting GPU memory. No effect on computed results.

    Only call this in processes that train cuML models: CatBoost/XGBoost do
    not allocate through RMM, and the default pool size (half of free GPU
    memory) would starve them. A small initial pool grows on demand.
    """
    try:
        import rmm
        rmm.reinitialize(pool_allocator=True, initial_pool_size=2 * 1024**3)
    except Exception as e:
        print(f"[WARN] RMM pool allocator not enabled: {e}")


def _outer_fold_worker(payload):
    """Run one outer fold — Optuna tuning on the outer-train split, then train
    with best params and evaluate on the outer-test split. Executed in a fresh
    spawned process; returns only small aggregated results."""
    if payload['model_name'] == "RandomForest":
        _init_rmm_pool()

    model_instance = build_model(payload['model_name'], payload['random_seed'])

    best_params, _, _, _, _ = run_hyperparam_tuning(
        X=payload['X_outer_train'],
        y=payload['y_outer_train'],
        subjects=payload['subjects_outer_train'],
        subject_time=payload['subject_time_outer_train'],
        model_instance=model_instance,
        model_name=payload['model_name'],
        params_search_space=payload['search_space'],
        config=payload['config'],
        folds=payload['inner_folds'],
        reps=1,  # inner reps=1; outer reps handle repetition
        n_trials=payload['n_trials'],
        study_suffix=payload['study_suffix']
    )

    # ─── Train on full outer train, evaluate on outer test ───
    X_train_norm, X_test_norm, _ = normalize_features(
        train_data=payload['X_outer_train'],
        test_data=payload['X_outer_test'],
        feature_cols=payload['feature_cols']
    )

    final_model = safe_clone(model_instance)
    final_model.set_params(**best_params)
    final_model.fit(X_train_norm, payload['y_outer_train'])

    window_preds = final_model.predict(X_test_norm)

    # Aggregate by SubjectTime
    eval_df = pd.DataFrame({
        'subject_time': payload['subject_time_outer_test'].values,
        'true': payload['y_outer_test'].values,
        'pred': window_preds
    })
    subject_level = eval_df.groupby('subject_time').mean()

    fold_rmse = float(np.sqrt(
        mean_squared_error(subject_level['true'], subject_level['pred'])
    ))

    return {
        'best_params': best_params,
        'fold_rmse': fold_rmse,
        'subject_true': subject_level['true'].tolist(),
        'subject_pred': subject_level['pred'].tolist(),
        'subject_times': subject_level.index.tolist(),
    }


def main(smoke_test=False):
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
 
    random_seed         = config['random_seed']
    params_search_space = config['params_search_space']
    models_path         = os.path.join(ROOT_PATH, config['result_paths']['models_path'])
    plot_path           = os.path.join(ROOT_PATH, config['result_paths']['figure_path'])
    output_path         = os.path.join(ROOT_PATH, config['result_paths']['output_path'])
    data_path           = os.path.join(ROOT_PATH, config['data_paths']['feature_path'])

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Make checkpoints
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Smoke test configs; otherwise, use the default configs
    if smoke_test:
        outer_folds = 2
        inner_folds = 2
        reps        = 2
        n_trials    = 3
        suffix      = "smoke_test"
        print("\n" + "="*60)
        print("[SMOKE TEST] Running with reduced settings:")
        print(f"  Subjects    : 5")
        print(f"  Outer Folds : {outer_folds}")
        print(f"  Inner Folds : {inner_folds}")
        print(f"  Reps        : {reps}")
        print(f"  Trials      : {n_trials}")
        print(f"  Targets     : first target per score type only")
        print("="*60 + "\n")
    else:
        outer_folds = config['folds']       # 5
        inner_folds = config['folds']       # 5
        reps        = config['reps']        # 5
        n_trials    = config['n_trials']
        suffix      = "full"
 
    smoke_test_errors = []
 
    # ─── WORK-UNIT SELECTION (SLURM job array) ───
    # Each (dataset, target) is an independent unit. Under a job array,
    # SLURM_ARRAY_TASK_ID selects exactly one unit, so N array tasks run on N
    # GPUs in parallel with identical per-unit computation (results unchanged).
    # With no array id set, the full set runs serially (both datasets).
    dataset_targets = {
        "MoCA":     config['moca_targets'],
        "Minibest": config['minibest_targets'],
    }
    all_units = [(nm, tg) for nm in ["MoCA", "Minibest"] for tg in dataset_targets[nm]]

    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_id is not None:
        idx = int(array_id)
        if not (0 <= idx < len(all_units)):
            raise SystemExit(
                f"SLURM_ARRAY_TASK_ID={idx} out of range 0..{len(all_units) - 1}; "
                f"use --array=0-{len(all_units) - 1}."
            )
        selected_units = [all_units[idx]]
        print(f"[ARRAY] Task {idx} → dataset='{all_units[idx][0]}' "
              f"target='{all_units[idx][1]}'")
    else:
        selected_units = all_units
        print(f"[SERIAL] Running all {len(all_units)} (dataset, target) units.")

    names_to_run = list(dict.fromkeys(nm for nm, _ in selected_units))

    for name in names_to_run:
        if name == "MoCA":
            df = pd.read_csv(os.path.join(data_path, "ALL_COHORTS_features_full_MoCA.csv"))
            targets         = config['moca_targets']
            exclude_targets = config['moca_exclude']
        elif name == "Minibest":
            df = pd.read_csv(os.path.join(data_path, "ALL_COHORTS_features_full_Minibest.csv"))
            targets         = config['minibest_targets']
            exclude_targets = config['minibest_exclude']

        # Restrict to the targets selected for this dataset (all of them unless a
        # job array picked a single unit); config order is preserved.
        _selected = [tg for (nm, tg) in selected_units if nm == name]
        targets = [tg for tg in targets if tg in _selected]

        exclude_cols = config['exclude_cols']
 
        df['SubjectTime'] = df['Subject'].astype(str) + "_" + df['Time'].astype(str)
 
        if smoke_test:
            all_subjects = df['Subject'].unique()
            rng = np.random.default_rng(random_seed)
            smoke_subjects = rng.choice(all_subjects, size=min(10, len(all_subjects)), replace=False)
            df = df[df['Subject'].isin(smoke_subjects)].reset_index(drop=True)
            targets = targets[:1]
            print(f"[SMOKE TEST] {name}: using {len(smoke_subjects)} subjects, target='{targets[0]}'")
            print(f"  Subjects: {list(smoke_subjects)}")
 
        print(f"Targets: {targets}")
 
        for target in targets:
            print(f"\n{'='*25} Processing Target: {target} ({name}) {'='*25}")
            try:
                y_target = df[target]

                # Drop identifiers, de-identified temporal metadata 
                cols_to_drop = list(set(NON_FEATURE_COLS) | set(exclude_cols) | set(exclude_targets))
                X_all_feats  = df.drop(columns=cols_to_drop, errors="ignore")

                print("\nColumns in X_all_feats after dropping the columns:", X_all_feats.columns)

                valid_target_idx = y_target.notna()
                X            = X_all_feats[valid_target_idx]
                y            = y_target[valid_target_idx]
                subjects     = df.loc[X.index, 'Subject']
                subject_time = df.loc[X.index, 'SubjectTime']

                
                nan_counts = X.isna().sum()
                nan_features = nan_counts[nan_counts > 0]
                if len(nan_features) > 0:
                    print(f"  Features with NaN (handled natively by models):")
                    for feat, count in nan_features.sort_values(ascending=False).items():
                        print(f"    {feat}: {count:,} ({100*count/len(X):.1f}%)")
                print(f"-----------------------------------\n")
 
                n_subjects = subjects.nunique()
                n_subject_times = subject_time.nunique()
 
                # =============================================================
                # NESTED CROSS-VALIDATION
                # =============================================================
                model_names = ["RandomForest", "CatBoost", "XGBoost"]

                model_performance = []

                # None until a model actually runs and normalizes the full data;
                # kept across the loop so the SHAP-data save at the end still
                # works when every model was resumed from a checkpoint.
                X_all_norm = None

                for model_name in model_names:
                    # ─── Resume: skip models already finished in a prior run ───
                    ckpt_path = os.path.join(
                        checkpoint_dir, f"{name}_{target}_{model_name}_{suffix}.joblib"
                    )
                    if os.path.exists(ckpt_path):
                        model_performance.append(joblib.load(ckpt_path))
                        print(f"[RESUME] {model_name} for '{target}' already complete "
                              f"— loaded checkpoint {ckpt_path}")
                        continue

                    print(f"\n========== {model_name}: Nested CV for {target} ==========")
 
                    current_search_space = params_search_space[model_name]
 
                    # Storage for all outer fold results across all reps
                    all_outer_rmses = []
                    all_outer_true = []
                    all_outer_pred = []
                    all_outer_subject_times = []
                    all_best_params = []
 
                    for rep in range(reps):
                        print(f"\n--- Rep {rep+1}/{reps} ---")
 
                        indices = np.arange(len(X))
                        np.random.seed(random_seed + rep)
                        np.random.shuffle(indices)
                        X_shuffled            = X.iloc[indices].reset_index(drop=True)
                        y_shuffled            = y.iloc[indices].reset_index(drop=True)
                        y_shuffled.name       = y.name
                        subjects_shuffled     = subjects.iloc[indices].reset_index(drop=True)
                        subject_time_shuffled = subject_time.iloc[indices].reset_index(drop=True)
 
                        # ─── Outer loop: GroupKFold by Subject ───
                        outer_gkf = GroupKFold(n_splits=outer_folds)
                        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
                            outer_gkf.split(X_shuffled, y_shuffled, subjects_shuffled)
                        ):
                            print(f"  Outer Fold {fold_idx+1}/{outer_folds}")
 
                            X_outer_train            = X_shuffled.iloc[outer_train_idx]
                            y_outer_train            = y_shuffled.iloc[outer_train_idx]
                            y_outer_train.name       = y.name
                            subjects_outer_train     = subjects_shuffled.iloc[outer_train_idx]
                            subject_time_outer_train = subject_time_shuffled.iloc[outer_train_idx]
 
                            X_outer_test             = X_shuffled.iloc[outer_test_idx]
                            y_outer_test             = y_shuffled.iloc[outer_test_idx]
                            subjects_outer_test      = subjects_shuffled.iloc[outer_test_idx]
                            subject_time_outer_test  = subject_time_shuffled.iloc[outer_test_idx]
 
                            test_subjects = subjects_outer_test.unique()
                            print(f"    Test subjects ({len(test_subjects)}): {list(test_subjects)}")

                            # ─── Tuning + evaluation in a fresh spawned process ───
                            payload = {
                                'model_name': model_name,
                                'random_seed': random_seed,
                                'X_outer_train': X_outer_train,
                                'y_outer_train': y_outer_train,
                                'subjects_outer_train': subjects_outer_train,
                                'subject_time_outer_train': subject_time_outer_train,
                                'X_outer_test': X_outer_test,
                                'y_outer_test': y_outer_test,
                                'subject_time_outer_test': subject_time_outer_test,
                                'feature_cols': X.columns.tolist(),
                                'search_space': current_search_space,
                                'config': config,
                                'inner_folds': inner_folds,
                                'n_trials': n_trials,
                                'study_suffix': f"rep{rep}_fold{fold_idx}",
                            }
                            with ProcessPoolExecutor(max_workers=1, mp_context=_MP_CTX) as executor:
                                fold_result = executor.submit(_outer_fold_worker, payload).result()

                            all_best_params.append(fold_result['best_params'])
                            all_outer_rmses.append(fold_result['fold_rmse'])
                            all_outer_true.extend(fold_result['subject_true'])
                            all_outer_pred.extend(fold_result['subject_pred'])
                            all_outer_subject_times.extend(fold_result['subject_times'])

                            print(f"    Outer Fold RMSE: {fold_result['fold_rmse']:.4f}")
 
                    # ─── Aggregate results across all reps and outer folds ───
                    mean_rmse = np.mean(all_outer_rmses)
                    std_err   = stats.sem(all_outer_rmses)
                    ci = stats.t.interval(
                        confidence=config["confidence_level"],
                        df=len(all_outer_rmses) - 1,
                        loc=mean_rmse,
                        scale=std_err
                    )
 
                    print(f"\n--- {model_name} Nested CV Results for '{target}' ---")
                    print(f"  Mean RMSE:           {mean_rmse:.4f}")
                    print(f"  95% CI:              ({ci[0]:.4f}, {ci[1]:.4f})")
                    print(f"  Num outer folds:     {len(all_outer_rmses)}")
 
                    # ─── Option B: Average across reps before plotting ───
                    results_df = pd.DataFrame({
                        'subject_time': all_outer_subject_times,
                        'true': all_outer_true,
                        'pred': all_outer_pred
                    })
                    subject_avg = results_df.groupby('subject_time').mean()
 
                    print(f"  Unique subject-timepoints in plot: {len(subject_avg)}")
 
                    # Plot with pre-aggregated data
                    if len(subject_avg) > 0:
                        plot_bland_altman(
                            y_true=subject_avg['true'],
                            y_pred=subject_avg['pred'],
                            model_name=model_name, target_name=target,
                            plot_dir=plot_path,
                            subjects=None, 
                            stage="Nested_CV"
                        )
                        plot_pearson_correlation(
                            y_true=subject_avg['true'],
                            y_pred=subject_avg['pred'],
                            model_name=model_name, target_name=target,
                            plot_dir=plot_path,
                            subjects=None, 
                            stage="Nested_CV"
                        )
                        print(f"[PLOT] Finished plotting for {model_name} on {target}.")
 
                    ## ─── Save final model ───
                    # Pick params from the fold with lowest RMSE
                    best_fold_idx = np.argmin(all_outer_rmses)
                    final_best_params = all_best_params[best_fold_idx]
 
                    # Retrain on all data for the saved model
                    X_all_norm, _, final_scaler = normalize_features(
                        train_data=X,
                        test_data=X,
                        feature_cols=X.columns.tolist()
                    )
                    final_model_all = build_model(model_name, random_seed)
                    final_model_all.set_params(**final_best_params)
                    final_model_all.fit(X_all_norm, y)
 
                    model_path = os.path.join(
                        models_path, f"{model_name}_{name}_{target}_{suffix}.joblib"
                    )
                    joblib.dump(final_model_all, model_path)
                    print(f"Final model saved to: {model_path}")
 
                    perf_entry = {
                        model_name: {
                            "rmse": mean_rmse,
                            "confidence_interval": ci,
                            "std_error": std_err,
                            "performance_metrics": {
                                "mean_rmse": mean_rmse,
                                "confidence_interval": ci,
                                "std_error": std_err
                            },
                            "best_params": final_best_params,
                        }
                    }
                    model_performance.append(perf_entry)

                    joblib.dump(perf_entry, ckpt_path)
                    print(f"[CHECKPOINT] Saved {model_name} for '{target}' to {ckpt_path}")

                # ─── Select best model by nested CV RMSE ───
                best_model_dict = min(
                    model_performance,
                    key=lambda x: list(x.values())[0]['rmse']
                )
                best_model_name = list(best_model_dict.keys())[0]
                best_model_info = best_model_dict[best_model_name]
 
                print("\n--- Overall Best Model Performance ---")
                print(f"Best model for '{target}': {best_model_name}")
                print(f"Best Parameters: {best_model_info['best_params']}")
                ci = best_model_info['confidence_interval']
                print(f"  Mean Nested CV RMSE: {best_model_info['rmse']:.4f}")
                print(f"  95% CI:              ({ci[0]:.4f}, {ci[1]:.4f})")
 
                model_performance_path = os.path.join(
                    output_path,
                    f"{name}_{target}_all_models_performance_{suffix}.json"
                )
                with open(model_performance_path, 'w') as f:
                    def default_converter(o):
                        if isinstance(o, np.ndarray): return o.tolist()
                        if isinstance(o, tuple): return list(o)
                        raise TypeError(
                            f"Object of type {o.__class__.__name__} is not JSON serializable"
                        )
                    json.dump(model_performance, f, indent=4, default=default_converter)
                print(f"All model performance details saved to: {model_performance_path}")
 
                # Save normalized data for SHAP analysis.
                xtrain_path = os.path.join(
                    data_path, f"X_train_norm_{name}_{target}_{suffix}.joblib"
                )
                if X_all_norm is None:
                    X_all_norm, _, _ = normalize_features(
                        train_data=X, test_data=X, feature_cols=X.columns.tolist()
                    )
                joblib.dump(X_all_norm, xtrain_path)
                print(f"Normalized data saved to: {xtrain_path}")
 
            except Exception as e:
                msg = f"[ERROR] {name} / {target}: {type(e).__name__}: {e}"
                print(msg)
                smoke_test_errors.append(msg)
                continue
 
    if smoke_test:
        print("\n" + "="*60)
        print("[SMOKE TEST] Complete.")
        if smoke_test_errors:
            print(f"  {len(smoke_test_errors)} error(s) found:")
            for err in smoke_test_errors:
                print(f"    {err}")
        else:
            print("  All code paths completed without errors.")
            print("  Set smoke_test=False in __main__ to run the full experiment.")
        print("="*60 + "\n")


if __name__ == "__main__":
    main(smoke_test=False)