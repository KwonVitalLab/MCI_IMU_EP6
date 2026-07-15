"""
Fairness Analysis Pipeline using Cross-Validated Predictions.

Original fairness setting:
  1. Repeated GroupKFold CV generates out-of-fold predictions.
  2. GroupKFold splits by Subject.
  3. Feature rows with NaN values are removed before fitting.
  4. Window-level predictions are aggregated to the subject level.
  5. Wasserstein Barycenter mitigation is applied to all subject-level
     cross-validated predictions as a post-processing step.

Output:
  - fairness_result_{Assessment}_{Target}_{SensitiveAttr}.csv
"""

import os
import json
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error

from holisticai.bias.metrics import regression_bias_metrics
from holisticai.bias.mitigation import WassersteinBarycenter

from cuml.ensemble import RandomForestRegressor as cuRF
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from utils import normalize_features


# =============================================================================
# CONFIG
# =============================================================================

ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH = os.path.join(ROOT_PATH, "config.yaml")

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

output_path = os.path.join(ROOT_PATH, config["result_paths"]["output_path"])
fairness_path = os.path.join(ROOT_PATH, config["result_paths"]["fairness_path"])
data_path = os.path.join(ROOT_PATH, config["data_paths"]["feature_path"])
label_path = os.path.join(ROOT_PATH, config["data_paths"]["label_path"])

exclude_cols = config["exclude_cols"]
random_seed = config["random_seed"]
n_folds = config["folds"]
n_reps = config["reps"]

os.makedirs(fairness_path, exist_ok=True)

np.random.seed(random_seed)


NON_FEATURE_COLS = [
    "Subject",
    "Cohort",
    "Time",
    "SubjectTime",
    "VisitIndex",
    "CohortProgramDay",
    "Window Start TimeOfDay",
    "Window Start Elapsed (s)",
    "Window Start Datetime",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(name):
    if name == "MoCA":
        source_file = "ALL_COHORTS_features_full_MoCA.csv"

        moca_label_file = pd.read_csv(
            os.path.join(label_path, "Final_MoCA.csv")
        )
        moca_sub_label_file = pd.read_csv(
            os.path.join(label_path, "Final_MoCA_Sub.csv")
        )

        label_file = pd.merge(
            moca_label_file,
            moca_sub_label_file,
            on="Subject",
            how="outer",
        )

        columns_to_unify = ["MoCA", "Age", "Race", "Ethnicity", "Sex"]

        for col in columns_to_unify:
            label_file[col] = label_file[f"{col}_x"].fillna(
                label_file[f"{col}_y"]
            )
            label_file.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)

        label_file = label_file.drop_duplicates(subset="Subject", keep="first")

    elif name == "Minibest":
        source_file = "ALL_COHORTS_features_full_Minibest.csv"

        label_file = pd.read_csv(
            os.path.join(label_path, "Final_Minibest.csv")
        )
        label_file = label_file.drop_duplicates(subset="Subject", keep="first")

    else:
        raise ValueError(f"Unknown assessment name: {name}")

    feature_df = pd.read_csv(os.path.join(data_path, source_file))

    demographics = label_file[["Subject", "Sex", "Race"]].copy()
    assert demographics["Subject"].is_unique

    df = pd.merge(feature_df, demographics, on="Subject", how="left")

    print(f"\nLoaded {name} data: {df.shape}")

    return df


def clean_data(df, cols_to_drop, target, sensitive_cols):
    """
    Clean data using the fairness setting.

    Important:
      - Feature rows with NaNs are dropped.
      - Target NaNs are dropped.
      - Missing sensitive attributes are dropped.
      - Aggregation is performed later at the subject level.
    """
    y_target = df[target]
    sensitive_att_df = df[sensitive_cols]

    X_all_feats = df.drop(columns=cols_to_drop, errors="ignore")

    # -------------------------------------------------------------------------
    # Original fairness behavior: remove rows with feature NaNs
    # -------------------------------------------------------------------------
    X_cleaned = X_all_feats.dropna()

    removed_n = len(X_all_feats) - len(X_cleaned)

    y_aligned = y_target.loc[X_cleaned.index]
    subjects_aligned = df.loc[X_cleaned.index, "Subject"]
    sensitive_att_aligned = sensitive_att_df.loc[X_cleaned.index]

    # -------------------------------------------------------------------------
    # Drop missing target values
    # -------------------------------------------------------------------------
    valid_target = y_aligned.notna()

    X = X_cleaned.loc[valid_target]
    y = y_aligned.loc[valid_target]
    subjects = subjects_aligned.loc[valid_target]
    sensitive_att = sensitive_att_aligned.loc[valid_target]

    # -------------------------------------------------------------------------
    # Drop missing sensitive attributes
    # -------------------------------------------------------------------------
    valid_sensitive = sensitive_att.notna().all(axis=1)

    X = X.loc[valid_sensitive]
    y = y.loc[valid_sensitive]
    subjects = subjects.loc[valid_sensitive]
    sensitive_att = sensitive_att.loc[valid_sensitive]

    subject_demo = pd.DataFrame({
        "Subject": subjects.values,
        "Sex": sensitive_att["Sex"].values,
        "Race": sensitive_att["Race"].values,
    }).drop_duplicates("Subject")

    object_cols = X.select_dtypes(include="object").columns.tolist()
    if object_cols:
        raise ValueError(f"Object dtype columns remain in X: {object_cols}")

    return X, y, subjects, sensitive_att


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_best_model_info(name, target):
    model_performance_path = os.path.join(
        output_path,
        f"{name}_{target}_all_models_performance_full.json",
    )

    try:
        with open(model_performance_path, "r") as file:
            model_performance = json.load(file)
    except FileNotFoundError:
        print(f"ERROR: Performance file not found for {target}.")
        print(f"Expected path: {model_performance_path}")
        return None, None

    best_model_dict = min(
        model_performance,
        key=lambda x: list(x.values())[0]["rmse"],
    )

    best_model_name = list(best_model_dict.keys())[0]
    best_model_info = best_model_dict[best_model_name]
    best_params = best_model_info.get("best_params", {})

    print(f"Best model: {best_model_name}")
    print(f"Best params: {best_params}")

    return best_model_name, best_params


def instantiate_model(model_name, params):
    if model_name == "RandomForest":
        rf_params = {k: v for k, v in params.items() if k != "n_jobs"}
        return cuRF(random_state=random_seed, **rf_params)

    if model_name == "XGBoost":
        return XGBRegressor(
            device="cuda",
            tree_method="hist",
            n_jobs=1,
            random_state=random_seed,
            verbosity=0,
            **params,
        )

    if model_name == "CatBoost":
        return CatBoostRegressor(
            bootstrap_type="Bernoulli",
            task_type="GPU",
            verbose=0,
            random_state=random_seed,
            **params,
        )

    raise ValueError(f"Unknown model: {model_name}")


def safe_clone(model):
    try:
        cloned = clone(model)

        if hasattr(cloned, "n_features_in_"):
            raise RuntimeError("clone() returned a fitted model")

        return cloned

    except Exception:
        return type(model)(**model.get_params())


def preds_to_numpy(preds):
    if hasattr(preds, "to_pandas"):
        preds = preds.to_pandas()

    if type(preds).__module__.startswith("cupy"):
        preds = preds.get()

    return np.asarray(preds).ravel()


# =============================================================================
# STAGE 1: CROSS-VALIDATED SUBJECT-LEVEL PREDICTIONS
# =============================================================================

def run_cv_predictions(X, y, subjects, sensitive_att, model_name, params):
    """
    Generate repeated GroupKFold predictions and aggregate them to subject level.

    Grouping:
      - GroupKFold uses Subject as the grouping variable.
      - All windows from the same subject remain in the same fold.

    Aggregation:
      - Validation predictions are averaged per subject within each fold.
      - Repeated predictions are averaged again across repetitions.
    """
    model_instance = instantiate_model(model_name, params)

    subject_data = {}
    subject_demographics = {}

    for rep in tqdm(range(n_reps), desc="CV Reps", leave=False):
        indices = np.arange(len(X))

        rng = np.random.default_rng(random_seed + rep)
        rng.shuffle(indices)

        X_shuf = X.iloc[indices]
        y_shuf = y.iloc[indices]
        subjects_shuf = subjects.iloc[indices]
        sensitive_shuf = sensitive_att.iloc[indices]

        gkf = GroupKFold(n_splits=n_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X_shuf, y_shuf, subjects_shuf)
        ):
            X_train = X_shuf.iloc[train_idx]
            X_val = X_shuf.iloc[val_idx]
            y_train = y_shuf.iloc[train_idx]
            y_val = y_shuf.iloc[val_idx]
            subjects_val = subjects_shuf.iloc[val_idx]
            sensitive_val = sensitive_shuf.iloc[val_idx]

            X_train_norm, X_val_norm, _ = normalize_features(
                train_data=X_train,
                test_data=X_val,
                feature_cols=X.columns.tolist(),
            )

            model_fold = safe_clone(model_instance)
            model_fold.fit(X_train_norm, y_train)

            val_preds = preds_to_numpy(model_fold.predict(X_val_norm))

            val_df = pd.DataFrame({
                "subject": subjects_val.values,
                "y_true": y_val.values,
                "y_pred": val_preds,
                "Sex": sensitive_val["Sex"].values,
                "Race": sensitive_val["Race"].values,
            })

            val_subject = (
                val_df.groupby("subject")
                .agg({
                    "y_true": "mean",
                    "y_pred": "mean",
                    "Sex": "first",
                    "Race": "first",
                })
                .reset_index()
            )

            for _, row in val_subject.iterrows():
                subject = row["subject"]

                if subject not in subject_data:
                    subject_data[subject] = {
                        "y_true": [],
                        "y_pred": [],
                    }
                    subject_demographics[subject] = {
                        "Sex": row["Sex"],
                        "Race": row["Race"],
                    }

                subject_data[subject]["y_true"].append(row["y_true"])
                subject_data[subject]["y_pred"].append(row["y_pred"])

    rows = []

    for subject, data in subject_data.items():
        rows.append({
            "subject": subject,
            "y_true": np.mean(data["y_true"]),
            "y_pred": np.mean(data["y_pred"]),
            "Sex": subject_demographics[subject]["Sex"],
            "Race": subject_demographics[subject]["Race"],
        })

    result_df = pd.DataFrame(rows)

    return result_df


# =============================================================================
# STAGE 2: WASSERSTEIN BARYCENTER MITIGATION
# =============================================================================

def get_group_masks(result_df, att_name):
    if att_name == "Sex":
        group_a = result_df["Sex"] == "Male"
        group_b = result_df["Sex"] == "Female"
        label_a, label_b = "Male", "Female"

    elif att_name == "Race":
        group_a = result_df["Race"] == "White"
        group_b = result_df["Race"] != "White"
        label_a, label_b = "White", "Non-White"

    else:
        raise ValueError(f"Unknown sensitive attribute: {att_name}")

    return group_a, group_b, label_a, label_b


def apply_wb_mitigation(result_df, att_name):
    group_a, group_b, label_a, label_b = get_group_masks(result_df, att_name)

    n_a = int(group_a.sum())
    n_b = int(group_b.sum())

    print(f"    WB mitigation for {att_name}: {label_a}={n_a}, {label_b}={n_b}")

    if n_a == 0 or n_b == 0:
        print("    WARNING: Empty group. Cannot mitigate.")
        return result_df["y_pred"].values

    if min(n_a, n_b) < 2:
        print("    WARNING: Very small group size. WB may be unreliable.")

    try:
        wb = WassersteinBarycenter()
        wb.fit(result_df["y_pred"], group_a, group_b)

        transformed = wb.transform(result_df["y_pred"], group_a, group_b)
        y_mitigated = transformed["y_pred"]

        if hasattr(y_mitigated, "values"):
            return y_mitigated.values

        return np.asarray(y_mitigated).ravel()

    except Exception as error:
        print(f"    WARNING: WB failed for {att_name}: {error}")
        return result_df["y_pred"].values


# =============================================================================
# FAIRNESS METRICS
# =============================================================================

def compute_fairness_metrics(result_df, att_name, pred_col="y_pred"):
    group_a, group_b, label_a, label_b = get_group_masks(result_df, att_name)

    n_a = int(group_a.sum())
    n_b = int(group_b.sum())


    if n_a == 0 or n_b == 0:
        print("    WARNING: Empty group. Cannot compute fairness metrics.")
        return None

    metrics_df = regression_bias_metrics(
        group_a,
        group_b,
        result_df[pred_col],
        result_df["y_true"],
        metric_type="group",
    )

    # Keep only the requested fairness metrics.
    metrics_to_keep = [
        "Average Score Difference",
        "Statistical Parity AUC",
    ]

    missing_metrics = [
        metric for metric in metrics_to_keep
        if metric not in metrics_df.index
    ]

    if missing_metrics:
        raise ValueError(
            f"Expected fairness metrics not found: {missing_metrics}. "
            f"Available metrics: {metrics_df.index.tolist()}"
        )

    metrics_df = metrics_df.loc[metrics_to_keep].copy()

    rmse = root_mean_squared_error(
        result_df["y_true"],
        result_df[pred_col],
    )

    metrics_df.loc["RMSE"] = {
        "Value": rmse,
    }

    return metrics_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    sensitive_cols = config.get("sensitive_cols", ["Sex", "Race"])

    print("\n" + "=" * 80)
    print("FAIRNESS ANALYSIS")
    print("=" * 80)

    for name in ["MoCA", "Minibest"]:
        df = load_data(name)

        if name == "MoCA":
            targets = config["moca_targets"]
            exclude_targets = config["moca_exclude"]

        elif name == "Minibest":
            targets = config["minibest_targets"]
            exclude_targets = config["minibest_exclude"]

        else:
            raise ValueError(f"Unknown assessment name: {name}")

        cols_to_drop = list(
            set(NON_FEATURE_COLS)
            | set(exclude_cols)
            | set(exclude_targets)
            | set(sensitive_cols)
        )

        for target in tqdm(targets, desc=f"{name} Targets", leave=False):
            print(f"\n{'=' * 25} {target} ({name}) {'=' * 25}")

            X, y, subjects, sensitive_att = clean_data(
                df=df,
                target=target,
                cols_to_drop=cols_to_drop,
                sensitive_cols=sensitive_cols,
            )

            if X.empty:
                print(f"No data for target '{target}'. Skipping.")
                continue

            model_name, params = get_best_model_info(name, target)

            if model_name is None:
                continue

            # -----------------------------------------------------------------
            # Stage 1: CV predictions
            # -----------------------------------------------------------------
            print(f"\nStage 1: {n_reps} reps x {n_folds} folds with {model_name}")

            result_df = run_cv_predictions(
                X=X,
                y=y,
                subjects=subjects,
                sensitive_att=sensitive_att,
                model_name=model_name,
                params=params,
            )

            # -----------------------------------------------------------------
            # Stage 2: WB mitigation
            # -----------------------------------------------------------------
            print("\nStage 2: Wasserstein Barycenter mitigation")

            for att_name in sensitive_cols:
                mit_col = f"y_pred_mit_{att_name.lower()}"
                result_df[mit_col] = apply_wb_mitigation(
                    result_df=result_df,
                    att_name=att_name,
                )

            # -----------------------------------------------------------------
            # Fairness metrics
            # -----------------------------------------------------------------
            for att_name in sensitive_cols:
                print(f"\n  {'#' * 10} {att_name} {'#' * 10}")

                mit_col = f"y_pred_mit_{att_name.lower()}"

                print("  --- Baseline ---")
                baseline_df = compute_fairness_metrics(
                    result_df=result_df,
                    att_name=att_name,
                    pred_col="y_pred",
                )

                print("  --- Mitigated (WB) ---")
                mitigated_df = compute_fairness_metrics(
                    result_df=result_df,
                    att_name=att_name,
                    pred_col=mit_col,
                )

                parts = []

                if baseline_df is not None:
                    baseline_df = baseline_df.copy()
                    baseline_df.rename(
                        columns={"Value": f"Baseline_{att_name}"},
                        inplace=True,
                    )
                    parts.append(baseline_df[f"Baseline_{att_name}"])
                
                # {"Metric": ["ASD", "SPAUC", "RMSE"], "Baseline_Sex": [....]}

                if mitigated_df is not None:
                    mitigated_df = mitigated_df.copy()
                    mitigated_df.rename(
                        columns={"Value": f"Mitigated_{att_name}"},
                        inplace=True,
                    )
                    parts.append(mitigated_df[f"Mitigated_{att_name}"])

                # {"Metric": ["ASD", "SPAUC", "RMSE"], "Mitigated_Sex": [....]}

                if parts:
                    result_out = pd.concat(parts, axis=1)
                    result_out["is_mitigated"] = (
                        result_out[f"Mitigated_{att_name}"].abs()
                        < result_out[f"Baseline_{att_name}"].abs()
                    )
                else:
                    result_out = pd.DataFrame()

                fairness_result_path = os.path.join(
                    fairness_path,
                    f"fairness_result_{name}_{target}_{att_name}.csv",
                )

                result_out.to_csv(fairness_result_path)
                print(f"  Saved: {fairness_result_path}")


if __name__ == "__main__":
    main()