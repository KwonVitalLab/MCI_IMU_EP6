"""
SHAP feature importance on held-out windows, using the saved outer-fold models.

What it does (for one dataset / target / model, default Minibest / Minibest / XGBoost)
  1. Rebuilds X exactly as train_models_revision.py (same dropped columns, same rows).
  2. For every rep x fold model in fold_models:
       - test rows  = row_ids of that rep/fold in window_preds_*.csv.gz
       - normalization refit on that fold's outer-train participants with the same
         normalize_features() call as training
       - CHECK: model predictions on the rebuilt test windows must match the saved
         window predictions (confirms rows and normalization are identical)
       - exact TreeSHAP (XGBoost pred_contribs / CatBoost ShapValues)
       - CHECK: SHAP values + bias sum to the prediction
  3. Subject-fair aggregation (same rule as the manuscript):
       mean |SHAP| per participant over that participant's held-out windows,
       averaged over reps, then averaged across participants with equal weight.
  4. Stability across the rep x fold models: how often each feature ranks in the top k.
  5. Direction: Spearman correlation between feature value and SHAP value
     (held-out windows), i.e. whether higher feature values push predictions up or down.

Outputs ({out_dir})
  shap_importance_{dataset}_{target}_{model}.csv    importance, SD across participants,
                                                    top-k frequency, direction
  shap_subject_{dataset}_{target}_{model}.csv       per-participant mean |SHAP|
  shap_fold_{dataset}_{target}_{model}.csv          per-model (rep x fold) importance
  shap_bar_... .png / shap_beeswarm_... .png        (beeswarm only if `shap` is installed)

Usage
  python run_shap.py --config config.yaml \
      --preds_dir results/performance/predictions \
      --fold_models_dir results/performance/fold_models \
      --out_dir results/shap \
      [--dataset Minibest --target Minibest --model XGBoost] [--max_windows_per_subject 2000]
"""

import argparse
import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from utils import normalize_features

NON_FEATURE_COLS = [  # identical to train_models_revision.py
    "Subject", "Cohort", "Time", "SubjectTime",
    "VisitIndex", "CohortProgramDay",
    "Window Start TimeOfDay", "Window Start Elapsed (s)",
    "Window Start Datetime",
]
SOURCE_FILES = {"MoCA": "ALL_COHORTS_features_full_MoCA.csv",
                "Minibest": "ALL_COHORTS_features_full_Minibest.csv"}


# =============================================================================
# Data (rebuilt exactly as in train_models_revision.py)
# =============================================================================
_DF_CACHE = {}


def build_X(config, root, dataset, target):
    data_path = os.path.join(root, config["data_paths"]["feature_path"])
    if dataset not in _DF_CACHE:                       # read once, reuse for every target
        _DF_CACHE[dataset] = pd.read_csv(os.path.join(data_path, SOURCE_FILES[dataset]))
    df = _DF_CACHE[dataset]
    exclude_targets = config["moca_exclude"] if dataset == "MoCA" else config["minibest_exclude"]
    cols_to_drop = list(set(NON_FEATURE_COLS) | set(config["exclude_cols"]) | set(exclude_targets))
    X_all = df.drop(columns=cols_to_drop, errors="ignore")
    valid = df[target].notna()
    X = X_all[valid]                         # original df index = row_id in window_preds
    subjects = df.loc[X.index, "Subject"].astype(str)
    y = df.loc[X.index, target]
    return X, y, subjects


def as_frame(A, columns, index):
    if isinstance(A, pd.DataFrame):
        return A
    return pd.DataFrame(np.asarray(A), columns=columns, index=index)


# =============================================================================
# SHAP per model type (exact TreeSHAP; last column = bias / expected value)
# =============================================================================
def _rf_tree_dicts(rf, X_train):
    """Trees of a (cuML-converted) sklearn forest as shap dict models, with node
    covers recomputed by routing the training windows through each tree (the
    converted trees do not carry valid node sample counts)."""
    Xt = np.asarray(X_train, dtype=np.float32)
    n_trees = len(rf.estimators_)
    trees = []
    n_zero_leaves, n_leaves = 0, 0
    for est in rf.estimators_:
        t = est.tree_
        # leaf counts via apply() (safe traversal), then sum children bottom-up
        leaf = est.apply(Xt)
        cover = np.bincount(leaf, minlength=t.node_count).astype(np.float64)
        left_, right_ = t.children_left, t.children_right
        is_leaf = left_ == -1
        zero = is_leaf & (cover == 0)
        n_zero_leaves += int(zero.sum()); n_leaves += int(is_leaf.sum())
        cover[zero] = 1e-6          # leaves no training window reaches: negligible weight
        for i in range(t.node_count - 1, -1, -1):
            if not is_leaf[i]:
                cover[i] = cover[left_[i]] + cover[right_[i]]
        mgl = getattr(t, "missing_go_to_left", None)
        default = np.where(np.asarray(mgl).astype(bool), left_, right_) if mgl is not None else right_.copy()
        trees.append(dict(
            children_left=left_, children_right=right_, children_default=default.astype(np.int32),
            features=t.feature.astype(np.int32), thresholds=t.threshold.astype(np.float64),
            values=t.value[:, 0, :].astype(np.float64) / n_trees,   # forest = mean of trees
            node_sample_weight=cover))
    print(f"    cover: {n_zero_leaves:,} of {n_leaves:,} leaves not reached by training windows "
          f"({n_zero_leaves / max(n_leaves, 1):.2%})", flush=True)
    return {"trees": trees}


def shap_values(model, model_name, Xn, X_train=None):
    if model_name == "XGBoost":
        import xgboost as xgb
        booster = model.get_booster()
        try:
            booster.set_param({"device": "cpu"})
        except Exception:
            pass
        names = booster.feature_names
        dm = xgb.DMatrix(np.asarray(Xn, dtype=np.float32), missing=np.nan,
                         feature_names=names if names is not None else None)
        contrib = booster.predict(dm, pred_contribs=True)
    elif model_name == "CatBoost":
        from catboost import Pool
        contrib = model.get_feature_importance(Pool(Xn), type="ShapValues")
    elif model_name == "RandomForest":
        # import shap
        # explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        # sv = np.asarray(explainer.shap_values(np.asarray(Xn, dtype=float), check_additivity=False), dtype=float)
        # ev = float(np.ravel(explainer.expected_value)[0])
        # return sv, np.full(len(sv), ev)
        import shap
        from joblib import Parallel, delayed
        tree_model = _rf_tree_dicts(model, X_train)
        Xa = np.asarray(Xn, dtype=np.float32).astype(np.float64)   # same precision as sklearn
        n_jobs = len(os.sched_getaffinity(0))
        chunks = [ix for ix in np.array_split(np.arange(len(Xa)), n_jobs) if len(ix)]
        print(f"    TreeSHAP: {len(Xa):,} windows, {len(chunks)} chunks, n_jobs={n_jobs}", flush=True)

        def _one(ix):
            ex = shap.TreeExplainer(tree_model, feature_perturbation="tree_path_dependent")
            sv_ = np.asarray(ex.shap_values(Xa[ix], check_additivity=False), dtype=np.float64)
            return sv_, float(np.ravel(ex.expected_value)[0])

        res = Parallel(n_jobs=n_jobs)(delayed(_one)(ix) for ix in chunks)
        sv = np.vstack([r[0] for r in res])
        return sv, np.full(len(sv), res[0][1])
    else:
        raise SystemExit(f"[ERROR] unsupported model type {type(model)} for {model_name}")
    contrib = np.asarray(contrib, dtype=float)
    return contrib[:, :-1], contrib[:, -1]


def build_model(model_name, random_seed):
    """Identical to train_models_revision.build_model (lazy imports)."""
    if model_name == "RandomForest":
        from cuml.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=random_seed)
    if model_name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_state=random_seed, bootstrap_type="Bernoulli",
                                 task_type="GPU", verbose=0)
    if model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=random_seed, tree_method="hist", device="cuda", n_jobs=1)
    raise ValueError(model_name)


def fold_sources(args, config, unit, X, subjects, model_name):
    """Saved fold models where present; refit the missing ones from fold_summary
    (same test participants, same per-fold best_params, same row order and seed)."""
    saved = {}
    for mf in glob.glob(os.path.join(args.fold_models_dir, f"{unit}_rep*_fold*.joblib")):
        rep, fold = map(int, re.search(r"_rep(\d+)_fold(\d+)\.joblib$", mf).groups())
        saved[(rep, fold)] = mf
    fs_file = os.path.join(args.preds_dir, f"fold_summary_{unit}.csv")
    if not os.path.exists(fs_file):
        if not saved:
            raise SystemExit(f"[ERROR] no fold models and no {os.path.abspath(fs_file)}")
        for (rep, fold), mf in sorted(saved.items()):
            yield rep, fold, (lambda mf=mf: joblib.load(mf)), False
        return
    fs = pd.read_csv(fs_file)
    seed = config["random_seed"]
    for _, r in fs.sort_values(["rep", "fold"]).iterrows():
        rep, fold = int(r["rep"]), int(r["fold"])
        if (rep, fold) in saved:
            yield rep, fold, (lambda mf=saved[(rep, fold)]: joblib.load(mf)), False
            continue
        test_subj = set(str(t) for t in json.loads(r["test_subjects"]))
        params = json.loads(r["best_params"])

        def load(rep=rep, test_subj=test_subj, params=params):
            order = X.index[np.random.default_rng(seed + rep).permutation(len(X))]
            tr = order[~subjects.loc[order].isin(test_subj).to_numpy()]
            X_tr_n, _, _ = normalize_features(train_data=X.loc[tr], test_data=X.loc[tr[:1]],
                                              feature_cols=X.columns.tolist())
            m = build_model(model_name, seed)
            m.set_params(**params)
            m.fit(X_tr_n, _Y[0].loc[tr])
            return dict(model=m, feature_cols=X.columns.tolist(),
                        train_subjects=sorted(set(subjects.loc[tr])))
        print(f"  [REFIT] rep{rep} fold{fold}: no saved model, rebuilding from fold_summary", flush=True)
        yield rep, fold, load, True


_Y = [None]   # target series for the refit loader (set in run_one)


def predict(model, Xn):
    p = model.predict(Xn)
    if hasattr(p, "to_numpy"):
        p = p.to_numpy()
    if type(p).__module__.startswith("cupy"):
        p = p.get()
    return np.asarray(p, dtype=float).ravel()


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--fold_models_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="Minibest")
    ap.add_argument("--target", default="Minibest",
                    help="a target name, or 'all' for every target of --dataset in config")
    ap.add_argument("--model", default="best",
                    help="'best' = lowest mean outer-fold RMSE in fold_summary (as in the main "
                         "analysis), or a model name (XGBoost / CatBoost / RandomForest)")
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_windows_per_subject", type=int, default=None,
                    help="optional cap per participant per fold for speed (seeded)")
    ap.add_argument("--pred_tol", type=float, default=1e-3,
                    help="max allowed |rebuilt - saved| window prediction difference")
    ap.add_argument("--beeswarm_per_subject", type=int, default=300,
                    help="windows per participant in the beeswarm sample (rep 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=str, default=None,
                    help="comma-separated reps to process, e.g. '0' (quick beeswarm-only run). "
                         "The importance CSV then covers only these reps: use the full-run CSV for Fig 7.")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(args.config))
    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    targets = ((config["moca_targets"] if args.dataset == "MoCA" else config["minibest_targets"])
               if args.target == "all" else [args.target])
    failed = []
    for target in targets:
        try:
            model_name = args.model if args.model != "best" else best_model(args, target)
            print(f"\n{'=' * 20} {args.dataset} / {target} / {model_name} {'=' * 20}")
            run_one(args, config, root, target, model_name)
        except SystemExit as e:              # keep going with the other targets
            print(f"[FAILED] {target}: {e}")
            failed.append(target)
    if failed:
        raise SystemExit(f"\n[DONE WITH ERRORS] failed targets: {failed}")
    print(f"\n[DONE] {len(targets)} target(s): {targets}")


def best_model(args, target):
    files = glob.glob(os.path.join(args.preds_dir, f"fold_summary_{args.dataset}_{target}_*_{args.suffix}.csv"))
    if not files:
        raise SystemExit(f"[ERROR] no fold_summary files for {args.dataset}/{target} in {args.preds_dir}")
    fs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return fs.groupby("model")["test_rmse"].mean().idxmin()


def run_one(args, config, root, target, model_name):
    tag = f"{args.dataset}_{target}_{model_name}"
    unit = f"{args.dataset}_{target}_{model_name}_{args.suffix}"

    X, y, subjects = build_X(config, root, args.dataset, target)
    feats = X.columns.tolist()
    wp_file = os.path.join(args.preds_dir, f"window_preds_{unit}.csv.gz")
    if not os.path.exists(wp_file):
        raise SystemExit(f"[ERROR] missing {os.path.abspath(wp_file)}")
    wp = pd.read_csv(wp_file, usecols=["rep", "fold", "row_id", "pred"])
    _Y[0] = y
    sources = list(fold_sources(args, config, unit, X, subjects, model_name))
    if args.reps is not None:
        keep_reps = {int(r) for r in args.reps.split(",")}
        sources = [src for src in sources if src[0] in keep_reps]
    print(f"{tag}: {len(X):,} windows, {subjects.nunique()} participants, {len(sources)} fold models")

    rng = np.random.default_rng(args.seed)
    subj_rows, fold_rows = [], []
    dir_num = np.zeros(len(feats)); dir_n = 0                  # for pooled direction estimate
    bees_X, bees_S, bees_B, bees_subj = [], [], [], []         # sample for the beeswarm

    for rep, fold, loader, refit in sources:
        bundle = loader()
        if refit:   # save the rebuilt fold model so later runs load it instead of refitting
            os.makedirs(args.fold_models_dir, exist_ok=True)
            joblib.dump(bundle, os.path.join(args.fold_models_dir, f"{unit}_rep{rep}_fold{fold}.joblib"))
        model = bundle["model"]
        cuml_model = None
        if type(model).__module__.startswith("cuml"):
            # Explain the SAME trees with shap's standard TreeExplainer: convert the saved
            # cuML forest to scikit-learn (no retraining). Checked against saved preds below.
            if not hasattr(model, "as_sklearn"):
                raise SystemExit("[ERROR] this cuML version has no as_sklearn(); cannot convert "
                                 "the saved RandomForest without retraining.")
            cuml_model, model = model, model.as_sklearn()
        if bundle.get("feature_cols") and list(bundle["feature_cols"]) != feats:
            raise SystemExit(f"[ERROR] feature columns differ from training (rep{rep} fold{fold})")

        w = wp[(wp["rep"] == rep) & (wp["fold"] == fold)]
        test_idx = w["row_id"].to_numpy()
        train_idx = X.index[subjects.isin(set(bundle["train_subjects"]))]
        X_tr_n, X_te_n, _ = normalize_features(train_data=X.loc[train_idx], test_data=X.loc[test_idx],
                                               feature_cols=feats)
        X_te_n = as_frame(X_te_n, feats, test_idx)
        subj_te = subjects.loc[test_idx].to_numpy()

        # CHECK 1: rebuilt inputs reproduce the saved out-of-fold predictions
        # (for cuML RF this checks the converted sklearn forest, i.e. conversion + NaN handling)
        p = predict(model, np.asarray(X_te_n, dtype=float) if cuml_model is not None else X_te_n)
        dmax = float(np.max(np.abs(p - w["pred"].to_numpy())))
        pcorr = float(np.corrcoef(p, w["pred"].to_numpy())[0, 1]) if np.std(p) > 0 else float("nan")
        if refit:
            # refit models may differ slightly from the originals (GPU non-determinism):
            # record agreement instead of stopping
            if not (pcorr >= 0.99):
                print(f"  [WARN] rep{rep} fold{fold}: refit model agrees poorly with the saved "
                      f"predictions (r = {pcorr:.4f}, max |diff| = {dmax:.3g})")
        elif dmax > args.pred_tol:
            raise SystemExit(f"[ERROR] rep{rep} fold{fold}: rebuilt predictions differ from saved "
                             f"window preds (max |diff| = {dmax:.4g}). Rows or normalization differ"
                             + (", or the cuML->sklearn conversion does not reproduce the model "
                                "(e.g. NaN handling)." if cuml_model is not None else "."))

        keep = np.arange(len(test_idx))
        if args.max_windows_per_subject:
            keep = np.concatenate([
                rng.choice(ix, size=min(len(ix), args.max_windows_per_subject), replace=False)
                for ix in pd.Series(np.arange(len(subj_te))).groupby(subj_te).groups.values()])
            keep.sort()
        Xk = X_te_n.iloc[keep]
        sv, bias = shap_values(model, model_name, Xk, X_train=X_tr_n)

        # CHECK 2: local accuracy (SHAP + bias = prediction)
        add_err = float(np.max(np.abs(sv.sum(1) + bias - p[keep])))
        if add_err > max(args.pred_tol, 1e-2):
            print(f"  [WARN] rep{rep} fold{fold}: SHAP additivity error {add_err:.3g}")

        A = pd.DataFrame(np.abs(sv), columns=feats)
        A["Subject"] = subj_te[keep]
        per_subj = A.groupby("Subject").mean()                 # mean |SHAP| per participant
        per_subj.insert(0, "rep", rep); per_subj.insert(1, "fold", fold)
        subj_rows.append(per_subj.reset_index())
        fold_imp = per_subj[feats].mean(axis=0)                 # participant-weighted, this model
        fold_rows.append(dict(rep=rep, fold=fold, n_test_subjects=len(per_subj), refit=refit,
                              pred_corr_with_saved=pcorr, max_pred_diff=dmax, max_additivity_err=add_err, **fold_imp.to_dict()))

        # direction: Spearman(feature value, SHAP) per feature, weighted by windows
        Xv = Xk.to_numpy(dtype=float)
        for j in range(len(feats)):
            ok = np.isfinite(Xv[:, j])
            if ok.sum() > 10 and np.nanstd(Xv[ok, j]) > 0:
                dir_num[j] += stats.spearmanr(Xv[ok, j], sv[ok, j])[0] * ok.sum()
        dir_n += len(keep)
        if rep == 0:   # beeswarm sample: rep 0 covers every participant once; cap per participant
            sk = subj_te[keep]
            s = np.concatenate([rng.choice(ix, size=min(len(ix), args.beeswarm_per_subject), replace=False)
                                for ix in pd.Series(np.arange(len(sk))).groupby(sk).groups.values()])
            bees_X.append(Xk.iloc[s]); bees_S.append(sv[s]); bees_B.append(bias[s]); bees_subj.append(sk[s])
        print(f"  rep{rep} fold{fold}: {len(keep):,} windows, "
              f"{'refit ' if refit else ''}pred agreement r={pcorr:.5f} max|diff|={dmax:.2e}, "
              f"additivity {add_err:.2e}")

    subj = pd.concat(subj_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    subj_mean = subj.groupby("Subject")[feats].mean()          # average over reps
    imp = pd.DataFrame({
        "feature": feats,
        "mean_abs_shap": subj_mean.mean(axis=0).to_numpy(),    # equal weight per participant
        "sd_across_participants": subj_mean.std(axis=0).to_numpy(),
    })
    ranks = folds[feats].rank(axis=1, ascending=False)
    imp[f"top{args.top_k}_freq"] = [(ranks[f] <= args.top_k).mean() for f in feats]
    imp["median_rank_across_models"] = [ranks[f].median() for f in feats]
    imp["direction_spearman"] = dir_num / max(dir_n, 1)
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))

    imp.to_csv(os.path.join(args.out_dir, f"shap_importance_{tag}.csv"), index=False)
    subj_mean.reset_index().to_csv(os.path.join(args.out_dir, f"shap_subject_{tag}.csv"), index=False)
    folds.to_csv(os.path.join(args.out_dir, f"shap_fold_{tag}.csv"), index=False)

    if folds["refit"].any():
        print(f"\n[REFIT] agreement with saved out-of-fold predictions: "
              f"median r = {folds['pred_corr_with_saved'].median():.5f}, "
              f"min r = {folds['pred_corr_with_saved'].min():.5f}, "
              f"max |diff| = {folds['max_pred_diff'].max():.3g}")
    pd.set_option("display.width", 200)
    print(f"\nTop {args.top_k} features ({len(folds)} fold models, {len(subj_mean)} participants):")
    print(imp.head(args.top_k)[["rank", "feature", "mean_abs_shap", "sd_across_participants",
                                f"top{args.top_k}_freq", "median_rank_across_models",
                                "direction_spearman"]].round(4).to_string(index=False))

    # ---- plots ----
    # Bar of the reportable (full) aggregation
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = imp.head(args.top_k)[::-1]
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(top) + 1))
        ax.barh(top["feature"], top["mean_abs_shap"], color="#1E88E5")
        ax.set_xlabel("Mean |SHAP value| (participant-averaged, held-out windows)")
        ax.set_title(f"{args.dataset} {target} ({model_name})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"shap_bar_{tag}.png"), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] bar plot failed: {e}")

    np.savez_compressed(
        os.path.join(args.out_dir, f"shap_beeswarm_sample_{tag}.npz"),
        values=np.vstack(bees_S), base=np.concatenate(bees_B),
        data=pd.concat(bees_X).to_numpy(dtype=float),
        subjects=np.concatenate(bees_subj).astype(str), features=np.array(feats))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        name = "Mini-BESTest" if target == "Minibest" else target
        title = f"Top {args.top_k} Important Features for {name}"
        pdir = os.path.join(args.out_dir, "plots")
        os.makedirs(pdir, exist_ok=True)
        plt.rcParams.update({"font.size": 14, "axes.titlesize": 20, "axes.labelsize": 18,
                             "xtick.labelsize": 14, "ytick.labelsize": 14})
        top = imp.head(args.top_k).reset_index(drop=True)

        # bar chart
        b = top[::-1]
        plt.figure(figsize=(10, 8))
        plt.barh(np.arange(len(b)), b["mean_abs_shap"].to_numpy(), color="#1E88E5")
        plt.yticks(np.arange(len(b)), b["feature"], fontsize=14)
        plt.xlabel("Mean |SHAP value|", fontsize=18)
        plt.title(title, fontsize=20)
        plt.tight_layout()
        for ext in ("png", "tif"):
            plt.savefig(os.path.join(pdir, f"{tag}_shap_bar.{ext}"), dpi=300, bbox_inches="tight")
        plt.close("all")

        # beeswarm
        import shap
        idx = [feats.index(f) for f in top["feature"]]
        expl = shap.Explanation(values=np.vstack(bees_S)[:, idx], base_values=np.concatenate(bees_B),
                                data=pd.concat(bees_X).to_numpy(dtype=float)[:, idx],
                                feature_names=list(top["feature"]))
        plt.figure()
        shap.plots.beeswarm(expl, max_display=args.top_k, order=np.arange(len(idx)), show=False)
        plt.title(title, fontsize=16)
        for ext in ("png", "tif"):
            plt.savefig(os.path.join(pdir, f"{tag}_shap_beeswarm.{ext}"), dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"beeswarm sample: {len(expl.values):,} windows (rep 0, <= {args.beeswarm_per_subject} per participant)")
    except ImportError as e:
        print(f"[INFO] paper figures skipped ({e}).")
    except Exception as e:
        print(f"[WARN] paper figures failed: {e}")
    print(f"\nSaved to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()