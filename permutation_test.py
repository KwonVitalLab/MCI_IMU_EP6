"""
Permutation test (R1.2): is model performance better than chance, and is the observed
(negative) correlation unusual under "no relationship"?

Design
  - Same data, feature columns, window->SubjectTime aggregation, and outer folds as
    train_models_revision.py (subject-level folds, seed = random_seed + rep).
  - Labels are permuted ACROSS PARTICIPANTS, keeping each participant's baseline/exit
    label pair together (permutation within groups of participants with the same
    timepoint pattern). All windows of a SubjectTime take its permuted label.
  - Hyper-parameters are FIXED to the per-fold best params found in the nested CV
    (fold_summary_*.csv); Optuna is not rerun per permutation (computationally
    infeasible). State this in Methods. Permutation 0 = original labels (observed).
  - Statistics per permutation (SubjectTime level, window predictions averaged):
    RMSE, Lin's CCC, Pearson r.

p-values (with the +1 correction; smallest possible = 1/(n_perm+1))
  RMSE : P(RMSE_perm <= RMSE_obs)        better-than-chance error
  CCC  : P(CCC_perm  >= CCC_obs)         better-than-chance agreement
  r    : P(r_perm    >= r_obs)           better-than-chance positive association
  r_lower: P(r_perm  <= r_obs)           is the observed r unusually NEGATIVE vs chance?

Resumable and chunkable (SLURM array): each permutation is appended to a CSV as soon as
it finishes; finished permutation ids are skipped on restart.

Usage (run from the folder with config.yaml and train_models_revision.py)
  # one chunk (e.g., SLURM array task i -> perms i*100 .. i*100+99)
  python permutation_test.py run --dataset MoCA --target MoCA --model RandomForest \
      --preds_dir results/performance/predictions --out_dir results/performance/permutation \
      --perm_start 0 --perm_end 100
  # after all chunks: p-values + histogram
  python permutation_test.py summarize --out_dir results/performance/permutation
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Shared statistics
# =============================================================================
def ccc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    cov = np.mean((y - y.mean()) * (p - p.mean()))
    return 2 * cov / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)


def stats_row(true_st, pred_st):
    y, p = np.asarray(true_st, float), np.asarray(pred_st, float)
    r = stats.pearsonr(y, p)[0] if np.std(p) > 0 else np.nan
    return dict(rmse=float(np.sqrt(np.mean((p - y) ** 2))), ccc=float(ccc(y, p)), r=float(r))


# =============================================================================
# Label permutation (participant level, baseline/exit pairs kept together)
# =============================================================================
def permuted_st_labels(st_labels, st_subject, st_time, rng):
    """st_labels: Series indexed by SubjectTime. Returns a permuted Series (same index)."""
    df = pd.DataFrame({"label": st_labels, "Subject": st_subject, "Time": st_time})
    pattern = df.groupby("Subject")["Time"].apply(lambda t: tuple(sorted(map(str, t))))
    new = {}
    for _, subs in pattern.groupby(pattern):
        subs = list(subs.index)
        donors = rng.permutation(subs)
        for recv, don in zip(subs, donors):
            dl = df[df["Subject"] == don].set_index("Time")["label"]
            for st, row in df[df["Subject"] == recv].iterrows():
                new[st] = dl[row["Time"]]
    return pd.Series(new).reindex(st_labels.index)


# =============================================================================
# run
# =============================================================================
def run(args):
    import yaml
    import train_models_revision as T    
    from utils import normalize_features

    root = os.path.dirname(os.path.abspath(T.__file__))
    with open(os.path.join(root, "config.yaml")) as f:
        config = yaml.safe_load(f)
    seed = config["random_seed"]
    n_folds = config["folds"]

    # ── data, exactly as in train_models_revision.main ──
    fname = {"MoCA": "ALL_COHORTS_features_full_MoCA.csv",
             "Minibest": "ALL_COHORTS_features_full_Minibest.csv"}[args.dataset]
    excl = config["moca_exclude"] if args.dataset == "MoCA" else config["minibest_exclude"]
    df = pd.read_csv(os.path.join(root, config["data_paths"]["feature_path"], fname))
    df["SubjectTime"] = df["Subject"].astype(str) + "_" + df["Time"].astype(str)
    drop = list(set(T.NON_FEATURE_COLS) | set(config["exclude_cols"]) | set(excl))
    X_all = df.drop(columns=drop, errors="ignore")
    ok = df[args.target].notna()
    X = X_all[ok]
    y = df.loc[ok, args.target].astype(float)
    subjects = df.loc[ok, "Subject"]
    subject_time = df.loc[ok, "SubjectTime"]
    feat_cols = X.columns.tolist()

    st_first = pd.DataFrame({"label": y, "Subject": subjects.astype(str),
                             "Time": df.loc[ok, "Time"].astype(str), "st": subject_time}).groupby("st").first()

    # ── fixed per-fold params from the nested CV ──
    fs = pd.read_csv(os.path.join(
        args.preds_dir, f"fold_summary_{args.dataset}_{args.target}_{args.model}_full.csv"))
    fs = fs[fs["rep"].isin(range(args.reps))]
    params = {(int(r.rep), int(r.fold)): json.loads(r.best_params) for r in fs.itertuples()}

    # ── folds  ──
    fold_sets = {}
    for rep in range(args.reps):
        fold_sets[rep] = list(T.subject_group_folds(subjects.reset_index(drop=True), n_folds, seed + rep))

    if args.model == "RandomForest":
        T._init_rmm_pool()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.target}_{args.model}"
    out_csv = os.path.join(args.out_dir, f"perm_{tag}_{args.perm_start:05d}-{args.perm_end:05d}.csv")
    done = set(pd.read_csv(out_csv)["perm"]) if os.path.exists(out_csv) else set()
    with open(os.path.join(args.out_dir, f"design_{tag}.json"), "w") as f:
        json.dump(dict(dataset=args.dataset, target=args.target, model=args.model, reps=args.reps,
                       folds=n_folds, seed=seed, n_windows=int(len(X)), n_subject_times=int(len(st_first)),
                       n_subjects=int(subjects.nunique()),
                       permutation="participant-level, baseline/exit pairs kept, within timepoint pattern",
                       hyperparameters="fixed to per-fold best params"), f, indent=2)

    Xr = X.reset_index(drop=True)
    str_ = subject_time.reset_index(drop=True)
    for perm in range(args.perm_start, args.perm_end):
        if perm in done:
            continue
        t0 = time.time()
        if perm == 0:
            st_lab = st_first["label"]
        else:
            st_lab = permuted_st_labels(st_first["label"], st_first["Subject"], st_first["Time"],
                                        np.random.default_rng(100_000 + perm))
        y_perm = str_.map(st_lab).astype(float)

        rep_stats, rep_preds = [], []
        for rep in range(args.reps):
            preds = pd.Series(np.nan, index=Xr.index)
            for k, (tr, te) in enumerate(fold_sets[rep]):
                Xtr, Xte, _ = normalize_features(train_data=Xr.iloc[tr], test_data=Xr.iloc[te],
                                                 feature_cols=feat_cols)
                model = T.build_model(args.model, seed)
                model.set_params(**params[(rep, k)])
                model.fit(Xtr, y_perm.iloc[tr])
                preds.iloc[te] = np.asarray(model.predict(Xte), float).ravel()
            st_pred = preds.groupby(str_).mean()
            rep_preds.append(st_pred)
        # average over reps per SubjectTime, then compute statistics
        st_pred = pd.concat(rep_preds, axis=1).mean(axis=1)
        row = dict(perm=perm, **stats_row(st_lab.loc[st_pred.index], st_pred), seconds=round(time.time() - t0, 1))
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(f"[perm {perm}] RMSE {row['rmse']:.3f}  CCC {row['ccc']:+.3f}  r {row['r']:+.3f}  "
              f"({row['seconds']}s)", flush=True)


# =============================================================================
# summarize
# =============================================================================
def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = glob.glob(os.path.join(args.out_dir, "perm_*.csv"))
    if not files:
        raise SystemExit(f"[ERROR] no perm_*.csv in {args.out_dir}")
    allp = []
    for f in files:
        d = pd.read_csv(f)
        # perm_{dataset}_{target}_{model}_{start}-{end}.csv -> tag = {dataset}_{target}_{model}
        d["tag"] = "_".join(os.path.basename(f)[len("perm_"):].split("_")[:-1])
        allp.append(d)
    allp = pd.concat(allp, ignore_index=True).drop_duplicates(["tag", "perm"])

    rows = []
    for tag, d in allp.groupby("tag"):
        if 0 not in set(d["perm"]):
            print(f"[WARN] {tag}: observed run (perm 0) missing; include perm_start 0 in one chunk.")
            continue
        obs = d[d["perm"] == 0].iloc[0]
        null = d[d["perm"] > 0]
        n = len(null)
        p = lambda cond: (int(cond.sum()) + 1) / (n + 1)
        rows.append(dict(
            tag=tag, n_perm=n,
            rmse_obs=obs["rmse"], rmse_null_median=null["rmse"].median(), p_rmse=p(null["rmse"] <= obs["rmse"]),
            ccc_obs=obs["ccc"], ccc_null_median=null["ccc"].median(), p_ccc=p(null["ccc"] >= obs["ccc"]),
            r_obs=obs["r"], r_null_median=null["r"].median(),
            r_null_2p5=null["r"].quantile(0.025), r_null_97p5=null["r"].quantile(0.975),
            p_r_greater=p(null["r"] >= obs["r"]), p_r_lower=p(null["r"] <= obs["r"]),
        ))
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, col, lab in zip(axs, ["rmse", "ccc", "r"], ["RMSE", "Lin's CCC", "Pearson r"]):
            ax.hist(null[col].dropna(), bins=30, color="#9AB6DA")
            ax.axvline(obs[col], color="#D0622F", lw=2, label="observed")
            ax.set_xlabel(lab); ax.spines[["top", "right"]].set_visible(False)
        axs[0].set_ylabel("Permutations"); axs[2].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{tag} (n_perm = {n})", fontsize=10); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"null_{tag}.png"), dpi=200); plt.close(fig)

    res = pd.DataFrame(rows)
    # Primary vs exploratory analyses:
    #   primary     = total scores (target name == dataset name, e.g. MoCA_MoCA_*),
    #                 reported without adjustment;
    #   exploratory = subdomain scores, Holm-adjusted within each assessment
    #                 (MoCA subdomains together, Mini-BESTest subdomains together).
    if len(res):
        res["family"] = res["tag"].str.split("_").str[0]
        is_total = res.apply(lambda r: r["tag"].startswith(f"{r['family']}_{r['family']}_"), axis=1)
        res["analysis"] = np.where(is_total, "primary", "exploratory")
        for col in ["p_rmse", "p_ccc", "p_r_greater"]:
            res[col + "_holm"] = np.nan
            for _, idx in res[res["analysis"] == "exploratory"].groupby("family").groups.items():
                pv = res.loc[idx, col].to_numpy(float)
                order = np.argsort(pv)
                k = len(pv)
                adj = np.empty(k)
                running = 0.0
                for rank, j in enumerate(order):
                    running = max(running, min(1.0, (k - rank) * pv[j]))
                    adj[j] = running
                res.loc[idx, col + "_holm"] = adj
    res.to_csv(os.path.join(args.out_dir, "permutation_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=["MoCA", "Minibest"])
    r.add_argument("--target", required=True)
    r.add_argument("--model", required=True, choices=["RandomForest", "CatBoost", "XGBoost"])
    r.add_argument("--preds_dir", required=True, help="predictions folder (for fold_summary best params)")
    r.add_argument("--out_dir", required=True)
    r.add_argument("--perm_start", type=int, default=0)
    r.add_argument("--perm_end", type=int, default=1001, help="exclusive; 0 = observed, 1..N = permutations")
    r.add_argument("--reps", type=int, default=1,
                   help="CV reps per permutation")
    s = sub.add_parser("summarize")
    s.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()