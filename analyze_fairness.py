"""
Fairness metrics with 95% bootstrap CIs.

Outputs ({out_dir})
  fairness_ci_{dataset}_{target}_{attr}.csv   same layout as the original CSVs
                                              (+ CI columns and group sizes)
  fairness_ci_all.csv                         everything in one long table
  subject_level_{dataset}_{target}.csv        subject-level predictions used

Usage
  python analyze_fairness.py --config config.yaml \
      --preds_dir results/performance/predictions_rev1 \
      --out_dir results/performance/fairness_rev1
"""

import argparse
import contextlib
import glob
import io
import os
import warnings

import numpy as np
import pandas as pd
import yaml


warnings.filterwarnings("ignore")

METRICS = ["Average Score Difference", "Statistical Parity AUC"]


# =============================================================================
# Demographics
# =============================================================================
def load_demographics(label_path, dataset):
    if dataset == "MoCA":
        a = pd.read_csv(os.path.join(label_path, "Final_MoCA.csv"))
        b = pd.read_csv(os.path.join(label_path, "Final_MoCA_Sub.csv"))
        lab = pd.merge(a, b, on="Subject", how="outer")
        for col in ["MoCA", "Age", "Race", "Ethnicity", "Sex"]:
            if f"{col}_x" in lab:
                lab[col] = lab[f"{col}_x"].fillna(lab[f"{col}_y"])
                lab = lab.drop(columns=[f"{col}_x", f"{col}_y"])
    else:
        lab = pd.read_csv(os.path.join(label_path, "Final_Minibest.csv"))
    lab = lab.drop_duplicates(subset="Subject", keep="first")
    demo = lab[["Subject", "Sex", "Race"]].copy()
    demo["Subject"] = demo["Subject"].astype(str)
    return demo


def group_masks(df, attr):
    """
    Group definitions (ordered (interest, reference))
    interest: Female / Non-White;
    reference: Male / White
    """
    if attr == "Sex":
        return df["Sex"] == "Female", df["Sex"] == "Male", "Female", "Male"
    if attr == "Race":
        return df["Race"] != "White", df["Race"] == "White", "Non-White", "White"
    raise ValueError(attr)


# =============================================================================
# Metrics 
# =============================================================================
def fairness_values(df, attr, pred_col):
    from holisticai.bias.metrics import regression_bias_metrics
    ga, gb, _, _ = group_masks(df, attr)
    if ga.sum() == 0 or gb.sum() == 0:
        return {m: np.nan for m in METRICS + ["RMSE"]}
    try:
        # holisticai prints quantile warnings for tiny groups (unrelated to ASD / SP AUC)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            m = regression_bias_metrics(ga, gb, df[pred_col], df["y_true"], metric_type="group")
        out = {k: float(m.loc[k, "Value"]) for k in METRICS}
    except Exception:
        out = {k: np.nan for k in METRICS}
    out["RMSE"] = float(np.sqrt(np.mean((df[pred_col] - df["y_true"]) ** 2)))
    return out


# def wb_mitigate(df, attr):
#     from holisticai.bias.mitigation import WassersteinBarycenter
#     ga, gb, _, _ = group_masks(df, attr)
#     if ga.sum() == 0 or gb.sum() == 0:
#         return df["y_pred"].to_numpy()
#     try:
#         wb = WassersteinBarycenter()
#         with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
#             wb.fit(df["y_pred"], ga, gb)
#             y = wb.transform(df["y_pred"], ga, gb)["y_pred"]
#         return np.asarray(getattr(y, "values", y), dtype=float).ravel()
#     except Exception:
#         return np.full(len(df), np.nan)


def wb_mitigate(df, attr):
    ga, gb, _, _ = group_masks(df, attr)        
    ga, gb = ga.to_numpy(), gb.to_numpy()
    y = df["y_pred"].to_numpy(dtype=float)
    subj = df["Subject"].astype(str).to_numpy()
    if ga.sum() == 0 or gb.sum() == 0:
        return y
    groups = [ga, gb]
    w = [g.sum() / (ga.sum() + gb.sum()) for g in groups]
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        keep = subj != subj[i]                     # drop this participant (all copies)
        own = ga if ga[i] else gb
        ref = y[own & keep]                        # own group without participant i
        if len(ref) == 0 or any((g & keep).sum() == 0 for g in groups):
            return np.full(len(y), np.nan)         # a group would be empty
        # mid-rank quantile level of y_i among the other members of its group
        less = np.sum(ref < y[i]); ties = np.sum(ref == y[i])
        u = (less + 0.5 * ties + 0.5) / (len(ref) + 1)
        out[i] = sum(wk * np.quantile(y[g & keep], u) for wk, g in zip(w, groups))
    return out


def evaluate(df, attr):
    """Baseline and WB-mitigated metrics for one (possibly resampled) subject table."""
    d = df.reset_index(drop=True).copy()
    base = fairness_values(d, attr, "y_pred")
    d["y_pred_mit"] = wb_mitigate(d, attr)
    mit = fairness_values(d, attr, "y_pred_mit") if d["y_pred_mit"].notna().all() else \
        {k: np.nan for k in METRICS + ["RMSE"]}
    return base, mit


# =============================================================================
# Subject-level predictions from rev1 files
# =============================================================================
def subject_level(s):
    s = s.assign(wp=s["pred"] * s["n_windows"], wt=s["true"] * s["n_windows"])
    per_rep = s.groupby(["rep", "Subject"]).agg(wp=("wp", "sum"), wt=("wt", "sum"), n=("n_windows", "sum"))
    per_rep["y_pred"] = per_rep["wp"] / per_rep["n"]
    per_rep["y_true"] = per_rep["wt"] / per_rep["n"]
    return per_rep.groupby("Subject")[["y_true", "y_pred"]].mean().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--attrs", default="Sex,Race")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(args.config))
    with open(args.config) as f:
        config = yaml.safe_load(f)
    label_path = os.path.join(root, config["data_paths"]["label_path"])
    attrs = [a.strip() for a in args.attrs.split(",") if a.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    sf = glob.glob(os.path.join(args.preds_dir, "subject_preds_*.csv"))
    ff = glob.glob(os.path.join(args.preds_dir, "fold_summary_*.csv"))
    if not sf or not ff:
        raise SystemExit(f"[ERROR] no prediction files in {os.path.abspath(args.preds_dir)}")
    subj = pd.concat([pd.read_csv(f) for f in sf], ignore_index=True)
    subj["Subject"] = subj["Subject"].astype(str)
    folds = pd.concat([pd.read_csv(f) for f in ff], ignore_index=True)
    # best model per target: lowest mean outer-fold RMSE (same rule as the pipeline)
    m = folds.groupby(["dataset", "target", "model"])["test_rmse"].mean().reset_index()
    best = m.loc[m.groupby(["dataset", "target"])["test_rmse"].idxmin()]

    demos = {ds: load_demographics(label_path, ds) for ds in best["dataset"].unique()}
    rng = np.random.default_rng(args.seed)
    long_rows = []

    for _, b in best.iterrows():
        ds, tg, mdl = b["dataset"], b["target"], b["model"]
        s = subj[(subj["dataset"] == ds) & (subj["target"] == tg) & (subj["model"] == mdl)]
        sl = subject_level(s).merge(demos[ds], on="Subject", how="left").dropna(subset=["Sex", "Race"])
        sl.to_csv(os.path.join(args.out_dir, f"subject_level_{ds}_{tg}.csv"), index=False)

        for attr in attrs:
            ga, gb, la, lb = group_masks(sl, attr)
            ia, ib = np.flatnonzero(ga.to_numpy()), np.flatnonzero(gb.to_numpy())
            base, mit = evaluate(sl, attr)

            boot = {("Baseline", k): [] for k in METRICS + ["RMSE"]}
            boot.update({("Mitigated", k): [] for k in METRICS + ["RMSE"]})
            for _ in range(args.n_boot):
                idx = np.r_[rng.choice(ia, len(ia), replace=True), rng.choice(ib, len(ib), replace=True)]
                bb, mm = evaluate(sl.iloc[idx], attr)
                for k in METRICS + ["RMSE"]:
                    boot[("Baseline", k)].append(bb[k])
                    boot[("Mitigated", k)].append(mm[k])

            table = []
            for k in METRICS + ["RMSE"]:
                row = {"Metric": k}
                for cond, point in [("Baseline", base), ("Mitigated", mit)]:
                    v = np.array(boot[(cond, k)], float)
                    v = v[~np.isnan(v)]
                    lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
                    row[f"{cond}_{attr}"] = point[k]
                    row[f"{cond}_{attr}_CI_low"] = lo
                    row[f"{cond}_{attr}_CI_high"] = hi
                    long_rows.append(dict(dataset=ds, target=tg, model=mdl, attribute=attr, condition=cond,
                                          metric=k, value=point[k], ci_low=lo, ci_high=hi,
                                          group_a=la, n_a=len(ia), group_b=lb, n_b=len(ib),
                                          n_valid_boot=len(v),
                                          # SP AUC is a distance (>= 0): resampling tiny groups
                                          # inflates it, so its percentile CI can sit above the
                                          # point estimate. Flag those cases for reporting.
                                          point_in_ci=bool(lo <= point[k] <= hi) if len(v) else False))
                row["is_mitigated"] = abs(row[f"Mitigated_{attr}"]) < abs(row[f"Baseline_{attr}"])
                table.append(row)
            out = pd.DataFrame(table).set_index("Metric")
            out.attrs = {}
            out.to_csv(os.path.join(args.out_dir, f"fairness_ci_{ds}_{tg}_{attr}.csv"))
            print(f"{ds:8s} {tg:36s} {attr:4s} ({la}={len(ia)}, {lb}={len(ib)})  "
                  f"ASD {base['Average Score Difference']:+.2f} "
                  f"[{out.loc['Average Score Difference', f'Baseline_{attr}_CI_low']:+.2f}, "
                  f"{out.loc['Average Score Difference', f'Baseline_{attr}_CI_high']:+.2f}]  "
                  f"SP-AUC {base['Statistical Parity AUC']:.2f} "
                  f"[{out.loc['Statistical Parity AUC', f'Baseline_{attr}_CI_low']:.2f}, "
                  f"{out.loc['Statistical Parity AUC', f'Baseline_{attr}_CI_high']:.2f}]", flush=True)

    pd.DataFrame(long_rows).to_csv(os.path.join(args.out_dir, "fairness_ci_all.csv"), index=False)
    print(f"\nDone. {args.out_dir}/fairness_ci_all.csv")


if __name__ == "__main__":
    main()