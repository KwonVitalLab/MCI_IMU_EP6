"""
Post-hoc revision analyses for the IMU/MCI manuscript (PLOS Digital Health, rev1).

Reads the files written by train_models_revision.py in {preds_dir}:
    subject_preds_*.csv, window_preds_*.csv.gz, fold_summary_*.csv

WHAT IT PRODUCES ({out_dir}/tables, {out_dir}/figures, {out_dir}/summary.md)
  Per target (best model by mean outer-fold RMSE, as in the pipeline):
    - Lin's CCC (95% bootstrap CI) and its components          R1.2
    - Calibration slope (95% bootstrap CI), pred ~ observed     R2.3
    - Density-based reference predictor calibration slope       R2.3, R2.4
    - Bland-Altman proportional bias (diff ~ mean)              R2.3
  Total scores only:
    - Fig. 4: Bland-Altman, no label overlap                    R2.3, R2.6
    - Fig. 5: equal x/y ranges + 45-degree line                 R2.4
    - Quartile-binned calibration plot (+ reference predictor)  R2.3
    - Number of windows per subject-timepoint (table + hist)    R1.3
    - Individual baseline->exit score changes                   R1.4
    - 1/2/3-month data closest to baseline/exit                 R1.4

Usage
    python revision_analyses.py --preds_dir results/predictions \
        --out_dir revision_results \
        --moca_total <MoCA total target name> --minibest_total <Mini-BESTest total name>
If the total-score names are omitted, targets containing "total" are used.
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

C_BASE = "#2F6DB5"   # baseline / primary
C_EXIT = "#D0622F"   # exit / secondary
C_REF = "#7A7A7A"    # reference / annotations


# =============================================================================
# Metrics
# =============================================================================
def ccc_parts(y, p):
    """Lin's CCC = r * C_b. C_b (bias correction) combines the scale shift
    v = SD_pred / SD_obs and the location shift u."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    my, mp = y.mean(), p.mean()
    sy, sp = y.std(), p.std()          # population SDs (Lin 1989)
    cov = np.mean((y - my) * (p - mp))
    ccc = 2 * cov / (sy**2 + sp**2 + (my - mp) ** 2)
    r = cov / (sy * sp) if sy > 0 and sp > 0 else np.nan
    v = sp / sy if sy > 0 else np.nan
    u = (mp - my) / np.sqrt(sy * sp) if sy > 0 and sp > 0 else np.nan
    cb = ccc / r if r and not np.isnan(r) else np.nan
    return dict(ccc=ccc, ccc_r=r, ccc_cb=cb, ccc_scale_shift=v, ccc_location_shift=u)


def ols(x, y):
    res = stats.linregress(np.asarray(x, float), np.asarray(y, float))
    return res.slope, res.intercept, res.pvalue


def point_metrics(df):
    """df: one row per SubjectTime with columns true, pred."""
    y, p = df["true"].to_numpy(float), df["pred"].to_numpy(float)
    out = dict(n=len(df), n_subjects=df["Subject"].nunique())
    out["rmse"] = float(np.sqrt(np.mean((p - y) ** 2)))
    out["pearson_r"], out["pearson_p"] = stats.pearsonr(y, p)
    out.update(ccc_parts(y, p))
    out["calib_slope"], out["calib_intercept"], _ = ols(y, p)   # pred ~ observed
    diff, mean = y - p, (p + y) / 2   # observed - predicted, as in the original manuscript
    out["ba_mean_bias"] = float(diff.mean())
    out["ba_loa_low"] = float(diff.mean() - 1.96 * diff.std(ddof=1))
    out["ba_loa_high"] = float(diff.mean() + 1.96 * diff.std(ddof=1))
    out["ba_prop_slope"], _, out["ba_prop_p"] = ols(mean, diff)
    out["range_true"] = f"{y.min():.1f}-{y.max():.1f}"
    out["range_pred"] = f"{p.min():.1f}-{p.max():.1f}"
    return out


def cluster_bootstrap(df, fn, B=2000, seed=0):
    """Percentile CI, resampling SUBJECTS (both timepoints move together)."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in df.groupby("Subject")}
    subs = np.array(list(groups))
    vals = []
    for _ in range(B):
        pick = rng.choice(subs, size=len(subs), replace=True)
        d = pd.concat([groups[s] for s in pick], ignore_index=True)
        try:
            vals.append(fn(d))
        except Exception:
            continue
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) else (np.nan, np.nan)


# =============================================================================
# Loading
# =============================================================================
def load(preds_dir):
    subj_files = glob.glob(os.path.join(preds_dir, "subject_preds_*.csv"))
    fold_files = glob.glob(os.path.join(preds_dir, "fold_summary_*.csv"))
    if not subj_files or not fold_files:
        raise SystemExit(
            f"[ERROR] No subject_preds_*.csv / fold_summary_*.csv found in:\n"
            f"        {os.path.abspath(preds_dir)}\n"
            f"        (exists: {os.path.isdir(preds_dir)})\n"
            f"  --preds_dir must be the predictions/ folder inside the pipeline's\n"
            f"  config output_path. Locate it with:  find . -name 'subject_preds_*.csv' | head"
        )
    subj = pd.concat([pd.read_csv(f) for f in subj_files], ignore_index=True)
    folds = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)
    for c in ["Subject", "Time", "subject_time"]:
        subj[c] = subj[c].astype(str)
    return subj, folds


def load_windows(preds_dir, dataset, target, model):
    f = glob.glob(os.path.join(preds_dir, f"window_preds_{dataset}_{target}_{model}_*.csv.gz"))
    if not f:
        return None
    w = pd.read_csv(f[0])
    for c in ["Subject", "Time", "SubjectTime"]:
        if c in w:
            w[c] = w[c].astype(str)
    return w


def best_models(folds):
    """Same selection rule as the pipeline: lowest mean outer-fold RMSE."""
    m = folds.groupby(["dataset", "target", "model"]).agg(
        mean_test_rmse=("test_rmse", "mean"), n_folds=("test_rmse", "size")).reset_index()
    best = m.loc[m.groupby(["dataset", "target"])["mean_test_rmse"].idxmin()]
    return m, best.reset_index(drop=True)


def rep_average(s):
    """Average predictions across reps -> one row per SubjectTime (as in the paper figures)."""
    agg = dict(true=("true", "mean"), pred=("pred", "mean"), Subject=("Subject", "first"),
               Time=("Time", "first"), n_windows=("n_windows", "first"))
    if "pred_ref" in s:
        agg["pred_ref"] = ("pred_ref", "mean")
    return s.groupby("subject_time").agg(**agg).reset_index()


def time_order(w, s):
    """Return (baseline_label, exit_label). Uses CohortProgramDay if available."""
    times = sorted(s["Time"].unique())
    if len(times) < 2:
        return times[0], None
    if w is not None and "CohortProgramDay" in w:
        d = w.groupby("Time")["CohortProgramDay"].mean().sort_values()
        return str(d.index[0]), str(d.index[-1])
    low = [t for t in times if t.lower() in ("early", "baseline", "entry", "pre", "0")]
    return (low[0], [t for t in times if t != low[0]][0]) if low else (times[0], times[1])


# =============================================================================
# Density-based reference predictor (R2.3, R2.4)
# =============================================================================
def add_reference_predictor(s, bandwidth):
    """For each held-out SubjectTime, predict a Gaussian-kernel weighted average of
    the TRAINING-fold labels centred at its TRUE score. Its only error is being
    pulled toward where training labels are dense. `bandwidth` in score units."""
    s = s.copy()
    s["pred_ref"] = np.nan
    for rep, srep in s.groupby("rep"):
        for fold, test in srep.groupby("fold"):
            train_y = srep.loc[srep["fold"] != fold, "true"].to_numpy(float)
            ty = test["true"].to_numpy(float)
            w = np.exp(-0.5 * ((train_y[None, :] - ty[:, None]) / bandwidth) ** 2)
            s.loc[test.index, "pred_ref"] = (w @ train_y) / w.sum(1)
    return s


def silverman_bw(y):
    y = np.asarray(y, float)
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    return 0.9 * min(y.std(ddof=1), iqr / 1.34) * len(y) ** (-0.2)


# =============================================================================
# Figures
# =============================================================================
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path + ".png", dpi=300); fig.savefig(path + ".pdf"); plt.close(fig)


def fig_scatter(a, target, m, base_t, path):
    """Fig. 5 (R2.4): same x/y range, 45-degree identity line."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    lo = np.floor(min(a["true"].min(), a["pred"].min())) - 1
    hi = np.ceil(max(a["true"].max(), a["pred"].max())) + 1
    for d, c, lab in [(a[a["Time"] == base_t], C_BASE, "Baseline"), (a[a["Time"] != base_t], C_EXIT, "Exit")]:
        if len(d):
            ax.scatter(d["true"], d["pred"], s=26, color=c, alpha=0.85, label=lab, edgecolor="white", lw=0.5)
    ax.plot([lo, hi], [lo, hi], color=C_REF, ls="--", lw=1, label="Identity (y = x)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Observed score"); ax.set_ylabel("Predicted score"); ax.set_title(target, fontsize=11)
    ax.text(0.03, 0.97, f"r = {m['pearson_r']:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _style(ax); _save(fig, path)


def fig_bland_altman(a, target, m, path):
    """Fig. 4 (R2.3, R2.6): proportional-bias line; legend below the axes (no overlap)."""
    y, p = a["true"].to_numpy(float), a["pred"].to_numpy(float)
    mean, diff = (y + p) / 2, y - p   # observed - predicted, as in the original manuscript figure
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(mean, diff, s=26, color=C_BASE, alpha=0.85, edgecolor="white", lw=0.5)
    ax.axhline(m["ba_mean_bias"], color="black", lw=1, label=f"Mean bias {m['ba_mean_bias']:.2f}")
    for v in (m["ba_loa_low"], m["ba_loa_high"]):
        ax.axhline(v, color=C_REF, ls="--", lw=1)
    ax.plot([], [], color=C_REF, ls="--", label=f"95% LoA [{m['ba_loa_low']:.2f}, {m['ba_loa_high']:.2f}]")
    xs = np.linspace(mean.min(), mean.max(), 50)
    sl, ic, pv = ols(mean, diff)
    ax.plot(xs, ic + sl * xs, color=C_EXIT, lw=1.4, label=f"Proportional bias: slope {sl:.2f} (p={pv:.3g})")
    ax.set_xlabel("Mean of observed and predicted"); ax.set_ylabel("Observed − predicted")
    ax.set_title(target, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False, ncol=1)
    _style(ax); _save(fig, path)


def quartile_bins(a, col):
    """Quartiles of the observed score (~11 per bin at N=44), ties broken by order."""
    q = pd.qcut(a["true"].rank(method="first"), 4, labels=False)
    g = a.assign(q=q).groupby("q")
    return pd.DataFrame({"obs_mean": g["true"].mean(), "pred_mean": g[col].mean(),
                         "pred_se": g[col].sem(), "n": g.size()})


def fig_calibration(a, target, m, path):
    b = quartile_bins(a, "pred")
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    lo = np.floor(a["true"].min()) - 1; hi = np.ceil(a["true"].max()) + 1
    ax.plot([lo, hi], [lo, hi], color=C_REF, ls="--", lw=1, label="Identity (predicted = observed)")
    ax.errorbar(b["obs_mean"], b["pred_mean"], yerr=1.96 * b["pred_se"], fmt="o-", color=C_BASE,
                capsize=3, label=f"Model (slope {m['calib_slope']:.2f})")
    r = quartile_bins(a, "pred_ref")
    ax.plot(r["obs_mean"], r["pred_mean"], "s:", color=C_EXIT,
            label=f"Density reference (slope {m['ref_calib_slope']:.2f})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Mean observed score (quartile bins)"); ax.set_ylabel("Mean predicted score")
    ax.set_title(target, fontsize=11); ax.legend(loc="lower right", fontsize=8, frameon=False)
    _style(ax); _save(fig, path)
    return b, r


def fig_score_change(wide, target, path):
    """R1.4: individual baseline -> exit observed scores."""
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    for _, r in wide.iterrows():
        ax.plot([0, 1], [r["baseline"], r["exit"]], color=C_REF, alpha=0.6, lw=0.9, marker="o", ms=3)
    ax.plot([0, 1], [wide["baseline"].mean(), wide["exit"].mean()], color="black", lw=2.5, marker="o",
            label="Group mean")
    ax.set_xticks([0, 1], ["Baseline", "Exit"]); ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel("Observed score"); ax.set_title(target, fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _style(ax); _save(fig, path)


def fig_windows(a, target, path):
    """R1.3: distribution of windows per subject-timepoint."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.hist(a["n_windows"], bins=20, color=C_BASE, alpha=0.85)
    ax.set_xlabel("Windows per subject-timepoint"); ax.set_ylabel("Count"); ax.set_title(target, fontsize=11)
    _style(ax); _save(fig, path)


# =============================================================================
# 1/2/3-month data near baseline / exit (R1.4)
# =============================================================================
def month_restricted(w, base_t, exit_t, month_days):
    """Baseline: 1st month / 1st-2nd months / all 3 months (days since the cohort's
    first window). Exit: 6th / 5th-6th / last 3 months (days before the cohort's
    last window). Same trained models; only the windows aggregated per
    subject-timepoint change."""
    if "CohortProgramDay" not in w:
        return None
    w = w.copy()
    if "Cohort" in w:
        first = w.groupby("Cohort")["CohortProgramDay"].transform("min")
        last = w.groupby("Cohort")["CohortProgramDay"].transform("max")
    else:
        first, last = w["CohortProgramDay"].min(), w["CohortProgramDay"].max()
    w["dist"] = np.where(w["Time"] == base_t, w["CohortProgramDay"] - first, last - w["CohortProgramDay"])
    rows = []
    for months in (1, 2, 3):
        sub = w if months == 3 else w[w["dist"] < months * month_days]
        s = sub.groupby(["rep", "SubjectTime"]).agg(true=("true", "mean"), pred=("pred", "mean"),
                                                    Subject=("Subject", "first"),
                                                    Time=("Time", "first")).reset_index()
        a = s.groupby("SubjectTime").agg(true=("true", "mean"), pred=("pred", "mean"),
                                         Subject=("Subject", "first"), Time=("Time", "first")).reset_index()
        for label, d in [("baseline", a[a["Time"] == base_t]), ("exit", a[a["Time"] == exit_t]), ("both", a)]:
            if len(d) < 3:
                continue
            y, p = d["true"].to_numpy(float), d["pred"].to_numpy(float)
            rows.append(dict(months=months, timepoint=label, n=len(d),
                             rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                             pearson_r=stats.pearsonr(y, p)[0], ccc=ccc_parts(y, p)["ccc"]))
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--out_dir", default="revision_results")
    ap.add_argument("--moca_total", default=None)
    ap.add_argument("--minibest_total", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--model", default=None,
                    help="force one model (e.g. XGBoost) instead of the best per target; use to compare runs")
    ap.add_argument("--month_days", type=float, default=30.0)
    ap.add_argument("--ref_bandwidth", default="rmse",
                    help="'rmse' (model RMSE), 'silverman', or a number in score units")
    args = ap.parse_args()

    tdir, fdir = os.path.join(args.out_dir, "tables"), os.path.join(args.out_dir, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)

    subj, folds = load(args.preds_dir)
    if args.model:
        folds = folds[folds["model"] == args.model]
        if folds.empty:
            raise SystemExit(f"[ERROR] model {args.model} not found in {args.preds_dir}")
    _, best = best_models(folds)

    totals = set()
    for ds, name in [("MoCA", args.moca_total), ("Minibest", args.minibest_total)]:
        cand = [name] if name else [t for t in best.loc[best["dataset"] == ds, "target"] if "total" in t.lower()]
        if cand:
            totals.add((ds, cand[0]))
    print("Total-score targets:", sorted(totals))

    summary_rows, md = [], ["# Revision analyses summary\n\n"]
    for _, b in best.iterrows():
        ds, tg, mdl = b["dataset"], b["target"], b["model"]
        s = subj[(subj["dataset"] == ds) & (subj["target"] == tg) & (subj["model"] == mdl)].copy()
        a0 = rep_average(s)
        rmse_model = float(np.sqrt(np.mean((a0["pred"] - a0["true"]) ** 2)))
        bw = {"rmse": rmse_model, "silverman": silverman_bw(a0["true"])}.get(args.ref_bandwidth)
        bw = float(args.ref_bandwidth) if bw is None else bw
        s = add_reference_predictor(s, bw)
        a = rep_average(s)

        m = point_metrics(a)
        ref = point_metrics(a.assign(pred=a["pred_ref"]))
        m["ref_bandwidth"] = bw
        m["ref_calib_slope"] = ref["calib_slope"]
        m["model_minus_ref_slope"] = m["calib_slope"] - m["ref_calib_slope"]

        for key, fn in [("ccc", lambda d: ccc_parts(d["true"], d["pred"])["ccc"]),
                        ("calib_slope", lambda d: ols(d["true"], d["pred"])[0]),
                        ("ref_calib_slope", lambda d: ols(d["true"], d["pred_ref"])[0])]:
            lo, hi = cluster_bootstrap(a, fn, B=args.n_boot)
            m[f"{key}_ci_low"], m[f"{key}_ci_high"] = lo, hi

        summary_rows.append(dict(dataset=ds, target=tg, best_model=mdl, **m))
        a.to_csv(os.path.join(tdir, f"subject_level_{ds}_{tg}_{mdl}.csv"), index=False)

        if (ds, tg) not in totals:
            continue

        # ── Total scores ─────────────────────────────────────────────────────
        w = load_windows(args.preds_dir, ds, tg, mdl)
        base_t, exit_t = time_order(w, s)
        fig_scatter(a, tg, m, base_t, os.path.join(fdir, f"fig5_scatter_{tg}"))
        fig_bland_altman(a, tg, m, os.path.join(fdir, f"fig4_bland_altman_{tg}"))
        bins, ref_bins = fig_calibration(a, tg, m, os.path.join(fdir, f"calibration_{tg}"))
        bins.to_csv(os.path.join(tdir, f"calibration_bins_model_{tg}.csv"))
        ref_bins.to_csv(os.path.join(tdir, f"calibration_bins_reference_{tg}.csv"))

        # R1.3 windows per subject-timepoint
        a[["Subject", "Time", "n_windows"]].sort_values(["Subject", "Time"]).to_csv(
            os.path.join(tdir, f"windows_per_subject_{tg}.csv"), index=False)
        fig_windows(a, tg, os.path.join(fdir, f"windows_{tg}"))
        a.groupby("Time")["n_windows"].describe().to_csv(os.path.join(tdir, f"windows_summary_{tg}.csv"))

        # R1.4 individual changes + 1/2/3-month experiment
        change_txt, month_txt = "", ""
        if exit_t is not None:
            wide = a.pivot_table(index="Subject", columns="Time", values="true").rename(
                columns={base_t: "baseline", exit_t: "exit"}).dropna()
            wide["change"] = wide["exit"] - wide["baseline"]
            wide.to_csv(os.path.join(tdir, f"score_change_{tg}.csv"))
            fig_score_change(wide, tg, os.path.join(fdir, f"score_change_{tg}"))
            change_txt = (f"- Baseline→exit (n={len(wide)}): group mean {wide['baseline'].mean():.1f} → "
                          f"{wide['exit'].mean():.1f}; individual change range "
                          f"{wide['change'].min():+.1f} to {wide['change'].max():+.1f}\n")
            months = month_restricted(w, base_t, exit_t, args.month_days) if w is not None else None
            if months is not None:
                months.to_csv(os.path.join(tdir, f"month_restricted_{tg}.csv"), index=False)
                month_txt = "- 1/2/3-month data:\n\n" + months.round(3).to_string(index=False) + "\n"

        md.append(f"## {tg} ({ds}, best model: {mdl})\n\n")
        md.append(f"- N = {m['n']} subject-timepoints / {m['n_subjects']} subjects\n")
        md.append(f"- RMSE {m['rmse']:.2f}; Pearson r {m['pearson_r']:.2f} (p={m['pearson_p']:.3g})\n")
        md.append(f"- CCC {m['ccc']:.2f} [{m['ccc_ci_low']:.2f}, {m['ccc_ci_high']:.2f}] = r {m['ccc_r']:.2f} × "
                  f"C_b {m['ccc_cb']:.2f} (scale shift SD_pred/SD_obs {m['ccc_scale_shift']:.2f}, "
                  f"location shift {m['ccc_location_shift']:.2f})\n")
        md.append(f"- Observed range {m['range_true']} vs predicted {m['range_pred']}\n")
        md.append(f"- Calibration slope {m['calib_slope']:.2f} [{m['calib_slope_ci_low']:.2f}, "
                  f"{m['calib_slope_ci_high']:.2f}]; density reference slope {m['ref_calib_slope']:.2f} "
                  f"[{m['ref_calib_slope_ci_low']:.2f}, {m['ref_calib_slope_ci_high']:.2f}] "
                  f"(bandwidth {bw:.2f})\n")
        md.append(f"- Bland-Altman: mean bias {m['ba_mean_bias']:.2f}, 95% LoA [{m['ba_loa_low']:.2f}, "
                  f"{m['ba_loa_high']:.2f}], proportional-bias slope {m['ba_prop_slope']:.2f} "
                  f"(p={m['ba_prop_p']:.3g})\n")
        md.append(f"- Windows per subject-timepoint: median {a['n_windows'].median():.0f} "
                  f"(range {a['n_windows'].min():.0f}-{a['n_windows'].max():.0f})\n")
        md.append(change_txt + month_txt + "\n")

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(os.path.join(tdir, "summary_all_targets.csv"), index=False)
    cols = ["dataset", "target", "best_model", "n", "rmse", "pearson_r", "ccc", "ccc_ci_low", "ccc_ci_high",
            "calib_slope", "ref_calib_slope", "ba_prop_slope", "ba_prop_p"]
    tab = summ[[c for c in cols if c in summ]].round(3)
    try:
        tab_txt = tab.to_markdown(index=False)
    except ImportError:          # tabulate not installed
        tab_txt = tab.to_string(index=False)
    md.append("## All targets\n\n" + tab_txt + "\n")
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("".join(md))
    print(f"Done. See {args.out_dir}/summary.md")


if __name__ == "__main__":
    main()