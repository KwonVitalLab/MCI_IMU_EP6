# MCI_IMU_EP6
## Wearable sensing for quantifying cognitive and balance functions in naturalistic movements of older adults with mild cognitive impairment in therapeutic environments

This repository contains the analysis pipeline for our study on estimating clinical assessment scores — MoCA (cognition) and Mini-BESTest (balance) — from waist-mounted IMU recordings collected during unstructured break and lunch periods of the CEP. 
Gait and signal features are extracted from short windows of accelerometer/gyroscope data, tree-based regressors are trained with nested, subject-aware cross-validation, and the resulting models are examined for feature importance (SHAP) and demographic fairness.

Preprint Link: [Wearable sensing for quantifying cognitive and balance functions in naturalistic movements of older adults with mild cognitive impairment in therapeutic environments](https://www.medrxiv.org/content/10.64898/2026.06.18.26355980v1)

## 1. Get the data

Download the de-identified dataset from PhysioNet ([TODO: add link]), make the dataset folders, and place the
files in these folders at the repository root:

| File | Destination |
| --- | --- |
| `ALL_COHORTS_combined_clean_data.csv` | `SourceData/` |
| `cohort_midpoints.csv` | `MetaData/` |
| `Final_MoCA.csv`, `Final_MoCA_Sub.csv`, `Final_Minibest.csv` | `LabelData/` |

All other directories (`processed/`, `results/`) are created automatically.

## 2. Requirements

**A CUDA GPU is required.** Steps 4–6 below train cuML's RandomForest and run
CatBoost/XGBoost on the GPU, and will not fall back to CPU. Steps 1–3 are CPU-only.

## 3. Run the pipeline

Run in order — each step consumes the previous step's output.

```bash
python process_imu.py        # 1
python extract_features.py   # 2
python assign_labels.py      # 3
python train_models.py       # 4
python run_shap.py           # 5
python analyze_fairness.py   # 6
```

1. **`process_imu.py`** — Loads the raw source recordings, keeps only the break/lunch
   windows, and splits each cohort into early- vs late-program halves using the
   cohort midpoints. Writes `ALL_COHORTS_{early,late}_data.csv` to `processed/`.

2. **`extract_features.py`** — Slides fixed-size windows over each subject's signals
   and computes gait, ECDF, and statistical features per window. Writes
   `ALL_COHORTS_{early,late}_features_full.csv` to `processed/features/`.

3. **`assign_labels.py`** — Joins MoCA and Mini-BESTest scores onto the feature
   windows by subject and period, producing per-period and combined labeled files
   (`ALL_COHORTS_features_full_{MoCA,Minibest}.csv`).

4. **`train_models.py`** — Trains RandomForest, CatBoost, and XGBoost for every target
   score using nested cross-validation (outer GroupKFold by subject, inner Optuna
   tuning). Saves trained models, performance JSONs, and agreement plots to `results/`.

5. **`run_shap.py`** — Loads the best model per target and computes SHAP feature
   importances, aggregated subject-fairly so no single subject dominates. Writes
   importance JSONs and plots to `results/shap/`.

6. **`analyze_fairness.py`** — Generates cross-validated predictions and measures
   regression bias across sensitive attributes (Sex, Race), then applies Wasserstein
   Barycenter post-processing as mitigation. Writes `fairness_result_*.csv` to
   `results/fairness/`.

### Supporting files

- **`config.yaml`** — All paths, target/exclusion lists, windowing parameters, CV
  settings, and hyperparameter search spaces. Edit this rather than the scripts.
- **`utils.py`** — Shared helpers (signal filtering, windowing, stride estimation,
  normalization, subject-aware splits, SHAP, plotting). Not run directly.

## Notes

- `extract_features.py` and `train_models.py` accept `smoke_test=True` in `main()`
  to run on a small subject subset first — useful for verifying the pipeline end to
  end before a full run.
- `train_models.py` checkpoints each finished model, so an interrupted run resumes
  where it left off. It also supports SLURM job arrays: set `SLURM_ARRAY_TASK_ID` to
  run one (dataset, target) unit per GPU in parallel.
