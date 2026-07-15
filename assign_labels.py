'''
assign_labels.py

Standalone script for assigning MoCA and Minibest labels to extracted
feature files. Run this after the feature-extraction step has completed.

Reads:
  ALL_COHORTS_early_features_full.csv
  ALL_COHORTS_late_features_full.csv

Writes (6 files):
  ALL_COHORTS_early_features_full_MoCA.csv
  ALL_COHORTS_early_features_full_Minibest.csv
  ALL_COHORTS_late_features_full_MoCA.csv
  ALL_COHORTS_late_features_full_Minibest.csv
  ALL_COHORTS_features_full_MoCA.csv         (early + late combined)
  ALL_COHORTS_features_full_Minibest.csv     (early + late combined)
'''

import os
import yaml
import numpy as np
import pandas as pd

from pathlib import Path


# =============================================================================
# LABEL ASSIGNMENT FUNCTIONS.
# =============================================================================

ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH =  os.path.join(ROOT_PATH, 'config.yaml')

def assign_moca_labels(features_df, time, moca_labels_df, config, moca_sub_labels_df=None):
    # Canonical output column names (shared across early and late)
    canonical_targets = config['moca_targets']  # ["MoCA", "Memory_Score", ...]

    if time == "early":
        # Early period: only the total MoCA score exists (no subscores at entry)
        source_cols = config['moca_early_targets']          # ["MoCA"]
        canonical   = canonical_targets[:len(source_cols)]  # ["MoCA"]
        score_map   = moca_labels_df.set_index("Subject")[source_cols]

    elif time == "late":
        # Late period: total score + all subscores
        main_source = config['moca_late_targets']           # ["MoCA Post"]
        sub_source  = config['moca_sub_late_targets']       # ["Memory Score", ...]
        source_cols = main_source + sub_source              # all source names
        canonical   = canonical_targets                     # all canonical names

        main_map = moca_labels_df.set_index("Subject")[main_source]
        if moca_sub_labels_df is not None:
            sub_map   = moca_sub_labels_df.set_index("Subject")[sub_source]
            score_map = pd.concat([main_map, sub_map], axis=1)
        else:
            score_map = main_map

    else:
        return features_df

    # Merge then rename source columns → canonical names
    final_df = pd.merge(features_df, score_map, on="Subject", how="left")
    rename_map = dict(zip(source_cols, canonical))
    final_df = final_df.rename(columns=rename_map)

    # Early rows have no subscores — fill those columns with NaN so the
    # combined file has consistent columns across both periods
    for col in canonical_targets:
        if col not in final_df.columns:
            final_df[col] = np.nan

    return final_df


def assign_minibest_labels(features_df, time, minibest_labels_df, config):
    # Canonical output column names (shared across early and late)
    canonical_targets = config['minibest_targets']  # ["Minibest", "Anticipatory_Left", ...]

    if time == "early":
        main_source = config['minibest_early_targets']      # ["Minibest Baseline"]
        sub_source  = config['minibest_sub_early_targets']  # ["b_minibest_mot_anticipatory_left", ...]
    elif time == "late":
        main_source = config['minibest_late_targets']       # ["Minibest Exit"]
        sub_source  = config['minibest_sub_late_targets']   # ["e_minibest_mot_anticipatory_left", ...]
    else:
        return features_df

    source_cols = main_source + sub_source  # all source names for this period
    canonical   = canonical_targets         # all canonical names (same order)

    main_map  = minibest_labels_df.set_index("Subject")[main_source]
    sub_map   = minibest_labels_df.set_index("Subject")[sub_source]
    score_map = pd.concat([main_map, sub_map], axis=1)

    # Merge then rename source columns → canonical names
    final_df   = pd.merge(features_df, score_map, on="Subject", how="left")
    rename_map = dict(zip(source_cols, canonical))
    final_df   = final_df.rename(columns=rename_map)

    return final_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    data_path = os.path.join(ROOT_PATH, config['data_paths']['feature_path'])
    label_path = os.path.join(ROOT_PATH, config['data_paths']['label_path'])
    output_dir = data_path

    # -------------------------------------------------------------------------
    # Load label files
    # -------------------------------------------------------------------------
    moca_df     = pd.read_csv(os.path.join(label_path, "Final_MoCA.csv"))
    moca_sub_df = pd.read_csv(os.path.join(label_path, "Final_MoCA_Sub.csv"))
    minibest_df = pd.read_csv(os.path.join(label_path, "Final_Minibest.csv"))

    # -------------------------------------------------------------------------
    # Process early and late, then combine into single output files
    # Each row is tagged with a 'Time' column ('early' or 'late') so the
    # two periods remain distinguishable in the combined file.
    # -------------------------------------------------------------------------
    moca_parts = []
    minibest_parts = []

    for time_period in ["early", "late"]:
        features_filename = f"ALL_COHORTS_{time_period}_features_full.csv"
        features_path = os.path.join(output_dir, features_filename)

        if not os.path.exists(features_path):
            print(f"WARNING: Features file not found, skipping: {features_path}")
            continue

        print(f"\nLoading {features_filename}...")
        features_df = pd.read_csv(features_path)
        features_df['Time'] = time_period
        print(f"  Shape: {features_df.shape}, Unique subjects: {features_df['Subject'].nunique()}")

        # MoCA
        features_with_moca = assign_moca_labels(
            features_df=features_df,
            time=time_period,
            moca_labels_df=moca_df,
            config=config,
            moca_sub_labels_df=moca_sub_df
        )
        moca_parts.append(features_with_moca)
        moca_individual_out = os.path.join(output_dir, f"ALL_COHORTS_{time_period}_features_full_MoCA.csv")
        features_with_moca.to_csv(moca_individual_out, index=False)
        print(f"  Saved MoCA ({time_period}): {os.path.basename(moca_individual_out)}  shape={features_with_moca.shape}")

        # Minibest
        features_with_minibest = assign_minibest_labels(
            features_df=features_df,
            time=time_period,
            minibest_labels_df=minibest_df,
            config=config
        )
        minibest_parts.append(features_with_minibest)
        minibest_individual_out = os.path.join(output_dir, f"ALL_COHORTS_{time_period}_features_full_Minibest.csv")
        features_with_minibest.to_csv(minibest_individual_out, index=False)
        print(f"  Saved Minibest ({time_period}): {os.path.basename(minibest_individual_out)}  shape={features_with_minibest.shape}")

    # -------------------------------------------------------------------------
    # Concatenate and save combined files
    # -------------------------------------------------------------------------
    if moca_parts:
        moca_combined = pd.concat(moca_parts, ignore_index=True)
        moca_out = os.path.join(output_dir, "ALL_COHORTS_features_full_MoCA.csv")
        moca_combined.to_csv(moca_out, index=False)
        print(f"\nSaved combined MoCA file: {os.path.basename(moca_out)}")

    if minibest_parts:
        minibest_combined = pd.concat(minibest_parts, ignore_index=True)
        minibest_out = os.path.join(output_dir, "ALL_COHORTS_features_full_Minibest.csv")
        minibest_combined.to_csv(minibest_out, index=False)
        print(f"\nSaved combined Minibest file: {os.path.basename(minibest_out)}")

    print("\nLabel assignment complete.")


if __name__ == "__main__":
    main()