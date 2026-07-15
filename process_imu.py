"""
IMU preprocessing for the de-identified CEP waist-mounted IMU dataset.

WHAT THIS SCRIPT DOES
    1. Loads the released, cleaned IMU source recordings.
    2. Keeps only the unstructured break/lunch periods (via TimeOfDay).
    3. Splits each cohort into early- vs late-program records, which the
       associated study pairs with enrollment (baseline) vs exit labels.

config.yaml SCHEMA
    data_paths:
      source_file:           ./SourceData/ALL_COHORTS_combined_clean_data.csv
      cohort_midpoints_file: ./MetaData/cohort_midpoints.csv
      output_path:           ./processed
"""

import os
import yaml
import pandas as pd

from pathlib import Path


# --- Configuration ----------------------------------------------------------

ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH =  os.path.join(ROOT_PATH, 'config.yaml')

USEFUL_COLS = [
    "Subject", "Cohort",
    "CollectionYear", "VisitIndex", "CohortProgramDay", "TimeOfDay",
    "RawAccelX", "RawAccelY", "RawAccelZ",
    "RawGyroX", "RawGyroY", "RawGyroZ",
]

# Break/lunch windows analyzed in the study (clock time, no date).
# Each entry is (start, end), inclusive, as HH:MM:SS strings.
BREAK_WINDOWS = [
    ("10:55:00", "11:10:00"),
    ("12:05:00", "12:45:00"),
]


# --- Pipeline steps ---------------------------------------------------------

def filter_break_times(input_df, windows=BREAK_WINDOWS, time_col="TimeOfDay"):
    """
    Keep only rows whose clock time falls inside one of the break/lunch windows.

    Uses the de-identified TimeOfDay column. Times are parsed as how much time 
    has passed since the start of the day.

    Input:
        - input_df: DataFrame containing `time_col` (HH:MM:SS.ffffff strings)
        - windows:  list of (start, end) inclusive clock-time strings
    Output:
        - filtered_df: rows recorded within any of the break windows
    """
    # clock time -> time since midnight
    time_of_day = pd.to_timedelta(input_df[time_col])  

    in_any_window = pd.Series(False, index=input_df.index)
    for start, end in windows:
        in_any_window |= (time_of_day >= pd.to_timedelta(start)) & \
                         (time_of_day <= pd.to_timedelta(end))

    filtered_df = input_df[in_any_window].copy()

    return filtered_df


def load_cohort_midpoints(midpoints_path, cohort_col="Cohort", midpoint_col="MidpointProgramDay"):
    """
    Load the cohort-relative early/late cut-off from the release metadata.

    Input:
        - midpoints_path: path to MetaData/cohort_midpoints.csv
    Output:
        - midpoints_df: DataFrame with [cohort_col, midpoint_col]
    """
    midpoints_df = pd.read_csv(midpoints_path)

    missing_cols = {cohort_col, midpoint_col} - set(midpoints_df.columns)
    if missing_cols:
        raise ValueError(f"{midpoints_path} is missing required column(s): {sorted(missing_cols)}")

    return midpoints_df[[cohort_col, midpoint_col]]


def split_early_late(input_df, midpoints_df, cohort_col="Cohort",
                     day_col="CohortProgramDay", midpoint_col="MidpointProgramDay"):
    """
    Split records into early- vs late-program halves per cohort.

    A row is "early" if its CohortProgramDay is before the cohort midpoint and
    "late" otherwise. The merged midpoint column is dropped from the
    outputs so both splits keep the original source schema.

    Input:
        - input_df: DataFrame with [cohort_col, day_col] (e.g., break-time rows)
        - midpoints_df: DataFrame with [cohort_col, midpoint_col]
    Output:
        - (early_df, late_df)
    """
    merged = input_df.merge(midpoints_df, on=cohort_col, how="left")

    unmatched = merged.loc[merged[midpoint_col].isna(), cohort_col].unique()
    if len(unmatched):
        print(f"Warning: no midpoint found for cohort(s) {sorted(unmatched)}; "
              f"their rows are excluded from both splits.")

    is_early = merged[day_col] < merged[midpoint_col]
    early_df = merged[is_early].drop(columns=[midpoint_col])
    late_df = merged[~is_early & merged[midpoint_col].notna()].drop(columns=[midpoint_col])

    return early_df, late_df



if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    source_file = os.path.join(ROOT_PATH, config["data_paths"]["source_file"])
    midpoints_file = os.path.join(ROOT_PATH, config["data_paths"]["cohort_midpoints_file"])
    output_path = os.path.join(ROOT_PATH, config["data_paths"]["processed_path"])
    os.makedirs(output_path, exist_ok=True)

    filtered_path = os.path.join(output_path, "ALL_COHORTS_filtered_break_time_data.csv")
    early_path = os.path.join(output_path, "ALL_COHORTS_early_data.csv")
    late_path = os.path.join(output_path, "ALL_COHORTS_late_data.csv")

    # Load the cleaned source recordings and keep only break/lunch periods.
    if os.path.exists(filtered_path):
        print(f"Found existing filtered file, loading: {filtered_path}")
        filtered_df = pd.read_csv(filtered_path)
    else:
        print(f"Loading cleaned source recordings: {source_file}")
        source_df = pd.read_csv(source_file, usecols=USEFUL_COLS)
        filtered_df = filter_break_times(source_df)
        filtered_df.to_csv(filtered_path, index=False)
        print(f"Saved break-time data to {filtered_path}")

    # Split each cohort into early- vs late-program records using the
    # cohort-relative midpoint.
    midpoints_df = load_cohort_midpoints(midpoints_file)
    early_df, late_df = split_early_late(filtered_df, midpoints_df)

    early_df.to_csv(early_path, index=False)
    print(f"Early data saved in {early_path}")
    late_df.to_csv(late_path, index=False)
    print(f"Late data saved in {late_path}")

    print("============= Complete Data Preprocessing =============")