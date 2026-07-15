"""
Window-level feature extraction from the preprocessed break-time IMU data.

WHAT THIS SCRIPT DOES
    1. Loads the early- and late-program break-time recordings produced by
       process_imu.py (ALL_COHORTS_early_data.csv / ALL_COHORTS_late_data.csv).
    2. Slides fixed-size windows over each subject's accelerometer/gyroscope
       signals and computes, per window:
         - gait features (cadence, stride duration/length, gait velocity,
           symmetry, variability, RMS acceleration/jerk, entropy, ...),
         - ECDF percentile features of the filtered vertical acceleration,
         - general statistical/time-series features.
       Stride-length estimation uses a sex-stratified leg length, so the
       MoCA/Minibest label files are read for the Subject -> Sex mapping.
    3. Clips numeric feature outliers to Q1/Q3 +/- 3*IQR and writes one
       feature table per split:
         ALL_COHORTS_early_features_full.csv
         ALL_COHORTS_late_features_full.csv

    Windowing parameters (window_size, stride, fs) and estimated leg lengths
    come from config.yaml. Run with main(smoke_test=True) to verify the
    pipeline on only the 10 highest-volume subjects.
"""

import itertools
import os
import sys
import yaml
import pandas as pd
import numpy as np

from scipy.signal import find_peaks, welch
from scipy.stats import variation, entropy, kurtosis, skew
from scipy.fft import fft
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from pathlib import Path


ROOT_PATH = Path(__file__).resolve().parent
CONFIG_PATH =  os.path.join(ROOT_PATH, 'config.yaml')

from utils import (
    compute_entropy_from_hist,
    sliding_window,
    range_valid_or_nan,
    butter_filter,
    calculate_stride_length,
    timeofday_to_seconds,
)


# ----------------------------
# Helper Functions
# ----------------------------

def extract_ecdf_features(signal):
    """
    Compute ECDF-based features (percentiles and basic stats) on a 1D NumPy array.
    Returns a dict of features.
    """
    # remove NaN
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return {}

    feats = {}
    for q in [0, 5, 25, 50, 75, 95, 100]:
        feats[f'ECDF_{q}'] = np.percentile(signal, q)
    feats['ECDF_mean'] = np.mean(signal)
    feats['ECDF_std'] = np.std(signal)

    return feats


def compute_statistical_features(signal):
    """
    Compute a collection of statistical features on a 1D NumPy array signal.
    Returns a dict of features.
    """
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return {}

    n = len(signal)
    diffs = np.diff(signal)
    fft_coeffs = np.abs(fft(signal))

    feats = {
        'abs_energy': np.sum(signal**2),
        'absolute_maximum': np.max(np.abs(signal)),
        'absolute_sum_of_changes': np.sum(np.abs(diffs)),
        'autocorrelation_lag1': np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else np.nan,
        'binned_entropy': entropy(np.histogram(signal, bins=10)[0] + 1e-8),
        'mean_abs_change': np.mean(np.abs(diffs)),
        'skewness': skew(signal, nan_policy='omit'),
        'kurtosis': kurtosis(signal, nan_policy='omit'),
        'fourier_entropy': entropy(welch(signal, nperseg=min(256, n))[1][:10] + 1e-8),
        'mean_second_derivative_central': np.mean(np.diff(signal, n=2)) if n > 2 else np.nan,
        'number_peaks': len(find_peaks(signal, distance=5)[0]),
        'fft_mean': np.mean(fft_coeffs),
        'fft_var': np.var(fft_coeffs),
        'change_quantiles': np.nanquantile(signal, 0.75) - np.nanquantile(signal, 0.25),
        'longest_strike_above_mean': max((sum(1 for _ in g) for k, g in itertools.groupby(signal > np.mean(signal)) if k), default=0)
    }

    # Linear trend
    if n > 1:
        X = np.arange(n).reshape(-1, 1)
        lr = LinearRegression().fit(X, signal)
        feats['linear_trend_slope'] = lr.coef_[0]
        feats['linear_trend_intercept'] = lr.intercept_
    else:
        feats['linear_trend_slope'] = np.nan
        feats['linear_trend_intercept'] = np.nan

    return feats


def extract_gait_features(window, fs, sex, config):
    gait_features = {}

    window_duration = window['Time From Subject Start Time'].iloc[-1] - window['Time From Subject Start Time'].iloc[0]
    if window_duration <= 0:
        return {}

    # 1) Preprocess accelerometer axes
    ax_raw = window['RawAccelX'].values
    ay_raw = window['RawAccelY'].values
    az_raw = window['RawAccelZ'].values

    # High pass filtering
    ax_hp = butter_filter(ax_raw, cutoff=0.3, fs=fs, btype="high")
    ay_hp = butter_filter(ay_raw, cutoff=0.3, fs=fs, btype="high")
    az_hp = butter_filter(az_raw, cutoff=0.3, fs=fs, btype="high")

    # Low pass filtering
    ax_filt = butter_filter(ax_hp, cutoff=5.0, fs=fs, btype="low")
    ay_filt = butter_filter(ay_hp, cutoff=5.0, fs=fs, btype="low")
    az_filt = butter_filter(az_hp, cutoff=5.0, fs=fs, btype="low")

    num_steps = None

    # Peak-based features (cadence, stride duration/variability, stance & swing time)
    try:
        peaks, _ = find_peaks(az_filt, distance=int(fs * 0.5))
        if len(peaks) >= 2:
            num_steps = len(peaks)

            # Cadence
            cadence = (num_steps / window_duration) * 60  # Convert steps/sec to steps/min

            gait_features['Cadence (steps/min)'] = range_valid_or_nan(cadence, 0, 150)

            # Stride Durations
            step_times = window['Time From Subject Start Time'].iloc[peaks].values
            step_durations = np.diff(step_times)
            stride_durations = [step_durations[i] + step_durations[i+1] for i in range(len(step_durations) - 1)]

            if len(step_durations) >= 2:
                # Separate step durations into left (even indices) and right (odd indices)
                left_steps = step_durations[0::2]
                right_steps = step_durations[1::2]

                if len(left_steps) > 0 and len(right_steps) > 0:
                    mean_left = np.mean(left_steps)
                    mean_right = np.mean(right_steps)

                    # Gait Symmetry Index (GSI) for Step Time
                    gsi = (abs(mean_left - mean_right) / (0.5 * (mean_left + mean_right))) * 100
                    gait_features['Gait Symmetry Index (GSI)'] = gsi

            if stride_durations:
                avg_stride_duration = np.mean(stride_durations)
                gait_features['Stride Duration (s)'] = range_valid_or_nan(avg_stride_duration, 0.7, float('inf'))

                stride_variability = variation(stride_durations)
                gait_features['Stride Variability'] = stride_variability

                # Stance and Swing Time
                # Derived from the biphasic gait-cycle model (~60% stance / 40% swing),
                # following the manuscript (stance/swing are not directly measured).
                validated_stride = gait_features['Stride Duration (s)']
                gait_features['Stance Time (s)'] = validated_stride * 0.6
                gait_features['Swing Time (s)'] = validated_stride * 0.4

                # Stride Time Variability (as Coefficient of Variation)
                mean_stride_dur = np.mean(stride_durations)
                std_stride_dur = np.std(stride_durations)
                if mean_stride_dur > 0:
                    gait_features['Stride Time CV (%)'] = (std_stride_dur / mean_stride_dur) * 100
    except Exception as e:
        print(f"DEBUG: Could not calculate temporal features (peaks): {e}")

    # Stride Length
    try:
        if sex == "Male":
            leg_length = config['male_estimated_leg_length']
        elif sex == "Female":
            leg_length = config['female_estimated_leg_length']
        else:
            print("WARNING: This subject doesn't have a Sex column! Assume this subject is female.")
            leg_length = config['female_estimated_leg_length']

        # Get the average stride length
        avg_stride_length, avg_gait_velocity = calculate_stride_length(
            az_raw=az_raw,
            ay_raw=ay_raw,
            leg_length=leg_length,
            fs=fs
        )

        if not np.isnan(avg_stride_length):
            gait_features['Stride Length (m)'] = avg_stride_length

        if not np.isnan(avg_gait_velocity):
            gait_features['Gait Velocity (m/s)'] = avg_gait_velocity

    except Exception as e:
        print(f"DEBUG: Could not calculate stride length (Zijlstra): {e}")

    # RMS Vertical Acceleration
    try:
        rms_vertical_accel = np.sqrt(np.mean(az_filt ** 2))
        gait_features['RMS Vertical Acceleration (g)'] = rms_vertical_accel
    except Exception as e:
        print(f"DEBUG: Could not calculate RMS vertical accel: {e}")

    # Movement Smoothness (Jerk)
    try:
        svm_accel = np.sqrt(ax_filt**2 + ay_filt**2 + az_filt**2)

        jerk = np.diff(svm_accel) / (1 / fs)

        gait_features["RMS Jerk"] = np.sqrt(np.mean(jerk**2))
    except Exception as e:
        print(f"DEBUG: Could not calculate RMS Jerk: {e}")

    # Gyroscope-related features
    try:
        # Apply a low-pass filter.
        # Gyroscope axes do not need a high-pass filter since they have no
        # gravitational DC component.
        gyro_x = butter_filter(window['RawGyroX'].values, cutoff=5.0, fs=fs, btype="low")
        gyro_y = butter_filter(window['RawGyroY'].values, cutoff=5.0, fs=fs, btype="low")
        gyro_z = butter_filter(window['RawGyroZ'].values, cutoff=5.0, fs=fs, btype="low")

        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        if len(gyro_mag) > 1:
            gait_features['Direction Change'] = np.mean(np.abs(np.diff(gyro_mag)))
        gait_features['Entropy Orientation'] = compute_entropy_from_hist(gyro_mag)
    except Exception as e:
        print(f"DEBUG: Could not calculate gyro features: {e}")

    try:
        gait_features['Entropy Vertical Acceleration'] = compute_entropy_from_hist(az_filt)
    except Exception as e:
        print(f"DEBUG: Could not calculate velocity entropy: {e}")

    return gait_features

# ----------------------------
# Main Function
# ----------------------------


def main(smoke_test=False):
    # -------------------------------------------------------------------------
    # SMOKE TEST MODE
    # When smoke_test=True, runs on only the top 10 subjects (by data volume)
    # of the early-data file to quickly verify the pipeline runs without errors.
    # Set to False for the full run over both early and late data.
    # -------------------------------------------------------------------------
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    data_path = os.path.join(ROOT_PATH, config['data_paths']['processed_path'])
    label_path = os.path.join(ROOT_PATH, config['data_paths']['label_path'])
    output_dir = os.path.join(data_path, "features")
    os.makedirs(output_dir, exist_ok=True)

    early_data_file_name = "ALL_COHORTS_early_data.csv"
    late_data_file_name = "ALL_COHORTS_late_data.csv"

    early_data_path = os.path.join(data_path, early_data_file_name)
    late_data_path = os.path.join(data_path, late_data_file_name)

    # the Subject -> Sex mapping is needed, to select a sex-stratified leg length for stride estimation.
    moca_file = "Final_MoCA.csv"
    minibest_file = "Final_Minibest.csv"

    moca_path = os.path.join(label_path, moca_file)
    minibest_path = os.path.join(label_path, minibest_file)

    moca_df = pd.read_csv(moca_path)
    minibest_df = pd.read_csv(minibest_path)

    print("Creating Subject-to-Sex mapping...")
    sex_info_moca = moca_df[['Subject', 'Sex']]
    sex_info_minibest = minibest_df[['Subject', 'Sex']]
    combined_sex_info = pd.concat(
        [sex_info_moca, sex_info_minibest]
    ).drop_duplicates().set_index('Subject')
    print("Mapping created successfully.")

    window_size = config['sliding_window']['window_size']
    stride = config['sliding_window']['stride']
    fs = config['sliding_window']['fs']

    for path in [early_data_path, late_data_path]:
        df = pd.read_csv(path)

        # Calculate the elapsed time from CohortProgramDay and TimeOfDay
        df['_continuous_sec'] = df['CohortProgramDay'].astype(float) * 86400.0 + timeofday_to_seconds(df['TimeOfDay'])

        # Drop rows whose temporal fields could not be parsed 
        df.dropna(subset=['_continuous_sec'], inplace=True)
        df.sort_values(['Subject', '_continuous_sec'], inplace=True)

        # Elapsed time from each subject's first available sample 
        first_time = df.groupby('Subject')['_continuous_sec'].transform('min')
        df['Time From Subject Start Time'] = df['_continuous_sec'] - first_time
        df.drop(columns=['_continuous_sec'], inplace=True)

        print("\nStarting feature extraction...")
        all_features_list = []

        grouped_data = df.groupby(['Cohort', 'Subject'])
        base_filename = os.path.basename(path).replace('_data.csv', '')

        if smoke_test:
            # Select top 10 subjects by number of samples (data volume)
            grouped_data = list(grouped_data)
            grouped_data = sorted(grouped_data, key=lambda x: len(x[1]), reverse=True)[:10]
            print(f"\n[SMOKE TEST] Running on top 10 subjects by data volume:")
            for (c, s), d in grouped_data:
                print(f"  Subject {s} (Cohort {c}): {len(d)} samples")

        for (cohort, subject), data in tqdm(grouped_data, desc=f"Subjects in {base_filename}"):
            data = data.reset_index(drop=True)
            if len(data) < 100: continue

            try:
                subject_sex = combined_sex_info.loc[subject, 'Sex']
            except KeyError:
                print(f"Warning: Sex for subject {subject} not found. Skipping subject.")
                continue

            windows = sliding_window(
                df=data,
                window_size=window_size,
                stride=stride
            )

            for window in tqdm(windows, desc=f"Windows for {subject}", leave=False):
                vertical_accel_raw = window['RawAccelZ'].values
                az_hp = butter_filter(vertical_accel_raw, cutoff=0.3, fs=fs, btype="high")
                az_filt = butter_filter(az_hp, cutoff=5.0, fs=fs, btype="low")

                gait_features = extract_gait_features(
                    window=window,
                    fs=fs,
                    sex=subject_sex,
                    config=config
                )
                ecdf_features = extract_ecdf_features(
                    signal=az_filt
                )
                stats_features = compute_statistical_features(
                    signal=az_filt
                )

                base_info = {
                    "Cohort": cohort,
                    "Subject": subject,
                    "VisitIndex": window['VisitIndex'].iloc[0],
                    "CohortProgramDay": window['CohortProgramDay'].iloc[0],
                    "Window Start TimeOfDay": window['TimeOfDay'].iloc[0],
                    "Window Start Elapsed (s)": window['Time From Subject Start Time'].iloc[0],
                }

                base_info.update(gait_features)
                base_info.update(ecdf_features)
                base_info.update(stats_features)

                all_features_list.append(base_info)

        print(f"Finished feature extraction. Found {len(all_features_list)} valid windows.")

        # Save results to CSV for validation
        df_out = pd.DataFrame(all_features_list)

        non_feature_cols = [
            'Cohort', 'Subject', 'VisitIndex', 'CohortProgramDay',
            'Window Start TimeOfDay', 'Window Start Elapsed (s)'
        ]
        feature_cols = [c for c in df_out.columns if c not in non_feature_cols]
        numeric_cols = df_out[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        Q1         = df_out[numeric_cols].quantile(0.25)
        Q3         = df_out[numeric_cols].quantile(0.75)
        IQR        = Q3 - Q1
        clip_lower = Q1 - 3 * IQR
        clip_upper = Q3 + 3 * IQR

        n_clipped = (
            (df_out[numeric_cols] < clip_lower) | (df_out[numeric_cols] > clip_upper)
        ).sum().sum()

        df_out[numeric_cols] = df_out[numeric_cols].clip(
            lower=clip_lower, upper=clip_upper, axis=1
        )

        suffix = "smoke_test" if smoke_test else "full"
        out_path = os.path.join(output_dir, f"{base_filename}_features_{suffix}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"Saved features to: {out_path}")

        if smoke_test:
            print(f"\n[SMOKE TEST] Completed on {len(grouped_data)} subjects.")
            print("\n[SMOKE TEST] Set smoke_test=False in main() to run the full pipeline.")
            return


if __name__ == "__main__":
    main()