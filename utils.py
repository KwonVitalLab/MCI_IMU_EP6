"""
Shared utility functions used across the pipeline scripts (not run directly).

WHAT THIS MODULE PROVIDES
    - De-identified time helper: convert the TimeOfDay column to seconds
      since midnight (timeofday_to_seconds).
    - Signal processing and feature-extraction helpers used by
      extract_features.py: Butterworth filtering, gap-aware sliding windows,
      range validation, histogram entropy, and Zijlstra-based stride length /
      gait velocity estimation.
    - ML pipeline helpers used by train_models.py, run_shap.py, and
      analyze_fairness.py: subject-aware train/test splits, z-score feature
      normalization, and SHAP analysis with subject-fair aggregation.
    - Plotting functions: SHAP summary/bar plots, Pearson-correlation
      scatter plots, and Bland-Altman agreement plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from scipy.signal import butter, filtfilt
from scipy.stats import entropy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from pathlib import Path


# Path configuration
ROOT_PATH = Path(__file__).resolve().parent

# ----------------------------
# DE-IDENTIFIED TIME HELPERS
# ----------------------------

def timeofday_to_seconds(time_series):
    """
    Convert 'HH:MM:SS.ffffff' clock-time strings to seconds since midnight.

    Parsing as a timedelta interprets each value as elapsed time since midnight, 
    giving a numeric axis that can be compared/sorted and combined with CohortProgramDay 
    to rebuild a continuous per-subject timeline.

    Parameters
    ----------
    time_series : pd.Series or array-like of str
        Clock-time strings (HH:MM:SS[.ffffff]).

    Returns
    -------
    pd.Series of float
        Seconds since midnight (NaN where the value could not be parsed).
    """
    return pd.to_timedelta(time_series).dt.total_seconds()


# ----------------------------
# FEATURE EXTRACTIONS
# ----------------------------

def range_valid_or_nan(value, low, high):
    """
    Return value if within [low, high], else np.nan.

    Used for physiological range validation of gait features.
    """
    return value if low <= value <= high else np.nan


def butter_filter(data, cutoff, fs, btype, order=2):
    """
    Apply a zero-phase Butterworth filter.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    if norm_cutoff >= 1.0:
        return data  # cutoff at or above Nyquist — filter is invalid, skip
    b, a = butter(order, norm_cutoff, btype=btype)
    return filtfilt(b, a, data)


def calculate_stride_length(az_raw, ay_raw, leg_length, fs):
    """
    Compute mean stride length using Zijlstra & Hof (2003) method.

    Parameters
    ----------
    az_raw : np.ndarray
        Raw vertical acceleration (Z-axis, aligned with gravity).
    ay_raw : np.ndarray
        Raw anterior-posterior acceleration (Y-axis, aligned forward at setup).
    leg_length : float
        Estimated leg length in meters.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    mean_stride_length : float
        Mean stride length in meters. NaN if computation fails.
    mean_gait_velocity : float
        Mean gait velocity in m/s (stride length / stride duration). NaN if fails.
    """
    ZIJLSTRA_IC_LP_CUTOFF = 2.0    # Hz
    ZIJLSTRA_DISP_HP_CUTOFF = 0.1  # Hz
    ZIJLSTRA_PEAK_MIN_DIST_S = 0.3 # seconds
    ZIJLSTRA_MIN_STEPS = 2         

    min_samples = int(0.5 * fs)
    peak_min_distance = max(int(ZIJLSTRA_PEAK_MIN_DIST_S * fs), 1)

    if len(az_raw) < min_samples or len(ay_raw) < min_samples:
        return np.nan, np.nan

    # ── Step 1: IC detection from AP acceleration (Zijlstra method) ──
    try:
        ay_lp = butter_filter(ay_raw, ZIJLSTRA_IC_LP_CUTOFF, fs, btype="low")
    except Exception:
        return np.nan, np.nan

    # Find zero-crossings: positive → negative
    zero_crossings = []
    for i in range(1, len(ay_lp)):
        if ay_lp[i - 1] > 0 and ay_lp[i] <= 0:
            zero_crossings.append(i)

    if len(zero_crossings) < 1:
        return np.nan, np.nan

    # For each zero-crossing, find the peak immediately preceding it
    ic_indices = []
    for zc in zero_crossings:
        search_start = max(0, zc - int(0.5 * fs))
        segment = ay_lp[search_start:zc]
        if len(segment) < 2:
            continue
        local_peak = np.argmax(segment)
        peak_idx = search_start + local_peak
        if ic_indices and (peak_idx - ic_indices[-1]) < peak_min_distance:
            continue
        ic_indices.append(peak_idx)

    if len(ic_indices) < 3:
        # Need at least 3 ICs to form 2 step lengths → 1 stride
        return np.nan, np.nan

    # ── Step 2: Displacement from vertical acceleration ──────────────
    try:
        az_lp = butter_filter(az_raw, ZIJLSTRA_IC_LP_CUTOFF, fs, btype="low")
        az_dynamic = az_lp - np.mean(az_lp)
        velocity = np.cumsum(az_dynamic * 9.81) / fs
        displacement = np.cumsum(velocity) / fs
        displacement = butter_filter(
            displacement, ZIJLSTRA_DISP_HP_CUTOFF, fs, btype="high"
        )
    except Exception:
        return np.nan, np.nan

    # ── Step 3: Compute step lengths from IC-segmented displacement ──
    step_lengths = []
    step_durations = []
    for i in range(len(ic_indices) - 1):
        seg = displacement[ic_indices[i]:ic_indices[i + 1]]
        if len(seg) < 2:
            continue
        h = np.max(seg) - np.min(seg)
        arg = 2 * leg_length * h - h**2
        if arg > 0:
            step_lengths.append(2 * np.sqrt(arg))
            step_durations.append((ic_indices[i + 1] - ic_indices[i]) / fs)

    if len(step_lengths) < ZIJLSTRA_MIN_STEPS:
        return np.nan, np.nan

    # ── Step 4: Convert step lengths to stride lengths ───────────────
    # Stride = two consecutive steps (left→right + right→left)
    stride_lengths = []
    stride_durations = []
    for i in range(len(step_lengths) - 1):
        stride_lengths.append(step_lengths[i] + step_lengths[i + 1])
        stride_durations.append(step_durations[i] + step_durations[i + 1])

    if len(stride_lengths) == 0:
        return np.nan, np.nan

    mean_stride_length = np.mean(stride_lengths)
    mean_stride_duration = np.mean(stride_durations)

    if mean_stride_duration > 0:
        mean_gait_velocity = mean_stride_length / mean_stride_duration
    else:
        mean_gait_velocity = np.nan

    return mean_stride_length, mean_gait_velocity


def compute_entropy_from_hist(signal, bins=20):
    """Shannon entropy computed over a histogram of the signal."""
    valid_signal = signal[np.isfinite(signal)]
    if valid_signal.size == 0:
        return np.nan
    counts, _ = np.histogram(valid_signal, bins=bins, density=False)
    probs = counts / np.sum(counts)
    return entropy(probs + 1e-8)


def sliding_window(df, window_size=100, stride=50, gap_threshold=60):
    """
    Generates overlapping windows of size `window_size` over a DataFrame sorted by Time.
    Discards windows where any internal time gap > gap_threshold seconds.
    """
    if len(df) < window_size:
        return []
    windows = []
    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start : start + window_size]
        time_diffs = window['Time From Subject Start Time'].diff().fillna(0)
        if time_diffs.max() > gap_threshold:
            continue
        windows.append(window)
    return windows


# ----------------------------
# PIPELINE
# ----------------------------


def subject_aware_train_test_split(X, y, groups, test_size=0.2, random_state=42):
    # Get unique subjects and split them
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state
    )

    # Create boolean masks for train and test sets
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)

    # Apply masks to X and y
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test


# Perform z-score normalization on training set and transform both sets
def normalize_features(train_data, test_data, feature_cols):
    if train_data.empty:
        raise ValueError("Training data is empty.")
    if test_data.empty:
        raise ValueError("Test data is empty.")
    if not feature_cols:
        raise ValueError("Feature columns list is empty.")

    # copy=False scales in place on the owned copies made below, halving the
    # peak memory of this function. The computed values are unchanged.
    scaler = RobustScaler(copy=False)

    # Extract feature data. copy=True guarantees an owned buffer — to_numpy can
    # otherwise return a view of the DataFrame block, which in-place scaling
    # must never touch.
    X_train = train_data[feature_cols].to_numpy(dtype=np.float64, copy=True)
    X_test = test_data[feature_cols].to_numpy(dtype=np.float64, copy=True)

    # Fit scaler on training data and transform
    X_train_norm = scaler.fit_transform(X_train)

    # Apply the same transformation on test set
    X_test_norm = scaler.transform(X_test)

    # Verify output shapes
    if X_train_norm.shape[0] != len(train_data) or X_test_norm.shape[0] != len(test_data):
        raise ValueError("Normalization altered the number of samples")
    if X_train_norm.shape[1] != len(feature_cols) or X_test_norm.shape[1] != len(feature_cols):
        raise ValueError("Normalization altered the number of features")

    X_train_norm_df = pd.DataFrame(X_train_norm, index=train_data.index, columns=feature_cols)
    X_test_norm_df = pd.DataFrame(X_test_norm, index=test_data.index, columns=feature_cols)

    return X_train_norm_df, X_test_norm_df, scaler    


# Run the SHAP analysis to find which features are most important for our regression tasks
def run_shap_analysis(model_instance, model_name, X_train, X_test, target_name, assess_name, config, subjects=None):
    import cupy
    import cuml

    feature_names = X_train.columns

    # 2. Calculate SHAP values on the GPU
    # Both cuML and shap TreeExplainer use tree_path_dependent 
    if model_name == "RandomForest":
        X_test_np_array = X_test.to_numpy()
        X_test_gpu_array = cupy.asarray(X_test_np_array)

        # cuML's explainer
        explainer = cuml.explainer.TreeExplainer(model=model_instance)
        shap_values_gpu = explainer.shap_values(X_test_gpu_array)
        base_values_gpu = explainer.expected_value

        # 3. Convert all components from GPU (CuPy) to CPU (NumPy)
        shap_values_np = cupy.asnumpy(shap_values_gpu)

        # The base_values (expected_value) can be a scalar or an array
        if hasattr(base_values_gpu, 'get'):
            base_values_np = base_values_gpu.get() # .get() is robust for cupy scalars
        else:
            base_values_np = cupy.asnumpy(base_values_gpu)

        base_values_np = np.atleast_1d(base_values_np)

        # 4. Create a CPU-based SHAP Explanation object
        shap_explanation_cpu = shap.Explanation(
            values=shap_values_np,
            base_values=base_values_np,
            data=X_test_np_array,
            feature_names=feature_names
        )
    else:
        # Standard SHAP explainer
        explainer = shap.TreeExplainer(
            model=model_instance, 
            feature_perturbation="tree_path_dependent"
        )
        shap_explanation_cpu = explainer(X_test)
    
    shap_path = os.path.join(ROOT_PATH, config['result_paths']['shap_path'])
    plot_path = os.path.join(shap_path, "plots")
    # result_path = os.path.join(shap_path, "results")

    os.makedirs(plot_path, exist_ok=True)

    plot_shap_single_target(
        shap_explanation=shap_explanation_cpu,
        target_name=target_name,
        dataset_name=assess_name,
        model_name=model_name,
        plots_dir=plot_path,
        subjects=subjects 
    )

    return shap_explanation_cpu

# ----------------------------
# Plotting Functions
# ----------------------------


def plot_shap_single_target(shap_explanation, target_name, dataset_name, model_name, plots_dir, subjects=None):
    """
    Generates and saves SHAP summary plots for a single-target model.

    Beeswarm: window-level (shows full distribution of SHAP values).
    Bar: subject-fair mean |SHAP| (each subject weighted equally).

    Args:
        shap_explanation (shap.Explanation): The explanation object from explainer(X).
        target_name (str): The name of the target variable being plotted.
        dataset_name (str): The name of the dataset (e.g., 'MoCA', 'Minibest').
        model_name (str): The name of the model (e.g., 'RandomForest').
        plots_dir (str): The directory where the plot images will be saved.
        subjects (pd.Series or np.ndarray, optional): Subject IDs for subject-fair aggregation.
    """
    os.makedirs(plots_dir, exist_ok=True)
    feature_names = shap_explanation.feature_names

    if target_name == "Minibest":
        target_name = "Mini-BESTest"

    # --- 1. Beeswarm Plot (window-level) ---
    top10_idx = np.argsort(-np.abs(shap_explanation.values).mean(axis=0))[:10]
    shap_top10 = shap_explanation[:, top10_idx]

    plt.rcParams.update({
        'font.size':       14,
        'axes.titlesize':  20,
        'axes.labelsize':  18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    plt.figure()
    shap.plots.beeswarm(shap_top10, max_display=10)
    plt.title(f"Top 10 Important Features for {target_name}", fontsize=16)

    beeswarm_filename = f"{dataset_name}_{model_name}_{target_name}_shap_beeswarm.png"
    shap_beeswarm_path = os.path.join(plots_dir, beeswarm_filename)
    plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP beeswarm plot saved to {shap_beeswarm_path}")

    # --- 2. Bar Plot (subject-fair if subjects provided) ---
    if subjects is not None:
        # Extract raw SHAP values
        if isinstance(shap_explanation.values, np.ndarray):
            shap_vals = shap_explanation.values
        else:
            shap_vals = np.array(shap_explanation.values)

        # Aggregate: mean |SHAP| per subject, then mean across subjects
        shap_abs_df = pd.DataFrame(np.abs(shap_vals), columns=feature_names)
        shap_abs_df['subject'] = np.asarray(subjects)
        subject_mean_shap = shap_abs_df.groupby('subject').mean()
        fair_mean_abs_shap = subject_mean_shap.mean(axis=0)

        # Sort by importance for plotting
        top10 = fair_mean_abs_shap.sort_values(ascending=False).head(10)
        sorted_top10 = top10.sort_values(ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_top10)), sorted_top10.values, color='#1E88E5')
        plt.yticks(range(len(sorted_top10)), sorted_top10.index, fontsize=14)
        plt.xlabel("Mean |SHAP value|", fontsize=18)
        plt.title(f"Top 10 Important Features for {target_name}", fontsize=20)
        plt.tight_layout()
    else:
        # Fall back to default shap bar plot (window-weighted)
        plt.figure()
        shap.plots.bar(shap_explanation, max_display=10)
        plt.title(f"Top 10 Important Features for {target_name}")

    bar_filename = f"{dataset_name}_{model_name}_{target_name}_shap_bar.png"
    shap_bar_path = os.path.join(plots_dir, bar_filename)
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar chart saved to {shap_bar_path}")
    

def plot_pearson_correlation(y_true, y_pred, model_name, target_name, plot_dir, subjects=None, stage=None):
    """
    Generates and saves a Pearson correlation plot with a line of best fit.
    
    If subjects are provided, aggregates to one point per subject before
    computing correlation — this gives the correct effective N and p-value
    for repeated CV where each subject appears multiple times.

    Args:
        y_true (pd.Series or np.ndarray): The ground truth values.
        y_pred (pd.Series or np.ndarray): The values predicted by the model.
        model_name (str): The name of the model (e.g., 'RandomForest').
        target_name (str): The name of the target variable (e.g., 'MoCA_Total').
        plot_dir (str): The directory where the plot image will be saved.
        subjects (pd.Series or np.ndarray, optional): Subject IDs for aggregation.
    """
    # Ensure save directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Subject-level aggregation if subjects are provided
    if subjects is not None:
        agg_df = pd.DataFrame({
            'subject': np.asarray(subjects),
            'y_true': np.asarray(y_true),
            'y_pred': np.asarray(y_pred),
        })
        subj_agg = agg_df.groupby('subject').mean()
        y_true_plot = subj_agg['y_true'].values
        y_pred_plot = subj_agg['y_pred'].values
    else:
        y_true_plot = np.asarray(y_true)
        y_pred_plot = np.asarray(y_pred)

    # Calculate Pearson correlation
    r, p_value = stats.pearsonr(y_true_plot, y_pred_plot)

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 9))

    sns.regplot(
        x=y_true_plot, 
        y=y_pred_plot, 
        ax=ax,
        scatter_kws={'alpha': 0.6, 's': 50},
        line_kws={'color': 'red', 'linestyle': '--'},
    )

    # Set plot labels and title
    ax.set_xlabel(f"True {target_name} Values", fontsize=12)
    ax.set_ylabel(f"Predicted {target_name} Values", fontsize=12)
    # ax.set_title(f"Pearson Correlation for {model_name} on {target_name}", fontsize=14, weight='bold')
    
    # Add annotation for r, p-value, and N
    n_label = len(y_true_plot)
    ax.text(0.05, 0.95, f'$r = {r:.2f}$\n$p = {p_value:.3f}$\n$N = {n_label}$',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    title = f"Pearson Correlation for {model_name} on {target_name}"
    if stage:
        title += f" ({stage})"
    ax.set_title(title, fontsize=14, weight='bold')
    
    # Filename uses underscore version
    stage_suffix = f"_{stage.replace(' ', '_').replace('-', '_')}" if stage else ""
    file_path = os.path.join(plot_dir, f"pearson_plot_{model_name}_{target_name}{stage_suffix}.png")
    plt.savefig(file_path, dpi=300)
    print(f"Saved Pearson plot to: {file_path}")
    plt.close(fig)


def plot_bland_altman(y_true, y_pred, model_name, target_name, plot_dir, subjects=None, stage=None):
    """
    Generates and saves a Bland-Altman plot.
    
    If subjects are provided, aggregates to one point per subject before
    computing agreement statistics.

    Args:
        y_true (pd.Series or np.ndarray): The ground truth values.
        y_pred (pd.Series or np.ndarray): The values predicted by the model.
        model_name (str): The name of the model (e.g., 'RandomForest').
        target_name (str): The name of the target variable (e.g., 'MoCA_Total').
        plot_dir (str): The directory where the plot image will be saved.
        subjects (pd.Series or np.ndarray, optional): Subject IDs for aggregation.
    """
    # Ensure save directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Subject-level aggregation if subjects are provided
    if subjects is not None:
        agg_df = pd.DataFrame({
            'subject': np.asarray(subjects),
            'y_true': np.asarray(y_true),
            'y_pred': np.asarray(y_pred),
        })
        subj_agg = agg_df.groupby('subject').mean()
        y_true = subj_agg['y_true'].values
        y_pred = subj_agg['y_pred'].values
    else:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

    # Calculate the necessary values
    differences = y_true - y_pred
    averages = (y_true + y_pred) / 2.0
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # 95% limits of agreement
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create the scatter plot
    ax.scatter(averages, differences, alpha=0.6, s=50, edgecolor='k', facecolor='c')

    # Plot bias and limits of agreement lines
    ax.axhline(mean_diff, color='r', linestyle='--', linewidth=2, label=f'Bias (Mean Diff): {mean_diff:.2f}')
    ax.axhline(upper_limit, color='g', linestyle='--', linewidth=2, label=f'Upper Limit (+1.96 SD): {upper_limit:.2f}')
    ax.axhline(lower_limit, color='g', linestyle='--', linewidth=2, label=f'Lower Limit (-1.96 SD): {lower_limit:.2f}')

    # Set plot labels and title
    n_label = len(y_true)
    ax.set_xlabel(f"Average of True and Predicted {target_name}", fontsize=12)
    ax.set_ylabel(f"Difference (True - Predicted)", fontsize=12)
    # ax.set_title(f"Bland-Altman Plot for {model_name} on {target_name} (N={n_label})", fontsize=14, weight='bold')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the figure
    title = f"Bland-Altman Plot for {model_name} on {target_name}"
    if stage:
        title += f" ({stage})"
    ax.set_title(title, fontsize=14, weight='bold')
    
    # Filename uses underscore version
    stage_suffix = f"_{stage.replace(' ', '_').replace('-', '_')}" if stage else ""
    file_path = os.path.join(plot_dir, f"bland_altman_{model_name}_{target_name}{stage_suffix}.png")
    plt.savefig(file_path, dpi=300)
    print(f"Saved Bland-Altman plot to: {file_path}")
    plt.close(fig)