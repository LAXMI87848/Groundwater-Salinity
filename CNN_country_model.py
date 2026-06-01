# ===========================
# CNN time-split with GroupKFold, leak-free scaling, and SHAP
# ===========================

import os
import random
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# NEW (Keras 3)
from keras import Sequential
from keras.layers import Conv1D, Dense, Flatten, Input, LeakyReLU, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import shap
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.impute import SimpleImputer


# ---------------------------
# Config and give your own path for output directory where you want to save your results
# ---------------------------
output_dir = r"D:\Python\India Project\output_figures\dataset2910\CNN\NEW\All_basins" # used in my case
os.makedirs(output_dir, exist_ok=True)
CV_plots_dir = os.path.join(output_dir, "CV_plots")
os.makedirs(CV_plots_dir, exist_ok=True)
GSA_dir = os.path.join(output_dir, "GSA8192")
os.makedirs(GSA_dir, exist_ok=True)


# Load the data shared and give the path in #excelFilePath, where it is present
sheet_name = "Sheet1"
excelFilePath = r'D:\Python\India Project\GWS_Model info\Dataset_to_share.xlsx' # used in my case

# Random seeds
random.seed(1235)
np.random.seed(1235)
tf.random.set_seed(1235)

# -----------------------------
# Fixed per-basin weights + helpers (same as LSTM version)
# -----------------------------
RAW_BASIN_WEIGHTS = {
    "Carmel":           0.516,
    "Coast":            0.302,
    "Galil West":       0.837,
    "Sea of galilee":   1.332,
    "mountin east":     1.531,
    "Yarkatan":         1.292,
    "Negev and Arava":  1.189,
}


# =============================================
# SEPARATE ACTUAL VS PREDICTED PLOT WITH STATISTICS (from snippet; filename added)
# =============================================
def create_actual_vs_predicted_plot(y_test, y_pred, metrics, output_dir, model_name="XGBoost",
                                    filename="actual_vs_predicted_detailed.png"):
    """
    Create a separate actual vs predicted plot with performance statistics in legend
    (Format matches the provided snippet exactly; only 'filename' was added.)
    """
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = plt.scatter(y_test, y_pred, alpha=0.6, s=50, c='blue',
                          edgecolors='black', linewidth=0.5)

    # Set equal aspect ratio
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add regression line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "g-", alpha=0.8, linewidth=2, label='Regression Line')

    # Calculate additional statistics
    residuals = y_test - y_pred
    mape = np.mean(np.abs(residuals / y_test)) * 100  # Mean Absolute Percentage Error
    rmse = np.sqrt(metrics['mse'])

    # Create detailed statistics text for legend
    stats_text = (
        f'Performance Statistics:\n'
        f'R² = {metrics["r2"]:.4f}\n'
        f'RMSE = {rmse:.2f} mg/l\n'
        f'MAE = {metrics["mae"]:.2f} mg/l\n'
        f'MSE = {metrics["mse"]:.2f}\n'
        # f'MAPE = {mape:.2f}%\n'
        # f'n = {len(y_test)} samples'
    )

    # Add statistics as text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontfamily='monospace')

    # Plot formatting
    plt.xlabel('Actual Salinity (mg/l)', fontsize=12)
    plt.ylabel('Predicted Salinity (mg/l)', fontsize=12)
    plt.title(f'Actual vs Predicted Salinity - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add legend for lines only
    plt.legend(loc='lower right', fontsize=10)

    # Set equal axis limits
    plt.axis('equal')
    buffer = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Detailed actual vs predicted plot saved: {filename}")

    return {'mape': mape, 'rmse': rmse}


def fixed_basin_weights_for(basins: np.ndarray,
                            normalize_to_mean_one: bool = False,
                            warn_unknown: bool = True) -> np.ndarray:
    b = basins.astype(str)
    w = np.array([RAW_BASIN_WEIGHTS.get(x, 1.0) for x in b], dtype=float)
    if warn_unknown:
        missing = sorted(set(x for x in b if x not in RAW_BASIN_WEIGHTS))
        if missing:
            print(f"[warn] No fixed weight for: {missing} — using 1.0")
    if normalize_to_mean_one and w.size:
        w /= np.nanmean(w)
    w[~np.isfinite(w)] = 1.0
    return w

def safe_rmse(y_true, y_pred, sample_weight=None):
    try:
        return mean_squared_error(y_true, y_pred, sample_weight=sample_weight, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    w = np.asarray(w).reshape(-1)
    w = np.where(np.isfinite(w), w, 0.0)
    ws = w.sum()
    if ws <= 0:
        return np.nan
    ybar = (w * y_true).sum() / ws
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - ybar) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-12)

# ---------------------------
# Sobol Global Sensitivity (SALib) for sequence CNN
# ---------------------------
def _feature_bounds_from_train(X_train_raw_3d, lower_q=1.0, upper_q=99.0):
    """
    Compute per-feature bounds from TRAIN sequences only, aggregating over (samples,time).
    Using robust percentiles avoids extreme outliers.
    Returns: (F,2) array of [low, high] for each feature.
    """
    Ntr, T, F = X_train_raw_3d.shape
    flat = X_train_raw_3d.reshape(-1, F)
    lows  = np.nanpercentile(flat, lower_q, axis=0)
    highs = np.nanpercentile(flat, upper_q, axis=0)
    # guard degenerate ranges
    mask_bad = ~np.isfinite(lows) | ~np.isfinite(highs) | (highs <= lows)
    # fallback: use nanmean ± small epsilon
    means = np.nanmean(flat, axis=0)
    eps = np.maximum(1e-8, 0.01 * np.where(np.isfinite(means), np.abs(means), 1.0))
    lows[mask_bad]  = np.where(np.isfinite(means), means - eps, -eps)[mask_bad]
    highs[mask_bad] = np.where(np.isfinite(means), means + eps,  eps)[mask_bad]
    return np.vstack([lows, highs]).T  # (F,2)


def run_sobol_sensitivity(
    model,
    scaler,                 # MinMaxScaler fitted on TRAIN
    X_train_raw_3d,         # TRAIN ONLY, shape (Ntr, T, F)
    feature_names,
    time_steps,
    output_dir,
    n_base=1000,            # Saltelli base sampler size
    lower_q=1.0, upper_q=99.0,
    second_order=False,     # True -> S2 CSV + heatmap
    top_display=None        # None shows all; int caps the number of features in plots
):
    """
    Runs Sobol GSA on the CNN and saves:
      CSVs:  sobol_first_order.csv, sobol_total_order.csv, (optional) sobol_second_order_pairs.csv
      Plots: sobol_S1_bar.png, sobol_ST_bar.png, sobol_S1_ST_combined.png, (optional) sobol_S2_heatmap.png
    """
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except Exception as e:
        raise ImportError("SALib is required for Sobol analysis. Please `pip install SALib`.") from e

    import matplotlib.pyplot as plt

    feature_names = [str(f) for f in feature_names]
    F = len(feature_names)
    assert X_train_raw_3d.ndim == 3 and X_train_raw_3d.shape[2] == F, "Dim mismatch for TRAIN 3D array."

    # 1) bounds from TRAIN only (robust percentiles)
    bounds = _feature_bounds_from_train(X_train_raw_3d, lower_q, upper_q)  # (F,2)
    problem = {"num_vars": F, "names": feature_names, "bounds": bounds.tolist()}

    # 2) Saltelli sampling in RAW space (compat across SALib versions)
    np.random.seed(42)  # SALib uses NumPy RNG internally
    try:
        X_samples = saltelli.sample(problem, N=n_base, calc_second_order=second_order)
    except TypeError:
        # older SALib may expect positionals
        X_samples = saltelli.sample(problem, n_base, second_order)

    # 3) Scale and tile into sequences (Ns, T, F)
    X_scaled_2d = scaler.transform(np.nan_to_num(X_samples, nan=0.0, posinf=0.0, neginf=0.0))
    X_seq = np.tile(X_scaled_2d[:, None, :], (1, time_steps, 1)).astype(np.float32)

    # 4) Predict from model
    y_pred = model.predict(X_seq, verbose=0).reshape(-1)

    # 5) Sobol analysis
    Si = sobol.analyze(problem, y_pred, calc_second_order=second_order, print_to_console=False)

    # 6) Save CSVs
    s1_df = pd.DataFrame({
        "feature": feature_names,
        "S1":  Si["S1"],
        "S1_conf": Si["S1_conf"]
    }).sort_values("S1", ascending=False, kind="mergesort")
    st_df = pd.DataFrame({
        "feature": feature_names,
        "ST":  Si["ST"],
        "ST_conf": Si["ST_conf"]
    }).sort_values("ST", ascending=False, kind="mergesort")

    s1_df.to_csv(os.path.join(GSA_dir, "sobol_first_order.csv"), index=False)
    st_df.to_csv(os.path.join(GSA_dir, "sobol_total_order.csv"), index=False)

    # Save bounds + sample head
    pd.DataFrame(bounds, columns=["lower", "upper"]).assign(feature=feature_names) \
      .loc[:, ["feature", "lower", "upper"]] \
      .to_csv(os.path.join(GSA_dir, "sobol_feature_bounds_from_train.csv"), index=False)
    pd.DataFrame(X_samples, columns=feature_names) \
      .head(10).to_csv(os.path.join(GSA_dir, "sobol_sample_head10.csv"), index=False)

    # === GSA results table (exactly as in snippet) ===
    gsa_results = (
        pd.DataFrame({
            'feature': feature_names,
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf']
        })
        .sort_values('ST', ascending=False, kind='mergesort')
        .reset_index(drop=True)
    )

    # Save GSA results to match snippet behavior
    gsa_results.to_csv(os.path.join(GSA_dir, "gsa_results.csv"), index=False)

    # === Plot format: EXACT match to the provided snippet ===
    # Top 15 features by ST
    features_plot = gsa_results.head(15).copy()
    y_pos = np.arange(len(features_plot))

    # ----- Total Sensitivity (ST) -----
    plt.figure(figsize=(12, 8))
    plt.barh(
        y_pos,
        features_plot['ST'],
        xerr=features_plot['ST_conf'],
        alpha=0.7,
        color='steelblue',
        ecolor='black',
        capsize=5
    )
    plt.yticks(y_pos, features_plot['feature'])
    plt.xlabel('Total Sensitivity Index (ST)')
    plt.title('Global Sensitivity Analysis - Total Sensitivity Indices (Top 15)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(GSA_dir, "gsa_total_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ----- First-Order Sensitivity (S1) -----
    plt.figure(figsize=(12, 8))
    plt.barh(
        y_pos,
        features_plot['S1'],
        xerr=features_plot['S1_conf'],
        alpha=0.7,
        color='lightcoral',
        ecolor='black',
        capsize=5
    )
    plt.yticks(y_pos, features_plot['feature'])
    plt.xlabel('First-Order Sensitivity Index (S1)')
    plt.title('Global Sensitivity Analysis - First-Order Sensitivity Indices (Top 15)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(GSA_dir, "gsa_first_order_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ----- Combined plot: EXACTLY as in your snippet -----
    # Combined plot
    plt.figure(figsize=(14, 10))
    x = np.arange(len(features_plot))  # (kept to match snippet exactly, even if unused)
    width = 0.35

    plt.barh(y_pos - width / 2, features_plot['S1'], width,
             label='First-Order (S1)', alpha=0.7, color='lightcoral')
    plt.barh(y_pos + width / 2, features_plot['ST'], width,
             label='Total (ST)', alpha=0.7, color='steelblue')

    plt.yticks(y_pos, features_plot['feature'])
    plt.xlabel('Sensitivity Index')
    plt.title('Global Sensitivity Analysis - First-Order vs Total Sensitivity Indices')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(GSA_dir, "gsa_combined_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close()


    # 9) Optional S2 heatmap
    if second_order and "S2" in Si and isinstance(Si["S2"], np.ndarray):
        rows = []
        for i in range(F):
            for j in range(i+1, F):
                rows.append({
                    "feature_i": feature_names[i],
                    "feature_j": feature_names[j],
                    "S2":  Si["S2"][i, j],
                    "S2_conf": Si["S2_conf"][i, j]
                })
        s2_df = pd.DataFrame(rows).sort_values("S2", ascending=False, kind="mergesort")
        s2_df.to_csv(os.path.join(GSA_dir, "sobol_second_order_pairs.csv"), index=False)

        cap = top_display if top_display is not None else min(30, F)
        top_idx = st_df["feature"].tolist()[:cap]  # order by ST
        idx_map = {f: i for i, f in enumerate(feature_names)}
        sel = [idx_map[f] for f in top_idx]

        mat = np.zeros((len(sel), len(sel)), dtype=float); mat[:] = np.nan
        for a, ia in enumerate(sel):
            for b, ib in enumerate(sel):
                if ib <= ia:
                    continue
                mat[a, b] = Si["S2"][ia, ib]

        plt.figure(figsize=(1.2 + 0.5*len(sel), 1.2 + 0.5*len(sel)))
        im = plt.imshow(mat, aspect="auto")
        plt.xticks(range(len(sel)), top_idx, rotation=35, ha="right")
        plt.yticks(range(len(sel)), top_idx)
        plt.title("Sobol Second-Order (S2) — Upper Triangle (Top features)")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.ax.tick_params(labelsize=8)
        plt.xticks(np.arange(-.5, len(sel), 1), minor=True)
        plt.yticks(np.arange(-.5, len(sel), 1), minor=True)
        plt.grid(which="minor", linestyle="-", linewidth=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(GSA_dir, "sobol_S2_heatmap.png"), dpi=300)
        plt.close()

    return {"S1": s1_df, "ST": st_df}

# =============================================
# SEPARATE ACTUAL VS PREDICTED PLOT WITH STATISTICS (from snippet; filename added)
# =============================================
def create_actual_vs_predicted_plot(y_test, y_pred, metrics, output_dir, model_name="XGBoost",
                                    filename="actual_vs_predicted_detailed.png"):
    """
    Create a separate actual vs predicted plot with performance statistics in legend
    (Format matches the provided snippet exactly; only 'filename' was added.)
    """
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = plt.scatter(y_test, y_pred, alpha=0.6, s=50, c='blue',
                          edgecolors='black', linewidth=0.5)

    # Set equal aspect ratio
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add regression line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "g-", alpha=0.8, linewidth=2, label='Regression Line')

    # Calculate additional statistics
    residuals = y_test - y_pred
    mape = np.mean(np.abs(residuals / y_test)) * 100  # Mean Absolute Percentage Error
    rmse = np.sqrt(metrics['mse'])

    # Create detailed statistics text for legend
    stats_text = (
        f'Performance Statistics:\n'
        f'R² = {metrics["r2"]:.4f}\n'
        f'RMSE = {rmse:.2f} mg/l\n'
        f'MAE = {metrics["mae"]:.2f} mg/l\n'
        f'MSE = {metrics["mse"]:.2f}\n'
        # f'MAPE = {mape:.2f}%\n'
        # f'n = {len(y_test)} samples'
    )

    # Add statistics as text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontfamily='monospace')

    # Plot formatting
    plt.xlabel('Actual Salinity (mg/l)', fontsize=12)
    plt.ylabel('Predicted Salinity (mg/l)', fontsize=12)
    plt.title(f'Actual vs Predicted Salinity - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add legend for lines only
    plt.legend(loc='lower right', fontsize=10)

    # Set equal axis limits
    plt.axis('equal')
    buffer = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Detailed actual vs predicted plot saved: {filename}")

    return {'mape': mape, 'rmse': rmse}


# ---------------------------
# Data loading and preprocessing (no scaling here to avoid leakage)
# ---------------------------
df = pd.read_excel(excelFilePath, sheet_name=sheet_name)
# basin_data = df[df['Hydrological Basin'] == basin_name].copy()
basin_data = df.copy()

print(f"Found {len(basin_data)} samples for simulation")

# Ensure datetime and sort by entity then time
basin_data['Year'] = pd.to_datetime(basin_data['Year'], format='%Y')
basin_data = basin_data.sort_values(['Drill', 'Year'])

# Filter and feature selection
selected = basin_data.copy()
selected = selected[selected['Salinity - Cl (mg/l)'] <= 4000]
# Keep raw basin labels (for weighting) before one-hot encoding
basin_labels_all = selected['Hydrological Basin'].astype(str).values

X_df = selected.drop(
    columns=[
    'Salinity - Cl (mg/l)', 'Year',
    'Hightemp_fall', 'Hightemp_spring', 'Hightemp_summer', 'Hightemp_winter',
    'Tempanam_fall', 'Tempanam_spring', 'Tempanam_summer', 'Tempanam_winter',
    'Zcore_fall', 'Zcore_spring', 'Zcore_summer', 'Zcore_winter',
    'Elevation (m)',
    'salinity difference', 'salinity relative difference', 'Precip diff',
    'Cell name', 'Sub Basin', 'Drill',
    'precipitation ( #N of events Q1)', 'precipitation (  #N of events Q2)',
    'precipitation ( #N of events Q3)', 'precipitation ( #N of events Q4)',
    'precipitation (  #N of events Q5)',
    # 'Shoreline distance (m)',
    'National system - Groundwater (%)', 'National system - Surface water (%)', 'National system - Desalinated water (%)',
    # 'Fishponds (No.)',
    # 'LULC_joined value',
    # 'Distance to saline bodies (m)',
    # 'Distance to agricultural fields (m)',
    # "Hydrological basin",
    'Aquifer'
],
    errors='ignore'
)

X_df = pd.get_dummies(X_df, drop_first=True)

# Replace the manual list with a safe prefix drop:
X_df = X_df.drop(columns=[c for c in X_df.columns if c.startswith("Hydrological Basin_")],
                 errors="ignore")

feature_names = X_df.columns.tolist()
print(f"Prepared features with shape: {X_df.shape}")

# Targets and group/time ids
y_all = selected['Salinity - Cl (mg/l)'].values.reshape(-1, 1)
drill_ids_all = selected['Drill'].values
years_all = selected['Year'].dt.year.values

# Save raw (post-preprocessing) features to inspect later (no scaling to avoid leakage)
X_df.to_csv(os.path.join(output_dir, "raw_features_after_preprocessing.csv"), index=False)


# ---------------------------
# Sequence creation (no scaling here)
# ---------------------------
def create_sequences_grouped(
    X: np.ndarray, y: np.ndarray, drill_ids: np.ndarray, years: np.ndarray,
    time_steps: int = 3, basins: np.ndarray = None
):
    """
    Create (samples, time_steps, features) sequences grouped by drill id, preserving time order.
    Returns:
        X_seq: (N, T, F)
        y_seq: (N, 1)
        group_seq: (N,)
        year_seq: (N,)   # prediction year
        basins_seq: (N,) # basin label aligned with y_seq
    """
    X_seq, y_seq, group_seq, year_seq, basins_seq = [], [], [], [], []
    unique_drills = np.unique(drill_ids)

    for drill_id in unique_drills:
        idx = (drill_ids == drill_id)
        X_loc, y_loc, years_loc = X[idx], y[idx], years[idx]
        basins_loc = basins[idx] if basins is not None else np.array(["NA"] * np.sum(idx))

        if len(X_loc) <= time_steps:
            continue

        for i in range(len(X_loc) - time_steps):
            X_seq.append(X_loc[i:i + time_steps])
            y_seq.append(y_loc[i + time_steps])
            group_seq.append(drill_id)
            year_seq.append(years_loc[i + time_steps])
            basins_seq.append(basins_loc[i + time_steps])

    return (
        np.array(X_seq),
        np.array(y_seq),
        np.array(group_seq),
        np.array(year_seq),
        np.array(basins_seq, dtype=str)
    )


# ---------------------------
# Scaling helpers (fit on train only; transform any 3D array)
# ---------------------------
def fit_scaler_3d(train_3d: np.ndarray) -> MinMaxScaler:
    """Fit MinMaxScaler on features flattened across samples and time steps."""
    tshape = train_3d.shape
    flat = train_3d.reshape(-1, tshape[2])  # (N*T, F)
    scaler = MinMaxScaler()
    scaler.fit(flat)
    return scaler


def transform_with_scaler_3d(arr_3d: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    tshape = arr_3d.shape
    flat = arr_3d.reshape(-1, tshape[2])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(tshape)

def reset_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    K.clear_session()

# ---------------------------
# Model builder
# ---------------------------
def build_improved_model(input_shape, filters=32, activation='relu'):
    act_layer = LeakyReLU(negative_slope=0.01) if activation == 'leaky_relu' else tf.keras.layers.ReLU()
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(filters, kernel_size=3, padding='same'),
        BatchNormalization(),
        act_layer,
        Dropout(0.2),

        Conv1D(filters * 2, kernel_size=3, padding='same'),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01) if activation == 'leaky_relu' else tf.keras.layers.ReLU(),
        Dropout(0.2),

        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    return model


# ---------------------------
# SHAP Gradient (new API, data=background) — robust for TF 2.20 + Keras 3.11 + SHAP 0.47
# ---------------------------
def calculate_shap_values(
    model, scaler, X_seq_raw, feature_names,
    n_background=50, n_samples=100, time_steps=5,
    background_override=None,   # <-- add this
    sample_idx=None             # <-- and this
):
    """
    Computes gradient-based SHAP on 3D inputs (T,F) via shap.explainers.Gradient(model, data),
    aggregates across time, and saves:
      - aggregated_shap_values.png         (bar: mean |SHAP|)
      - shap_beeswarm_directional.png      (manual beeswarm)
      - mean_shap_values_absolute.csv
      - shap_values_feature_mean_abs.csv   (T x F, mean |SHAP| per time step)
    Returns: (S, F) per-sample aggregated |SHAP|.
    """
    import shap

    # --- hygiene
    feature_names = [str(f) for f in feature_names]

    # --- scale & check (NumPy only)
    X_seq = transform_with_scaler_3d(X_seq_raw, scaler).astype(np.float32)
    # sanitize inputs to avoid SHAP exploding on inf/nan
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)

    n, T, F = X_seq.shape
    if n == 0:
        raise ValueError("No sequences available for SHAP.")
    if T != time_steps:
        raise ValueError(f"time_steps mismatch: got {T}, expected {time_steps}")

    # --- ensure model graph exists (Keras 3)
    _ = model.predict(X_seq[:1], verbose=0)

    # --- background & test subsamples (NumPy)
    rng = np.random.default_rng(42)

    # ----- Background (TRAIN ONLY)
    if background_override is not None:
        background_np = background_override.astype(np.float32)
    else:
        # (fallback) random from the provided pool — not recommended if pool mixes train/test
        bg_idx = rng.choice(n, min(n_background, n), replace=False)
        background_np = X_seq[bg_idx]

    # ----- Evaluation set (TEST)
    if sample_idx is not None:
        test_samples_np = X_seq[sample_idx]
    else:
        # allow "all" to mean use the entire provided pool
        use_all = (n_samples is None) or (isinstance(n_samples, str) and n_samples.lower() == "all") \
                  or (isinstance(n_samples, int) and n_samples <= 0)
        if use_all:
            test_samples_np = X_seq
        else:
            ts_idx = rng.choice(n, min(n_samples, n), replace=False)
            test_samples_np = X_seq[ts_idx]

    # --- NEW API: pass DATA, not masker
    try:
        explainer = shap.explainers.Gradient(model, background_np)   # <- key change
        expl = explainer(test_samples_np)                            # shap.Explanation
        shap_values = np.asarray(expl.values, dtype=np.float32)      # (S,T,F) expected
    except Exception as e_new:
        # Last resort: legacy GradientExplainer
        try:
            expl_old = shap.GradientExplainer(model, background_np)
            sv = expl_old.shap_values(test_samples_np)
            shap_values = np.asarray(sv[0] if isinstance(sv, list) else sv, dtype=np.float32)
            print(f"Used legacy GradientExplainer due to: {e_new}")
        except Exception as e_old:
            raise RuntimeError(f"Failed SHAP initialization. New-API error: {e_new} ; Legacy error: {e_old}")

    # Some SHAP builds return an extra singleton output dim; squeeze if needed
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]  # (S,T,F)

    # --- aggregate across time -> (S,F)
    shap_abs_agg = np.mean(np.abs(shap_values), axis=1)  # (S,F)
    shap_raw_agg = np.mean(shap_values, axis=1)          # (S,F)
    feat_agg     = np.mean(test_samples_np, axis=1)      # (S,F)

    # ========= 1) BAR: mean |SHAP| (manual, robust) =========
    mean_abs_vals = np.mean(shap_abs_agg, axis=0)        # (F,)
    order = np.argsort(-mean_abs_vals)
    plt.figure(figsize=(10, max(6, 0.35*len(feature_names))))
    plt.barh(np.array(feature_names, dtype=object)[order][::-1], mean_abs_vals[order][::-1])
    plt.title("Aggregated SHAP Values (Mean Absolute Impact) — Gradient")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_shap_values.png"))
    plt.close()

    # CSV of mean |SHAP|
    # Save mean abs shap to CSV (with ranking)
    mean_abs_vals = np.mean(shap_abs_agg, axis=0)  # (F,)
    mean_abs_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_vals})
        .sort_values("mean_abs_shap", ascending=False, kind="mergesort")
    )
    mean_abs_df.insert(0, "rank", np.arange(1, len(mean_abs_df) + 1))
    mean_abs_df.to_csv(os.path.join(output_dir, "mean_shap_values_absolute.csv"), index=False)

    # 2) Beeswarm with directionality (defensive: new API → legacy fallback)
    values = np.asarray(shap_raw_agg, dtype=np.float32)  # (S, F)
    feats = np.asarray(feat_agg, dtype=np.float32)  # (S, F)
    # final guard
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    fnames = [str(f) for f in feature_names]

    # sanity checks
    assert values.ndim == 2 and feats.ndim == 2 and values.shape == feats.shape
    if not np.isfinite(values).all() or not np.isfinite(feats).all():
        raise ValueError("NaN/inf in SHAP values or features for beeswarm")

    plt.figure(figsize=(12, 8))
    try:
        # Preferred path (SHAP 0.47+): avoids the legacy pandas code
        expl = shap.Explanation(values=values, data=feats, feature_names=fnames)
        shap.plots.beeswarm(expl, show=False, max_display=len(fnames))
    except Exception:
        # Fallback to legacy summary_plot (wrap features in a DataFrame to keep it 2-D)
        feats_df = pd.DataFrame(feats, columns=fnames)
        shap.summary_plot(
            values, features=feats_df, feature_names=fnames,
            plot_type="dot", show=False, max_display=len(fnames)
        )

    plt.title("Directional SHAP Impact on Salinity Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_beeswarm_directional.png"), dpi=300)
    plt.close()

    # ========= 3) Per-time-step mean |SHAP| (T x F) =========
    shap_time_feature = np.mean(np.abs(shap_values), axis=0)  # (T,F)
    pd.DataFrame(shap_time_feature, columns=feature_names) \
      .to_csv(os.path.join(output_dir, "shap_values_feature_mean_abs.csv"), index=False)

    # ========= MEDIAN aggregation over time (NEW) =========
    # Per-sample, per-feature, aggregated over time steps
    shap_abs_median_agg = np.median(np.abs(shap_values), axis=1)  # (S, F)
    shap_raw_median_agg = np.median(shap_values, axis=1)          # (S, F)

    # Save per-sample median aggregates (handy for audits/interaction work)
    pd.DataFrame(shap_abs_median_agg, columns=feature_names) \
      .to_csv(os.path.join(output_dir, "shap_values_aggregated_abs_median_per_sample.csv"), index=False)
    pd.DataFrame(shap_raw_median_agg, columns=feature_names) \
      .to_csv(os.path.join(output_dir, "shap_values_aggregated_signed_median_per_sample.csv"), index=False)

    # Time × Feature MEDIAN(|SHAP|) (analogous to your mean T×F matrix)
    shap_time_feature_median = np.median(np.abs(shap_values), axis=0)  # (T, F)
    pd.DataFrame(shap_time_feature_median, columns=feature_names) \
      .to_csv(os.path.join(output_dir, "shap_values_feature_median_abs.csv"), index=False)

    # ========= Global comparison: MEAN vs MEDIAN =========
    # --- make global importance vectors finite ---
    def _finite(a):
        a = np.asarray(a, dtype=float)
        a[~np.isfinite(a)] = np.nan
        return a

    global_mean_abs   = _finite(np.mean(shap_abs_agg, axis=0))        # (F,)
    global_median_abs = _finite(np.mean(shap_abs_median_agg, axis=0)) # (F,)

    cmp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap":   global_mean_abs,
        "median_abs_shap": global_median_abs,
    })

    # --- ranks: place NaNs at the bottom, then cast safely to nullable Int64 ---
    n = len(cmp_df)
    rank_mean   = cmp_df["mean_abs_shap"].rank(ascending=False, method="min")   # float ranks (NaN where value is NaN)
    rank_median = cmp_df["median_abs_shap"].rank(ascending=False, method="min")

    # send NaNs to bottom = (n + 1), then convert to Int64 (nullable-safe)
    rank_mean   = rank_mean.fillna(n + 1).round().astype("Int64")
    rank_median = rank_median.fillna(n + 1).round().astype("Int64")

    cmp_df["rank_mean"]   = rank_mean
    cmp_df["rank_median"] = rank_median
    cmp_df["rank_delta"]  = (rank_median.astype("float") - rank_mean.astype("float")).round().astype("Int64")

    # % change (guard zeros & NaNs)
    eps = 1e-12
    denom = cmp_df["mean_abs_shap"].where(np.isfinite(cmp_df["mean_abs_shap"]), np.nan)
    cmp_df["pct_change_median_vs_mean"] = 100.0 * (
        (cmp_df["median_abs_shap"] - cmp_df["mean_abs_shap"]) / (denom + eps)
    )

    # sort & save
    cmp_df = cmp_df.sort_values("median_abs_shap", ascending=False, kind="mergesort").reset_index(drop=True)
    cmp_df.to_csv(os.path.join(output_dir, "mean_vs_median_shap_ranking.csv"), index=False)

    # correlation after ranks exist
    rank_corr = cmp_df["rank_mean"].astype("float").corr(cmp_df["rank_median"].astype("float"), method="spearman")
    print(f"Spearman rank corr (mean vs median): {rank_corr:.3f}")


    # ========= Plots for comparison =========
    # (1) Scatter: Median vs Mean (global |SHAP|)
    plt.figure(figsize=(7.5, 7))
    plt.scatter(cmp_df["mean_abs_shap"], cmp_df["median_abs_shap"], s=35, alpha=0.8, edgecolors='black', linewidth=0.4)
    lo = float(min(cmp_df["mean_abs_shap"].min(), cmp_df["median_abs_shap"].min()))
    hi = float(max(cmp_df["mean_abs_shap"].max(), cmp_df["median_abs_shap"].max()))
    pad = 0.03 * (hi - lo + 1e-12)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'r--', linewidth=1.5, label='y = x')
    plt.xlabel("Global importance (mean |SHAP|)")
    plt.ylabel("Global importance (median |SHAP|)")
    plt.title("Median vs Mean Aggregation over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_importance_scatter_mean_vs_median.png"), dpi=300)
    plt.close()

    # (2) Grouped bars: Top-20 by median, show mean vs median side-by-side
    top_k = min(20, len(cmp_df))
    top = cmp_df.head(top_k).copy()
    y = np.arange(top_k)
    width = 0.4

    plt.figure(figsize=(12, max(6, 0.4 * top_k)))
    plt.barh(y - width/2, top["median_abs_shap"], height=width, label="Median over time")
    plt.barh(y + width/2, top["mean_abs_shap"],   height=width, label="Mean over time")
    plt.yticks(y, top["feature"])
    plt.gca().invert_yaxis()
    plt.xlabel("Global importance (|SHAP|)")
    plt.title("Top Features by Median Aggregation (with Mean for comparison)")
    plt.grid(True, axis='x', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top20_mean_vs_median_bars.png"), dpi=300, bbox_inches="tight")
    plt.close()




    return shap_abs_agg, shap_raw_agg, feat_agg


# ---------------------------
# Training & evaluation with time-based split
# ---------------------------
def train_and_evaluate_time_split(
    X_seq_raw, y_seq, group_ids, years, feature_names, test_years=6, time_steps=5,
    basin_seq=None
):

    """
    Time split by holding out the most recent `test_years` (or fewer if not available).
    Uses GroupKFold on the training period to avoid drill leakage.
    Performs leak-free scaling inside each fold and for the final train/test fit.
    """
    unique_years = np.sort(np.unique(years))
    if len(unique_years) < 2:
        raise ValueError("Not enough distinct years to perform a time-based split.")

    # Select last k years for test (with guard)
    k = min(test_years, len(unique_years))
    test_years_set = set(unique_years[-k:])
    train_mask = ~np.isin(years, list(test_years_set))
    test_mask = np.isin(years, list(test_years_set))

    assert basin_seq is not None, "basin_seq must be provided for weighting."
    basin_train = basin_seq[train_mask]
    basin_test = basin_seq[test_mask]

    # ---- Leak-free imputation on X_seq_raw (shape: N,T,F)
    X_train_seq = X_seq_raw[train_mask]  # (Ntr,T,F)

    # compute per-feature means over (samples,time) on TRAIN ONLY
    feat_means = np.nanmean(X_train_seq.reshape(-1, X_train_seq.shape[2]), axis=0)  # (F,)
    # replace NaN means with 0 as a last resort
    feat_means = np.where(np.isfinite(feat_means), feat_means, 0.0)

    def impute_3d(arr, means):
        arr = np.array(arr, dtype=np.float32, copy=True)
        # broadcast means over (N,T,F)
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = np.take(means, np.where(mask)[2])
        return arr

    X_seq_raw = impute_3d(X_seq_raw, feat_means)

    X_train_raw, X_test_raw = X_seq_raw[train_mask], X_seq_raw[test_mask]
    y_train, y_test = y_seq[train_mask], y_seq[test_mask]
    train_groups, test_groups = group_ids[train_mask], group_ids[test_mask]

    print("\nTime-based split:")
    print(f"Training samples: {len(X_train_raw)} (years not in {sorted(test_years_set)})")
    print(f"Test samples:     {len(X_test_raw)} (years in {sorted(test_years_set)})")
    print(f"Unique train drills: {len(np.unique(train_groups))}")
    print(f"Unique test drills:  {len(np.unique(test_groups))}")

    time_steps_detected = X_seq_raw.shape[1]
    input_dim = X_seq_raw.shape[2]
    assert time_steps_detected == time_steps, f"time_steps mismatch: got {time_steps_detected}, expected {time_steps}"

    optimizer_configs = {
        'adam': {'class': tf.keras.optimizers.Adam, 'args': {'learning_rate': 0.001}}
    }
    activations = ['relu']
    filter_sizes = [64]

    best_results = {}
    all_plots_data = []

    # Callbacks
    val_callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_mae", mode="min"),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_mae", mode="min")
    ]

    final_callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="loss"),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor="loss")
    ]

    fold_maes, fold_r2s, fold_rmses = [], [], []
    fold_maes_w, fold_r2s_w, fold_rmses_w = [], [], []
    fold_plots = []

    for filters in filter_sizes:
        for optimizer_name, optimizer_config in optimizer_configs.items():
            for activation in activations:
                print(f"\nTraining CNN with {filters} filters, {activation} activation, {optimizer_name} optimizer...")

                # GroupKFold on training only
                n_groups = len(np.unique(train_groups))
                if n_groups < 2:
                    raise ValueError(f"Need at least 2 groups for GroupKFold, got {n_groups}.")
                n_splits = min(5, n_groups)
                gkf = GroupKFold(n_splits=n_splits)

                fold_maes, fold_r2s = [], []
                fold_plots = []

                for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train_raw, y_train, groups=train_groups), 1):
                    X_tr_raw, X_va_raw = X_train_raw[tr_idx], X_train_raw[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                    # Fit scaler on train fold only
                    scaler_fold = fit_scaler_3d(X_tr_raw)
                    X_tr = transform_with_scaler_3d(X_tr_raw, scaler_fold)
                    X_va = transform_with_scaler_3d(X_va_raw, scaler_fold)

                    # Basin weights aligned to train/val folds
                    basins_tr_fold = basin_train[tr_idx]
                    basins_va_fold = basin_train[va_idx]
                    sw_tr = fixed_basin_weights_for(basins_tr_fold)
                    sw_va = fixed_basin_weights_for(basins_va_fold)

                    # No group overlap check
                    train_drills = np.unique(train_groups[tr_idx])
                    val_drills = np.unique(train_groups[va_idx])
                    assert len(set(train_drills) & set(val_drills)) == 0, \
                        "GroupKFold failed: overlapping drills between train and validation sets."

                    reset_everything(42)

                    # Build and compile model
                    model = build_improved_model((time_steps, input_dim), filters, activation)
                    optimizer = optimizer_config['class'](**optimizer_config['args'])
                    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                    # Ensure model is built
                    _ = model.predict(X_tr[:1], verbose=0)

                    # Fit
                    history = model.fit(
                        X_tr, y_tr,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_va, y_va, sw_va),  # weighted validation
                        callbacks=val_callbacks,
                        verbose=0,
                        sample_weight=sw_tr  # weighted training
                    )

                    # Validate
                    # Unweighted
                    y_hat = model.predict(X_va, verbose=0).ravel()
                    y_true = y_va.ravel()
                    # Unweighted
                    val_mae = mean_absolute_error(y_true, y_hat)
                    val_rmse = sqrt(mean_squared_error(y_true, y_hat))
                    val_r2 = r2_score(y_true, y_hat)
                    fold_maes.append(val_mae)
                    fold_rmses.append(val_rmse)
                    fold_r2s.append(val_r2)

                    # Weighted
                    val_rmse_w = safe_rmse(y_true, y_hat, sample_weight=sw_va)
                    val_mae_w = mean_absolute_error(y_true, y_hat, sample_weight=sw_va)
                    val_r2_w = weighted_r2(y_true, y_hat, sw_va)
                    fold_maes_w.append(val_mae_w)
                    fold_rmses_w.append(val_rmse_w)
                    fold_r2s_w.append(val_r2_w)


                    # Store data for plotting
                    fold_data = {
                        'filters': filters,
                        'optimizer': optimizer_name,
                        'activation': activation,
                        'fold': fold,
                        'y_val': y_true,
                        'y_pred': y_hat,
                        'history': history.history,
                        'model': model,
                        'rmse': val_rmse,
                        'mae': val_mae,
                        'r2': val_r2,
                        'rmse_w': val_rmse_w,
                        'mae_w': val_mae_w,
                        'r2_w': val_r2_w,
                        'train_drills': len(train_drills),
                        'val_drills': len(val_drills)
                    }

                    fold_plots.append(fold_data)


                    # Plot training history for this fold
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.title(f'Loss (Fold {fold})')
                    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['mae'], label='Train MAE')
                    plt.plot(history.history['val_mae'], label='Validation MAE')
                    plt.title(f'MAE (Fold {fold})')
                    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()

                    plt.suptitle(f'Training History: {filters} filters, {optimizer_name}, {activation} (Fold {fold})')
                    plt.tight_layout()
                    plot_filename = os.path.join(CV_plots_dir, f"history_f{filters}_{optimizer_name}_{activation}_fold{fold}.png")
                    plt.savefig(plot_filename)
                    plt.close()

                avg_mae = float(np.mean(fold_maes)) if fold_maes else np.nan
                avg_rmse = float(np.mean(fold_rmses)) if fold_rmses else np.nan
                avg_r2 = float(np.mean(fold_r2s)) if fold_r2s else np.nan

                avg_mae_w = float(np.mean(fold_maes_w)) if fold_maes_w else np.nan
                avg_rmse_w = float(np.mean(fold_rmses_w)) if fold_rmses_w else np.nan
                avg_r2_w = float(np.mean(fold_r2s_w)) if fold_r2s_w else np.nan

                print(f"Avg Validation (unweighted): MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")
                print(f"Avg Validation (weighted):   MAE={avg_mae_w:.4f}, RMSE={avg_rmse_w:.4f}, R²={avg_r2_w:.4f}")

                # ---- Final fit on all training data and evaluate on test set
                # Fit scaler on all training data (raw)
                final_scaler = fit_scaler_3d(X_train_raw)
                X_train = transform_with_scaler_3d(X_train_raw, final_scaler)
                X_test  = transform_with_scaler_3d(X_test_raw,  final_scaler)

                final_model = build_improved_model((time_steps, input_dim), filters, activation)
                optimizer = optimizer_config['class'](**optimizer_config['args'])
                final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                sw_train_final = fixed_basin_weights_for(basin_train)

                final_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    callbacks=final_callbacks,
                    verbose=0,
                    sample_weight=sw_train_final
                )

                y_test_pred = final_model.predict(X_test, verbose=0).ravel()
                y_test_true = y_test.ravel()

                sw_test = fixed_basin_weights_for(basin_test)

                # Unweighted
                test_mae = mean_absolute_error(y_test_true, y_test_pred)
                test_rmse = sqrt(mean_squared_error(y_test_true, y_test_pred))
                test_r2 = r2_score(y_test_true, y_test_pred)

                # Weighted (headline if you prefer)
                test_rmse_w = safe_rmse(y_test_true, y_test_pred, sample_weight=sw_test)
                test_mae_w = mean_absolute_error(y_test_true, y_test_pred, sample_weight=sw_test)
                test_r2_w = weighted_r2(y_test_true, y_test_pred, sw_test)

                print(f"Test (weighted)   MAE={test_mae_w:.4f}, RMSE={test_rmse_w:.4f}, R²={test_r2_w:.4f}")
                print(f"Test (unweighted) MAE={test_mae:.4f},  RMSE={test_rmse:.4f},  R²={test_r2:.4f}")

                best_results[(filters, optimizer_name, activation)] = {
                    # validation (averaged across folds)
                    'val_mae': avg_mae,
                    'val_rmse': avg_rmse,
                    'val_r2': avg_r2,
                    'val_mae_w': avg_mae_w,
                    'val_rmse_w': avg_rmse_w,
                    'val_r2_w': avg_r2_w,
                    # test
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'test_mae_w': test_mae_w,
                    'test_rmse_w': test_rmse_w,
                    'test_r2_w': test_r2_w,
                    # artifacts
                    'model': final_model,
                    'scaler': final_scaler,
                    'test_pred': (y_test_true, y_test_pred)
                }

                all_plots_data.extend(fold_plots)

    return best_results, all_plots_data

def compute_and_plot_shap_interactions_top5(
    shap_values_per_sample: np.ndarray,  # (S, F) signed SHAP aggregated over time
    feature_values_per_sample: np.ndarray,  # (S, F) aggregated feature values
    feature_names: list,
    output_dir: str,
    top_k: int = 5,
    partners_per_feature: int = 10
):
    """
    Ranks interaction partners for each of the top_k features using SHAP's
    approximate_interactions heuristic, saves a CSV, dependence-style plots,
    and a compact Top5×Top5 heatmap.
    """
    import shap

    fnames = [str(f) for f in feature_names]
    S, F = shap_values_per_sample.shape
    assert feature_values_per_sample.shape == (S, F), "Features and SHAP arrays must align."

    # --- pick top-k by mean |SHAP| ---
    mean_abs = np.mean(np.abs(shap_values_per_sample), axis=0)
    top_idx = np.argsort(-mean_abs)[:min(top_k, F)]
    top_feats = [fnames[i] for i in top_idx]

    rows = []
    mat = np.zeros((len(top_idx), len(top_idx)), dtype=float)

    # SHAP API compatibility (utils vs. top-level)
    _approx = getattr(shap, "approximate_interactions",
                      getattr(shap.utils, "approximate_interactions"))

    # Build an Explanation for modern scatter plotting
    expl_for_scatter = shap.Explanation(
        values=shap_values_per_sample, data=feature_values_per_sample, feature_names=fnames
    )

    for r, f_ind in enumerate(top_idx):
        # ranked partner indices for feature f_ind
        ranked = _approx(f_ind, shap_values_per_sample, feature_values_per_sample)
        ranked = [j for j in ranked if j != f_ind][:partners_per_feature]

        # log partner table + compute a simple comparable score
        for j in ranked:
            score = float(np.mean(np.abs(
                shap_values_per_sample[:, f_ind] * feature_values_per_sample[:, j]
            )))
            rows.append({"feature": fnames[f_ind],
                         "partner": fnames[j],
                         "approx_interaction_score": score})

        # fill Top5×Top5 matrix
        for c, g_ind in enumerate(top_idx):
            mat[r, c] = np.nan if g_ind == f_ind else float(np.mean(np.abs(
                shap_values_per_sample[:, f_ind] * feature_values_per_sample[:, g_ind]
            )))

        # dependence-style scatter colored by strongest partner
        if ranked:
            strongest = ranked[0]
            try:
                plt.figure(figsize=(8, 6))
                shap.plots.scatter(expl_for_scatter[:, f_ind],
                                   color=expl_for_scatter[:, strongest],
                                   show=False)
                plt.title(f"Interaction: {fnames[f_ind]} × {fnames[strongest]}")
                plt.tight_layout()
                safe_name = fnames[f_ind].replace(' ', '_')
                partner_name = fnames[strongest].replace(' ', '_')
                plt.savefig(os.path.join(output_dir,
                            f"dependence_{safe_name}__color_by_{partner_name}.png"),
                            dpi=300)
                plt.close()
            except Exception:
                # legacy fallback
                feats_df = pd.DataFrame(feature_values_per_sample, columns=fnames)
                shap.dependence_plot(f_ind, shap_values_per_sample, feats_df,
                                     feature_names=fnames, interaction_index=strongest,
                                     show=False)
                plt.title(f"Interaction: {fnames[f_ind]} × {fnames[strongest]}")
                plt.tight_layout()
                safe_name = fnames[f_ind].replace(' ', '_')
                partner_name = fnames[strongest].replace(' ', '_')
                plt.savefig(os.path.join(output_dir,
                            f"dependence_{safe_name}__color_by_{partner_name}.png"),
                            dpi=300)
                plt.close()

    # save interaction rankings
    inter_df = pd.DataFrame(rows).sort_values(
        ["feature", "approx_interaction_score"], ascending=[True, False]
    )
    inter_df.to_csv(os.path.join(output_dir, "shap_approx_interactions_top5.csv"), index=False)

    # compact Top5×Top5 heatmap with small fonts that fit
    tick_fs, title_fs, cb_fs = 8, 12, 8
    fig, ax = plt.subplots(
        figsize=(1.1 + 0.5*len(top_feats), 1.1 + 0.5*len(top_feats)),
        constrained_layout=True
    )
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(top_feats))); ax.set_yticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(top_feats, fontsize=tick_fs)
    ax.set_title("Approx. Interaction Strength (Top 5 × Top 5)", fontsize=title_fs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=cb_fs)
    # faint grid for readability
    ax.set_xticks(np.arange(-.5, len(top_feats), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(top_feats), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.savefig(os.path.join(output_dir, "shap_interactions_heatmap_top5.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    try:
        # Prepare arrays
        X_np = X_df.values.astype(float)
        y_np = y_all.astype(float)
        drill_ids = drill_ids_all
        years = years_all

        # Create sequences (unscaled)
        TIME_STEPS = 5
        X_seq_raw, y_seq, group_ids, year_seq, basin_seq = create_sequences_grouped(
            X_np, y_np, drill_ids, years, time_steps=TIME_STEPS, basins=basin_labels_all
        )

        print(f"Created {len(X_seq_raw)} sequences with shape {X_seq_raw.shape}")

        # Reconstruct the same time split you used (test_years=6)
        unique_years = np.sort(np.unique(year_seq))
        k = min(6, len(unique_years))
        test_years_set = set(unique_years[-k:])
        test_mask = np.isin(year_seq, list(test_years_set))
        train_mask = ~test_mask

        # Train & evaluate
        best_results_covariate, all_plots_data = train_and_evaluate_time_split(
            X_seq_raw, y_seq, group_ids, year_seq, feature_names, test_years=6, time_steps=TIME_STEPS,
            basin_seq=basin_seq
        )


        # Pick best by validation R2 (guard NaNs)
        def safe_metric(k):
            m = best_results_covariate[k]
            v = m.get('val_r2', m.get('r2', np.nan))  # prefer new key, fallback to old
            return -1e9 if (v is None or np.isnan(v)) else v


        best_model_key = max(best_results_covariate, key=safe_metric)
        best_filters, best_optimizer, best_activation = best_model_key
        best_info = best_results_covariate[best_model_key]

        print(f"\nBest Model: {best_filters} filters, {best_optimizer} optimizer, {best_activation} activation")
        print(f"Best Validation MAE (uw): {best_info.get('val_mae', np.nan):.4f}")
        print(f"Best Validation RMSE (uw): {best_info.get('val_rmse', np.nan):.4f}")
        print(f"Best Validation R²  (uw): {best_info.get('val_r2', np.nan):.4f}")
        print(f"Best Validation MAE (w):  {best_info.get('val_mae_w', np.nan):.4f}")
        print(f"Best Validation RMSE (w): {best_info.get('val_rmse_w', np.nan):.4f}")
        print(f"Best Validation R²  (w):  {best_info.get('val_r2_w', np.nan):.4f}")

        print(f"Test MAE: {best_info['test_mae']:.4f}")
        print(f"Test R²: {best_info['test_r2']:.4f}")

        # ===== Replace the "Actual vs Predicted (all folds)" block with this =====
        best_model_data = [d for d in all_plots_data
                           if d['filters'] == best_filters
                           and d['optimizer'] == best_optimizer
                           and d['activation'] == best_activation]

        # Concatenate y_true / y_pred from all folds of the best configuration
        y_val_all = []
        y_hat_all = []
        for fd in best_model_data:
            y_val_all.append(np.ravel(fd['y_val']))
            y_hat_all.append(np.ravel(fd['y_pred']))

        if len(y_val_all) > 0:
            y_val_all = np.concatenate(y_val_all).astype(float)
            y_hat_all = np.concatenate(y_hat_all).astype(float)

            # Compute metrics (unweighted, same as your snippet expects)
            val_r2 = r2_score(y_val_all, y_hat_all)
            val_mae = mean_absolute_error(y_val_all, y_hat_all)
            val_mse = mean_squared_error(y_val_all, y_hat_all)

            metrics_val = {'r2': val_r2, 'mae': val_mae, 'mse': val_mse}

            create_actual_vs_predicted_plot(
                y_test=y_val_all,
                y_pred=y_hat_all,
                metrics=metrics_val,
                output_dir=output_dir,
                model_name=f'CNN ({best_filters}, {best_optimizer}, {best_activation}) - Validation (CV)',
                filename='actual_vs_predicted_validation.png'
            )

        # ===== Replace the "Test set: Actual vs Predicted" block with this =====
        y_test_true, y_test_pred = best_info['test_pred']
        y_test_true = np.ravel(y_test_true).astype(float)
        y_test_pred = np.ravel(y_test_pred).astype(float)

        test_r2 = r2_score(y_test_true, y_test_pred)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)
        test_mse = mean_squared_error(y_test_true, y_test_pred)

        metrics_test = {'r2': test_r2, 'mae': test_mae, 'mse': test_mse}

        create_actual_vs_predicted_plot(
            y_test=y_test_true,
            y_pred=y_test_pred,
            metrics=metrics_test,
            output_dir=output_dir,
            model_name=f'CNN',
            filename='actual_vs_predicted_test.png'
        )

        # Save performance summary (weighted + unweighted, val + test)
        perf_rows = []
        for (filters, optimizer, activation), m in best_results_covariate.items():
            perf_rows.append({
                'Filters': filters,
                'Optimizer': optimizer,
                'Activation': activation,
                # Validation (CV averages)
                'Val MAE': m.get('val_mae'),
                'Val RMSE': m.get('val_rmse'),
                'Val R2': m.get('val_r2'),
                'Val MAE (w)': m.get('val_mae_w'),
                'Val RMSE (w)': m.get('val_rmse_w'),
                'Val R2 (w)': m.get('val_r2_w'),
                # Test (held-out years)
                'Test MAE': m.get('test_mae'),
                'Test RMSE': m.get('test_rmse'),
                'Test R2': m.get('test_r2'),
                'Test MAE (w)': m.get('test_mae_w'),
                'Test RMSE (w)': m.get('test_rmse_w'),
                'Test R2 (w)': m.get('test_r2_w'),
            })

        perf_df = pd.DataFrame(perf_rows).sort_values('Val R2', ascending=False)
        print("\nModel Performance Summary (weighted + unweighted):")
        print(perf_df)
        perf_df.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)

        # -------- SHAP on best model (using its scaler)
        print("\nPerforming SHAP analysis on the best model...")
        best_model = best_info['model']
        best_scaler = best_info['scaler']
        # Train-only background (pre-scaled)
        X_bg_train_scaled = transform_with_scaler_3d(X_seq_raw[train_mask], best_scaler).astype(np.float32)
        test_idx_all = np.where(test_mask)[0]

        shap_abs, shap_signed, feat_agg = calculate_shap_values(
            best_model, best_scaler, X_seq_raw, feature_names,
            n_background=None,  # ignored since we pass background_override
            n_samples="all",  # evaluate all selected rows
            time_steps=TIME_STEPS,
            background_override=X_bg_train_scaled,  # TRAIN baseline (no leakage)
            sample_idx=test_idx_all  # ALL TEST sequences
        )

        # Save per-sample aggregated |SHAP| (n_samples × F)
        pd.DataFrame(shap_abs, columns=feature_names) \
            .to_csv(os.path.join(output_dir, "shap_values_aggregated_abs_per_sample.csv"), index=False)

        # Also save signed SHAP and aggregated feature values (aligned rows)
        pd.DataFrame(shap_signed, columns=feature_names) \
            .to_csv(os.path.join(output_dir, "shap_values_aggregated_signed_per_sample.csv"), index=False)
        pd.DataFrame(feat_agg, columns=feature_names) \
            .to_csv(os.path.join(output_dir, "aggregated_feature_values_per_sample.csv"), index=False)

        # === NEW: interaction analysis for the top-5 features ===
        compute_and_plot_shap_interactions_top5(
            shap_values_per_sample=shap_signed,  # signed SHAP (S,F)
            feature_values_per_sample=feat_agg,  # aggregated features (S,F)
            feature_names=feature_names,
            output_dir=output_dir,
            top_k=5,
            partners_per_feature=10
        )

        # Print top 10 by mean |SHAP|
        mean_abs = np.mean(np.abs(shap_signed), axis=0)
        top10 = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(10)
        print("\nTop 10 influential features (mean absolute SHAP):")
        for feat, val in top10.items():
            print(f"{feat}: {val:.4f}")

        # Save per-sample aggregated abs SHAP (n_samples x F)
        shap_abs_df = pd.DataFrame(shap_abs, columns=feature_names)
        shap_abs_df.to_csv(os.path.join(output_dir, "shap_values_aggregated_abs_per_sample.csv"), index=False)

        # Print top 10 features by mean abs shap
        mean_abs = shap_abs_df.abs().mean(axis=0).sort_values(ascending=False)
        print("\nTop 10 influential features (mean absolute SHAP):")
        for feat, val in mean_abs.head(10).items():
            print(f"{feat}: {val:.4f}")

        # -------- SOBOL Global Sensitivity (on best model) --------
        print("\nRunning Sobol global sensitivity analysis (SALib)...")
        # TRAIN-only 3D array for bounds (leak-free)
        X_train_only_3d = X_seq_raw[train_mask]  # (Ntr, T, F)

        try:
            sobol_out = run_sobol_sensitivity(
                model=best_model,
                scaler=best_scaler,
                X_train_raw_3d=X_train_only_3d,
                feature_names=feature_names,
                time_steps=TIME_STEPS,
                output_dir=output_dir,
                n_base=8192,           # adjust for accuracy vs. runtime
                lower_q=1.0, upper_q=99.0,
                second_order=False      # set True if you also want S2 (costly)
            )

            print("\nSobol GSA saved:")
            print(f" - {os.path.join(GSA_dir, 'sobol_first_order.csv')}")
            print(f" - {os.path.join(GSA_dir, 'sobol_total_order.csv')}")
            print(f" - {os.path.join(GSA_dir, 'sobol_feature_bounds_from_train.csv')}")

        except ImportError as ie:
            print(f"[warn] Sobol GSA skipped: {ie}")
        except Exception as e:
            print(f"[warn] Sobol GSA failed: {e}")


    except Exception as e:
        print(f"Error occurred: {str(e)}")
