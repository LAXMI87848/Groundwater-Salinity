
import os
import math
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Dropout
from tensorflow.keras import regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

import shap

try:
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample
    SALIB_AVAILABLE = True
except Exception:
    SALIB_AVAILABLE = False


# =========================
# Setup output directories
# =========================
output_dir = r"E:\Python\India Project\output_figures\dataset2910\ANN\All_basins"
os.makedirs(output_dir, exist_ok=True)
shap_dir = os.path.join(output_dir, "shap_plots");          os.makedirs(shap_dir, exist_ok=True)
cv_dir   = os.path.join(output_dir, "cross_validation");    os.makedirs(cv_dir, exist_ok=True)
gsa_dir   = os.path.join(output_dir, "GSA");    os.makedirs(gsa_dir, exist_ok=True)


# ==============================
# Data loading and preprocessing
# ==============================
sheet_name   = "Sheet1"
basin_name   = "Galil West"  # Coast, Yarkatan, Negev and Arava, Sea of galilee, mountin east, Carmel, Galil West
excelFilePath = r"E:\Python\India Project\GWS_Model info\dataset_29.10.xlsx"

df = pd.read_excel(excelFilePath, sheet_name)
# basin_data = df[df["Hydrological Basin"] == basin_name].copy()
basin_data = df.copy()

# ========= Fixed basin weights (same as LSTM/CNN) =========
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
# SEPARATE ACTUAL VS PREDICTED PLOT WITH STATISTICS (snippet-exact)
# =============================================
def create_actual_vs_predicted_plot(y_test, y_pred, metrics, output_dir, model_name="XGBoost",
                                    filename="actual_vs_predicted_detailed.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

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

    # Calculate additional statistics (matches snippet)
    residuals = y_test - y_pred
    mape = np.mean(np.abs(residuals / y_test)) * 100  # Mean Absolute Percentage Error
    rmse = np.sqrt(metrics['mse'])

    # Text box
    stats_text = (
        f'Performance Statistics:\n'
        f'R² = {metrics["r2"]:.4f}\n'
        f'RMSE = {rmse:.2f} mg/l\n'
        f'MAE = {metrics["mae"]:.2f} mg/l\n'
        f'MSE = {metrics["mse"]:.2f}\n'
        # f'MAPE = {mape:.2f}%\n'
        # f'n = {len(y_test)} samples'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontfamily='monospace')

    # Labels / legend / grid
    plt.xlabel('Actual Salinity (mg/l)', fontsize=12)
    plt.ylabel('Predicted Salinity (mg/l)', fontsize=12)
    plt.title(f'Actual vs Predicted Salinity - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)

    # Equal axes
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

# Convert Year to datetime
basin_data["Year"] = pd.to_datetime(basin_data["Year"], format="%Y")

# =================
# Set random seeds
# =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =================
# Data preprocessing
# =================
selected = basin_data[(basin_data["Salinity - Cl (mg/l)"] <= 4000)].copy()

# Keep basin id (string) to compute weights later (do NOT one-hot it)
basin_ids_all = selected["Hydrological Basin"].astype(str).values

# Keep a raw feature frame (we'll one-hot AFTER splitting to avoid leakage)
DROP_COLS = [
    "Salinity - Cl (mg/l)", "Year",
    "Hightemp_fall", "Hightemp_spring", "Hightemp_summer", "Hightemp_winter",
    "Tempanam_fall", "Tempanam_spring", "Tempanam_summer", "Tempanam_winter",
    "Zcore_fall", "Zcore_spring", "Zcore_summer", "Zcore_winter",
    "Elevation (m)",
    "salinity difference", "salinity relative difference", "Precip diff",
    "Cell name", "Sub Basin", "Drill",
    "precipitation ( #N of events Q1)", "precipitation (  #N of events Q2)",
    "precipitation ( #N of events Q3)", "precipitation ( #N of events Q4)",
    "precipitation (  #N of events Q5)",
    "National system - Groundwater (%)", "National system - Surface water (%)", "National system - Desalinated water (%)",
    # "Shoreline distance (m)",
    # "Distance to saline bodies (m)",
    # "Fishponds (No.)",
    # "LULC_joined value",
    "Aquifer",
]
X_raw = selected.drop(columns=DROP_COLS, errors="ignore")

y_all      = selected["Salinity - Cl (mg/l)"].values.astype(np.float32)
drill_ids  = selected["Drill"].values
years_dt   = selected["Year"].values  # datetime64
years_only = selected["Year"].dt.year.values  # ints for printing if needed

print(f"Total samples before split: {len(X_raw)}")

# ===================================
# Time-based train-test split (robust)
# ===================================
unique_years = np.sort(np.unique(years_dt))
k = min(6, max(1, len(unique_years) - 1))  # keep at least one year for train
split_year = unique_years[-k]

train_mask = years_dt < split_year
test_mask  = years_dt >= split_year

print(f"\nTime-based split using split_year = {split_year.astype('datetime64[Y]')}")
print(f"Training set years: < {split_year}, Test set years: >= {split_year}")

X_train_df = pd.get_dummies(X_raw[train_mask].copy(), drop_first=True)
X_test_df  = pd.get_dummies(X_raw[test_mask].copy(),  drop_first=True)

# Align test columns to train columns
X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)

# Drop ALL one-hot basin indicators (model shouldn’t “see” basin while we weight by it)
X_train_df = X_train_df.drop(columns=[c for c in X_train_df.columns if c.startswith("Hydrological Basin_")], errors="ignore")
X_test_df  = X_test_df.drop(columns=[c for c in X_test_df.columns  if c.startswith("Hydrological Basin_")],  errors="ignore")

feature_names = list(X_train_df.columns)

y_train = y_all[train_mask]
y_test  = y_all[test_mask]
drill_ids_train = drill_ids[train_mask]
drill_ids_test  = drill_ids[test_mask]

basin_ids_train = basin_ids_all[train_mask]
basin_ids_test  = basin_ids_all[test_mask]

print(f"Training samples: {len(X_train_df)}, Test samples: {len(X_test_df)}")
print(f"Unique train drills: {len(np.unique(drill_ids_train))}, Unique test drills: {len(np.unique(drill_ids_test))}")
print(f"Num features after one-hot (train columns): {len(feature_names)}")

# ===========================
# Keep all columns: Impute, then Scale
# ===========================
# Replace inf with NaN so imputers can process them
X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan)

# Identify columns that are all-NaN in TRAIN (median can't be computed for these)
all_nan_cols = X_train_df.columns[X_train_df.isna().all()]
if len(all_nan_cols):
    print(f"Imputing {len(all_nan_cols)} all-NaN TRAIN columns with constant 0.0: "
          f"{list(all_nan_cols)[:8]}{' ...' if len(all_nan_cols)>8 else ''}")
    # Fill TRAIN & TEST with a neutral constant (choose per domain if desired)
    X_train_df.loc[:, all_nan_cols] = 0.0
    X_test_df.loc[:, all_nan_cols]  = 0.0

# Median-impute the rest (fit on TRAIN only)
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train_df)
X_test_imputed  = imputer.transform(X_test_df)

# Scale AFTER imputation (fit on TRAIN only)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_imputed).astype(np.float32)
X_test  = scaler.transform(X_test_imputed).astype(np.float32)


feature_names = list(X_train_df.columns)

# Final safety checks
assert np.isfinite(X_train).all(), "Non-finite in X_train after impute/scale"
assert np.isfinite(X_test).all(),  "Non-finite in X_test after impute/scale"

# Save preprocessed (imputed+scaled)
pd.concat([
    pd.DataFrame(X_train, columns=feature_names).assign(_split="train"),
    pd.DataFrame(X_test,  columns=feature_names).assign(_split="test")
], ignore_index=True).to_csv(os.path.join(output_dir, "preprocessed_data.csv"), index=False)

# ==========================
# Basin weighting utility
# ==========================
def make_basin_weights(basin_arr: np.ndarray, clip_low: float = 0.5, clip_high: float = 3.0) -> np.ndarray:
    """
    Per-sample weights = inverse basin frequency (normalized within the given array),
    clipped to a reasonable range to stabilize training.
    """
    freq = pd.Series(basin_arr).value_counts(normalize=True)
    inv = {k: 1.0 / v for k, v in freq.items()}
    w = np.array([inv[b] for b in basin_arr], dtype=float)
    return np.clip(w, clip_low, clip_high)

# ==========================================
# Recursive Feature Elimination (for ranking)
# ==========================================
print("\nRunning Recursive Feature Elimination (ranking only)...")
try:
    rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
    rfe_selector.fit(X_train, y_train)
    rfe_results = pd.DataFrame({
        "Feature": feature_names,
        "RFE Ranking": rfe_selector.ranking_,
        "Selected": rfe_selector.support_
    }).sort_values("RFE Ranking", ascending=True)
    rfe_results.to_csv(os.path.join(output_dir, "rfe_results.csv"), index=False)
except Exception as e:
    print(f"RFE skipped due to error: {e}")

# ==========================================
# Global Sensitivity Analysis (Sobol', optional)
# ==========================================
if SALIB_AVAILABLE:
    try:
        print("\nRunning Global Sensitivity Analysis (Sobol)...")
        problem = {
            "num_vars": X_train.shape[1],
            "names": feature_names,
            "bounds": [[0.0, 1.0]] * X_train.shape[1]
        }

        # -- Sampling --
        N_base = 8192
        # Using Sobol sequence sampler (you can switch to Saltelli if you prefer)
        param_values = sobol_sample.sample(problem, N_base, calc_second_order=False)

        # -- Surrogate (fast) model for GSA --
        surrogate_model = LinearRegression().fit(X_train, y_train)
        Y = surrogate_model.predict(param_values)

        # -- Analyze (returns S1/ST and their confidence intervals) --
        si = sobol_analyze.analyze(
            problem, Y, calc_second_order=False, print_to_console=False
            # conf_level defaults to 0.95; you can pass conf_level=0.90 if desired
        )

        # Build tidy DataFrame (handles possible Nones by converting to NaN)
        def _to_arr(key):
            arr = si.get(key, None)
            if arr is None:
                return np.full(len(feature_names), np.nan)
            return np.asarray(arr, dtype=float)

        S1      = _to_arr("S1")
        S1_conf = _to_arr("S1_conf")
        ST      = _to_arr("ST")
        ST_conf = _to_arr("ST_conf")

        # === Rebuild GSA results to match snippet column names/order ===
        gsa_results = (
            pd.DataFrame({
                'feature': feature_names,
                'S1': S1,
                'S1_conf': S1_conf,
                'ST': ST,
                'ST_conf': ST_conf
            })
            .sort_values('ST', ascending=False)
            .reset_index(drop=True)
        )
        gsa_results.to_csv(os.path.join(gsa_dir, "gsa_results.csv"), index=False)

        # === Plot GSA results EXACTLY like the snippet ===
        features_plot = gsa_results.head(15).copy()  # Top 15 features
        y_pos = np.arange(len(features_plot))

        # ----- Total Sensitivity (ST) -----
        plt.figure(figsize=(12, 8))
        plt.barh(
            y_pos, features_plot['ST'],
            xerr=features_plot['ST_conf'],
            alpha=0.7, color='steelblue', ecolor='black', capsize=5
        )
        plt.yticks(y_pos, features_plot['feature'])
        plt.xlabel('Total Sensitivity Index (ST)')
        plt.title('Global Sensitivity Analysis - Total Sensitivity Indices (Top 15)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(gsa_dir, "gsa_total_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ----- First-Order Sensitivity (S1) -----
        plt.figure(figsize=(12, 8))
        plt.barh(
            y_pos, features_plot['S1'],
            xerr=features_plot['S1_conf'],
            alpha=0.7, color='lightcoral', ecolor='black', capsize=5
        )
        plt.yticks(y_pos, features_plot['feature'])
        plt.xlabel('First-Order Sensitivity Index (S1)')
        plt.title('Global Sensitivity Analysis - First-Order Sensitivity Indices (Top 15)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(gsa_dir, "gsa_first_order_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ----- Combined plot (grouped bars; no error bars) -----
        plt.figure(figsize=(14, 10))
        x = np.arange(len(features_plot))
        width = 0.35

        plt.barh(y_pos - width / 2, features_plot['S1'], width,
                 label='First-Order (S1)', alpha=0.7, color='lightcoral', zorder=2)
        plt.barh(y_pos + width / 2, features_plot['ST'], width,
                 label='Total (ST)', alpha=0.7, color='steelblue', zorder=2)

        plt.yticks(y_pos, features_plot['feature'])
        plt.xlabel('Sensitivity Index')
        plt.title('Global Sensitivity Analysis - First-Order vs Total Sensitivity Indices')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x', zorder=0)

        # ✅ Place legend below the top bars, slightly offset from the right edge
        plt.legend(
            loc='upper right',
            bbox_to_anchor=(0.98, 0.88),  # move legend lower inside the plot
            frameon=True,
            framealpha=0.9,
            borderpad=0.4,
            handlelength=1.2,
            handletextpad=0.6,
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(os.path.join(gsa_dir, "gsa_combined_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"GSA results saved to: {os.path.join(gsa_dir, 'gsa_results.csv')}")
        print(f"GSA plots saved to:   {gsa_dir}")

    except Exception as e:
        print(f"GSA skipped due to error: {e}")
else:
    print("\nSALib not available; skipping Sobol' sensitivity analysis.")

# =======================
# Model builder (regularized)
# =======================
def build_ann(input_dim: int,
              hidden_layers: int = 3,
              units: int = 64,
              activation: str = "relu",
              l2_strength: float = 1e-4,
              dropout_rate: float = 0.2) -> tf.keras.Model:
    """Build a regularized feedforward ANN."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(hidden_layers):
        if activation == "leaky_relu":
            model.add(Dense(units, kernel_regularizer=regularizers.l2(l2_strength)))
            model.add(LeakyReLU(negative_slope=0.01))
        else:
            model.add(Dense(units, activation=activation, kernel_regularizer=regularizers.l2(l2_strength)))
        if dropout_rate and dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # linear output
    return model

# =======================
# Grouped Cross-Validation
# =======================
unique_drills_train = np.unique(drill_ids_train)
do_cv = len(unique_drills_train) >= 2
cv_results = []
cv_memory = []  # <-- hold in-memory info including indices for best-final validation reuse


if do_cv:
    n_splits = min(5, len(unique_drills_train))
    gkf = GroupKFold(n_splits=n_splits)
    print(f"\nStarting {n_splits}-fold GroupKFold CV by drill...")

    # Hyperparameter grid (kept modest)
    optimizers  = ["adam"]  # can add "sgd" etc.
    activations = ["relu", "leaky_relu"]
    hidden_layers_list = [1, 3, 5]
    neurons_per_layer  = [64, 128]
    l2_strengths       = [1e-4]   # adjust if needed
    dropout_rates      = [0.2]    # adjust if needed

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=drill_ids_train), start=1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        drills_tr, drills_va = drill_ids_train[tr_idx], drill_ids_train[va_idx]
        basins_tr, basins_va = basin_ids_train[tr_idx], basin_ids_train[va_idx]
        print(f"\n--- Fold {fold}/{n_splits} ---")
        print(f"Train fold: {len(X_tr)} samples, drills={len(np.unique(drills_tr))} | "
              f"Val fold: {len(X_va)} samples, drills={len(np.unique(drills_va))}")

        best_mae_fold = float("inf")
        best_hist = None
        best_label = None

        for optimizer in optimizers:
            for activation in activations:
                for num_layers in hidden_layers_list:
                    for units in neurons_per_layer:
                        for l2_val in l2_strengths:
                            for dr in dropout_rates:
                                model = build_ann(
                                    input_dim=X_tr.shape[1],
                                    hidden_layers=num_layers,
                                    units=units,
                                    activation=activation,
                                    l2_strength=l2_val,
                                    dropout_rate=dr
                                )
                                model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

                                callbacks_cv = [
                                    tf.keras.callbacks.EarlyStopping(
                                        monitor="val_loss", patience=10, restore_best_weights=True, min_delta=1e-4
                                    ),
                                    tf.keras.callbacks.ReduceLROnPlateau(
                                        monitor="val_loss", factor=0.5, patience=5
                                    ),
                                ]

                                # Basin weights aligned to this fold
                                sw_tr = fixed_basin_weights_for(basins_tr)
                                sw_va = fixed_basin_weights_for(basins_va)

                                hist = model.fit(
                                    X_tr, y_tr,
                                    epochs=100, batch_size=32,
                                    validation_data=(X_va, y_va, sw_va),  # <— weighted validation
                                    sample_weight=sw_tr,  # <— weighted training
                                    callbacks=callbacks_cv,
                                    verbose=0
                                )

                                # Validation predictions (unweighted + weighted metrics)
                                y_va_pred = model.predict(X_va, verbose=0).ravel()

                                # Unweighted
                                val_mae = mean_absolute_error(y_va, y_va_pred)
                                val_rmse = math.sqrt(mean_squared_error(y_va, y_va_pred))
                                val_r2 = r2_score(y_va, y_va_pred)

                                # Weighted
                                val_rmse_w = safe_rmse(y_va, y_va_pred, sample_weight=sw_va)
                                val_mae_w = mean_absolute_error(y_va, y_va_pred, sample_weight=sw_va)
                                val_r2_w = weighted_r2(y_va, y_va_pred, sw_va)

                                hist = model.fit(
                                    X_tr, y_tr,
                                    epochs=100, batch_size=32,
                                    validation_data=(X_va, y_va),
                                    sample_weight = make_basin_weights(basins_tr),
                                    callbacks=callbacks_cv,
                                    verbose=0
                                )

                                # Best epoch & generalization gap
                                best_epoch = int(np.argmin(hist.history["val_loss"]))
                                gen_gap = float(hist.history["val_loss"][best_epoch] - hist.history["loss"][best_epoch])

                                # Validation metrics
                                y_va_pred = model.predict(X_va, verbose=0).ravel()
                                val_mae  = mean_absolute_error(y_va, y_va_pred)
                                val_rmse = math.sqrt(mean_squared_error(y_va, y_va_pred))
                                val_r2   = r2_score(y_va, y_va_pred)


                                label = f"opt={optimizer}, act={activation}, L={num_layers}, U={units}, l2={l2_val}, dr={dr}"
                                cv_results.append({
                                    "fold": fold,
                                    "optimizer": optimizer,
                                    "activation": activation,
                                    "hidden_layers": num_layers,
                                    "units": units,
                                    "l2": l2_val,
                                    "dropout": dr,
                                    "best_epoch": best_epoch,
                                    "gen_gap": gen_gap,
                                    "val_mae": float(val_mae),
                                    "val_rmse": float(val_rmse),
                                    "val_r2": float(val_r2),
                                    "val_mae_w": float(val_mae_w),
                                    "val_rmse_w": float(val_rmse_w),
                                    "val_r2_w": float(val_r2_w),
                                })

                                cv_memory.append({
                                    "fold": fold,
                                    "optimizer": optimizer,
                                    "activation": activation,
                                    "hidden_layers": num_layers,
                                    "units": units,
                                    "l2": l2_val,
                                    "dropout": dr,
                                    "val_mae": float(val_mae),
                                    "val_r2": float(val_r2),
                                    "tr_idx": tr_idx,  # indices into X_train/y_train
                                    "va_idx": va_idx
                                })

                                if val_mae < best_mae_fold:
                                    best_mae_fold = val_mae
                                    best_hist = hist
                                    best_label = label

        # Save best training history plot for this fold
        if best_hist is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(best_hist.history["loss"], label="Train Loss")
            plt.plot(best_hist.history["val_loss"], label="Val Loss")
            plt.title(f"Best Training History - Fold {fold}\n({best_label})")
            plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()
            plt.savefig(os.path.join(cv_dir, f"training_history_fold{fold}.png"), dpi=300, bbox_inches="tight")
            plt.close()

    # Save raw CV results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(output_dir, "cross_validation_results.csv"), index=False)

    # Summaries by config
    group_cols = ["optimizer", "activation", "hidden_layers", "units", "l2", "dropout"]
    cv_summary = (pd.DataFrame(cv_results)
    .groupby(group_cols, as_index=False)
    .agg(
        mean_val_mae=("val_mae", "mean"),
        std_val_mae=("val_mae", "std"),
        mean_val_rmse=("val_rmse", "mean"),
        mean_val_r2=("val_r2", "mean"),
        mean_val_mae_w=("val_mae_w", "mean"),
        mean_val_rmse_w=("val_rmse_w", "mean"),
        mean_val_r2_w=("val_r2_w", "mean"),
        mean_gen_gap=("gen_gap", "mean"),
    ))
    cv_summary = cv_summary.sort_values(["mean_val_mae", "mean_gen_gap"]).reset_index(drop=True)
    cv_summary.to_csv(os.path.join(output_dir, "cross_validation_summary.csv"), index=False)

    print("\nCross-validation completed!")
    print("Top CV configs:\n", cv_summary.head())

    # Select best params by mean MAE, then by smallest gen gap
    best_row = cv_summary.iloc[0]
    best_optimizer = str(best_row["optimizer"])
    best_params = {
        "activation":     str(best_row["activation"]),
        "hidden_layers":  int(best_row["hidden_layers"]),
        "units":          int(best_row["units"]),
        "l2":             float(best_row["l2"]),
        "dropout":        float(best_row["dropout"]),
    }
else:
    print("\nNot enough unique drill groups for GroupKFold (need >= 2). Skipping CV; using defaults.")
    best_optimizer = "adam"
    best_params = {"activation": "relu", "hidden_layers": 3, "units": 64, "l2": 1e-4, "dropout": 0.2}
    cv_df = pd.DataFrame(columns=[
        "fold","optimizer","activation","hidden_layers","units","l2","dropout",
        "best_epoch","gen_gap","val_mae","val_rmse","val_r2"
    ])
    cv_summary = pd.DataFrame(columns=[
        "optimizer","activation","hidden_layers","units","l2","dropout",
        "mean_val_mae","std_val_mae","mean_val_rmse","mean_val_r2","mean_gen_gap"
    ])

print(f"\nBest parameters for final model: {best_params} (optimizer={best_optimizer})")


# ==========================
# Final validation selection (best CV fold OR temporal fallback)
# ==========================
X_tr_final, y_tr_final, X_va_final, y_va_final = None, None, None, None
basin_tr_final, basin_va_final = None, None
final_val_strategy = None

if do_cv and len(cv_memory):
    # 1) Try to reuse the best fold among entries that match best hyperparams
    matches = [r for r in cv_memory if
               r["optimizer"] == best_optimizer and
               r["activation"] == best_params["activation"] and
               r["hidden_layers"] == best_params["hidden_layers"] and
               r["units"] == best_params["units"] and
               float(r["l2"]) == float(best_params["l2"]) and
               float(r["dropout"]) == float(best_params["dropout"])]
    if len(matches):
        best_fold_entry = min(matches, key=lambda r: r["val_mae"])
        tr_idx_final, va_idx_final = best_fold_entry["tr_idx"], best_fold_entry["va_idx"]
        X_tr_final, y_tr_final = X_train[tr_idx_final], y_train[tr_idx_final]
        X_va_final, y_va_final = X_train[va_idx_final], y_train[va_idx_final]
        basin_tr_final, basin_va_final = basin_ids_train[tr_idx_final], basin_ids_train[va_idx_final]
        final_val_strategy = f"best_cv_fold (fold={best_fold_entry['fold']})"
        print(f"\nFinal validation uses best CV fold: fold={best_fold_entry['fold']} "
              f"| train={len(X_tr_final)}, val={len(X_va_final)}")

# 2) Temporal validation fallback (inside training years) if we couldn't reuse a CV fold
if X_tr_final is None:
    years_train_dt = years_dt[train_mask]  # datetime64 array aligned to X_train/y_train
    unique_train_years = np.sort(np.unique(years_train_dt))
    if len(unique_train_years) >= 2:
        # choose last ~20% of training years as validation, but at least 1 year and leaving ≥1 year for training
        n_years = len(unique_train_years)
        n_val_years = max(1, int(round(0.2 * n_years)))
        n_val_years = min(n_val_years, n_years - 1)  # leave ≥1 year for training
        val_years_set = set(unique_train_years[-n_val_years:])
        va_mask_train = np.isin(years_train_dt, list(val_years_set))
        tr_mask_train = ~va_mask_train
        # ensure non-empty
        if tr_mask_train.sum() > 0 and va_mask_train.sum() > 0:
            X_tr_final, y_tr_final = X_train[tr_mask_train], y_train[tr_mask_train]
            X_va_final, y_va_final = X_train[va_mask_train], y_train[va_mask_train]
            basin_tr_final, basin_va_final = basin_ids_train[tr_mask_train], basin_ids_train[va_mask_train]
            final_val_strategy = f"temporal_within_train (val_years={sorted(val_years_set)})"
            print(f"\nFinal validation uses temporal split within training years: "
                  f"train={len(X_tr_final)}, val={len(X_va_final)} | "
                  f"val years={sorted([str(y.astype('datetime64[Y]')) for y in val_years_set])}")

# 3) Last resort: random 10% validation if no groups/years allow a temporal split
if X_tr_final is None:
    print("\nWARNING: Could not form group-aware/temporal validation; using random validation_split=0.1.")
    X_tr_final, y_tr_final = X_train, y_train
    X_va_final, y_va_final = None, None
    basin_tr_final, basin_va_final = basin_ids_train, None
    final_val_strategy = "random_split_0.1"

print(f"Final validation strategy: {final_val_strategy}")


# ==========================
# Train final model (full train)
# ==========================
final_model = build_ann(
    input_dim=X_train.shape[1],
    hidden_layers=best_params["hidden_layers"],
    units=best_params["units"],
    activation=best_params["activation"],
    l2_strength=best_params["l2"],
    dropout_rate=best_params["dropout"]
)
final_model.compile(optimizer=best_optimizer, loss="mse", metrics=["mae"])

callbacks_final = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, min_delta=1e-4),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
]

if X_va_final is not None:
    sw_tr_final = fixed_basin_weights_for(basin_tr_final)
    sw_va_final = fixed_basin_weights_for(basin_va_final)
    hist_final = final_model.fit(
        X_tr_final, y_tr_final,
        epochs=300, batch_size=32,
        validation_data=(X_va_final, y_va_final, sw_va_final),  # weighted val
        sample_weight=sw_tr_final,                               # weighted train
        callbacks=callbacks_final, verbose=0
    )
else:
    # Keras can't pass val sample_weight when using validation_split, but train is still weighted
    sw_tr_final = fixed_basin_weights_for(basin_tr_final)
    hist_final = final_model.fit(
        X_tr_final, y_tr_final,
        epochs=300, batch_size=32,
        validation_split=0.1,
        sample_weight=sw_tr_final,
        callbacks=callbacks_final, verbose=0
    )


# Generalization gap at best epoch
best_epoch_final = int(np.argmin(hist_final.history["val_loss"]))
gen_gap_final = float(hist_final.history["val_loss"][best_epoch_final] - hist_final.history["loss"][best_epoch_final])
print(f"\nFinal training best epoch: {best_epoch_final} | generalization gap: {gen_gap_final:.6f}")

# ==========================
# Evaluate on test and train
# ==========================
y_test_pred  = final_model.predict(X_test, verbose=0).ravel()
y_train_pred = final_model.predict(X_train, verbose=0).ravel()

# Weighted evaluation
sw_test  = fixed_basin_weights_for(basin_ids_test)
sw_train = fixed_basin_weights_for(basin_ids_train)

def metrics_report_w(y_true, y_pred, w):
    r2_w   = weighted_r2(y_true, y_pred, w)
    rmse_w = safe_rmse(y_true, y_pred, sample_weight=w)
    mae_w  = mean_absolute_error(y_true, y_pred, sample_weight=w)
    rmae_w = mae_w / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    return r2_w, rmse_w, mae_w, rmae_w

test_r2_w,  test_rmse_w,  test_mae_w,  test_rmae_w  = metrics_report_w(y_test,  y_test_pred,  sw_test)
train_r2_w, train_rmse_w, train_mae_w, train_rmae_w = metrics_report_w(y_train, y_train_pred, sw_train)

print(f"\nFINAL Test (weighted) -> R²={test_r2_w:.4f}, RMSE={test_rmse_w:.2f}, MAE={test_mae_w:.2f}, RMAE={test_rmae_w:.4f}")
print(f"Train (weighted)      -> R²={train_r2_w:.4f}, RMSE={train_rmse_w:.2f}, MAE={train_mae_w:.2f}, RMAE={train_rmae_w:.4f}")


def metrics_report(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    rmae = mae / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    return r2, rmse, mae, rmae

test_r2, test_rmse, test_mae, test_rmae   = metrics_report(y_test,  y_test_pred)
train_r2, train_rmse, train_mae, train_rmae = metrics_report(y_train, y_train_pred)

print(f"\nFINAL Test metrics -> R²={test_r2:.4f}, RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, RMAE={test_rmae:.4f}")
print(f"Train metrics       -> R²={train_r2:.4f}, RMSE={train_rmse:.2f}, MAE={train_mae:.2f}, RMAE={train_rmae:.4f}")

# -------- Per-basin test metrics (macro across basins) --------
print("\nPer-basin Test metrics:")
per_basin_mae, per_basin_rmse, per_basin_r2 = [], [], []
for b in np.unique(basin_ids_test):
    m = (basin_ids_test == b)
    if m.sum() == 0:
        continue
    r2_b, rmse_b, mae_b, _ = metrics_report(y_test[m], y_test_pred[m])
    per_basin_mae.append(mae_b); per_basin_rmse.append(rmse_b); per_basin_r2.append(r2_b)
    print(f"  {b:>22s}: n={m.sum():4d} | R²={r2_b: .4f} RMSE={rmse_b: .2f} MAE={mae_b: .2f}")
if per_basin_mae:
    print(f"\nMacro (mean over basins) -> R²={np.nanmean(per_basin_r2):.4f}, RMSE={np.nanmean(per_basin_rmse):.2f}, MAE={np.nanmean(per_basin_mae):.2f}")


# Save CV + final metrics summary
final_results = {
    "optimizer": best_optimizer,
    "activation": best_params["activation"],
    "hidden_layers": best_params["hidden_layers"],
    "units": best_params["units"],
    "l2": best_params["l2"],
    "dropout": best_params["dropout"],

    # CV (unweighted)
    "cv_val_mae_mean":  (cv_summary["mean_val_mae"].iloc[0]   if len(cv_summary) else np.nan),
    "cv_val_rmse_mean": (cv_summary["mean_val_rmse"].iloc[0]  if len(cv_summary) else np.nan),
    "cv_val_r2_mean":   (cv_summary["mean_val_r2"].iloc[0]    if len(cv_summary) else np.nan),

    # CV (weighted)
    "cv_val_mae_mean_w":  (cv_summary["mean_val_mae_w"].iloc[0]   if len(cv_summary) else np.nan),
    "cv_val_rmse_mean_w": (cv_summary["mean_val_rmse_w"].iloc[0]  if len(cv_summary) else np.nan),
    "cv_val_r2_mean_w":   (cv_summary["mean_val_r2_w"].iloc[0]    if len(cv_summary) else np.nan),

    # Train (unweighted)
    "train_mae":  float(train_mae),  "train_rmse":  float(train_rmse),  "train_r2":  float(train_r2),  "train_rmae":  float(train_rmae),
    # Train (weighted)
    "train_mae_w":float(train_mae_w),"train_rmse_w":float(train_rmse_w),"train_r2_w":float(train_r2_w),"train_rmae_w":float(train_rmae_w),

    # Test (unweighted)
    "test_mae":   float(test_mae),   "test_rmse":   float(test_rmse),   "test_r2":   float(test_r2),   "test_rmae":   float(test_rmae),
    # Test (weighted)
    "test_mae_w": float(test_mae_w), "test_rmse_w": float(test_rmse_w), "test_r2_w": float(test_r2_w), "test_rmae_w": float(test_rmae_w),

    "final_best_epoch": int(best_epoch_final),
    "final_gen_gap": float(gen_gap_final),
    "final_val_strategy": final_val_strategy,

    # Macro per-basin (unweighted; keep as you had)
    "macro_test_mae_over_basins": float(np.nanmean(per_basin_mae)) if len(per_basin_mae) else np.nan,
    "macro_test_rmse_over_basins": float(np.nanmean(per_basin_rmse)) if len(per_basin_rmse) else np.nan,
    "macro_test_r2_over_basins": float(np.nanmean(per_basin_r2)) if len(per_basin_r2) else np.nan,
}

pd.DataFrame([final_results]).to_csv(os.path.join(output_dir, "final_performance_summary.csv"), index=False)


# ==========================
# Plots: Training curves
# ==========================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_final.history["loss"], label="Train Loss")
plt.plot(hist_final.history["val_loss"], label="Val Loss")
plt.axvline(best_epoch_final, color="r", linestyle="--", label="Best Epoch")
plt.title("Final Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_final.history["mae"], label="Train MAE")
plt.plot(hist_final.history["val_mae"], label="Val MAE")
plt.title("Final MAE"); plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_training_history.png"), dpi=300, bbox_inches="tight")
plt.close()

# ==========================
# Plots: Actual vs Predicted (snippet-exact style)
# ==========================
# TEST
test_metrics_dict = {
    'r2':  r2_score(y_test, y_test_pred),
    'mae': mean_absolute_error(y_test, y_test_pred),
    'mse': mean_squared_error(y_test, y_test_pred),
}
create_actual_vs_predicted_plot(
    y_test=y_test.astype(float),
    y_pred=y_test_pred.astype(float),
    metrics=test_metrics_dict,
    output_dir=output_dir,
    model_name=f'ANN',
    filename='actual_vs_predicted_test.png'
)

# TRAIN
train_metrics_dict = {
    'r2':  r2_score(y_train, y_train_pred),
    'mae': mean_absolute_error(y_train, y_train_pred),
    'mse': mean_squared_error(y_train, y_train_pred),
}
create_actual_vs_predicted_plot(
    y_test=y_train.astype(float),
    y_pred=y_train_pred.astype(float),
    metrics=train_metrics_dict,
    output_dir=output_dir,
    model_name=f'ANN ({best_optimizer}) - Training Set',
    filename='actual_vs_predicted_train.png'
)

# ============ SHAP Analysis ============
print("\nRunning SHAP analysis with GradientExplainer...")
# Build background with mild stratification by drill (cap total size)
background_indices = []
drills = np.unique(drill_ids_train)
max_bg = 256
samples_per_drill = max(1, max_bg // max(1, len(drills)))
for d in drills:
    idx = np.where(drill_ids_train == d)[0]
    take = min(samples_per_drill, len(idx))
    if take > 0:
        background_indices.extend(np.random.choice(idx, size=take, replace=False))
background_indices = np.unique(background_indices)
background = X_train[background_indices]

# Prefer GradientExplainer for eager TF2
explainer = shap.GradientExplainer(final_model, background)

test_shap_values = explainer.shap_values(X_test)
if isinstance(test_shap_values, list):
    test_shap_values = test_shap_values[0]
test_shap_values = np.asarray(test_shap_values)
# Handle possible shapes: (N, 1, D) or (N, D)
if test_shap_values.ndim == 3:
    if test_shap_values.shape[1] == 1:
        test_shap_values = test_shap_values[:, 0, :]
    elif test_shap_values.shape[2] == 1:
        test_shap_values = test_shap_values[:, :, 0]
    else:
        test_shap_values = np.squeeze(test_shap_values)

assert test_shap_values.shape == (X_test.shape[0], X_test.shape[1]), \
    f"Unexpected SHAP shape: {test_shap_values.shape} vs {(X_test.shape[0], X_test.shape[1])}"

# Summary bar
plt.figure(figsize=(12, 8))
shap.summary_plot(test_shap_values, X_test, feature_names=feature_names,
                  plot_type="bar", show=False, max_display=len(feature_names))
plt.gcf().suptitle(f"Global Feature Importance ({best_optimizer})")
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, f"global_shap_summary_bar_{best_optimizer}.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# Beeswarm
plt.figure(figsize=(12, 8))
shap.summary_plot(test_shap_values, X_test, feature_names=feature_names,
                  show=False, max_display=len(feature_names))
plt.gcf().suptitle(f"Global SHAP Value Distribution ({best_optimizer})")
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, f"global_shap_beeswarm_{best_optimizer}.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# Save mean |SHAP|
shap_df = pd.DataFrame(test_shap_values, columns=feature_names)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
pd.DataFrame(mean_abs_shap, columns=["mean_abs_shap"]).to_csv(
    os.path.join(shap_dir, "shap_values_absolute.csv"), index=True
)

# SHAP interaction (top 5 features)
print("Running SHAP interaction analysis (top 5 features)...")
interaction_dir = os.path.join(shap_dir, f"interaction_plots_{best_optimizer}")
os.makedirs(interaction_dir, exist_ok=True)
top_features = mean_abs_shap.head(5).index.tolist()

for i, feat1 in enumerate(top_features):
    for j, feat2 in enumerate(top_features):
        if i < j:
            plt.figure(figsize=(10, 8))
            shap.dependence_plot(
                feat1, test_shap_values, X_test,
                feature_names=feature_names,
                interaction_index=feat2,
                show=False
            )
            plt.title(f"SHAP Interaction: {feat1} × {feat2} ({best_optimizer})")
            plt.tight_layout()
            plt.savefig(os.path.join(interaction_dir, f"interaction_{i}_{j}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

print("\nTraining completed successfully!")
print(f"Final Test MAE: {test_mae:.4f}")
print(f"Final Test R²: {test_r2:.4f}")
