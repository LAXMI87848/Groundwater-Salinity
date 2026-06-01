import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, LeakyReLU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random
import os
import sys
import io
from datetime import datetime
import shap
from statsmodels.graphics.tsaplots import plot_acf
import csv
from sklearn.impute import SimpleImputer
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

# -----------------------------
# Output dirs # give your own path for output directory where you want to save the results
# -----------------------------
output_dir = r"E:\Python\India Project\output_figures\dataset2910\LSTM\All_basins" # used in my case
os.makedirs(output_dir, exist_ok=True)
loss_plots_dir = os.path.join(output_dir, "loss_plots")
os.makedirs(loss_plots_dir, exist_ok=True)
shap_plots_dir = os.path.join(output_dir, "shap_plots")
os.makedirs(shap_plots_dir, exist_ok=True)
residual_plots_dir = os.path.join(output_dir, "residual_plots")
os.makedirs(residual_plots_dir, exist_ok=True)
shap_csv_dir = os.path.join(output_dir, "shap_csv")
os.makedirs(shap_csv_dir, exist_ok=True)
# NEW: Sensitivity outputs
GSA_dir = os.path.join(output_dir, "GSA")
os.makedirs(GSA_dir, exist_ok=True)

# -----------------------------
# Fixed per-basin weights
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
# SEPARATE ACTUAL VS PREDICTED PLOT WITH STATISTICS (snippet-exact)
# =============================================
# =============================
# RFE helpers for LSTM
# =============================
def _slice_features(X_3d: np.ndarray, feat_idx: list[int]) -> np.ndarray:
    """Keep only the selected features across ALL time steps."""
    X_3d = np.asarray(X_3d)
    return X_3d[:, :, feat_idx]

def _build_lstm_model(n_features: int, time_steps: int,
                      hidden_units: int, activation: str, optimizer: str) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(time_steps, n_features)),
        LSTM(hidden_units, activation=activation, return_sequences=True),
        Dropout(0.2),
        LSTM(max(1, hidden_units // 2), activation=activation),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    if optimizer == 'sgd':
        optimizer_instance = tf.keras.optimizers.SGD(
            learning_rate=0.001, momentum=0.9, nesterov=True, clipnorm=1.0
        )
    else:
        optimizer_instance = tf.keras.optimizers.get(optimizer)
    model.compile(optimizer=optimizer_instance, loss='mse', metrics=['mae'])
    return model

def _cv_score_for_subset(X_train_full: np.ndarray, y_train: np.ndarray,
                         groups: np.ndarray, basins: np.ndarray,
                         feat_idx: list[int], time_steps: int,
                         hidden_units: int, activation: str, optimizer: str,
                         max_epochs: int = 40, batch_size: int = 64, verbose: int = 0,
                         n_splits: int = 3) -> float:
    """
    Train/evaluate using GroupKFold on a subset of features. Returns mean *weighted R^2* over folds.
    """
    X_sub = _slice_features(X_train_full, feat_idx)
    n_features_sub = X_sub.shape[2]
    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for tr_idx, va_idx in gkf.split(X_sub, y_train, groups=groups):
        model = _build_lstm_model(n_features_sub, time_steps, hidden_units, activation, optimizer)

        X_tr, X_va = X_sub[tr_idx], X_sub[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        sw_tr = fixed_basin_weights_for(basins[tr_idx])
        sw_va = fixed_basin_weights_for(basins[va_idx])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

        # IMPORTANT: pass validation sample weights in validation_data tuple
        model.fit(
            X_tr, y_tr,
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sw_tr,
            validation_data=(X_va, y_va, sw_va),
            callbacks=[es]
        )
        y_va_pred = model.predict(X_va, verbose=0).ravel()
        r2w = weighted_r2(y_va.ravel(), y_va_pred, sw_va)
        scores.append(r2w)

        # cleanup (avoid TF graph growth)
        tf.keras.backend.clear_session()

    return float(np.mean(scores)) if scores else np.nan

def run_lstm_rfe(
    X_train_full: np.ndarray, y_train: np.ndarray,
    groups: np.ndarray, basins: np.ndarray, feature_names: list[str],
    time_steps: int, optimizer: str, hidden_units: int, activation: str,
    out_dir: str, max_remove: int | None = None, min_features: int = 5,
    epochs_per_eval: int = 40, folds: int = 3, random_seed: int = 42
) -> dict:
    """
    Custom, model-based RFE:
      - Start with all features.
      - At each step, try removing each remaining feature; keep the removal that gives the highest CV weighted R².
      - Continue until min_features or max_remove reached.
    Saves ranking CSV and plots in `out_dir`.
    """
    rng = np.random.default_rng(random_seed)
    os.makedirs(out_dir, exist_ok=True)

    remaining = list(range(len(feature_names)))
    n_total = len(remaining)
    if max_remove is None:
        max_remove = max(0, n_total - min_features)

    # Baseline score with all features
    print("\n[RFE] Establishing baseline CV score with ALL features ...")
    best_subset_score = _cv_score_for_subset(
        X_train_full, y_train, groups, basins, remaining, time_steps,
        hidden_units, activation, optimizer,
        max_epochs=epochs_per_eval, n_splits=folds, verbose=0
    )
    print(f"[RFE] Baseline weighted R2 (all features): {best_subset_score:.4f}")

    elimination_records = []
    num_removed = 0

    while num_removed < max_remove and len(remaining) > min_features:
        print(f"\n[RFE] Iteration {num_removed+1} — {len(remaining)} features remaining")
        trial_scores = []
        # Try dropping each feature one-by-one
        for f in remaining:
            cand = [x for x in remaining if x != f]
            score = _cv_score_for_subset(
                X_train_full, y_train, groups, basins, cand, time_steps,
                hidden_units, activation, optimizer,
                max_epochs=epochs_per_eval, n_splits=folds, verbose=0
            )
            trial_scores.append((f, score))
            print(f"    - remove [{feature_names[f]}] -> mean wR2 = {score:.4f}")

        # pick the removal that gives highest score
        f_remove, best_score_if_removed = max(trial_scores, key=lambda t: t[1])
        print(f"[RFE] Best removal: {feature_names[f_remove]} (score {best_score_if_removed:.4f})")

        elimination_records.append({
            'step': num_removed + 1,
            'removed_feature': feature_names[f_remove],
            'remaining_after': len(remaining) - 1,
            'cv_weighted_r2': best_score_if_removed
        })
        remaining.remove(f_remove)
        best_subset_score = best_score_if_removed
        num_removed += 1

    # Compute ranks: 1 = most important (survivors), larger = less important (earlier eliminations)
    # Survivors get rank 1; eliminated get rank by reverse elimination order.
    ranks = {fn: 1 for fn in [feature_names[i] for i in remaining]}
    # Earlier elimination -> worse (larger) rank
    for idx, rec in enumerate(elimination_records, start=1):
        # feature eliminated at step idx; assign rank = (#survivors rank 1) + (remaining eliminations descending)
        # Simpler: give elimination rank = min_features + (num_removed - idx + 1)
        ranks[rec['removed_feature']] = min_features + (len(elimination_records) - idx + 1)

    # Prepare CSV
    df_rank = pd.DataFrame({
        'feature': feature_names,
        'rfe_rank': [ranks[fn] for fn in feature_names]
    }).sort_values(['rfe_rank','feature']).reset_index(drop=True)

    df_elim = pd.DataFrame(elimination_records)

    csv_path = os.path.join(out_dir, "rfe_ranking.csv")
    df_rank.to_csv(csv_path, index=False)

    elim_csv_path = os.path.join(out_dir, "rfe_elimination_trace.csv")
    df_elim.to_csv(elim_csv_path, index=False)

    # Plot: bar of ranks (lower = more important)
    plt.figure(figsize=(12, max(6, 0.25 * len(feature_names))))
    plot_df = df_rank.sort_values('rfe_rank', ascending=True)
    y_pos = np.arange(len(plot_df))
    plt.barh(y_pos, plot_df['rfe_rank'].values, alpha=0.8)
    plt.yticks(y_pos, plot_df['feature'].values)
    plt.xlabel('RFE Rank (1 = most important)')
    plt.title('LSTM RFE Feature Ranking')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "rfe_rankings.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot: CV performance vs #features kept
    if len(elimination_records) > 0:
        perf_df = pd.DataFrame({
            '#features_kept': [len(feature_names) - i for i in range(len(elimination_records)+1)],
            'cv_weighted_r2': [np.nan] * (len(elimination_records)+1)
        })
        # first point is the baseline (all features)
        perf_df.loc[0, 'cv_weighted_r2'] = elimination_records[0]['cv_weighted_r2'] \
            if len(elimination_records) else best_subset_score
        # subsequent points from the trace after each removal
        for i, rec in enumerate(elimination_records, start=1):
            perf_df.loc[i, 'cv_weighted_r2'] = rec['cv_weighted_r2']

        plt.figure(figsize=(10, 6))
        plt.plot(perf_df['#features_kept'], perf_df['cv_weighted_r2'], marker='o')
        plt.xlabel('# of Features Kept')
        plt.ylabel('Mean CV Weighted R²')
        plt.title('RFE: Performance vs. Number of Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        perf_png = os.path.join(out_dir, "rfe_perf_vs_num_features.png")
        plt.savefig(perf_png, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        perf_png = None

    print(f"[RFE] Saved ranking CSV: {csv_path}")
    print(f"[RFE] Saved elimination trace CSV: {elim_csv_path}")
    print(f"[RFE] Saved ranking plot: {png_path}")
    if perf_png:
        print(f"[RFE] Saved performance plot: {perf_png}")

    return {
        'ranking_csv': csv_path,
        'trace_csv': elim_csv_path,
        'ranking_png': png_path,
        'perf_png': perf_png,
        'kept_features': [feature_names[i] for i in remaining]
    }

def create_actual_vs_predicted_plot(y_test, y_pred, metrics, output_dir, model_name="LSTM",
                                    filename="actual_vs_predicted_detailed.png"):
    import numpy as np, matplotlib.pyplot as plt, os

    plt.figure(figsize=(10, 8))

    # Scatter
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)

    # Perfect line + regression line
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "g-", alpha=0.8, linewidth=2, label='Regression Line')

    # Stats box (exact text/format)
    residuals = y_test - y_pred
    mape = np.mean(np.abs(residuals / y_test)) * 100
    rmse = np.sqrt(metrics['mse'])
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

    # Labels/legend/grid
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

    return {'mape': mape, 'rmse': rmse}

def safe_rmse(y_true, y_pred, sample_weight=None):
    """RMSE that works across sklearn versions (with/without squared=)."""
    try:
        # newer sklearn
        return mean_squared_error(y_true, y_pred, sample_weight=sample_weight, squared=False)
    except TypeError:
        # older sklearn
        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))


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

# -----------------------------
# Logging tee
# -----------------------------
class Tee(io.TextIOWrapper):
    def __init__(self, *files):
        self.files = files
        super().__init__(io.BytesIO(), encoding=sys.stdout.encoding)

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
        return len(obj)

    def flush(self):
        for f in self.files:
            f.flush()

def run_sobol_analysis(
    model,
    X_train_scaled: np.ndarray,
    feature_names: list,
    time_steps: int,
    n_samples: int,
    GSA_dir: str,
    random_seed: int = 42,
    calc_second_order: bool = True
):
    """
    Runs Sobol GSA on the LSTM by sampling the scaled [0,1] feature space.
    For each sample vector (len = n_features), we construct a sequence by
    setting EVERY time step's value of feature j to that sampled value.
    Other structure comes from a baseline sequence (per-time, per-feature median).
    Saves CSVs and bar plots with error bars (S1/ST ± conf).

    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model expecting input shape (None, time_steps, n_features).
    X_train_scaled : np.ndarray
        Training data AFTER imputation & scaling, shape (n_samples, time_steps, n_features).
    feature_names : list[str]
        Names for the base features (length = n_features).
    time_steps : int
    n_samples : int
        Base Saltelli sample size (actual evaluations ~ n_samples*(2D+2)).
    out_csv_dir, out_plot_dir : str
    random_seed : int
    calc_second_order : bool
    """
    rng = np.random.default_rng(random_seed)

    n_features = len(feature_names)
    assert X_train_scaled.ndim == 3 and X_train_scaled.shape[2] == n_features, \
        f"X_train_scaled must be (N, {time_steps}, {n_features})"

    # Baseline: per-time, per-feature median (stays in scaled space)
    baseline = np.nanmedian(X_train_scaled, axis=0)  # (time_steps, n_features)

    # SALib problem—bounds are [0,1] (scaled)
    problem = {
        'num_vars': n_features,
        'names': feature_names,
        'bounds': [[0.0, 1.0]] * n_features
    }

    # Saltelli sampling
    np.random.seed(random_seed)
    param_values = sobol_sample.sample(problem, n_samples, calc_second_order=calc_second_order)
    N = param_values.shape[0]

    # Build batch of sequences for prediction
    # For each row x (len=n_features), set every time step's feature j to x[j]
    batch = np.repeat(baseline[None, ...], N, axis=0)  # (N, time_steps, n_features)
    for j in range(n_features):
        batch[:, :, j] = param_values[:, j].reshape(-1, 1)

    # Predict in one go
    y_pred = model.predict(batch, verbose=0).reshape(-1)

    # Sobol indices (analyze)
    Si = sobol_analyze.analyze(problem, y_pred, calc_second_order=calc_second_order, print_to_console=False)

    # --- Save CSVs ---
    s1 = np.asarray(Si['S1'])
    s1_conf = np.asarray(Si['S1_conf'])
    st = np.asarray(Si['ST'])
    st_conf = np.asarray(Si['ST_conf'])

    sobol_df = pd.DataFrame({
        'feature': feature_names,
        'S1': s1,
        'S1_conf': s1_conf,
        'ST': st,
        'ST_conf': st_conf
    }).sort_values('ST', ascending=False).reset_index(drop=True)

    sobol_csv_path = os.path.join(GSA_dir, 'sobol_indices.csv')
    sobol_df.to_csv(sobol_csv_path, index=False)

    # ---------- Optional S2 (pairwise) ----------
    s2_df = None
    s2_csv_path = None
    s2_plot = None  # <- initialize to avoid UnboundLocalError

    if calc_second_order and 'S2' in Si and Si['S2'] is not None:
        s2_mat = np.asarray(Si['S2'])
        s2_conf_mat = np.asarray(Si['S2_conf'])
        pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                pairs.append({
                    'feature_i': feature_names[i],
                    'feature_j': feature_names[j],
                    'S2': s2_mat[i, j],
                    'S2_conf': s2_conf_mat[i, j]
                })
        s2_df = pd.DataFrame(pairs).sort_values('S2', ascending=False).reset_index(drop=True)
        s2_csv_path = os.path.join(GSA_dir, 'sobol_S2_pairs.csv')
        s2_df.to_csv(s2_csv_path, index=False)
    else:
        s2_df = None
        s2_csv_path = None

    # === Save "gsa_results.csv" in snippet schema (sorted by ST) ===
    gsa_results = (
        pd.DataFrame({
            'feature': feature_names,
            'S1': s1, 'S1_conf': s1_conf,
            'ST': st, 'ST_conf': st_conf
        })
        .sort_values('ST', ascending=False)
        .reset_index(drop=True)
    )
    gsa_results.to_csv(os.path.join(GSA_dir, "gsa_results.csv"), index=False)

    # === Plot EXACTLY like your snippet ===
    features_plot = gsa_results.head(15).copy()
    y_pos = np.arange(len(features_plot))

    # Total sensitivity (ST)
    plt.figure(figsize=(12, 8))
    plt.barh(y_pos, features_plot['ST'], xerr=features_plot['ST_conf'],
             alpha=0.7, color='steelblue', ecolor='black', capsize=5)
    plt.yticks(y_pos, features_plot['feature'])
    plt.xlabel('Total Sensitivity Index (ST)')
    plt.title('Global Sensitivity Analysis - Total Sensitivity Indices (Top 15)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st_plot = os.path.join(GSA_dir, "gsa_total_sensitivity.png")
    plt.savefig(st_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # First-order sensitivity (S1)
    plt.figure(figsize=(12, 8))
    plt.barh(y_pos, features_plot['S1'], xerr=features_plot['S1_conf'],
             alpha=0.7, color='lightcoral', ecolor='black', capsize=5)
    plt.yticks(y_pos, features_plot['feature'])
    plt.xlabel('First-Order Sensitivity Index (S1)')
    plt.title('Global Sensitivity Analysis - First-Order Sensitivity Indices (Top 15)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    s1_plot = os.path.join(GSA_dir, "gsa_first_order_sensitivity.png")
    plt.savefig(s1_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # Combined S1 vs ST (grouped bars; snippet exact)
    plt.figure(figsize=(14, 10))
    x = np.arange(len(features_plot))  # kept to mirror snippet
    width = 0.35
    plt.barh(y_pos - width / 2, features_plot['S1'], width,
             label='First-Order (S1)', alpha=0.7, color='lightcoral')
    plt.barh(y_pos + width / 2, features_plot['ST'], width,
             label='Total (ST)', alpha=0.7, color='steelblue')
    plt.yticks(y_pos, features_plot['feature'])
    plt.xlabel('Sensitivity Index')
    plt.title('Global Sensitivity Analysis - First-Order vs Total Sensitivity Indices')
    plt.legend(loc='upper right')  # <-- top-right legend
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    combined_plot = os.path.join(GSA_dir, "gsa_combined_sensitivity.png")
    plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # Keep your earlier S2 CSV creation (if any). Return updated paths:
    return {
        'sobol_csv': os.path.join(GSA_dir, "gsa_results.csv"),
        's2_csv': s2_csv_path,
        'plots': {
            'ST_bar': st_plot,
            'S1_bar': s1_plot,
            'S1_ST_combined': combined_plot,
            'S2_heatmap': None  # unchanged; add later if you render a heatmap
        }
    }

    # Optional: Top 20 S2 heatmap (if available)
    if s2_df is not None and len(s2_df) > 0:
        top = s2_df.head(min(20, len(s2_df))).copy()
        # Create a square matrix for the top features that appear in pairs
        uniq_feats = sorted(set(top['feature_i']).union(set(top['feature_j'])))
        mat = pd.DataFrame(0.0, index=uniq_feats, columns=uniq_feats)
        for _, r in top.iterrows():
            mat.loc[r['feature_i'], r['feature_j']] = r['S2']
            mat.loc[r['feature_j'], r['feature_i']] = r['S2']

        plt.figure(figsize=(10, 8))
        im = plt.imshow(mat.values, aspect='auto')
        plt.colorbar(im, label='S2')
        plt.xticks(range(len(uniq_feats)), uniq_feats, rotation=75, ha='right')
        plt.yticks(range(len(uniq_feats)), uniq_feats)
        plt.title('Top Pairwise Second-order Sobol Indices (S2)')
        plt.tight_layout()
        s2_plot = os.path.join(GSA_dir, 'sobol_S2_top_heatmap.png')
        plt.savefig(s2_plot, dpi=300, bbox_inches='tight'); plt.close()

    # Return paths for logging
    return {
        'sobol_csv': sobol_csv_path,
        's2_csv': s2_csv_path,
        'plots': {
            'ST_bar': st_plot,
            'S1_bar': s1_plot,
            'S2_heatmap': s2_plot if (calc_second_order and s2_df is not None and len(s2_df) > 0) else None
        }
    }


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)

# -----------------------------
# Logging
# -----------------------------
log_file_path = os.path.join(output_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
original_stdout = sys.stdout

with open(log_file_path, 'w') as log_file:
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print("Starting salinity prediction model training")
        print(f"Log file: {log_file_path}")
        print(f"Current time: {datetime.now()}\n")

        # -----------------------------
        # Load / basic prep/ Give your own path where the data used is present
        # -----------------------------
        sheet_name = "Sheet1"
        #basin_name   = "Galil West"  # Coast, Yarkatan, Negev and Arava, Sea of galilee, mountin east, Carmel, Galil West
        excelFilePath = r'E:\Python\India Project\GWS_Model info\Dataset_to_share.xlsx' # used in my case
        df = pd.read_excel(excelFilePath, sheet_name)

        basin_data = df.copy()
        print(f"Found {len(basin_data)} samples for all basins")

        basin_data['Year'] = pd.to_datetime(basin_data['Year'], format='%Y')
        basin_data = basin_data.sort_values(['Drill', 'Year'])

        random.seed(42); np.random.seed(42); tf.random.set_seed(42)

        # -----------------------------
        # Features / target
        # -----------------------------
        basin_data = basin_data[basin_data['Salinity - Cl (mg/l)'] <= 4000]

        # Keep raw basin labels for weighting (then drop dummies of basin)
        basin_labels_all = basin_data['Hydrological Basin'].astype(str).values

        X = basin_data.drop(
            columns=[
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
                "Aquifer",
            ]
        )

        X = pd.get_dummies(X, drop_first=True)
        # drop one-hot basin indicators if present (we'll weight by the raw labels instead)
        X = X.drop(columns=[c for c in X.columns if c.startswith("Hydrological Basin_")], errors="ignore")

        feature_names = list(X.columns)
        y = basin_data['Salinity - Cl (mg/l)'].values.reshape(-1, 1)
        drill_ids = basin_data['Drill'].values
        years = basin_data['Year'].dt.year.values

        print(f"Created {len(X)} samples with shape {X.shape}")

        # -----------------------------
        # Sequences (keep basin labels per target)
        # -----------------------------
        def create_sequences_grouped(X, y, drill_ids, years, basins, time_steps=5):
            X_seq, y_seq, groups_seq, years_seq, basins_seq = [], [], [], [], []
            unique_drills = np.unique(drill_ids)
            for drill_id in unique_drills:
                mask = (drill_ids == drill_id)
                X_loc, y_loc = X[mask], y[mask]
                yr_loc, bs_loc = years[mask], basins[mask]
                if len(X_loc) <= time_steps:
                    continue
                for i in range(len(X_loc) - time_steps):
                    X_seq.append(X_loc[i:i + time_steps])
                    y_seq.append(y_loc[i + time_steps])
                    groups_seq.append(drill_id)
                    years_seq.append(yr_loc[i + time_steps])
                    basins_seq.append(bs_loc[i + time_steps])
            return np.array(X_seq), np.array(y_seq), np.array(groups_seq), np.array(years_seq), np.array(basins_seq)

        time_steps = 5
        X_raw = X.values.astype(np.float32)
        X_seq_raw, y_seq, group_ids, target_years, basin_seq = create_sequences_grouped(
            X_raw, y, drill_ids, years, basins=basin_labels_all, time_steps=time_steps
        )
        print(f"Created {len(X_seq_raw)} sequences with shape {X_seq_raw.shape}")

        # -----------------------------
        # Time split (last 6 years = test)
        # -----------------------------
        test_years = sorted(np.unique(target_years))[-6:]
        print(f"Using years {test_years} as test set")
        test_mask = np.isin(target_years, test_years)

        X_train_raw, X_test_raw = X_seq_raw[~test_mask], X_seq_raw[test_mask]
        y_train, y_test = y_seq[~test_mask], y_seq[test_mask]
        train_groups, test_groups = group_ids[~test_mask], group_ids[test_mask]
        basin_train, basin_test = basin_seq[~test_mask], basin_seq[test_mask]

        X_train_raw = np.where(np.isfinite(X_train_raw), X_train_raw, np.nan).astype(np.float32)
        X_test_raw = np.where(np.isfinite(X_test_raw), X_test_raw, np.nan).astype(np.float32)

        print(f"Train set size: {len(X_train_raw)}, Test set size: {len(X_test_raw)}")
        print(f"Train years: {sorted(np.unique(target_years[~test_mask]))}")
        print(f"Test years: {sorted(np.unique(target_years[test_mask]))}")

        # -----------------------------
        # Preprocess (fit on TRAIN only)
        # -----------------------------
        scaler = MinMaxScaler()
        imputer = SimpleImputer(strategy='median')

        n_features = X_train_raw.shape[2]
        X_train_flat = X_train_raw.reshape(-1, n_features)
        X_test_flat = X_test_raw.reshape(-1, n_features)

        X_train_imputed_flat = imputer.fit_transform(X_train_flat)
        X_test_imputed_flat = imputer.transform(X_test_flat)

        scaler.fit(X_train_imputed_flat)
        X_train = scaler.transform(X_train_imputed_flat).reshape(X_train_raw.shape)
        X_test = scaler.transform(X_test_imputed_flat).reshape(X_test_raw.shape)

        # Optional: export preprocessed full table (row-level)
        X_all_imputed = imputer.transform(X.values.astype(np.float32))
        X_scaled_full = scaler.transform(X_all_imputed)
        pd.DataFrame(X_scaled_full, columns=X.columns).to_csv(
            os.path.join(output_dir, "preprocessed_data.csv"), index=False
        )

        # -----------------------------
        # Model search (GroupKFold CV)
        # -----------------------------
        optimizers = ['adam']
        hidden_units_list = [50, 100]
        activations = ['relu']

        all_metrics = []
        best_results = {}

        for optimizer in optimizers:
            print(f"\n=== Evaluating optimizer: {optimizer} ===")
            best_cv_r2 = float("-inf")
            best_hparams = None

            for hidden_units in hidden_units_list:
                for activation in activations:
                    print(f"\nTesting with {hidden_units} LSTM units and {activation} activation")

                    gkf = GroupKFold(n_splits=5)
                    fold_metrics = []
                    fold_best_epochs = []

                    for fold, (train_index, val_index) in enumerate(gkf.split(X_train, y_train, groups=train_groups), 1):
                        model = Sequential([
                            Input(shape=(time_steps, n_features)),
                            LSTM(hidden_units, activation=activation, return_sequences=True),
                            Dropout(0.2),
                            LSTM(hidden_units // 2, activation=activation),
                            Dropout(0.2),
                            Dense(1, activation='linear')
                        ])

                        if optimizer == 'sgd':
                            optimizer_instance = tf.keras.optimizers.SGD(
                                learning_rate=0.001, momentum=0.9, nesterov=True, clipnorm=1.0
                            )
                        else:
                            optimizer_instance = tf.keras.optimizers.get(optimizer)

                        model.compile(optimizer=optimizer_instance, loss='mse', metrics=['mae'])

                        es = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=10, restore_best_weights=True
                        )

                        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                        train_drills = np.unique(train_groups[train_index])
                        val_drills = np.unique(val_groups := train_groups[val_index])
                        assert len(set(train_drills) & set(val_drills)) == 0, \
                            "GroupKFold failed - overlapping drills between train and validation sets"

                        # ---- NEW: per-fold basin weights (train + val)
                        basins_tr_fold = basin_train[train_index]
                        basins_va_fold = basin_train[val_index]
                        sw_tr = fixed_basin_weights_for(basins_tr_fold)
                        sw_va = fixed_basin_weights_for(basins_va_fold)

                        history = model.fit(
                            X_train_fold, y_train_fold,
                            validation_data=(X_val_fold, y_val_fold, sw_va),  # weighted val
                            epochs=100,
                            batch_size=64,
                            callbacks=[es],
                            verbose=0,
                            sample_weight=sw_tr  # weighted train
                        )

                        best_epoch = es.stopped_epoch - es.patience + 1
                        if best_epoch < 0:
                            best_epoch = len(history.history['val_loss']) - 1
                        fold_best_epochs.append(best_epoch)

                        # Unweighted CV metrics (kept for comparability)
                        y_val_pred = model.predict(X_val_fold, verbose=0).ravel()
                        r2 = r2_score(y_val_fold.ravel(), y_val_pred)
                        rmse = safe_rmse(y_val_fold.ravel(), y_val_pred)  # <- change
                        mae = mean_absolute_error(y_val_fold.ravel(), y_val_pred)

                        r2_w = weighted_r2(y_val_fold.ravel(), y_val_pred, sw_va)
                        rmse_w = safe_rmse(y_val_fold.ravel(), y_val_pred, sample_weight=sw_va)  # <- change
                        mae_w = mean_absolute_error(y_val_fold.ravel(), y_val_pred, sample_weight=sw_va)

                        fold_metrics.append({
                            'fold': fold,
                            'r2': r2, 'rmse': rmse, 'mae': mae,
                            'r2_w': r2_w, 'rmse_w': rmse_w, 'mae_w': mae_w,
                            'best_epoch': best_epoch,
                            'train_drills': len(train_drills),
                            'val_drills': len(val_drills)
                        })

                        print(
                            f"  Fold {fold}: "
                            f"R²={r2:.4f} (w {r2_w:.4f}), "
                            f"RMSE={rmse:.4f} (w {rmse_w:.4f}), "
                            f"MAE={mae:.4f} (w {mae_w:.4f}) "
                            f"(best epoch: {best_epoch})"
                        )
                        print(f"    Train drills: {len(train_drills)}, Val drills: {len(val_drills)}")

                        # Loss curves
                        plt.figure(figsize=(10, 6))
                        plt.plot(history.history['loss'], label='Training Loss')
                        plt.plot(history.history['val_loss'], label='Validation Loss')
                        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
                        plt.title(
                            f'Training vs Validation Loss\nOptimizer: {optimizer}, Units: {hidden_units}, Activation: {activation}, Fold: {fold}')
                        plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend()
                        loss_plot_path = os.path.join(
                            loss_plots_dir, f'loss_{optimizer}_{hidden_units}units_{activation}_fold{fold}.png'
                        )
                        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight'); plt.close()

                    # CV selection is still by unweighted R² (change if you prefer weighted)
                    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
                    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
                    avg_mae = np.mean([m['mae'] for m in fold_metrics])
                    avg_best_epoch = int(np.mean(fold_best_epochs))

                    all_metrics.append({
                        'optimizer': optimizer,
                        'hidden_units': hidden_units,
                        'activation': activation,
                        'avg_r2': avg_r2,
                        'avg_rmse': avg_rmse,
                        'avg_mae': avg_mae,
                        'fold_metrics': fold_metrics,
                        'best_epoch': avg_best_epoch
                    })

                    if avg_r2 > best_cv_r2:
                        best_cv_r2 = avg_r2
                        best_hparams = {
                            'hidden_units': hidden_units,
                            'activation': activation,
                            'best_epoch': avg_best_epoch,
                            'fold_metrics': fold_metrics
                        }

            # -----------------------------
            # Final model on all TRAIN (weighted)
            # -----------------------------
            print("\nTraining final model on all training data with best CV params...")
            final_model = Sequential([
                Input(shape=(time_steps, n_features)),
                LSTM(best_hparams['hidden_units'], activation=best_hparams['activation'], return_sequences=True),
                Dropout(0.2),
                LSTM(best_hparams['hidden_units'] // 2, activation=best_hparams['activation']),
                Dropout(0.2),
                Dense(1, activation='linear')
            ])
            if optimizer == 'sgd':
                optimizer_instance = tf.keras.optimizers.SGD(
                    learning_rate=0.001, momentum=0.9, nesterov=True, clipnorm=1.0
                )
            else:
                optimizer_instance = tf.keras.optimizers.get(optimizer)
            final_model.compile(optimizer=optimizer_instance, loss='mse', metrics=['mae'])

            final_epochs = max(1, best_hparams['best_epoch'])
            sw_train_final = fixed_basin_weights_for(basin_train)

            final_model.fit(
                X_train, y_train,
                epochs=final_epochs,
                batch_size=64,
                verbose=0,
                sample_weight=sw_train_final  # weighted train
            )

            # -----------------------------
            # Test evaluation (weighted headline + unweighted)
            # -----------------------------
            y_test_pred = final_model.predict(X_test, verbose=0).ravel()
            sw_test = fixed_basin_weights_for(basin_test)

            # weighted (headline)
            test_r2_w = weighted_r2(y_test.ravel(), y_test_pred, sw_test)
            test_rmse_w = safe_rmse(y_test.ravel(), y_test_pred, sample_weight=sw_test)  # <- change
            test_mae_w = mean_absolute_error(y_test.ravel(), y_test_pred, sample_weight=sw_test)

            # unweighted (secondary)
            test_r2 = r2_score(y_test.ravel(), y_test_pred)
            test_rmse = safe_rmse(y_test.ravel(), y_test_pred)  # <- change
            test_mae = mean_absolute_error(y_test.ravel(), y_test_pred)

            print(f"\n  Test Set Performance (final model):")
            print(f"  Weighted  -> R^2: {test_r2_w:.4f}, RMSE: {test_rmse_w:.4f}, MAE: {test_mae_w:.4f}")
            print(f"  Unweighted-> R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

            best_results[optimizer] = {
                'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae,
                'r2_w': test_r2_w, 'rmse_w': test_rmse_w, 'mae_w': test_mae_w,
                'model': final_model,
                'params': {
                    'optimizer': optimizer,
                    'hidden_units': best_hparams['hidden_units'],
                    'activation': best_hparams['activation'],
                    'best_epoch': best_hparams['best_epoch'],
                    'fold_metrics': best_hparams['fold_metrics'],
                    'test_years': test_years
                }
            }

            # Save metrics CSV (include weighted test cols)
            metrics_file = os.path.join(output_dir, "all_metrics.csv")
            with open(metrics_file, 'w', newline='') as csvfile:
                fieldnames = ['optimizer', 'hidden_units', 'activation',
                              'avg_r2', 'avg_rmse', 'avg_mae',
                              'best_epoch', 'fold_metrics',
                              'test_r2', 'test_rmse', 'test_mae',
                              'test_r2_w', 'test_rmse_w', 'test_mae_w']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for metric in all_metrics:
                    # append the final test metrics from best_results for this optimizer (if present)
                    br = best_results.get(metric['optimizer'], {})
                    row = metric.copy()
                    row.update({
                        'test_r2': br.get('r2', np.nan),
                        'test_rmse': br.get('rmse', np.nan),
                        'test_mae': br.get('mae', np.nan),
                        'test_r2_w': br.get('r2_w', np.nan),
                        'test_rmse_w': br.get('rmse_w', np.nan),
                        'test_mae_w': br.get('mae_w', np.nan),
                    })
                    writer.writerow(row)
            print(f"\nSaved all metrics to: {metrics_file}")

            # -----------------------------
            # RFE (NEW): model-based recursive feature elimination on TRAIN set
            # -----------------------------
            rfe_dir = os.path.join(output_dir, "RFE")
            os.makedirs(rfe_dir, exist_ok=True)

            # Use the CV-best hyperparams you just found for the final model
            rfe_epochs = min(max(10, best_hparams['best_epoch']), 40)  # keeps it reasonable
            rfe_results = run_lstm_rfe(
                X_train_full=X_train,  # (N, time_steps, n_features) scaled + imputed
                y_train=y_train,
                groups=train_groups,
                basins=basin_train,
                feature_names=feature_names,
                time_steps=time_steps,
                optimizer=optimizer,
                hidden_units=best_hparams['hidden_units'],
                activation=best_hparams['activation'],
                out_dir=rfe_dir,
                max_remove=None,  # remove until min_features remains
                min_features=5,  # <-- adjust to stop earlier if desired
                epochs_per_eval=rfe_epochs,
                folds=3,  # 3-fold GroupKFold for speed; set to 5 if you want
            )

            print("\n[RFE] Top kept features (rank=1):")
            print(rfe_results['kept_features'])

            # -----------------------------
            # Sobol Global Sensitivity (NEW)
            # -----------------------------
            print("\nStarting Sobol global sensitivity analysis...")

            # Choose base sample size; total evals ~ n_samples*(2D+2)
            # If features are many, you can start smaller (e.g., 256) and scale up.
            sobol_base_n = 25000

            sobol_paths = run_sobol_analysis(
                model=final_model,
                X_train_scaled=X_train,  # scaled & imputed, shape (N, time_steps, n_features)
                feature_names=feature_names,
                time_steps=time_steps,
                n_samples=sobol_base_n,
                GSA_dir=GSA_dir,
                random_seed=42,
                calc_second_order=True
            )

            print("Sobol outputs:")
            print(f"  S1/ST CSV: {sobol_paths['sobol_csv']}")
            if sobol_paths['s2_csv'] is not None:
                print(f"  S2 pairs CSV: {sobol_paths['s2_csv']}")
            print("  Plots:")
            for k, v in sobol_paths['plots'].items():
                if v: print(f"    {k}: {v}")

            # -----------------------------
            # Results + residuals + SHAP (unchanged except titles)
            # -----------------------------
            print("\n=== Final Results ===")
            for optimizer, result in best_results.items():
                print(f"\nOptimizer: {optimizer}")
                print(f"  Best LSTM Units: {result['params']['hidden_units']}")
                print(f"  Best Activation: {result['params']['activation']}")
                print(f"  Weighted Test -> R²: {result['r2_w']:.4f}, RMSE: {result['rmse_w']:.4f}, MAE: {result['mae_w']:.4f}")
                print(f"  Unweighted    -> R²: {result['r2']:.4f}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
                print(f"  Best Epoch: {result['params']['best_epoch']}")
                print(f"  Test Years: {result['params']['test_years']}")
                print("  Fold Metrics:")
                for fm in result['params']['fold_metrics']:
                    print(f"    Fold {fm['fold']}: R² {fm['r2']:.4f} (w {fm['r2_w']:.4f}), "
                          f"RMSE {fm['rmse']:.4f} (w {fm['rmse_w']:.4f}), "
                          f"MAE {fm['mae']:.4f} (w {fm['mae_w']:.4f}) | "
                          f"best_epoch {fm['best_epoch']} | "
                          f"train_drills {fm['train_drills']} | val_drills {fm['val_drills']}")

                y_pred = result['model'].predict(X_test, verbose=0).ravel()
                residuals = y_test.ravel() - y_pred

                # Build metrics dict (unweighted, to match snippet box)
                metrics_snip = {
                    'r2': r2_score(y_test.ravel(), y_pred),
                    'mae': mean_absolute_error(y_test.ravel(), y_pred),
                    'mse': mean_squared_error(y_test.ravel(), y_pred),
                }

                # Snippet-exact plot
                _ = create_actual_vs_predicted_plot(
                    y_test=y_test.ravel().astype(float),
                    y_pred=y_pred.astype(float),
                    metrics=metrics_snip,
                    output_dir=output_dir,
                    model_name=f"LSTM",
                    filename=f"actual_vs_predicted_{optimizer}.png"
                )
                print(f"  Saved plot to: {os.path.join(output_dir, f'actual_vs_predicted_{optimizer}.png')}")

                # Residuals: scatter
                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title(f'Residuals vs Predicted Values ({optimizer}) - Test Set')
                plt.xlabel('Predicted Salinity'); plt.ylabel('Residuals')
                resid_pred_path = os.path.join(residual_plots_dir, f'residuals_vs_predicted_{optimizer}.png')
                plt.savefig(resid_pred_path, dpi=300, bbox_inches='tight'); plt.close()
                print(f"  Saved residuals vs predicted plot to: {resid_pred_path}")

                # Weighted metrics (already printed above) — recompute here for clarity if needed
                sw_test_local = fixed_basin_weights_for(basin_test)
                y_pred_r = y_pred.ravel(); y_test_r = y_test.ravel()
                test_rmse_w = safe_rmse(y_test_r, y_pred_r, sample_weight=sw_test_local)  # <- change
                test_mae_w = mean_absolute_error(y_test_r, y_pred_r, sample_weight=sw_test_local)
                test_r2_w = weighted_r2(y_test_r, y_pred_r, sw_test_local)
                print(f"  Weighted R²: {test_r2_w:.4f} | Weighted RMSE: {test_rmse_w:.4f} | Weighted MAE: {test_mae_w:.4f}")

                # Residuals histogram
                plt.figure(figsize=(10, 6))
                plt.hist(residuals, bins=50, edgecolor='k')
                plt.title(f'Distribution of Residuals ({optimizer}) - Test Set')
                plt.xlabel('Residuals'); plt.ylabel('Frequency')
                resid_hist_path = os.path.join(residual_plots_dir, f'residuals_histogram_{optimizer}.png')
                plt.savefig(resid_hist_path, dpi=300, bbox_inches='tight'); plt.close()
                print(f"  Saved residuals histogram to: {resid_hist_path}")

                # ACF
                plt.figure(figsize=(10, 6))
                plot_acf(residuals, lags=20, alpha=0.05)
                plt.title(f'Residuals Autocorrelation ({optimizer}) - Test Set')
                plt.xlabel('Lag'); plt.ylabel('Autocorrelation')
                resid_acf_path = os.path.join(residual_plots_dir, f'residuals_acf_{optimizer}.png')
                plt.savefig(resid_acf_path, dpi=300, bbox_inches='tight'); plt.close()
                print(f"  Saved residuals ACF plot to: {resid_acf_path}")

                # -----------------------------
                # SHAP (kept as in your version)
                # -----------------------------
                print("\nStarting SHAP analysis with time-step aggregation...", flush=True)
                old_stdout = sys.stdout
                sys.stdout = sys.__stdout__
                try:
                    def _to_n_samples_time_features(sv, test_samples, time_steps, n_features):
                        sv = np.array(sv)
                        if isinstance(sv, list):
                            sv = sv[0]
                        if sv.ndim == 4 and sv.shape[0] == 1 and sv.shape[1] == test_samples.shape[0]:
                            sv = sv[0]
                        if sv.ndim == 4 and sv.shape[-1] == 1 and sv.shape[0] == test_samples.shape[0]:
                            sv = sv[..., 0]
                        sv = np.squeeze(sv)
                        if sv.ndim == 3:
                            if sv.shape[0] == test_samples.shape[0] and sv.shape[1] == time_steps and sv.shape[2] == n_features:
                                return sv
                            if sv.shape[0] == test_samples.shape[0] and sv.shape[1] == n_features and sv.shape[2] == time_steps:
                                return np.transpose(sv, (0, 2, 1))
                            if sv.shape[0] == time_steps and sv.shape[1] == n_features and sv.shape[2] == test_samples.shape[0]:
                                return np.transpose(sv, (2, 0, 1))
                        raise ValueError(f"Unexpected SHAP shape {sv.shape}. "
                                         f"Expected something compatible with (n_samples, {time_steps}, {n_features}).")

                    background = X_train[np.random.choice(X_train.shape[0], min(50, len(X_train)), replace=False)]
                    test_samples = X_test

                    explainer = shap.GradientExplainer(result['model'], background)
                    shap_values = explainer.shap_values(test_samples)

                    n_features_local = len(feature_names)
                    shap_values_3d = _to_n_samples_time_features(
                        shap_values, test_samples, time_steps, n_features_local
                    )

                    aggregated_shap = np.mean(shap_values_3d, axis=1)
                    test_samples_flat = test_samples.reshape(test_samples.shape[0], -1)
                    base_feature_names = feature_names

                    print("Exporting SHAP CSVs...")
                    metadata_path = os.path.join(shap_csv_dir, 'test_metadata.csv')
                    test_years_aligned = target_years[test_mask]
                    test_groups_aligned = test_groups
                    if not os.path.exists(metadata_path):
                        meta_df = pd.DataFrame({
                            'sample_index': np.arange(test_samples.shape[0]),
                            'group_id': test_groups_aligned,
                            'year': test_years_aligned
                        })
                        meta_df.to_csv(metadata_path, index=False)

                    agg_df = pd.DataFrame(aggregated_shap, columns=base_feature_names)
                    agg_df.insert(0, 'sample_index', np.arange(aggregated_shap.shape[0]))
                    agg_df.insert(1, 'group_id', test_groups_aligned)
                    agg_df.insert(2, 'year', test_years_aligned)
                    agg_path = os.path.join(shap_csv_dir, f'shap_aggregated_values_{_slug(optimizer)}.csv')
                    agg_df.to_csv(agg_path, index=False)

                    mean_abs = np.abs(aggregated_shap).mean(axis=0)
                    mean_signed = aggregated_shap.mean(axis=0)
                    global_imp_df = pd.DataFrame({
                        'feature': base_feature_names,
                        'mean_abs_shap': mean_abs,
                        'mean_signed_shap': mean_signed
                    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
                    global_imp_path = os.path.join(shap_csv_dir, f'shap_global_importance_{_slug(optimizer)}.csv')
                    global_imp_df.to_csv(global_imp_path, index=False)

                    mean_shap_per_timestep = np.mean(np.abs(shap_values_3d), axis=0)
                    ts_cols = [f't-{time_steps - t - 1}' for t in range(time_steps)]
                    heatmap_df = pd.DataFrame(mean_shap_per_timestep.T, index=base_feature_names, columns=ts_cols)
                    heatmap_df.insert(0, 'feature', heatmap_df.index)
                    heatmap_df.reset_index(drop=True, inplace=True)
                    heatmap_path_csv = os.path.join(shap_csv_dir, f'shap_time_step_importance_{_slug(optimizer)}.csv')
                    heatmap_df.to_csv(heatmap_path_csv, index=False)

                    mean_abs_agg_shap = pd.Series(mean_abs, index=base_feature_names).sort_values(ascending=False)
                    top_agg_features = mean_abs_agg_shap.head(5).index.tolist()
                    top_path = os.path.join(shap_csv_dir, f'shap_top_features_{_slug(optimizer)}.csv')
                    top_df = pd.DataFrame({
                        'feature': mean_abs_agg_shap.index,
                        'mean_abs_shap': mean_abs_agg_shap.values,
                        'rank': np.arange(1, len(mean_abs_agg_shap) + 1)
                    })
                    top_df.to_csv(top_path, index=False)

                    dep_dir = os.path.join(shap_csv_dir, 'dependence_data')
                    os.makedirs(dep_dir, exist_ok=True)
                    if len(top_agg_features) >= 2:
                        f1, f2 = top_agg_features[0], top_agg_features[1]
                        i1, i2 = base_feature_names.index(f1), base_feature_names.index(f2)
                        dep_df = pd.DataFrame({
                            'sample_index': np.arange(aggregated_shap.shape[0]),
                            f'shap_{f1}': aggregated_shap[:, i1],
                            f'{f1}_value_t0': test_samples_flat[:, i1],
                            f'{f2}_value_t0': test_samples_flat[:, i2]
                        })
                        dep_df.to_csv(os.path.join(dep_dir, f'dependence_{_slug(f1)}_vs_{_slug(f2)}_{_slug(optimizer)}.csv'), index=False)

                    # Plots
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        aggregated_shap,
                        test_samples_flat[:, :n_features_local],
                        feature_names=base_feature_names,
                        plot_type="bar",
                        show=False,
                        max_display=len(base_feature_names)
                    )
                    plt.title(f"Aggregated SHAP Feature Importance (Across All Time Steps)\n({optimizer}) - Test Set")
                    aggregated_path = os.path.join(shap_plots_dir, f'shap_aggregated_{optimizer}.png')
                    plt.savefig(aggregated_path, dpi=300, bbox_inches='tight'); plt.close()
                    print(f"Saved aggregated SHAP plot to: {aggregated_path}")

                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        aggregated_shap,
                        test_samples_flat[:, :n_features_local],
                        feature_names=base_feature_names,
                        plot_type="dot",
                        show=False,
                        max_display=len(base_feature_names)
                    )
                    plt.title(f"Aggregated SHAP Values (Direction Preserved)\n({optimizer}) - Test Set")
                    plt.savefig(os.path.join(shap_plots_dir, f'shap_aggregated_directional_{optimizer}.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()

                    plt.figure(figsize=(12, 8))
                    plt.imshow(mean_shap_per_timestep.T, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Mean Absolute SHAP Value')
                    plt.xlabel('Time Step (t-n)'); plt.ylabel('Feature')
                    plt.yticks(range(len(base_feature_names)), base_feature_names)
                    plt.xticks(range(time_steps), [f't-{time_steps - t - 1}' for t in range(time_steps)])
                    plt.title(f"SHAP Importance Across Time Steps\n({optimizer}) - Test Set")
                    heatmap_path = os.path.join(shap_plots_dir, f'shap_time_heatmap_{optimizer}.png')
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight'); plt.close()
                    print(f"Saved time-step heatmap to: {heatmap_path}")

                finally:
                    sys.stdout = old_stdout

    finally:
        sys.stdout = original_stdout

print(f"\nAll output has been saved to: {log_file_path}")
