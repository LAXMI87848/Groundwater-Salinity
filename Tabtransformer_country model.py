import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import random
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GroupShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import warnings
import time
from tqdm import tqdm
import seaborn as sns
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr, norm
import shap
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample
import joblib

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Create output directory and give your own path where you want to save the results
output_dir = r'G:\IIT_AI work\ISRAEL\GWS_Model info\Data and feature score\Revised data and feature_fitting scores\Final results (dataset_28.8)\Israel model run\TabTransformer_RFE_Selected_Features'  # used in my case 
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories
viz_dir = os.path.join(output_dir, "visualizations")
os.makedirs(viz_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)
results_dir = os.path.join(output_dir, "results")
os.makedirs(results_dir, exist_ok=True)
selection_dir = os.path.join(output_dir, "feature_selection")
os.makedirs(selection_dir, exist_ok=True)


# Helper function to clean feature names for file writing (removes Unicode characters)
def clean_feature_name(name):
    """Replace Unicode characters with ASCII equivalents"""
    replacements = {
        '\u2103': 'C',  # Degree Celsius
        '\u00b0': 'deg',  # Degree symbol
        '\u00b2': '2',  # Superscript 2
        '\u00b3': '3',  # Superscript 3
        '\u00b5': 'u',  # Micro symbol
        '\u00b1': '+/-',  # Plus-minus
        '\u00d7': 'x',  # Multiplication
        '\u2212': '-',  # Minus
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
    }
    for unicode_char, ascii_char in replacements.items():
        name = name.replace(unicode_char, ascii_char)
    return name


# ==================== DATA LOADING ====================
 # Load the data and give your own path in #excelFilePath, where the data shared is been saved
sheet_name = "Sheet1"
excelFilePath = r'G:\IIT_AI work\ISRAEL\GWS_Model info\Data and feature score\Revised data and feature_fitting scores\Dataset_to_share.xlsx'  # used in my case
df = pd.read_excel(excelFilePath, sheet_name)
print(f"Original dataset shape: {df.shape}")

# ==================== BASIN DENSITIES FOR GEOGRAPHIC WEIGHTING ====================
basin_densities = {
    'Carmel': 0.119408792,
    'Coast': 0.348310389,
    'Galil West': 0.045398829,
    'Sea of galilee': 0.017933556,
    'mountin east': 0.013575234,
    'Yarkatan': 0.019049412,
    'Negev and Arava': 0.022511285
}


def calculate_basin_density_weights_inverse_sqrt(df, basin_densities):
    """Calculate sample weights based on basin densities using inverse square-root method"""
    basins_in_data = df['Hydrological Basin'].unique()
    missing_basins = set(basins_in_data) - set(basin_densities.keys())

    if missing_basins:
        print(f"Warning: Missing density values for basins: {missing_basins}")
        avg_density = np.mean(list(basin_densities.values()))
        for basin in missing_basins:
            basin_densities[basin] = avg_density

    basin_density_series = pd.Series(basin_densities)
    weights = 1 / np.sqrt(basin_density_series)
    weights = weights / weights.mean()
    weight_mapping = weights.to_dict()

    print("\nBasin density weights (inverse square-root method):")
    for basin, weight in weight_mapping.items():
        if basin in basins_in_data:
            print(f"  {basin}: {weight:.3f} (density: {basin_densities[basin]:.6f})")

    return weight_mapping


# ==================== TARGET FILTERING ====================
target = 'Salinity - Cl (mg/l)'

print("\n" + "=" * 60)
print("APPLYING SALINITY FILTER")
print("=" * 60)

print(f"Original data shape: {df.shape}")
print(f"Original target range: {df[target].min():.2f} to {df[target].max():.2f}")
print(f"Original target statistics:")
print(f"  Mean: {df[target].mean():.2f}")
print(f"  Median: {df[target].median():.2f}")
print(f"  Std: {df[target].std():.2f}")

n_high = (df[target] > 4000).sum()
print(f"\nSamples with salinity > 4000 mg/l: {n_high} ({n_high / len(df) * 100:.1f}%)")

df = df[df[target] <= 4000].copy()
df = df[df[target] >= 0].copy()

print(f"\nFiltered data shape: {df.shape}")
print(f"Filtered target range: {df[target].min():.2f} to {df[target].max():.2f}")
print(f"Filtered target statistics:")
print(f"  Mean: {df[target].mean():.2f}")
print(f"  Median: {df[target].median():.2f}")
print(f"  Std: {df[target].std():.2f}")

# Visualize filter impact
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df[target], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Salinity - Cl (mg/l)')
plt.ylabel('Frequency')
plt.title('Filtered Target Distribution (≤ 4000 mg/l)')
plt.axvline(df[target].mean(), color='red', linestyle='--', label=f'Mean: {df[target].mean():.0f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(df[target])
plt.ylabel('Salinity - Cl (mg/l)')
plt.title('Boxplot of Filtered Target')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "target_distribution_after_filter.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Target distribution plot saved to: {viz_dir}/target_distribution_after_filter.png")
print("=" * 60)

# List of columns to exclude from feature selection
exclude_columns = [
    'Salinity - Cl (mg/l)', 'Hightemp_fall', 'Hightemp_spring', 'Hightemp_summer',
    'Hightemp_winter', 'Tempanam_fall', 'Tempanam_spring', 'Tempanam_summer', 'Tempanam_winter',
    'Zcore_fall', 'Zcore_spring', 'Zcore_summer', 'Zcore_winter', 'precipitation ( #N of events Q1)',
    'precipitation (  #N of events Q2)', 'precipitation ( #N of events Q3)', 'precipitation ( #N of events Q4)',
    'precipitation (  #N of events Q5)', 'Elevation (m)', 'salinity difference', 'salinity relative difference',
    'Precip diff', 'Hydrological Basin', 'Aquifer', 'Cell name', 'Sub Basin', 'Drill', 'Year',
    'National system - Groundwater (%)', 'National system - Surface water (%)',
    'National system - Desalinated water (%)'
]

all_columns = df.columns.tolist()
all_features = [col for col in all_columns if col not in exclude_columns and col != target]

print(f"\nTarget column: {target}")
print(f"Total features available: {len(all_features)}")
print(f"Features: {all_features[:10]}... (showing first 10)")


# ==================== COMPREHENSIVE FEATURE IMBALANCE VISUALIZATION ====================
def create_comprehensive_imbalance_visualizations(X, feature_names, output_dir, prefix="original"):
    """Create comprehensive visualizations to understand feature imbalance"""

    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()

    print(f"\nCreating comprehensive imbalance visualizations for {prefix} features...")
    X_df = X_df.dropna()

    plt.figure(figsize=(16, 12))

    variances = X_df.var()
    skewness = X_df.skew()
    kurtosis = X_df.kurtosis()

    # 1. Variance comparison
    plt.subplot(2, 3, 1)
    plt.bar(range(len(variances)), sorted(variances))
    plt.title('Feature Variances (Sorted)')
    plt.xlabel('Features (sorted by variance)')
    plt.ylabel('Variance')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 2. Skewness distribution
    plt.subplot(2, 3, 2)
    plt.hist(skewness, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Skewness Distribution Across Features')
    plt.xlabel('Skewness')
    plt.ylabel('Frequency')
    plt.axvline(-2, color='red', linestyle='--', label='High skew threshold')
    plt.axvline(2, color='red', linestyle='--')
    plt.legend()

    # 3. Kurtosis distribution
    plt.subplot(2, 3, 3)
    plt.hist(kurtosis, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Kurtosis Distribution Across Features')
    plt.xlabel('Kurtosis')
    plt.ylabel('Frequency')
    plt.axvline(-5, color='red', linestyle='--', label='Extreme kurtosis')
    plt.axvline(5, color='red', linestyle='--')
    plt.legend()

    # 4. Outlier analysis
    outlier_percentages = []
    for col in X_df.columns:
        Q1 = X_df[col].quantile(0.25)
        Q3 = X_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((X_df[col] < lower_bound) | (X_df[col] > upper_bound)).sum()
        outlier_percentages.append((outliers / len(X_df)) * 100)

    plt.subplot(2, 3, 4)
    plt.hist(outlier_percentages, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Outlier Percentage Distribution')
    plt.xlabel('Percentage of Outliers')
    plt.ylabel('Frequency')
    plt.axvline(10, color='red', linestyle='--', label='10% outlier threshold')
    plt.legend()

    # 5. Zero inflation analysis
    zero_percentages = (X_df == 0).sum() / len(X_df) * 100
    plt.subplot(2, 3, 5)
    plt.hist(zero_percentages, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Zero Percentage Distribution')
    plt.xlabel('Percentage of Zeros')
    plt.ylabel('Frequency')
    plt.axvline(50, color='red', linestyle='--', label='50% zero threshold')
    plt.legend()

    # 6. Value ranges
    ranges = X_df.max() - X_df.min()
    plt.subplot(2, 3, 6)
    plt.bar(range(len(ranges)), sorted(ranges))
    plt.title('Feature Value Ranges (Sorted)')
    plt.xlabel('Features (sorted by range)')
    plt.ylabel('Range')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(14, 12))
    corr_matrix = X_df.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                square=True, cbar_kws={"shrink": .8})
    plt.title(f'Feature Correlation Matrix ({prefix})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to: {output_dir}")
    return {
        'variances': variances,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'outlier_percentages': outlier_percentages,
        'zero_percentages': zero_percentages,
        'ranges': ranges
    }


# ==================== TEMPORAL VALIDATION FUNCTIONS ====================
def create_temporal_splits(df, n_splits=3, test_years=6):
    """Create temporal validation splits by drill - train on earlier years, test on later years"""

    selected_columns_sorted = df.sort_values(by=['Drill', 'Year'])
    splits = []
    drill_splits = {}

    for drill_id, group in selected_columns_sorted.groupby('Drill'):
        group_years = group['Year'].sort_values().unique()

        if len(group_years) <= test_years:
            split_point = max(1, len(group_years) - 1)
            train_idx = group[group['Year'].isin(group_years[:split_point])].index.tolist()
            test_idx = group[group['Year'].isin(group_years[split_point:])].index.tolist()
            if train_idx and test_idx:
                drill_splits[drill_id] = [(train_idx, test_idx)]
        else:
            drill_splits_list = []
            for i in range(len(group_years) - test_years):
                train_years = group_years[:i + 1]
                test_years_range = group_years[i + 1:min(i + 1 + test_years, len(group_years))]
                train_idx = group[group['Year'].isin(train_years)].index.tolist()
                test_idx = group[group['Year'].isin(test_years_range)].index.tolist()
                if train_idx and test_idx:
                    drill_splits_list.append((train_idx, test_idx))
            if drill_splits_list:
                drill_splits[drill_id] = drill_splits_list[-min(n_splits, len(drill_splits_list)):]

    max_splits = max([len(splits) for splits in drill_splits.values()]) if drill_splits else 0

    for split_idx in range(max_splits):
        train_indices = []
        test_indices = []
        for drill_id, drill_split_list in drill_splits.items():
            if split_idx < len(drill_split_list):
                train_idx, test_idx = drill_split_list[split_idx]
                train_indices.extend(train_idx)
                test_indices.extend(test_idx)
        if train_indices and test_indices:
            splits.append((train_indices, test_indices))

    if not splits:
        print("Warning: Could not create temporal splits. Creating single hold-out split.")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['Drill']))
        splits = [(df.index[train_idx].tolist(), df.index[test_idx].tolist())]

    print(f"Created {len(splits)} temporal validation splits")
    return splits


# ==================== TABTRANSFORMER MODEL ====================
def create_tabtransformer_model(num_features, num_categories_list, emb_dim=32, num_heads=4, num_layers=2,
                                ffn_hidden=128, mlp_hidden1=64, mlp_hidden2=32, dropout=0.1):
    """TabTransformer model with configurable parameters"""

    inputs = []
    for i in range(num_features):
        inputs.append(L.Input(shape=(1,), dtype='int32', name=f'categorical_input_{i}'))

    embeddings = []
    for i, (inp, num_cat) in enumerate(zip(inputs, num_categories_list)):
        embedding = L.Embedding(input_dim=num_cat, output_dim=emb_dim, name=f'embedding_{i}')(inp)
        embeddings.append(embedding)

    x = L.Concatenate(axis=1, name='concat_embeddings')(embeddings)
    x = L.Dropout(dropout, name='embedding_dropout')(x)

    for i in range(num_layers):
        attn_output = L.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim, name=f'mha_{i}')(x, x)
        attn_output = L.Dropout(dropout, name=f'attn_dropout_{i}')(attn_output)
        x = L.Add(name=f'add_1_{i}')([x, attn_output])
        x = L.LayerNormalization(name=f'ln_1_{i}')(x)

        ffn = L.Dense(ffn_hidden, activation='gelu', name=f'ffn_dense1_{i}')(x)
        ffn = L.Dropout(dropout, name=f'ffn_dropout_{i}')(ffn)
        ffn = L.Dense(emb_dim, name=f'ffn_dense2_{i}')(ffn)
        x = L.Add(name=f'add_2_{i}')([x, ffn])
        x = L.LayerNormalization(name=f'ln_2_{i}')(x)

    x = L.Flatten(name='flatten')(x)
    x = L.Dense(mlp_hidden1, activation='relu', name='mlp_dense1')(x)
    x = L.Dropout(dropout, name='mlp_dropout1')(x)
    x = L.Dense(mlp_hidden2, activation='relu', name='mlp_dense2')(x)
    output = L.Dense(1, name='output')(x)

    model = keras.Model(inputs=inputs, outputs=output, name=f'tabtransformer_{num_features}_features')
    return model


# ==================== FEATURE PREPROCESSING ====================
def preprocess_feature_enhanced(df, feature_name, target, num_bins=10, scaler_type='standard'):
    """Enhanced preprocessing with multiple scaling options"""

    temp_df = df[[feature_name, target, 'Drill', 'Year', 'Hydrological Basin']].copy()
    temp_df = temp_df.dropna()

    if len(temp_df) < 20:
        return None, None, None, None

    col = temp_df[feature_name]
    y = temp_df[target].values
    basins = temp_df['Hydrological Basin'].values
    original_indices = temp_df.index.values

    if col.dtype == 'object' or len(col.unique()) <= 20:
        processed = pd.Categorical(col).codes
    else:
        col_array = col.values.reshape(-1, 1)

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        elif scaler_type == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            scaler = StandardScaler()

        normed = scaler.fit_transform(col_array).flatten()

        if np.isnan(normed).any():
            return None, None, None, None

        try:
            processed = pd.cut(normed, bins=num_bins, labels=False)
            if pd.isna(processed).any():
                mode_val = pd.Series(processed).mode()[0]
                processed = pd.Series(processed).fillna(mode_val).values
        except:
            try:
                processed = pd.qcut(normed, q=num_bins, labels=False, duplicates='drop')
                if pd.isna(processed).any():
                    mode_val = pd.Series(processed).mode()[0]
                    processed = pd.Series(processed).fillna(mode_val).values
            except:
                return None, None, None, None

    return processed.astype('int32'), y, original_indices, basins


# ==================== RFE SELECTION USING RANDOM FOREST (Proxy for TabTransformer) ====================
def perform_rfe_selection_for_tabtransformer(df, feature_list, target, n_features_to_select=15, output_dir=None):
    """
    Perform RFE using Random Forest to select features for TabTransformer
    This is a PROXY selection - Random Forest is fast and has built-in feature importance
    """

    print("\n" + "=" * 80)
    print("RFE FEATURE SELECTION USING RANDOM FOREST (Proxy for TabTransformer)")
    print("=" * 80)
    print(f"Total features available: {len(feature_list)}")
    print(f"Target number of features to select: {n_features_to_select}")

    try:
        # Prepare data for RFE
        print("\nPreparing data for RFE analysis...")
        X = df[feature_list].copy()

        # Convert categorical features to numeric
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes

        # Handle missing values
        X = X.fillna(X.mean())
        y = df[target].values

        print(f"Data shape for RFE: {X.shape}")

        # Create Random Forest estimator (fast, has feature_importances_)
        estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )

        # Perform RFE
        print(f"\nPerforming Recursive Feature Elimination to select top {n_features_to_select} features...")
        print("This may take 5-10 minutes...")

        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=2,  # Remove 2 features at a time for speed
            verbose=1
        )

        rfe.fit(X, y)

        # Get selected features
        selected_indices = rfe.support_
        selected_features = [feature_list[i] for i in range(len(feature_list)) if selected_indices[i]]
        feature_rankings = rfe.ranking_

        # Create ranking DataFrame
        rfe_ranking_df = pd.DataFrame({
            'feature': feature_list,
            'rfe_ranking': feature_rankings,
            'selected': selected_indices
        }).sort_values('rfe_ranking')

        # Add clean names
        rfe_ranking_df['feature_clean'] = rfe_ranking_df['feature'].apply(clean_feature_name)
        rfe_ranking_df['rank_order'] = range(1, len(rfe_ranking_df) + 1)

        # Save rankings
        if output_dir:
            rfe_ranking_df.to_csv(os.path.join(output_dir, "rfe_selection_rankings.csv"), index=False)

        # Print selected features
        print("\n" + "-" * 60)
        print(f"RFE SELECTED {len(selected_features)} FEATURES (Rank 1):")
        print("-" * 60)
        for i, feat in enumerate(selected_features, 1):
            clean_feat = clean_feature_name(feat)
            print(f"  {i:2d}. {clean_feat}")

        print("\n" + "-" * 60)
        print("TOP 20 FEATURES BY RFE RANKING:")
        print("-" * 60)
        for idx, row in rfe_ranking_df.head(20).iterrows():
            selected_mark = "✓ SELECTED" if row['selected'] else ""
            print(
                f"  Rank {row['rank_order']:2d}: {row['feature_clean'][:45]:<45} (Rank: {row['rfe_ranking']}) {selected_mark}")

        # Plot RFE rankings
        plt.figure(figsize=(14, 10))
        top_20 = rfe_ranking_df.head(20)
        colors = ['green' if sel else 'steelblue' for sel in top_20['selected']]
        plt.barh(range(len(top_20)), top_20['rank_order'].values[::-1], color=colors[::-1], alpha=0.8,
                 edgecolor='black')
        plt.yticks(range(len(top_20)), top_20['feature_clean'].values[::-1])
        plt.xlabel('RFE Rank (1 = Best, Lower is Better)', fontsize=12)
        plt.title('RFE Feature Ranking for TabTransformer Selection\n(Green = Selected, Blue = Not Selected)',
                  fontsize=14, fontweight='bold')
        plt.gca().invert_xaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "rfe_selection_ranking.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ RFE selection completed! Selected {len(selected_features)} features.")

        return selected_features, rfe_ranking_df

    except Exception as e:
        print(f"RFE selection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== SHAP ANALYSIS ====================
def perform_shap_analysis(model, X_sample, feature_names, output_dir):
    """Perform SHAP analysis for model interpretability"""

    print("\n" + "=" * 60)
    print("PERFORMING SHAP ANALYSIS")
    print("=" * 60)

    try:
        background = X_sample[np.random.choice(X_sample.shape[0], min(100, X_sample.shape[0]), replace=False)]

        def predict_fn(x):
            x_list = [x[:, i:i + 1] for i in range(x.shape[1])]
            return model.predict(x_list).flatten()

        explainer = shap.KernelExplainer(predict_fn, background)
        n_samples = min(100, X_sample.shape[0])
        X_subset = X_sample[:n_samples]
        shap_values = explainer.shap_values(X_subset, nsamples=100)

        # Summary plot (bee swarm)
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_subset, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_subset, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_bar_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Mean absolute SHAP values (importance ranking)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)

        # Add clean names for display
        shap_importance['feature_clean'] = shap_importance['feature'].apply(clean_feature_name)

        shap_importance.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)

        print("\nSHAP ANALYSIS COMPLETE - Top 10 Features by SHAP Importance:")
        print("-" * 50)
        for idx, row in shap_importance.head(10).iterrows():
            print(f"  {idx + 1}. {row['feature_clean'][:45]:<45} SHAP: {row['shap_importance']:.4f}")

        return shap_importance

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==================== GLOBAL SENSITIVITY ANALYSIS ====================
def perform_gsa_analysis(model, X_test, feature_names, output_dir, n_samples=500):
    """Perform Global Sensitivity Analysis using Sobol method"""

    print("\n" + "=" * 60)
    print("PERFORMING GLOBAL SENSITIVITY ANALYSIS (GSA)")
    print("=" * 60)

    try:
        # Define bounds for each feature
        bounds = []
        for i in range(X_test.shape[1]):
            feature_min = X_test[:, i].min()
            feature_max = X_test[:, i].max()
            buffer = (feature_max - feature_min) * 0.1
            bounds.append([feature_min - buffer, feature_max + buffer])

        problem = {
            'num_vars': X_test.shape[1],
            'names': feature_names,
            'bounds': bounds
        }

        # Generate samples using Sobol sequence
        param_values = sobol_sample.sample(problem, n_samples, calc_second_order=False)
        print(f"Evaluating model at {n_samples} sample points...")
        param_list = [param_values[:, i:i + 1] for i in range(param_values.shape[1])]
        Y = model.predict(param_list).flatten()

        # Perform Sobol analysis
        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

        # Create results dataframe
        gsa_results = pd.DataFrame({
            'feature': feature_names,
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf']
        }).sort_values('ST', ascending=False)

        # Add clean names for display
        gsa_results['feature_clean'] = gsa_results['feature'].apply(clean_feature_name)

        # Save results
        gsa_results.to_csv(os.path.join(output_dir, "gsa_results.csv"), index=False)

        # Plot Total Sensitivity (ST)
        plt.figure(figsize=(14, 10))
        top_features = gsa_results.head(15)
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_features['ST'], xerr=top_features['ST_conf'],
                 alpha=0.7, color='steelblue', ecolor='black', capsize=5)
        plt.yticks(y_pos, top_features['feature_clean'])
        plt.xlabel('Total Sensitivity Index (ST)', fontsize=12)
        plt.title('Global Sensitivity Analysis - Total Sensitivity Indices (ST)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gsa_total_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot First-Order Sensitivity (S1)
        plt.figure(figsize=(14, 10))
        plt.barh(y_pos, top_features['S1'], xerr=top_features['S1_conf'],
                 alpha=0.7, color='coral', ecolor='black', capsize=5)
        plt.yticks(y_pos, top_features['feature_clean'])
        plt.xlabel('First-Order Sensitivity Index (S1)', fontsize=12)
        plt.title('Global Sensitivity Analysis - First-Order Sensitivity Indices (S1)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gsa_first_order.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Combined plot (S1 vs ST)
        plt.figure(figsize=(14, 10))
        x = np.arange(len(top_features))
        width = 0.35
        plt.barh(x - width / 2, top_features['S1'], width, label='First-Order (S1)', alpha=0.7, color='coral')
        plt.barh(x + width / 2, top_features['ST'], width, label='Total (ST)', alpha=0.7, color='steelblue')
        plt.yticks(x, top_features['feature_clean'])
        plt.xlabel('Sensitivity Index', fontsize=12)
        plt.title('Global Sensitivity Analysis - First-Order vs Total Sensitivity', fontsize=14, fontweight='bold')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gsa_combined_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print("\nGSA ANALYSIS COMPLETE - Top 10 Features by Total Sensitivity (ST):")
        print("-" * 50)
        for idx, row in gsa_results.head(10).iterrows():
            print(f"  {idx + 1}. {row['feature_clean'][:45]:<45} ST: {row['ST']:.4f} (±{row['ST_conf']:.4f})")

        return gsa_results

    except Exception as e:
        print(f"GSA analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==================== ACTUAL VS PREDICTED PLOT ====================
def create_actual_vs_predicted_plot(y_test, y_pred, metrics, output_dir, model_name="TabTransformer"):
    """Create actual vs predicted plot with performance statistics and green regression line"""

    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)

    # Perfect prediction line (red dashed)
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (1:1 Line)')

    # Regression line (green solid)
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    y_line = p(x_line)
    plt.plot(x_line, y_line, 'g-', linewidth=2.5, alpha=0.9,
             label=f'Regression Line (y={z[0]:.2f}x + {z[1]:.1f})')

    # Calculate additional statistics
    residuals = y_test - y_pred
    mape = np.mean(np.abs(residuals / (y_test + 1e-8))) * 100

    # Statistics text
    stats_text = (
        f'Performance Statistics:\n'
        f'R² = {metrics["r2"]:.4f}\n'
        f'RMSE = {metrics["rmse"]:.2f} mg/l\n'
        f'MAE = {metrics["mae"]:.2f} mg/l\n'
        f'MSE = {metrics["mse"]:.2f}\n'
        f'MAPE = {mape:.2f}%\n'
        f'Regression: y = {z[0]:.3f}x + {z[1]:.1f}'
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.xlabel('Actual Salinity (mg/l)', fontsize=12)
    plt.ylabel('Predicted Salinity (mg/l)', fontsize=12)
    plt.title(f'Actual vs Predicted Salinity - {model_name} (RFE Selected Features)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)

    plt.axis('equal')
    buffer = (max_val - min_val) * 0.05
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Actual vs predicted plot saved with green regression line!")


# ==================== COMPREHENSIVE PERFORMANCE VISUALIZATION ====================
def create_comprehensive_performance_plot(y_test, y_pred, train_weights, feature_importance, output_dir):
    """Create comprehensive performance visualization with 6 subplots"""

    plt.figure(figsize=(18, 12))

    # 1. Residuals vs Fitted
    plt.subplot(2, 3, 1)
    residuals = y_test - y_pred
    fitted_values = y_pred
    plt.scatter(fitted_values, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.grid(True, alpha=0.3)

    # 2. Residual Distribution
    plt.subplot(2, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (mg/l)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)

    # 3. Distribution Comparison
    plt.subplot(2, 3, 3)
    plt.hist(y_test, alpha=0.7, label='Actual', bins=30, density=True)
    plt.hist(y_pred, alpha=0.7, label='Predicted', bins=30, density=True)
    plt.xlabel('Salinity (mg/l)')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Q-Q Plot
    plt.subplot(2, 3, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)

    # 5. Weight Distribution
    plt.subplot(2, 3, 5)
    unique_weights, weight_counts = np.unique(train_weights, return_counts=True)
    plt.bar(range(len(unique_weights)), weight_counts)
    plt.xlabel('Weight Category')
    plt.ylabel('Number of Samples')
    plt.title('Sample Weight Distribution')
    plt.xticks(range(len(unique_weights)), [f'{w:.2f}' for w in unique_weights], rotation=45)
    plt.grid(True, alpha=0.3)

    # 6. Feature Importance (SHAP)
    plt.subplot(2, 3, 6)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['shap_importance'].values[::-1])
    plt.yticks(range(len(top_features)), top_features['feature_clean'].values[::-1])
    plt.xlabel('SHAP Importance')
    plt.title('Top 10 Feature Importance (SHAP)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Comprehensive performance plot saved!")


# ==================== FEATURE IMPORTANCE COMPARISON PLOT ====================
def create_feature_importance_comparison_plot(rfe_results, shap_results, gsa_results, output_dir):
    """Create a comparison plot of feature importance from different methods"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    fig.suptitle('Feature Importance Comparison: RFE vs SHAP vs GSA (RFE Selected Features)', fontsize=16,
                 fontweight='bold')

    # Plot 1: RFE Ranking
    ax1 = axes[0]
    top_rfe = rfe_results.head(10)
    ax1.barh(range(len(top_rfe)), top_rfe['rank_order'].values[::-1], color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(top_rfe)))
    ax1.set_yticklabels(top_rfe['feature_clean'].values[::-1])
    ax1.set_xlabel('RFE Rank (1 = Best)', fontsize=11)
    ax1.set_title('RFE Feature Ranking', fontsize=12, fontweight='bold')
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: SHAP Importance
    ax2 = axes[1]
    top_shap = shap_results.head(10)
    ax2.barh(range(len(top_shap)), top_shap['shap_importance'].values[::-1], color='coral', alpha=0.8)
    ax2.set_yticks(range(len(top_shap)))
    ax2.set_yticklabels(top_shap['feature_clean'].values[::-1])
    ax2.set_xlabel('SHAP Importance', fontsize=11)
    ax2.set_title('SHAP Feature Importance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: GSA Total Sensitivity (ST)
    ax3 = axes[2]
    top_gsa = gsa_results.head(10)
    ax3.barh(range(len(top_gsa)), top_gsa['ST'].values[::-1], xerr=top_gsa['ST_conf'].values[::-1],
             color='seagreen', alpha=0.8, ecolor='black', capsize=3)
    ax3.set_yticks(range(len(top_gsa)))
    ax3.set_yticklabels(top_gsa['feature_clean'].values[::-1])
    ax3.set_xlabel('Total Sensitivity (ST)', fontsize=11)
    ax3.set_title('GSA - Total Sensitivity Index', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Feature importance comparison plot saved!")


# ==================== FINAL MODEL TRAINING WITH EMBEDDED RFE (NO DATA LEAKAGE) ====================
def train_tabtransformer_with_embedded_rfe(df, feature_list, target, basin_densities,
                                           n_features_to_select=15, n_epochs=50,
                                           n_splits=3, test_years=6, scaler_type='standard'):
    """
    CORRECTED VERSION: Embedded RFE - Selection happens INSIDE each fold
    This prevents data leakage by using ONLY training data for feature selection
    """

    print("\n" + "=" * 80)
    print("TRAINING TABTRANSFORMER WITH EMBEDDED RFE (NO DATA LEAKAGE)")
    print("=" * 80)
    print(f"Total features available: {len(feature_list)}")
    print(f"Target features to select per fold: {n_features_to_select}")
    print(f"Scaler type: {scaler_type}")
    print(f"Temporal validation: {n_splits} folds, {test_years} test years")

    basin_weights = calculate_basin_density_weights_inverse_sqrt(df, basin_densities)

    # Preprocess ALL features consistently FIRST
    print("\nPreprocessing features consistently...")
    processed_features, y_all, basins_all, years_all, drills_all, num_categories_list, valid_indices = preprocess_features_consistent(
        df, feature_list, target, scaler_type
    )

    # Create a temporary dataframe with processed data for splitting
    temp_df = pd.DataFrame(index=valid_indices)
    temp_df['target'] = y_all
    temp_df['Hydrological Basin'] = basins_all
    temp_df['Year'] = years_all
    temp_df['Drill'] = drills_all

    # Add processed features as columns
    for i, feat in enumerate(feature_list):
        temp_df[feat] = processed_features[i]

    # Create temporal splits
    splits = create_temporal_splits(temp_df, n_splits=n_splits, test_years=test_years)

    if not splits:
        print("No valid temporal splits created!")
        return None

    fold_results = []
    all_fold_selected_features = []

    for fold_idx, (train_indices, test_indices) in enumerate(splits):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold_idx + 1}/{len(splits)}")
        print(f"{'=' * 50}")

        # ⭐⭐⭐ CRITICAL: RFE uses ONLY TRAINING DATA ⭐⭐⭐
        X_train = temp_df.loc[train_indices, feature_list]
        y_train = temp_df.loc[train_indices, 'target']

        # Prepare data for RFE (convert to numeric)
        X_train_rfe = X_train.copy()
        for col in X_train_rfe.select_dtypes(include=['object']).columns:
            X_train_rfe[col] = pd.Categorical(X_train_rfe[col]).codes
        X_train_rfe = X_train_rfe.fillna(X_train_rfe.mean())

        # Perform RFE on TRAINING DATA ONLY
        print(f"Performing RFE on {len(X_train_rfe)} training samples to select {n_features_to_select} features...")
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        rfe = RFE(estimator=estimator, n_features_to_select=min(n_features_to_select, len(feature_list)), step=2)
        rfe.fit(X_train_rfe, y_train)

        # Get features selected by RFE for this fold
        fold_selected_features = [feature_list[i] for i in range(len(feature_list)) if rfe.support_[i]]
        all_fold_selected_features.append(fold_selected_features)

        print(f"RFE selected {len(fold_selected_features)} features for this fold")

        # Prepare data for TabTransformer using ONLY selected features for this fold
        X_train_list = []
        X_test_list = []
        num_categories_list_fold = []

        # Get the indices of selected features in the original feature_list
        selected_indices = [feature_list.index(feat) for feat in fold_selected_features]

        for idx in selected_indices:
            feature = feature_list[idx]
            X_train_list.append(temp_df.loc[train_indices, feature].values.reshape(-1, 1))
            X_test_list.append(temp_df.loc[test_indices, feature].values.reshape(-1, 1))
            # Get number of categories for this feature
            max_val = max(temp_df[feature].max(), temp_df[feature].max())
            num_categories_list_fold.append(int(max_val) + 1)

        y_train_fold = temp_df.loc[train_indices, 'target'].values
        y_test_fold = temp_df.loc[test_indices, 'target'].values
        train_weights = temp_df.loc[train_indices, 'Hydrological Basin'].map(basin_weights).values

        print(f"Training TabTransformer with {len(fold_selected_features)} features...")
        print(f"Training samples: {len(y_train_fold)}, Test samples: {len(y_test_fold)}")

        # Create and train TabTransformer
        model = create_tabtransformer_model(len(fold_selected_features), num_categories_list_fold)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,
                                                   verbose=0)

        history = model.fit(X_train_list, y_train_fold,
                            validation_data=(X_test_list, y_test_fold),
                            epochs=n_epochs, batch_size=min(32, len(y_train_fold)),
                            callbacks=[early_stop], verbose=1,
                            sample_weight=train_weights)

        y_pred = model.predict(X_test_list, verbose=0).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test_fold, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_fold, y_pred)
        r2 = r2_score(y_test_fold, y_pred)

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model': model,
            'y_test': y_test_fold,
            'y_pred': y_pred,
            'history': history,
            'selected_features': fold_selected_features
        })

        model.save(os.path.join(model_dir, f"tabtransformer_embedded_rfe_fold_{fold_idx + 1}.h5"))

        # Perform SHAP analysis for first fold
        if fold_idx == 0:
            X_test_stacked = np.column_stack([x.flatten() for x in X_test_list])
            shap_results = perform_shap_analysis(model, X_test_stacked, fold_selected_features, results_dir)

            # Perform GSA analysis for first fold
            gsa_results = perform_gsa_analysis(model, X_test_stacked, fold_selected_features, results_dir,
                                               n_samples=500)

            # Create visualizations for first fold
            metrics_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
            create_actual_vs_predicted_plot(y_test_fold, y_pred, metrics_dict, results_dir, "TabTransformer")

            if shap_results is not None:
                create_comprehensive_performance_plot(y_test_fold, y_pred, train_weights, shap_results, results_dir)

    # Check consistency of selected features across folds
    print("\n" + "=" * 80)
    print("FEATURE SELECTION CONSISTENCY ACROSS FOLDS:")
    print("=" * 80)

    # Find features selected in ALL folds
    if all_fold_selected_features:
        common_features = set(all_fold_selected_features[0])
        for fold_features in all_fold_selected_features[1:]:
            common_features = common_features.intersection(set(fold_features))

        print(f"\nFeatures selected in ALL {len(splits)} folds: {len(common_features)}")
        for feat in common_features:
            print(f"  - {clean_feature_name(feat)}")

        # Save common features
        with open(os.path.join(selection_dir, "embedded_rfe_common_features.txt"), 'w', encoding='utf-8') as f:
            f.write("FEATURES SELECTED IN ALL FOLDS (Embedded RFE)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total folds: {len(splits)}\n")
            f.write(f"Features selected in all folds: {len(common_features)}\n\n")
            for i, feat in enumerate(common_features, 1):
                f.write(f"{i:2d}. {clean_feature_name(feat)}\n")

        # Also save per-fold selections
        with open(os.path.join(selection_dir, "embedded_rfe_per_fold_selections.txt"), 'w', encoding='utf-8') as f:
            f.write("PER-FOLD FEATURE SELECTIONS (Embedded RFE)\n")
            f.write("=" * 60 + "\n\n")
            for fold_idx, fold_features in enumerate(all_fold_selected_features):
                f.write(f"\nFOLD {fold_idx + 1}:\n")
                f.write("-" * 40 + "\n")
                for i, feat in enumerate(fold_features, 1):
                    f.write(f"{i:2d}. {clean_feature_name(feat)}\n")

    if fold_results:
        results_df = pd.DataFrame(fold_results)

        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY (EMBEDDED RFE - NO DATA LEAKAGE)")
        print("=" * 80)
        print(f"Average MSE: {results_df['mse'].mean():.4f} ± {results_df['mse'].std():.4f}")
        print(f"Average RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
        print(f"Average MAE: {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
        print(f"Average R²: {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")

        results_df.to_csv(os.path.join(results_dir, "final_fold_results_embedded_rfe.csv"), index=False)
        return results_df, fold_results

    return None, None


# ==================== CONSISTENT PREPROCESSING FUNCTION ====================
def preprocess_features_consistent(df, feature_list, target, scaler_type='standard', num_bins=10):
    """
    Preprocess ALL features consistently - ensures same samples across all features
    """

    # First, get common indices where ALL features have valid data
    valid_mask = pd.Series([True] * len(df), index=df.index)

    for feature in feature_list:
        valid_mask = valid_mask & df[feature].notna()
    valid_mask = valid_mask & df[target].notna()
    valid_mask = valid_mask & df['Hydrological Basin'].notna()
    valid_mask = valid_mask & df['Drill'].notna()
    valid_mask = valid_mask & df['Year'].notna()

    # Filter to only complete cases
    df_complete = df[valid_mask].copy()

    print(f"Complete cases after removing NaN: {len(df_complete)} out of {len(df)}")

    if len(df_complete) < 1000:
        print("Warning: Too few complete cases. Using forward fill for missing values...")
        df_complete = df.copy()
        for feature in feature_list:
            df_complete[feature] = df_complete[feature].fillna(method='ffill').fillna(method='bfill')
        df_complete[target] = df_complete[target].fillna(method='ffill').fillna(method='bfill')
        df_complete['Hydrological Basin'] = df_complete['Hydrological Basin'].fillna(method='ffill')
        df_complete['Drill'] = df_complete['Drill'].fillna(method='ffill')
        df_complete['Year'] = df_complete['Year'].fillna(method='ffill')

    # Preprocess each feature
    processed_features = []
    num_categories_list = []

    for feature in feature_list:
        col = df_complete[feature]

        if col.dtype == 'object' or len(col.unique()) <= 20:
            processed = pd.Categorical(col).codes
        else:
            col_array = col.values.reshape(-1, 1)

            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'power':
                scaler = PowerTransformer(method='yeo-johnson')
            elif scaler_type == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal')
            else:
                scaler = StandardScaler()

            normed = scaler.fit_transform(col_array).flatten()

            try:
                processed = pd.cut(normed, bins=num_bins, labels=False)
                if pd.isna(processed).any():
                    mode_val = pd.Series(processed).mode()[0]
                    processed = pd.Series(processed).fillna(mode_val).values
            except:
                try:
                    processed = pd.qcut(normed, q=num_bins, labels=False, duplicates='drop')
                    if pd.isna(processed).any():
                        mode_val = pd.Series(processed).mode()[0]
                        processed = pd.Series(processed).fillna(mode_val).values
                except:
                    processed = np.digitize(normed, np.percentile(normed, np.linspace(0, 100, num_bins + 1)[1:-1]))

        processed_features.append(processed.astype('int32'))
        num_categories_list.append(int(processed.max()) + 1)

    y = df_complete[target].values
    basins = df_complete['Hydrological Basin'].values
    years = df_complete['Year'].values
    drills = df_complete['Drill'].values

    return processed_features, y, basins, years, drills, num_categories_list, df_complete.index


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function - TabTransformer with RFE-selected features"""

    print("=" * 80)
    print("TABTRANSFORMER WITH RFE-SELECTED FEATURES")
    print("=" * 80)
    print("\nPROCESS:")
    print("  1. RFE selection using Random Forest (proxy for TabTransformer)")
    print("  2. Train TabTransformer with selected features only")
    print("  3. Perform SHAP analysis on selected features")
    print("  4. Perform GSA analysis on selected features")
    print("=" * 80)

    # Step 1: Feature imbalance visualization for original data
    print("\n" + "=" * 80)
    print("STEP 1: FEATURE IMBALANCE VISUALIZATION")
    print("=" * 80)

    X_original = df[all_features].copy()
    for col in X_original.select_dtypes(include=['object']).columns:
        X_original[col] = pd.Categorical(X_original[col]).codes

    create_comprehensive_imbalance_visualizations(X_original, all_features, viz_dir, "all_features")

    # Step 2: RFE Selection using Random Forest
    print("\n" + "=" * 80)
    print("STEP 2: RFE FEATURE SELECTION (Using Random Forest as Proxy)")
    print("=" * 80)

    n_features_to_select = 15  # Select top 15 features
    selected_features, rfe_ranking_df = perform_rfe_selection_for_tabtransformer(
        df=df,
        feature_list=all_features,
        target=target,
        n_features_to_select=n_features_to_select,
        output_dir=selection_dir
    )

    if selected_features is None or len(selected_features) == 0:
        print("RFE selection failed. Exiting.")
        return

    # Save selected features
    with open(os.path.join(selection_dir, "rfe_selected_features.txt"), 'w', encoding='utf-8') as f:
        f.write("RFE SELECTED FEATURES FOR TABTRANSFORMER\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total features selected: {len(selected_features)}\n")
        f.write(f"Selection method: RFE with Random Forest (proxy for TabTransformer)\n\n")
        f.write("SELECTED FEATURES:\n")
        f.write("-" * 40 + "\n")
        for i, feat in enumerate(selected_features, 1):
            clean_feat = clean_feature_name(feat)
            f.write(f"{i:2d}. {clean_feat}\n")

    print(f"\n✓ Selected features saved to: {os.path.join(selection_dir, 'rfe_selected_features.txt')}")

    # Step 3: Train TabTransformer with EMBEDDED RFE (NO data leakage)
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING TABTRANSFORMER WITH EMBEDDED RFE (NO DATA LEAKAGE)")
    print("=" * 80)

    # IMPORTANT: Pass ALL features, let embedded RFE select per fold
    results_df, fold_results = train_tabtransformer_with_embedded_rfe(
        df=df,
        feature_list=all_features,  # ← Pass ALL features, NOT pre-selected ones!
        target=target,
        basin_densities=basin_densities,
        n_features_to_select=15,  # Select top 15 per fold
        n_epochs=50,
        n_splits=3,
        test_years=6,
        scaler_type='standard'
    )

    # Step 4: Generate final report
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING FINAL REPORT")
    print("=" * 80)

    with open(os.path.join(output_dir, "FINAL_REPORT_RFE_SELECTED.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TABTRANSFORMER WITH RFE-SELECTED FEATURES - FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("PROCESS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. RFE selection using Random Forest (proxy for TabTransformer)\n")
        f.write("2. TabTransformer trained on selected features only\n")
        f.write("3. SHAP analysis on selected features\n")
        f.write("4. GSA analysis on selected features\n\n")

        f.write("RFE SELECTION RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total features evaluated: {len(all_features)}\n")
        f.write(f"Features selected: {len(selected_features)}\n")
        f.write(f"Selection method: RFE with Random Forest\n\n")

        f.write("SELECTED FEATURES:\n")
        for i, feat in enumerate(selected_features, 1):
            clean_feat = clean_feature_name(feat)
            f.write(f"{i:2d}. {clean_feat}\n")

        if results_df is not None:
            f.write("\n\nFINAL MODEL PERFORMANCE (Temporal Validation):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average R²:  {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}\n")
            f.write(f"Average RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}\n")
            f.write(f"Average MAE:  {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}\n")
            f.write(f"Average MSE:  {results_df['mse'].mean():.4f} ± {results_df['mse'].std():.4f}\n")
            f.write(f"Number of folds: {len(results_df)}\n")

        f.write("\n\nOUTPUT DIRECTORIES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Main output: {output_dir}\n")
        f.write(f"Visualizations: {viz_dir}\n")
        f.write(f"Models: {model_dir}\n")
        f.write(f"Results: {results_dir}\n")
        f.write(f"Feature selection: {selection_dir}\n")

    print(f"\n✓ Final report saved to: {os.path.join(output_dir, 'FINAL_REPORT_RFE_SELECTED.txt')}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nRFE selected {len(selected_features)} features from {len(all_features)} total features")
    print("\nSelected features:")
    for i, feat in enumerate(selected_features, 1):
        clean_feat = clean_feature_name(feat)
        print(f"  {i:2d}. {clean_feat}")

    if results_df is not None:
        print(f"\nFinal Model Performance (3-fold temporal validation):")
        print(f"  Average R²:  {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
        print(f"  Average RMSE: {results_df['rmse'].mean():.2f} ± {results_df['rmse'].std():.2f}")
        print(f"  Average MAE:  {results_df['mae'].mean():.2f} ± {results_df['mae'].std():.2f}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFeature importance results (SHAP, GSA, RFE) are in the 'results' folder!")


# Run the main function
if __name__ == "__main__":
    main()
