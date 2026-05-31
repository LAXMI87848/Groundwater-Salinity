import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import seaborn as sns
import os
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample
import shap
from imblearn.under_sampling import ClusterCentroids, TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from scipy import stats
import warnings
from collections import Counter
import time
import joblib
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings('ignore')

# Create output directory and give your own path where results needs to be saved
output_dir = r'G:\IIT_AI work\ISRAEL\GWS_Model info\Data and feature score\Revised data and feature_fitting scores\Final results (dataset_28.8)\Israel model run\RandomForest_country_final_single_run1' # used in my case
os.makedirs(output_dir, exist_ok=True)

# Create visualization directories upfront
viz_dir = os.path.join(output_dir, "imbalance_visualizations")
os.makedirs(viz_dir, exist_ok=True)

single_run_dir = os.path.join(output_dir, "single_run_results")
os.makedirs(single_run_dir, exist_ok=True)

sheet_name = "Sheet1"
# Load your data and give your own path in #excelFilePath, where the data shared is present
excelFilePath = r'G:\IIT_AI work\ISRAEL\GWS_Model info\Data and feature score\Revised data and feature_fitting scores\dataset_28.8.xlsx' # used in my case
df = pd.read_excel(excelFilePath, sheet_name)
basin_data = df['Hydrological Basin']

# Select and rename columns
selected_columns = df.copy()
print(selected_columns.columns)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Convert Salinity to binary if necessary and filter y <= 4000
y = selected_columns['Salinity - Cl (mg/l)']
selected_columns = selected_columns[y <= 4000]
y = selected_columns['Salinity - Cl (mg/l)']

print("Filtered y (y <= 4000):")
print(y)
print("\nValue counts after filtering:")
print(y.value_counts())

X = selected_columns.drop(columns=[
    'Salinity - Cl (mg/l)', 'Hightemp_fall', 'Hightemp_spring', 'Hightemp_summer',
    'Hightemp_winter', 'Tempanam_fall', 'Tempanam_spring', 'Tempanam_summer', 'Tempanam_winter',
    'Zcore_fall', 'Zcore_spring', 'Zcore_summer', 'Zcore_winter', 'precipitation ( #N of events Q1)',
    'precipitation (  #N of events Q2)', 'precipitation ( #N of events Q3)', 'precipitation ( #N of events Q4)',
    'precipitation (  #N of events Q5)', 'Elevation (m)', 'salinity difference', 'salinity relative difference',
    'Precip diff', 'Hydrological Basin', 'Aquifer', 'Cell name', 'Sub Basin', 'Drill', 'Year',
    'National system - Groundwater (%)', 'National system - Surface water (%)',
    'National system - Desalinated water (%)'
])

# Handle NaN values before one-hot encoding
print(f"NaN values in X before handling: {X.isna().sum().sum()}")
# Fill NaN values with mean for numerical columns
for col in X.select_dtypes(include=[np.number]).columns:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].mean())

X = pd.get_dummies(X, drop_first=True)  # One-hot encode

# Check for any remaining NaN values
print(f"NaN values in X after handling: {X.isna().sum().sum()}")

print("Number of rows in filtered data:", len(basin_data))
print("X shape:", X.shape)
print("y shape:", y.shape)

# Store original data for processing
X_original = X.copy()
y_original = y.copy()
selected_columns_original = selected_columns.copy()

# =============================================
# BASIN DENSITIES (PROVIDED BY USER)
# =============================================

basin_densities = {
    'Carmel': 0.119408792,
    'Coast': 0.348310389,
    'Galil West': 0.045398829,
    'Sea of galilee': 0.017933556,
    'mountin east': 0.013575234,
    'Yarkatan': 0.019049412,
    'Negev and Arava': 0.022511285
}

print("Basin densities to be used:")
for basin, density in basin_densities.items():
    print(f"  {basin}: {density}")


# =============================================
# COMPREHENSIVE FEATURE IMBALANCE VISUALIZATION
# =============================================

def create_comprehensive_imbalance_visualizations(X, feature_names, output_dir, prefix="original"):
    """Create comprehensive visualizations to understand feature imbalance"""

    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()

    print(f"\nCreating comprehensive imbalance visualizations for {prefix} features...")

    # Check for and handle NaN values
    nan_count = X_df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in features. Removing rows with NaN values for visualization.")
        X_df = X_df.dropna()

    # 1. Distribution Comparison Plot
    plt.figure(figsize=(16, 12))

    # Calculate metrics
    variances = X_df.var()
    skewness = X_df.skew()
    kurtosis = X_df.kurtosis()

    # 1.1 Variance comparison
    plt.subplot(2, 3, 1)
    plt.bar(range(len(variances)), sorted(variances))
    plt.title('Feature Variances (Sorted)')
    plt.xlabel('Features (sorted by variance)')
    plt.ylabel('Variance')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 1.2 Skewness distribution
    plt.subplot(2, 3, 2)
    plt.hist(skewness, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Skewness Distribution Across Features')
    plt.xlabel('Skewness')
    plt.ylabel('Frequency')
    plt.axvline(-2, color='red', linestyle='--', label='High skew threshold')
    plt.axvline(2, color='red', linestyle='--')
    plt.legend()

    # 1.3 Kurtosis distribution
    plt.subplot(2, 3, 3)
    plt.hist(kurtosis, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Kurtosis Distribution Across Features')
    plt.xlabel('Kurtosis')
    plt.ylabel('Frequency')
    plt.axvline(-5, color='red', linestyle='--', label='Extreme kurtosis')
    plt.axvline(5, color='red', linestyle='--')
    plt.legend()

    # 1.4 Outlier analysis
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

    # 1.5 Zero inflation analysis
    zero_percentages = (X_df == 0).sum() / len(X_df) * 100
    plt.subplot(2, 3, 5)
    plt.hist(zero_percentages, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Zero Percentage Distribution')
    plt.xlabel('Percentage of Zeros')
    plt.ylabel('Frequency')
    plt.axvline(50, color='red', linestyle='--', label='50% zero threshold')
    plt.legend()

    # 1.6 Value ranges
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

    # 2. Correlation Heatmap
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


# =============================================
# CREATE IMBALANCE VISUALIZATIONS FOR ORIGINAL DATA
# =============================================

print("Creating imbalance visualizations for original data...")
original_imbalance = create_comprehensive_imbalance_visualizations(
    X_original, X_original.columns.tolist(), viz_dir, "original"
)


# =============================================
# SIMPLIFIED FEATURE BALANCING FOR SINGLE RUN
# =============================================

def enhanced_feature_balancing_simple(X, y, random_state):
    """
    Simplified feature balancing for individual runs
    """
    # Remove constant features
    constant_filter = VarianceThreshold(threshold=0.01)
    X_filtered = constant_filter.fit_transform(X)
    remaining_indices = constant_filter.get_support(indices=True)
    remaining_features = [X.columns[i] for i in remaining_indices]

    # Remove highly correlated features
    def remove_highly_correlated_simple(X_array, feature_names, threshold=0.95):
        df_temp = pd.DataFrame(X_array, columns=feature_names)
        corr_matrix = df_temp.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = []
        for column in upper.columns:
            if any(upper[column] > threshold):
                to_drop.append(column)

        features_to_keep = [col for col in feature_names if col not in to_drop]
        X_filtered = np.delete(X_array, [feature_names.index(col) for col in to_drop], axis=1)

        return X_filtered, features_to_keep

    X_filtered, remaining_features = remove_highly_correlated_simple(
        X_filtered, remaining_features, threshold=0.95
    )

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Convert back to DataFrame
    X_processed = pd.DataFrame(X_scaled, index=X.index, columns=remaining_features)

    return X_processed, remaining_features


# =============================================
# CALCULATE BASIN DENSITY WEIGHTS USING INVERSE SQUARE-ROOT METHOD
# =============================================

def calculate_basin_density_weights_inverse_sqrt(selected_columns, basin_densities):
    """
    Calculate sample weights based on the provided basin densities using inverse square-root method
    """
    # Verify all basins in data are in the provided densities
    basins_in_data = selected_columns['Hydrological Basin'].unique()
    missing_basins = set(basins_in_data) - set(basin_densities.keys())

    if missing_basins:
        print(f"Warning: Missing density values for basins: {missing_basins}")
        print("Using average density for missing basins")
        avg_density = np.mean(list(basin_densities.values()))
        for basin in missing_basins:
            basin_densities[basin] = avg_density

    # Convert basin_densities to a pandas Series for easier manipulation
    basin_density_series = pd.Series(basin_densities)

    # Calculate weights using inverse square-root density
    weights = 1 / np.sqrt(basin_density_series)

    # Normalize weights to have mean = 1
    weights = weights / weights.mean()

    # Create weight mapping for each basin
    weight_mapping = weights.to_dict()

    print("Basin density weights (inverse square-root method):")
    for basin, weight in weight_mapping.items():
        if basin in basins_in_data:
            density_val = basin_densities[basin]
            print(f"  {basin}: {weight:.3f} (density: {density_val})")

    return weight_mapping


# =============================================
# SEPARATE ACTUAL VS PREDICTED PLOT WITH STATISTICS
# =============================================

def create_actual_vs_predicted_plot(y_test, y_pred, metrics, output_dir, model_name="Random Forest"):
    """
    Create a separate actual vs predicted plot with performance statistics in legend
    """
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = plt.scatter(y_test, y_pred, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)

    # Set equal aspect ratio
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
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
        #f'MAPE = {mape:.2f}%\n'
        #f'n = {len(y_test)} samples'
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
    buffer = (max_val - min_val) * 0.05
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted_detailed.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Detailed actual vs predicted plot saved!")

    return {
        'mape': mape,
        'rmse': rmse
    }


# =============================================
# GLOBAL SENSITIVITY ANALYSIS (GSA) FUNCTION
# =============================================

def perform_gsa_analysis(model, X_test, feature_names, output_dir, n_samples=1000):
    """
    Perform Global Sensitivity Analysis using Sobol method
    """
    print("Performing Global Sensitivity Analysis (GSA)...")

    try:
        # Define the problem for Sobol analysis
        problem = {
            'num_vars': len(feature_names),
            'names': feature_names,
            'bounds': [[-3, 3]] * len(feature_names)  # Wider bounds for standardized features
        }

        # Generate samples using Sobol sequence
        param_values = sobol_sample.sample(problem, n_samples, calc_second_order=False)

        # Evaluate model at sample points
        print(f"Evaluating model at {n_samples} sample points...")
        Y = model.predict(param_values)

        # Perform Sobol analysis
        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

        # Create results dataframe
        gsa_results = pd.DataFrame({
            'feature': feature_names,
            'S1': Si['S1'],  # First-order sensitivity indices
            'S1_conf': Si['S1_conf'],  # Confidence intervals for S1
            'ST': Si['ST'],  # Total sensitivity indices
            'ST_conf': Si['ST_conf']  # Confidence intervals for ST
        }).sort_values('ST', ascending=False)

        # Save GSA results
        gsa_results.to_csv(os.path.join(output_dir, "gsa_results.csv"), index=False)

        # Plot GSA results
        plt.figure(figsize=(12, 8))

        # Plot total sensitivity indices
        features_plot = gsa_results.head(15)  # Top 15 features
        y_pos = np.arange(len(features_plot))

        plt.barh(y_pos, features_plot['ST'], xerr=features_plot['ST_conf'],
                 alpha=0.7, color='steelblue', ecolor='black', capsize=5)
        plt.yticks(y_pos, features_plot['feature'])
        plt.xlabel('Total Sensitivity Index (ST)')
        plt.title('Global Sensitivity Analysis - Total Sensitivity Indices (Top 15)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gsa_total_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot first-order sensitivity indices
        plt.figure(figsize=(12, 8))
        plt.barh(y_pos, features_plot['S1'], xerr=features_plot['S1_conf'],
                 alpha=0.7, color='lightcoral', ecolor='black', capsize=5)
        plt.yticks(y_pos, features_plot['feature'])
        plt.xlabel('First-Order Sensitivity Index (S1)')
        plt.title('Global Sensitivity Analysis - First-Order Sensitivity Indices (Top 15)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gsa_first_order_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Combined plot
        plt.figure(figsize=(14, 10))
        x = np.arange(len(features_plot))
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
        plt.savefig(os.path.join(output_dir, "gsa_combined_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print("GSA analysis completed successfully!")
        return {
            'gsa_results': gsa_results,
            'success': True
        }

    except Exception as e:
        print(f"GSA analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


# =============================================
# SINGLE RUN TRAINING FUNCTION WITH RANDOM FOREST
# =============================================

def train_single_run_with_density_weights(X, y, selected_columns, basin_densities, random_state=42):
    """
    Train Random Forest model on a single run with basin density weighting using inverse square-root method
    """
    print("Calculating basin density weights using inverse square-root method...")
    basin_weights = calculate_basin_density_weights_inverse_sqrt(selected_columns, basin_densities)

    # Enhanced feature balancing
    X_processed, feature_names = enhanced_feature_balancing_simple(X, y, random_state)

    # Prepare train/test split using temporal validation
    selected_columns_sorted = selected_columns.sort_values(by=['Drill', 'Year', 'Cell name'])
    train_indices, test_indices = [], []

    for drill_id, group in selected_columns_sorted.groupby('Drill'):
        group_years = group['Year'].sort_values().unique()
        train_years = group_years[:-6] if len(group_years) > 6 else group_years[:1]
        test_years = group_years[-6:] if len(group_years) > 6 else group_years[-1:]

        train_indices.extend(group[group['Year'].isin(train_years)].index.tolist())
        test_indices.extend(group[group['Year'].isin(test_years)].index.tolist())

    # Create train/test splits
    train_data = selected_columns_sorted.loc[train_indices]
    test_data = selected_columns_sorted.loc[test_indices]

    y_train = train_data['Salinity - Cl (mg/l)']
    y_test = test_data['Salinity - Cl (mg/l)']

    X_train = X_processed.loc[train_data.index]
    X_test = X_processed.loc[test_data.index]

    # Calculate sample weights for training based on basin densities
    train_weights = train_data['Hydrological Basin'].map(basin_weights).values

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    # Final scaling
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)

    # Feature selection using RFE with Random Forest
    print("Performing feature selection with RFE...")
    rfe = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=random_state),
              n_features_to_select=min(20, X_train_scaled.shape[1]))

    # Fit RFE with sample weights
    rfe.fit(X_train_scaled, y_train, sample_weight=train_weights)

    # Get selected features
    selected_features = X_train.columns[rfe.support_]
    X_train_selected = X_train_scaled[:, rfe.support_]
    X_test_selected = X_test_scaled[:, rfe.support_]

    print(f"Selected {len(selected_features)} features for modeling")

    # Train final Random Forest model with sample weights
    print("Training Random Forest model with basin density weights...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Use sqrt for better performance
        bootstrap=True,
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train_selected, y_train, sample_weight=train_weights)

    # Make predictions and evaluate
    y_pred = model.predict(X_test_selected)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("RANDOM FOREST MODEL PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print("=" * 50)

    # Save model and artifacts
    joblib.dump(model, os.path.join(single_run_dir, "trained_model.pkl"))
    joblib.dump(final_scaler, os.path.join(single_run_dir, "feature_scaler.pkl"))
    joblib.dump(rfe, os.path.join(single_run_dir, "feature_selector.pkl"))

    # Feature importance from Random Forest
    feature_importance = model.feature_importances_

    # Save feature information
    feature_info = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    feature_info.to_csv(os.path.join(single_run_dir, "feature_importance.csv"), index=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_info)), feature_info['importance'][::-1])
    plt.yticks(range(len(feature_info)), feature_info['feature'][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(single_run_dir, "feature_importance_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate detailed actual vs predicted plot
    print("Creating detailed actual vs predicted plot...")
    metrics_dict = {
        'mse': mse,
        'r2': r2,
        'mae': mae
    }
    additional_metrics = create_actual_vs_predicted_plot(
        y_test, y_pred, metrics_dict, single_run_dir, "Random Forest"
    )

    # Create comprehensive performance plot (6 subplots)
    plt.figure(figsize=(15, 10))

    # 1. Residuals vs Fitted values (replacing duplicate actual vs predicted)
    plt.subplot(2, 3, 1)
    fitted_values = y_pred
    residuals = y_test - y_pred
    plt.scatter(fitted_values, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.grid(True, alpha=0.3)

    # 2. Residual plot
    plt.subplot(2, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Salinity (mg/l)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)

    # 3. Distribution comparison
    plt.subplot(2, 3, 3)
    plt.hist(y_test, alpha=0.7, label='Actual', bins=30, density=True)
    plt.hist(y_pred, alpha=0.7, label='Predicted', bins=30, density=True)
    plt.xlabel('Salinity (mg/l)')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Error distribution
    plt.subplot(2, 3, 4)
    errors = y_pred - y_test
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (mg/l)')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)

    # 5. Basin weight distribution
    plt.subplot(2, 3, 5)
    unique_weights, weight_counts = np.unique(train_weights, return_counts=True)
    plt.bar(range(len(unique_weights)), weight_counts)
    plt.xlabel('Weight Category')
    plt.ylabel('Number of Samples')
    plt.title('Sample Weight Distribution')
    plt.xticks(range(len(unique_weights)), [f'{w:.2f}' for w in unique_weights], rotation=45)
    plt.grid(True, alpha=0.3)

    # 6. Feature importance (top 10)
    plt.subplot(2, 3, 6)
    top_features = feature_info.head(10)
    plt.barh(range(len(top_features)), top_features['importance'][::-1])
    plt.yticks(range(len(top_features)), top_features['feature'][::-1])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')

    plt.tight_layout()
    plt.savefig(os.path.join(single_run_dir, "comprehensive_performance_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed performance metrics
    performance_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Residual': residuals,
        'Absolute_Error': np.abs(residuals),
        'Hydrological_Basin': test_data['Hydrological Basin'].values
    })
    performance_df.to_csv(os.path.join(single_run_dir, "detailed_performance.csv"), index=False)

    # Performance by basin
    basin_performance = performance_df.groupby('Hydrological_Basin').agg({
        'Actual': 'mean',
        'Predicted': 'mean',
        'Absolute_Error': 'mean',
        'Residual': ['mean', 'std']
    }).round(4)
    basin_performance.to_csv(os.path.join(single_run_dir, "performance_by_basin.csv"))

    # SHAP analysis for Random Forest
    print("Performing SHAP analysis for Random Forest...")
    try:
        # Use TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_selected)

        # For Random Forest, shap_values might be a list for multi-class, but for regression it's usually a single array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take the first array for regression

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_selected, feature_names=selected_features, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(single_run_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_selected, feature_names=selected_features,
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(single_run_dir, "shap_bar_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_importance = pd.DataFrame({
            'feature': selected_features,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)

        shap_importance.to_csv(os.path.join(single_run_dir, "shap_importance.csv"), index=False)
        print("SHAP analysis completed successfully!")

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # GLOBAL SENSITIVITY ANALYSIS (GSA)
    gsa_results = perform_gsa_analysis(model, X_test_selected, selected_features, single_run_dir)

    # Update run summary with additional metrics
    run_summary = pd.DataFrame([{
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': additional_metrics['mape'],
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_selected_features': len(selected_features),
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test),
        'random_state': random_state,
        'model_type': 'RandomForest',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth
    }])
    run_summary.to_csv(os.path.join(single_run_dir, "run_summary.csv"), index=False)

    # Save basin weights information
    basin_weights_df = pd.DataFrame(list(basin_weights.items()),
                                    columns=['Basin', 'Weight'])
    basin_weights_df = basin_weights_df.merge(
        pd.DataFrame(list(basin_densities.items()),
                     columns=['Basin', 'Density']),
        on='Basin'
    )

    # Calculate and add inverse square root values for verification
    basin_weights_df['Inverse_Sqrt_Density'] = 1 / np.sqrt(basin_weights_df['Density'])
    basin_weights_df['Normalized_Weight'] = basin_weights_df['Inverse_Sqrt_Density'] / basin_weights_df[
        'Inverse_Sqrt_Density'].mean()

    basin_weights_df.to_csv(os.path.join(single_run_dir, "basin_weights_detailed.csv"), index=False)

    # Additional Random Forest specific diagnostics
    print("\nRandom Forest Model Details:")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print(f"Number of features used: {model.n_features_in_}")

    return {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'selected_features': selected_features,
        'scaler': final_scaler,
        'feature_selector': rfe,
        'performance_df': performance_df,
        'basin_weights': basin_weights,
        'basin_densities': basin_densities,
        'gsa_results': gsa_results
    }


# =============================================
# MAIN EXECUTION - SINGLE RUN WITH RANDOM FOREST
# =============================================

if __name__ == "__main__":
    print("Starting single run with Random Forest and basin density weighting...")
    print("=" * 60)

    results = train_single_run_with_density_weights(
        X_original,
        y_original,
        selected_columns_original,
        basin_densities,
        random_state=42
    )

    if results:
        print("\n" + "=" * 60)
        print("RANDOM FOREST TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final R²: {results['r2']:.4f}")
        print(f"Final RMSE: {results['rmse']:.4f}")
        print(f"Final MAE: {results['mae']:.4f}")
        print(f"Number of features used: {len(results['selected_features'])}")
        print(f"Results saved to: {single_run_dir}")

        # Print top features
        print("\nTop 10 Most Important Features:")
        print("-" * 40)
        feature_importance_path = os.path.join(single_run_dir, "feature_importance.csv")
        if os.path.exists(feature_importance_path):
            top_features = pd.read_csv(feature_importance_path).head(10)
            for idx, row in top_features.iterrows():
                print(f"{idx + 1:2d}. {row['feature']}: {row['importance']:.4f}")

        # Print basin weights summary
        print("\nBasin Weights Applied:")
        print("-" * 40)
        basin_weights_path = os.path.join(single_run_dir, "basin_weights_detailed.csv")
        if os.path.exists(basin_weights_path):
            basin_weights_df = pd.read_csv(basin_weights_path)
            for _, row in basin_weights_df.iterrows():
                print(f"{row['Basin']}: Weight = {row['Weight']:.3f}, Density = {row['Density']:.6f}")

        # Print GSA results if available
        if results.get('gsa_results', {}).get('success', False):
            print("\nTop 10 Most Sensitive Features (GSA):")
            print("-" * 50)
            gsa_path = os.path.join(single_run_dir, "gsa_results.csv")
            if os.path.exists(gsa_path):
                gsa_top = pd.read_csv(gsa_path).head(10)
                for idx, row in gsa_top.iterrows():
                    print(f"{idx + 1:2d}. {row['feature']}: ST = {row['ST']:.4f} ± {row['ST_conf']:.4f}")

    print("\nAll results saved to:", output_dir)
