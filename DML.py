import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import KFold
import statsmodels.api as sm
import random
import os
import warnings

warnings.filterwarnings('ignore')

# Create output directory and give your own path where you want to save the results
output_dir = r'G:\IIT_AI work\ISRAEL\GWS_Model info\Data and feature score\Revised data and feature_fitting scores\Final results (dataset_28.8)\Israel model run\DML_new' # used in my case
os.makedirs(output_dir, exist_ok=True)

# Create plots directory
plots_dir = os.path.join(output_dir, "dml_interpretation_plots")
os.makedirs(plots_dir, exist_ok=True)

sheet_name = "Sheet1"
# Load your data and giveyour own path of location where the data shared is present
excelFilePath = r'G:\IIT_AI work\ISRAEL\GWS_Model info\Data and feature score\Revised data and feature_fitting scores\Dataset_to_share.xlsx' # used in my case
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
selected_columns = selected_columns[y <= 4000]  # Filter rows where y <= 4000
y = selected_columns['Salinity - Cl (mg/l)']  # Reassign y after filtering

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

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)  # Use drop_first=True to avoid dummy variable trap
# Remove constant features
X = X.loc[:, (X != X.iloc[0]).any()]

# Convert boolean columns to integers - FIXED: Handle NaN values first
print(f"NaN values in X before cleaning: {X.isna().sum().sum()}")

# Convert boolean columns to integers, but handle NaN values properly
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)
    elif pd.api.types.is_numeric_dtype(X[col]):
        # For numeric columns, fill NaN with median
        X[col] = X[col].fillna(X[col].median())

# Ensure all data is numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with NaN values in X and ensure y aligns
nan_mask = X.isna().any(axis=1)
X = X[~nan_mask]
y = y[~nan_mask]

# Debugging: Check the shapes and data types
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Data types in X:\n", X.dtypes)
print(f"NaN values in X after cleaning: {X.isna().sum().sum()}")

# Store column names before converting to numpy
feature_names = X.columns.tolist()

# Convert to numpy arrays
X_array = X.values
y_array = y.values

# Store the original DataFrame for plotting
X_df = X.copy()

# =============================================================================
# PROPER DML WITH CROSS-FITTING
# =============================================================================

print("\n" + "=" * 60)
print("IMPLEMENTING DOUBLE MACHINE LEARNING WITH CROSS-FITTING")
print("=" * 60)

# Parameters
n_folds = 5  # Number of folds for cross-fitting
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize dictionaries to store results
causal_effects = {}
p_values = {}
standard_errors = {}

# Iterate over each predictor (treatment variable)
for i in range(X_array.shape[1]):
    treatment_name = feature_names[i]
    print(f"\nProcessing treatment: {treatment_name} ({i + 1}/{X_array.shape[1]})")

    # Store residuals across folds
    all_residual_X = np.zeros(len(y_array))
    all_residual_Y = np.zeros(len(y_array))

    # Cross-fitting loop
    for train_idx, val_idx in kf.split(X_array):
        # Split data
        Z_train = np.delete(X_array[train_idx], i, axis=1)  # Confounders (all other predictors)
        Z_val = np.delete(X_array[val_idx], i, axis=1)

        T_train = X_array[train_idx, i]  # Treatment
        T_val = X_array[val_idx, i]

        Y_train = y_array[train_idx]  # Outcome
        Y_val = y_array[val_idx]

        # Step 1: Estimate E[T|Z] using Random Forest (nuisance function for treatment)
        rf_T = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_T.fit(Z_train, T_train)
        E_T_given_Z_val = rf_T.predict(Z_val)

        # Step 2: Estimate E[Y|Z] using Random Forest (nuisance function for outcome)
        rf_Y = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_Y.fit(Z_train, Y_train)
        E_Y_given_Z_val = rf_Y.predict(Z_val)

        # Step 3: Compute residuals (orthogonalized)
        residual_T = T_val - E_T_given_Z_val
        residual_Y = Y_val - E_Y_given_Z_val

        # Store residuals for this fold
        all_residual_X[val_idx] = residual_T
        all_residual_Y[val_idx] = residual_Y

    # Step 4: Estimate causal effect by regressing residual_Y on residual_X
    # (using all residuals from all folds)
    causal_model = sm.OLS(all_residual_Y, sm.add_constant(all_residual_X)).fit()

    # Store results
    causal_effects[treatment_name] = causal_model.params[1]
    p_values[treatment_name] = causal_model.pvalues[1]
    standard_errors[treatment_name] = causal_model.bse[1]

# Convert results to DataFrame
causal_effects_df = pd.DataFrame({
    'Predictor': causal_effects.keys(),
    'Conditional_Association': causal_effects.values(),  # Renamed from 'Causal Effect'
    'P_Value': p_values.values(),
    'Std_Error': standard_errors.values()
})

# Calculate 95% confidence intervals
causal_effects_df['CI_Lower'] = causal_effects_df['Conditional_Association'] - 1.96 * causal_effects_df['Std_Error']
causal_effects_df['CI_Upper'] = causal_effects_df['Conditional_Association'] + 1.96 * causal_effects_df['Std_Error']

# Sort by absolute Conditional Association
causal_effects_df = causal_effects_df.sort_values(by='Conditional_Association', ascending=False)

print("\n" + "=" * 60)
print("DML RESULTS (Conditional Associations)")
print("=" * 60)
print(causal_effects_df.head(20))

# =============================================================================
# INTERPRETIVE PLOTS
# =============================================================================

print("\nGenerating interpretive plots...")

# 1. Bar plot of conditional associations
plt.figure(figsize=(12, 8))
top_effects = causal_effects_df.nlargest(15, 'Conditional_Association')
colors = ['red' if x < 0 else 'blue' for x in top_effects['Conditional_Association']]
sns.barplot(data=top_effects, x='Conditional_Association', y='Predictor', palette=colors, legend=False)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.title('Top 15 Conditional Associations (DML with Cross-Fitting)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '1_conditional_associations_barplot.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Conditional associations with confidence intervals
plt.figure(figsize=(12, 10))
significant_df = causal_effects_df[causal_effects_df['P_Value'] < 0.05].sort_values('Conditional_Association')

if len(significant_df) > 0:
    colors = ['red' if x < 0 else 'blue' for x in significant_df['Conditional_Association']]
    plt.barh(significant_df['Predictor'], significant_df['Conditional_Association'], color=colors, alpha=0.7)

    # Add error bars for confidence intervals
    for idx, (_, row) in enumerate(significant_df.iterrows()):
        plt.errorbar(row['Conditional_Association'], idx,
                     xerr=[[row['Conditional_Association'] - row['CI_Lower']],
                           [row['CI_Upper'] - row['Conditional_Association']]],
                     fmt='none', color='black', capsize=3)

    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Conditional Association (DML Estimate)')
    plt.title('Significant Conditional Associations (p < 0.05) with 95% CI')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '2_significant_conditional_associations.png'), dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No significant associations found at p < 0.05")

# 3. P-value distribution
plt.figure(figsize=(10, 6))
plt.hist(causal_effects_df['P_Value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0.05, color='red', linestyle='--', label='p=0.05 significance threshold')
plt.xlabel('P-Value')
plt.ylabel('Frequency')
plt.title('Distribution of P-Values from DML Analysis')
plt.legend()
plt.savefig(os.path.join(plots_dir, '3_pvalue_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Volcano plot
plt.figure(figsize=(10, 8))
plt.scatter(causal_effects_df['Conditional_Association'], -np.log10(causal_effects_df['P_Value']),
            alpha=0.6, c=causal_effects_df['P_Value'] < 0.05, cmap='coolwarm')
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Conditional Association Size')
plt.ylabel('-log10(P-Value)')
plt.title('Volcano Plot: Conditional Association vs Statistical Significance')
plt.colorbar(label='Significant (p < 0.05)')
plt.legend()
plt.savefig(os.path.join(plots_dir, '4_volcano_plot.png'), dpi=300, bbox_inches='tight')
plt.show()


# 5. Feature Importance Comparison
def compare_importance_measures():
    # Traditional feature importance using Random Forest
    rf_direct = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_direct.fit(X_array, y_array)
    traditional_importance = pd.DataFrame({
        'feature': feature_names,
        'traditional_importance': rf_direct.feature_importances_
    })

    # Merge with conditional associations
    comparison_df = causal_effects_df.merge(traditional_importance,
                                            left_on='Predictor', right_on='feature')

    # Plot comparison
    plt.figure(figsize=(12, 10))
    plt.scatter(comparison_df['traditional_importance'],
                comparison_df['Conditional_Association'],
                alpha=0.7, s=100)

    # Add labels for top features
    for idx, row in comparison_df.nlargest(5, 'traditional_importance').iterrows():
        plt.annotate(row['Predictor'],
                     (row['traditional_importance'], row['Conditional_Association']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Traditional Feature Importance (Random Forest)')
    plt.ylabel('Conditional Association (DML)')
    plt.title('Comparison: Traditional Importance vs DML Conditional Associations')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, '5_importance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return comparison_df


comparison_df = compare_importance_measures()


# 6. Correlation Heatmap of Top Predictors
def plot_correlation_heatmap(top_n=10):
    causal_effects_df['Abs_Association'] = causal_effects_df['Conditional_Association'].abs()
    top_features = causal_effects_df.nlargest(top_n, 'Abs_Association')['Predictor'].tolist()

    X_top = X_df[top_features]
    corr_matrix = X_top.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix,
                xticklabels=top_features,
                yticklabels=top_features,
                cmap='coolwarm', center=0,
                annot=True, fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f'Correlation Matrix of Top {top_n} Predictors (by DML Association)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '6_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()


plot_correlation_heatmap(10)

# 7. Conditional Association Distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(causal_effects_df['Conditional_Association'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Conditional Association Size')
plt.ylabel('Frequency')
plt.title('Distribution of All Conditional Associations')

plt.subplot(1, 2, 2)
significant_effects_values = causal_effects_df[causal_effects_df['P_Value'] < 0.05]['Conditional_Association']
if len(significant_effects_values) > 0:
    plt.hist(significant_effects_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Conditional Association Size')
    plt.ylabel('Frequency')
    plt.title('Significant Conditional Associations (p < 0.05)')
else:
    plt.text(0.5, 0.5, 'No significant associations\n(p < 0.05)',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('No Significant Conditional Associations')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '7_association_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()


# 8. Comprehensive Summary Plot
def create_summary_plot():
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    # Top left: Conditional associations bar plot
    ax1 = fig.add_subplot(gs[0, 0])
    top_10 = causal_effects_df.nlargest(10, 'Abs_Association')
    colors = ['red' if x < 0 else 'blue' for x in top_10['Conditional_Association']]
    ax1.barh(range(len(top_10)), top_10['Conditional_Association'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['Predictor'])
    ax1.set_xlabel('Conditional Association Size')
    ax1.set_title('Top 10 Largest Conditional Associations (Absolute Value)')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    # Top right: P-value vs effect size
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(causal_effects_df['Conditional_Association'],
                          -np.log10(causal_effects_df['P_Value']),
                          c=causal_effects_df['P_Value'] < 0.05,
                          cmap='viridis', alpha=0.6, s=50)
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Conditional Association Size')
    ax2.set_ylabel('-log10(P-Value)')
    ax2.set_title('Association Size vs Statistical Significance')
    ax2.legend()

    # Bottom: Traditional vs conditional association
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(comparison_df['traditional_importance'],
                comparison_df['Conditional_Association'],
                alpha=0.7, s=80,
                c=comparison_df['P_Value'] < 0.05, cmap='coolwarm')

    for idx, row in comparison_df.nlargest(5, 'traditional_importance').iterrows():
        ax3.annotate(row['Predictor'],
                     (row['traditional_importance'], row['Conditional_Association']),
                     xytext=(8, 8), textcoords='offset points', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax3.set_xlabel('Traditional Feature Importance')
    ax3.set_ylabel('Conditional Association (DML)')
    ax3.set_title('Traditional Importance vs DML Conditional Associations')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '8_comprehensive_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()


create_summary_plot()

# 9. Save results to Excel
output_file_path = os.path.join(output_dir, 'dml_conditional_associations_results.xlsx')
with pd.ExcelWriter(output_file_path) as writer:
    causal_effects_df.to_excel(writer, sheet_name='All_Conditional_Associations', index=False)
    significant_df.to_excel(writer, sheet_name='Significant_Associations', index=False)
    comparison_df.to_excel(writer, sheet_name='Comparison_with_Traditional', index=False)

print(f"\nAnalysis complete!")
print(f"Results saved to: {output_file_path}")
print(f"Plots saved to: {plots_dir}")
print(f"Total features analyzed: {len(causal_effects_df)}")
print(f"Significant features (p < 0.05): {len(causal_effects_df[causal_effects_df['P_Value'] < 0.05])}")

# Print top 5 positive and negative conditional associations
print("\nTop 5 Positive Conditional Associations:")
print(causal_effects_df.nlargest(5, 'Conditional_Association')[['Predictor', 'Conditional_Association', 'P_Value']])

print("\nTop 5 Negative Conditional Associations:")
print(causal_effects_df.nsmallest(5, 'Conditional_Association')[['Predictor', 'Conditional_Association', 'P_Value']])

# =============================================================================
# SAVE TOP 10 PREDICTORS WITH LOWEST P-VALUES
# =============================================================================

print("\n" + "=" * 60)
print("SAVING TOP 10 PREDICTORS WITH LOWEST P-VALUES")
print("=" * 60)

top_10_low_pvalue = causal_effects_df.nsmallest(10, 'P_Value')

print("\nTop 10 Predictors with Lowest P-Values (Most Statistically Significant):")
print("-" * 80)
for i, (idx, row) in enumerate(top_10_low_pvalue.iterrows(), 1):
    significance = ""
    if row['P_Value'] < 0.001:
        significance = "***"
    elif row['P_Value'] < 0.01:
        significance = "**"
    elif row['P_Value'] < 0.05:
        significance = "*"
    print(
        f"{i:2d}. {row['Predictor']:50} Association: {row['Conditional_Association']:8.3f}  p-value: {row['P_Value']:.6f} {significance}")

# Save to Excel
output_file_pvalue = os.path.join(output_dir, 'top_10_lowest_pvalue_predictors.xlsx')
top_10_low_pvalue[['Predictor', 'Conditional_Association', 'P_Value', 'CI_Lower', 'CI_Upper']].to_excel(
    output_file_pvalue, index=False)

# Create visualization
plt.figure(figsize=(14, 10))
colors = ['green' if p < 0.05 else 'orange' for p in top_10_low_pvalue['P_Value']]
bars = plt.barh(range(len(top_10_low_pvalue)), top_10_low_pvalue['Conditional_Association'],
                color=colors, alpha=0.7, edgecolor='black')

plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)

# Create custom y-tick labels
y_tick_labels = []
for predictor, p_val, assoc in zip(top_10_low_pvalue['Predictor'],
                                   top_10_low_pvalue['P_Value'],
                                   top_10_low_pvalue['Conditional_Association']):
    label = f"{predictor}\np={p_val:.4f}, Assoc={assoc:.3f}"
    y_tick_labels.append(label)

plt.yticks(range(len(top_10_low_pvalue)), y_tick_labels)
plt.xlabel('Conditional Association (DML Estimate)', fontsize=12)
plt.ylabel('Predictor', fontsize=12)
plt.title('Top 10 Predictors with Lowest P-Values\n(Green: p < 0.05, Orange: p ≥ 0.05)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_10_lowest_pvalue_visualization.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Results saved successfully!")
print(f"📊 Excel file: {output_file_pvalue}")
print(f"🖼️  Visualization: {os.path.join(plots_dir, 'top_10_lowest_pvalue_visualization.png')}")

print(f"\n📈 Summary Statistics for Top 10 Low P-Value Predictors:")
print(f"   • Minimum P-Value: {top_10_low_pvalue['P_Value'].min():.6f}")
print(f"   • Maximum P-Value: {top_10_low_pvalue['P_Value'].max():.6f}")
print(f"   • Mean P-Value: {top_10_low_pvalue['P_Value'].mean():.6f}")
print(f"   • Number with p < 0.05: {len(top_10_low_pvalue[top_10_low_pvalue['P_Value'] < 0.05])}/10")
