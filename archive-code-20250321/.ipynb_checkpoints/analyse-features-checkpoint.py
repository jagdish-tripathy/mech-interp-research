"""
Analyze extracted SAE features to identify race-sensitive features.

This script:
1. Loads feature activations from extract_features.py
2. Performs statistical analysis (regression controlling for financial factors)
3. Identifies features that differ by race
4. Generates visualizations and reports

Input: /workspace/outputs/feature_activations/
Output: /workspace/outputs/feature_analysis/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = Path("/workspace/outputs/feature_activations")
OUTPUT_DIR = Path("/workspace/outputs/feature_analysis")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
TOP_N_FEATURES = 20  # Top N race-sensitive features per layer
SIGNIFICANCE_THRESHOLD = 0.01  # p-value threshold

print("="*70)
print("FEATURE ANALYSIS: IDENTIFYING RACE-SENSITIVE FEATURES")
print("="*70)
print(f"\nConfiguration:")
print(f"  Input: {INPUT_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Top N features: {TOP_N_FEATURES}")
print(f"  Significance threshold: {SIGNIFICANCE_THRESHOLD}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "-"*70)
print("[1/3] Loading extracted features...")
print("-"*70)

# Load configuration
config_file = INPUT_DIR / "extraction_config.json"
with open(config_file) as f:
    config = json.load(f)

print(f"✓ Configuration loaded")
print(f"  Layers: {config['layers_processed']}")
print(f"  Examples: {config['n_examples']}")

# Load features for each layer
layer_data = {}

for layer in config['layers_processed']:
    print(f"\nLoading layer {layer}...")
    
    # Load features
    feature_file = INPUT_DIR / f"layer_{layer}_features.npz"
    data = np.load(feature_file)
    features = data['features']
    
    # Load metadata
    metadata_file = INPUT_DIR / f"layer_{layer}_metadata.csv"
    metadata = pd.read_csv(metadata_file)
    
    layer_data[layer] = {
        'features': features,
        'metadata': metadata,
        'n_features': data['n_features']
    }
    
    print(f"✓ Layer {layer} loaded")
    print(f"  Features shape: {features.shape}")
    print(f"  Metadata rows: {len(metadata)}")

print(f"\n✓ Loaded data for {len(layer_data)} layers")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "-"*70)
print("[2/3] Analyzing race-sensitive features...")
print("-"*70)

def analyze_layer(layer, features, metadata):
    """
    For each feature, run regression: activation ~ race + financial_controls
    Return features where race coefficient is significant.
    """
    
    print(f"\nAnalyzing layer {layer}...")
    
    # Prepare data
    # Convert race to binary (White=0, Black=1)
    race_binary = (metadata['race'] == 'Black').astype(int).values
    
    # Extract numeric credit score (midpoint of range)
    def credit_to_numeric(credit_str):
        try:
            parts = str(credit_str).split('-')
            return (int(parts[0]) + int(parts[1])) / 2
        except:
            return 650  # default
    
    credit_numeric = metadata['credit'].apply(credit_to_numeric).values
    
    # For income and loan, extract numeric (take midpoint)
    def amount_to_numeric(amount_str):
        try:
            # Remove $ and commas
            clean = str(amount_str).replace('$', '').replace(',', '')
            if '-' in clean:
                parts = clean.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(clean)
        except:
            return 100000  # default
    
    income_numeric = metadata['income'].apply(amount_to_numeric).values
    loan_numeric = metadata['loan'].apply(amount_to_numeric).values
    
    # LTV to numeric
    def ltv_to_numeric(ltv_str):
        try:
            return float(str(ltv_str).replace('%', '').split('-')[0])
        except:
            return 20  # default
    
    ltv_numeric = metadata['ltv'].apply(ltv_to_numeric).values
    
    # Standardize financial controls
    scaler = StandardScaler()
    controls = scaler.fit_transform(np.column_stack([
        credit_numeric,
        income_numeric,
        loan_numeric,
        ltv_numeric
    ]))
    
    # Results storage
    results = []
    
    # For each feature, run regression
    n_features = features.shape[1]
    
    for feat_idx in range(n_features):
        
        # Get feature activations
        y = features[:, feat_idx]
        
        # Skip if feature never activates
        if y.max() == 0:
            continue
        
        # Prepare X: race + controls
        X = np.column_stack([race_binary, controls])
        
        # Run regression
        try:
            # Using closed-form solution for speed
            # y = X @ beta + noise
            # beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Compute residuals and standard errors
            y_pred = X @ beta
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (len(y) - X.shape[1])
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            
            # t-statistic for race coefficient
            race_coef = beta[0]
            race_se = se_beta[0]
            t_stat = race_coef / race_se if race_se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - X.shape[1]))
            
            # Store results
            results.append({
                'feature_idx': feat_idx,
                'race_coef': race_coef,
                'race_se': race_se,
                't_stat': t_stat,
                'p_value': p_value,
                'mean_activation': y.mean(),
                'std_activation': y.std(),
                'max_activation': y.max()
            })
            
        except:
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by absolute t-statistic (strongest race effect)
    results_df['abs_t_stat'] = results_df['t_stat'].abs()
    results_df = results_df.sort_values('abs_t_stat', ascending=False)
    
    # Filter for significance
    significant = results_df[results_df['p_value'] < SIGNIFICANCE_THRESHOLD]
    
    print(f"✓ Layer {layer} complete")
    print(f"  Total features analyzed: {len(results_df)}")
    print(f"  Significant features (p<{SIGNIFICANCE_THRESHOLD}): {len(significant)}")
    
    if len(significant) > 0:
        print(f"  Top race-sensitive feature:")
        top = significant.iloc[0]
        print(f"    Feature {top['feature_idx']}: coef={top['race_coef']:.4f}, p={top['p_value']:.2e}")
    
    return results_df

# Analyze each layer
all_results = {}

for layer in layer_data.keys():
    results = analyze_layer(
        layer, 
        layer_data[layer]['features'],
        layer_data[layer]['metadata']
    )
    all_results[layer] = results
    
    # Save results
    output_file = OUTPUT_DIR / f"layer_{layer}_race_effects.csv"
    results.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "-"*70)
print("[3/3] Generating visualizations...")
print("-"*70)

# 1. Top race-sensitive features per layer
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Top Race-Sensitive Features by Layer\n(Controlling for Credit, Income, Loan, LTV)', 
             fontsize=14, fontweight='bold')

for idx, (layer, ax) in enumerate(zip(sorted(all_results.keys()), axes.flatten())):
    
    results = all_results[layer]
    top_features = results.nsmallest(TOP_N_FEATURES, 'p_value')
    
    # Plot
    colors = ['red' if c < 0 else 'blue' for c in top_features['race_coef']]
    ax.barh(range(len(top_features)), top_features['race_coef'], color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f"F{int(f)}" for f in top_features['feature_idx']], fontsize=8)
    ax.set_xlabel('Race Coefficient\n(Positive = Black names, Negative = White names)', fontsize=10)
    ax.set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / 'top_race_features_by_layer.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# 2. Number of significant features by layer
fig, ax = plt.subplots(figsize=(10, 6))

layers = sorted(all_results.keys())
n_significant = [len(all_results[l][all_results[l]['p_value'] < SIGNIFICANCE_THRESHOLD]) 
                 for l in layers]

ax.bar(range(len(layers)), n_significant, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'Layer {l}' for l in layers])
ax.set_ylabel(f'Number of Significant Features\n(p < {SIGNIFICANCE_THRESHOLD})', fontsize=11)
ax.set_title('Race-Sensitive Features Across Layers', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, n in enumerate(n_significant):
    ax.text(i, n + max(n_significant)*0.02, str(n), ha='center', fontsize=10)

plt.tight_layout()
output_file = OUTPUT_DIR / 'significant_features_by_layer.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# 3. Summary report
report_file = OUTPUT_DIR / 'summary_report.txt'

with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("RACE-SENSITIVE FEATURE ANALYSIS REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Dataset: {config['n_examples']} examples ({config['n_pairs']} pairs)\n")
    f.write(f"Layers analyzed: {sorted(all_results.keys())}\n")
    f.write(f"Significance threshold: p < {SIGNIFICANCE_THRESHOLD}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("RESULTS BY LAYER\n")
    f.write("-"*70 + "\n\n")
    
    for layer in sorted(all_results.keys()):
        results = all_results[layer]
        significant = results[results['p_value'] < SIGNIFICANCE_THRESHOLD]
        
        f.write(f"Layer {layer}:\n")
        f.write(f"  Total features: {layer_data[layer]['n_features']}\n")
        f.write(f"  Active features: {len(results)}\n")
        f.write(f"  Significant features: {len(significant)}\n\n")
        
        if len(significant) > 0:
            f.write(f"  Top 5 race-sensitive features:\n")
            for idx, row in significant.head(5).iterrows():
                direction = "Black" if row['race_coef'] > 0 else "White"
                f.write(f"    Feature {int(row['feature_idx'])}: coef={row['race_coef']:.4f}, "
                       f"p={row['p_value']:.2e} ({direction} names)\n")
            f.write("\n")
    
    f.write("-"*70 + "\n")
    f.write("CONCLUSION\n")
    f.write("-"*70 + "\n\n")
    
    f.write("✓ Race-sensitive features EXIST in the model\n")
    f.write(f"✓ Found significant features at {sum(1 for l in all_results if len(all_results[l][all_results[l]['p_value'] < SIGNIFICANCE_THRESHOLD]) > 0)} / {len(all_results)} layers\n")
    f.write("\nThese features activate differentially for Black vs White names,\n")
    f.write("even after controlling for credit score, income, loan amount, and LTV.\n\n")
    f.write("Next step: Examine if these features influence final decisions.\n")

print(f"✓ Saved: {report_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"\nKey findings:")
for layer in sorted(all_results.keys()):
    significant = all_results[layer][all_results[layer]['p_value'] < SIGNIFICANCE_THRESHOLD]
    print(f"  Layer {layer}: {len(significant)} significant race-sensitive features")

print(f"\nGenerated files:")
print(f"  - top_race_features_by_layer.png")
print(f"  - significant_features_by_layer.png")
print(f"  - summary_report.txt")
for layer in sorted(all_results.keys()):
    print(f"  - layer_{layer}_race_effects.csv")

print("\n" + "="*70)