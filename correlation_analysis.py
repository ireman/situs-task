"""
Beverage Correlation Analysis
Analyzes correlation patterns between different beverage products
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_excel('monthly_beverage_orders 2018-2020.xlsx')

# Pivot the data to have beverages as columns and months as rows
pivot_data = data.pivot(
    index=['Year', 'Month'],
    columns='Name',
    values='Quantity'
)

print("=" * 80)
print("BEVERAGE CORRELATION ANALYSIS")
print("=" * 80)

print("\nDataset shape (after pivot):", pivot_data.shape)
print("Beverages:", list(pivot_data.columns))

# Calculate correlation matrix
correlation_matrix = pivot_data.corr()

print("\n" + "=" * 80)
print("CORRELATION MATRIX")
print("=" * 80)
print(correlation_matrix.round(3))

# Find highly correlated pairs (>0.7 or <-0.7, excluding self-correlation)
print("\n" + "=" * 80)
print("HIGHLY CORRELATED BEVERAGE PAIRS (|r| > 0.7)")
print("=" * 80)

high_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:
            high_correlations.append({
                'Beverage 1': correlation_matrix.columns[i],
                'Beverage 2': correlation_matrix.columns[j],
                'Correlation': corr_value
            })

if high_correlations:
    high_corr_df = pd.DataFrame(high_correlations).sort_values('Correlation', ascending=False)
    print(high_corr_df.to_string(index=False))
else:
    print("No beverage pairs found with correlation > 0.7")

# Find moderately correlated pairs (0.5-0.7)
print("\n" + "=" * 80)
print("MODERATELY CORRELATED BEVERAGE PAIRS (0.5 < |r| < 0.7)")
print("=" * 80)

moderate_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if 0.5 < abs(corr_value) <= 0.7:
            moderate_correlations.append({
                'Beverage 1': correlation_matrix.columns[i],
                'Beverage 2': correlation_matrix.columns[j],
                'Correlation': corr_value
            })

if moderate_correlations:
    mod_corr_df = pd.DataFrame(moderate_correlations).sort_values('Correlation', ascending=False)
    print(mod_corr_df.to_string(index=False))
else:
    print("No beverage pairs found with 0.5 < correlation < 0.7")

# Visualize correlation matrix
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Heatmap 1: Full correlation matrix
sns.heatmap(correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=axes[0],
            vmin=-1, vmax=1)
axes[0].set_title('Beverage Correlation Matrix\n(2018-2020)',
                  fontsize=14, fontweight='bold', pad=20)
axes[0].set_xlabel('')
axes[0].set_ylabel('')
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=9)

# Heatmap 2: Masked correlation (show only significant correlations)
mask = np.abs(correlation_matrix) < 0.3  # Mask weak correlations
masked_corr = correlation_matrix.copy()
masked_corr[mask] = 0

sns.heatmap(masked_corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=axes[1],
            vmin=-1, vmax=1)
axes[1].set_title('Strong Correlations Only\n(|r| ≥ 0.3)',
                  fontsize=14, fontweight='bold', pad=20)
axes[1].set_xlabel('')
axes[1].set_ylabel('')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('beverage_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("Visualization saved: beverage_correlation_analysis.png")
print("=" * 80)

# Additional insights
print("\n" + "=" * 80)
print("CORRELATION INSIGHTS")
print("=" * 80)

# Average correlation for each beverage
avg_corr = correlation_matrix.abs().mean().sort_values(ascending=False)
print("\nAverage absolute correlation with other beverages:")
for bev, corr in avg_corr.items():
    # Exclude self-correlation
    other_corr = correlation_matrix[bev].drop(bev).abs().mean()
    print(f"  {bev:30s}: {other_corr:.3f}")

# Identify product groups
print("\n" + "=" * 80)
print("PRODUCT FAMILY ANALYSIS")
print("=" * 80)

# Group by product families
coca_cola = [col for col in pivot_data.columns if 'Coca Cola' in col]
sprite = [col for col in pivot_data.columns if 'Sprite' in col or 'sprite' in col]
fuze = [col for col in pivot_data.columns if 'Fuze' in col]
fruit = [col for col in pivot_data.columns if any(f in col for f in ['Grape', 'Grapefruit'])]
other = [col for col in pivot_data.columns if col not in coca_cola + sprite + fuze + fruit]

families = {
    'Coca Cola Family': coca_cola,
    'Sprite Family': sprite,
    'Fuze Tea Family': fuze,
    'Fruit Juices': fruit,
    'Other': other
}

for family_name, beverages in families.items():
    if len(beverages) > 0:
        print(f"\n{family_name}:")
        print(f"  Products: {', '.join(beverages)}")
        if len(beverages) > 1:
            family_corr = correlation_matrix.loc[beverages, beverages]
            # Get average correlation within family (excluding diagonal)
            mask = np.triu(np.ones_like(family_corr), k=1).astype(bool)
            avg_within = family_corr.where(mask).mean().mean()
            print(f"  Average within-family correlation: {avg_within:.3f}")

# Time series plots for highly correlated pairs
if high_correlations:
    print("\n" + "=" * 80)
    print("Creating time series comparison plots for highly correlated pairs...")
    print("=" * 80)

    n_pairs = min(3, len(high_correlations))  # Show top 3 pairs
    fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 4*n_pairs))
    if n_pairs == 1:
        axes = [axes]

    for idx, pair_info in enumerate(high_corr_df.head(n_pairs).to_dict('records')):
        bev1 = pair_info['Beverage 1']
        bev2 = pair_info['Beverage 2']
        corr = pair_info['Correlation']

        # Create time index
        time_index = pd.date_time = pd.to_datetime(
            pivot_data.index.to_frame().assign(day=1)
        )

        ax = axes[idx]
        ax.plot(time_index, pivot_data[bev1], marker='o', label=bev1, linewidth=2)
        ax.plot(time_index, pivot_data[bev2], marker='s', label=bev2, linewidth=2)
        ax.set_title(f'{bev1} vs {bev2}\nCorrelation: {corr:.3f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('beverage_correlation_timeseries.png', dpi=300, bbox_inches='tight')
    print("Time series comparison saved: beverage_correlation_timeseries.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
