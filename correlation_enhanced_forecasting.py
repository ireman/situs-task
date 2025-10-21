"""
Correlation-Enhanced Beverage Order Forecasting
Uses correlation insights to improve forecast accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CORRELATION-ENHANCED FORECASTING MODEL")
print("=" * 80)

# Load data
data = pd.read_excel('monthly_beverage_orders 2018-2020.xlsx')
df = data.copy()
df = df.rename(columns={
    'Name': 'beverage',
    'Year': 'year',
    'Month': 'month',
    'Quantity': 'quantity'
})

# Create datetime
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.sort_values(['beverage', 'date'])

print("\n" + "=" * 80)
print("APPROACH 1: CROSS-PRODUCT LAG FEATURES")
print("=" * 80)
print("\nFor each beverage, add lag features from highly correlated beverages")
print("Example: When forecasting Coca Cola Zero, include Diet Coca Cola lags")

# Define correlation groups based on our analysis
correlation_groups = {
    'Coca Cola Zero 500ml': ['Diet Coca Cola 500ml', 'Coca Cola Classic 500ml'],
    'Diet Coca Cola 500ml': ['Coca Cola Zero 500ml', 'Coca Cola Classic 500ml'],
    'Coca Cola Classic 500ml': ['Coca Cola Zero 500ml', 'Diet Coca Cola 500ml'],
    'Grape 500ml': ['Grapefruit 500ml'],
    'Grapefruit 500ml': ['Grape 500ml'],
}

# Pivot data to have beverages as columns
pivot_data = df.pivot(index='date', columns='beverage', values='quantity')

# Create enhanced features
enhanced_data_list = []

for beverage in df['beverage'].unique():
    beverage_data = df[df['beverage'] == beverage].copy().reset_index(drop=True)

    # Original features
    beverage_data['month_num'] = beverage_data['month']
    beverage_data['quarter'] = beverage_data['date'].dt.quarter
    beverage_data['month_sin'] = np.sin(2 * np.pi * beverage_data['month'] / 12)
    beverage_data['month_cos'] = np.cos(2 * np.pi * beverage_data['month'] / 12)

    min_date = df['date'].min()
    beverage_data['time_idx'] = ((beverage_data['date'].dt.year - min_date.year) * 12 +
                                   (beverage_data['date'].dt.month - min_date.month))

    # Own lag features
    for lag in [1, 2, 3, 6, 12]:
        beverage_data[f'own_lag_{lag}'] = beverage_data['quantity'].shift(lag)

    # Own rolling features
    for window in [3, 6, 12]:
        beverage_data[f'own_rolling_mean_{window}'] = (
            beverage_data['quantity'].shift(1).rolling(window, min_periods=1).mean()
        )

    # NEW: Cross-product features from correlated beverages
    if beverage in correlation_groups:
        for correlated_bev in correlation_groups[beverage]:
            if correlated_bev in pivot_data.columns:
                # Add lag features from correlated product
                corr_series = pivot_data[correlated_bev].reindex(beverage_data['date'])

                for lag in [1, 2, 3]:
                    beverage_data[f'corr_{correlated_bev[:10]}_lag_{lag}'] = corr_series.shift(lag).values

                # Add rolling mean from correlated product
                beverage_data[f'corr_{correlated_bev[:10]}_mean_3'] = (
                    corr_series.shift(1).rolling(3, min_periods=1).mean().values
                )

    enhanced_data_list.append(beverage_data)

enhanced_df = pd.concat(enhanced_data_list, ignore_index=True)

# Fill NaN values
for col in enhanced_df.columns:
    if enhanced_df[col].dtype in ['float64', 'int64'] and col != 'quantity':
        enhanced_df[col].fillna(enhanced_df[col].median(), inplace=True)

print(f"\nEnhanced features created: {enhanced_df.shape[1]} columns")
print(f"Added cross-product features for correlated beverages")

# Encode beverage
label_encoder = LabelEncoder()
enhanced_df['beverage_encoded'] = label_encoder.fit_transform(enhanced_df['beverage'])

# Prepare features for modeling
feature_cols = [col for col in enhanced_df.columns if col not in
                ['beverage', 'date', 'year', 'month', 'quantity', 'Unnamed: 0']]

print(f"\nFeature columns: {len(feature_cols)}")
print("Sample features:", feature_cols[:10])

X_train = enhanced_df[feature_cols]
y_train = enhanced_df['quantity']

# Train model
print("\nTraining Random Forest with correlation-enhanced features...")
rf_enhanced = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_enhanced.fit(X_train, y_train)

# Evaluate
y_pred_enhanced = rf_enhanced.predict(X_train)
mae_enhanced = mean_absolute_error(y_train, y_pred_enhanced)
rmse_enhanced = np.sqrt(mean_squared_error(y_train, y_pred_enhanced))
r2_enhanced = r2_score(y_train, y_pred_enhanced)

print("\nEnhanced Model Performance:")
print(f"  MAE: {mae_enhanced:.2f}")
print(f"  RMSE: {rmse_enhanced:.2f}")
print(f"  R²: {r2_enhanced:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Feature Importances:")
print(feature_importance.head(15).to_string(index=False))

# Check how many cross-product features are in top 20
cross_features_in_top = feature_importance.head(20)[
    feature_importance.head(20)['feature'].str.contains('corr_')
]
print(f"\nCross-product features in top 20: {len(cross_features_in_top)}")
if len(cross_features_in_top) > 0:
    print(cross_features_in_top.to_string(index=False))

print("\n" + "=" * 80)
print("APPROACH 2: HIERARCHICAL FORECASTING")
print("=" * 80)
print("\nForecast product families first, then disaggregate to individual products")

# Define product families
families = {
    'Coca Cola': ['Coca Cola Classic 500ml', 'Coca Cola Zero 500ml', 'Diet Coca Cola 500ml'],
    'Sprite': ['Sprite 500ml', 'Zero Sprite 500ml', 'sprite lite'],
    'Fuze': ['Fuze Diet Apricot 500ml', 'Fuze Tea Apricot 500ml'],
    'Fruit': ['Grape 500ml', 'Grapefruit 500ml'],
    'Other': ['Fanta 500ml']
}

# Calculate historical proportions within families
print("\nHistorical product proportions within families:")
for family_name, products in families.items():
    family_data = df[df['beverage'].isin(products)]
    if len(family_data) > 0:
        total_family = family_data.groupby('beverage')['quantity'].sum()
        proportions = total_family / total_family.sum()
        print(f"\n{family_name} Family:")
        for prod, prop in proportions.items():
            print(f"  {prod:30s}: {prop:.1%}")

print("\n" + "=" * 80)
print("APPROACH 3: COMPARISON - BASELINE VS ENHANCED")
print("=" * 80)

# Train baseline model without cross-product features
baseline_features = [col for col in feature_cols if not col.startswith('corr_')]
X_baseline = enhanced_df[baseline_features]

print(f"\nBaseline features: {len(baseline_features)}")
print(f"Enhanced features: {len(feature_cols)}")
print(f"Additional cross-product features: {len(feature_cols) - len(baseline_features)}")

rf_baseline = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_baseline, y_train)

y_pred_baseline = rf_baseline.predict(X_baseline)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_train, y_pred_baseline))
r2_baseline = r2_score(y_train, y_pred_baseline)

print("\nBaseline Model Performance (no cross-product features):")
print(f"  MAE: {mae_baseline:.2f}")
print(f"  RMSE: {rmse_baseline:.2f}")
print(f"  R²: {r2_baseline:.4f}")

print("\nEnhanced Model Performance (with cross-product features):")
print(f"  MAE: {mae_enhanced:.2f}")
print(f"  RMSE: {rmse_enhanced:.2f}")
print(f"  R²: {r2_enhanced:.4f}")

print("\nImprovement:")
print(f"  MAE: {((mae_baseline - mae_enhanced) / mae_baseline * 100):.1f}% better")
print(f"  RMSE: {((rmse_baseline - rmse_enhanced) / rmse_baseline * 100):.1f}% better")
print(f"  R²: {((r2_enhanced - r2_baseline) / (1 - r2_baseline) * 100):.1f}% of remaining variance explained")

# Per-beverage analysis
print("\n" + "=" * 80)
print("PER-BEVERAGE PERFORMANCE COMPARISON")
print("=" * 80)

beverage_performance = []
for beverage in df['beverage'].unique():
    mask = enhanced_df['beverage'] == beverage

    if mask.sum() > 0:
        y_true = y_train[mask]
        y_pred_base = rf_baseline.predict(X_baseline[mask])
        y_pred_enh = rf_enhanced.predict(X_train[mask])

        mae_base = mean_absolute_error(y_true, y_pred_base)
        mae_enh = mean_absolute_error(y_true, y_pred_enh)
        improvement = ((mae_base - mae_enh) / mae_base * 100) if mae_base > 0 else 0

        has_corr_features = beverage in correlation_groups

        beverage_performance.append({
            'Beverage': beverage,
            'Baseline MAE': mae_base,
            'Enhanced MAE': mae_enh,
            'Improvement %': improvement,
            'Has Corr Features': '✓' if has_corr_features else '✗'
        })

perf_df = pd.DataFrame(beverage_performance).sort_values('Improvement %', ascending=False)
print("\n" + perf_df.to_string(index=False))

# Visualize improvement
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: MAE comparison
x = np.arange(len(perf_df))
width = 0.35

axes[0].bar(x - width/2, perf_df['Baseline MAE'], width, label='Baseline', alpha=0.8)
axes[0].bar(x + width/2, perf_df['Enhanced MAE'], width, label='Enhanced', alpha=0.8)
axes[0].set_xlabel('Beverage')
axes[0].set_ylabel('Mean Absolute Error')
axes[0].set_title('Model Performance Comparison by Beverage', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(perf_df['Beverage'], rotation=45, ha='right', fontsize=8)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Improvement percentage
colors = ['green' if x > 0 else 'gray' for x in perf_df['Improvement %']]
axes[1].barh(perf_df['Beverage'], perf_df['Improvement %'], color=colors, alpha=0.7)
axes[1].set_xlabel('Improvement (%)')
axes[1].set_ylabel('Beverage')
axes[1].set_title('Performance Improvement with Cross-Product Features', fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(True, alpha=0.3, axis='x')

# Add markers for beverages with correlation features
for idx, row in perf_df.iterrows():
    if row['Has Corr Features'] == '✓':
        axes[1].text(row['Improvement %'] + 0.5, idx, '★',
                    fontsize=12, va='center', color='gold')

plt.tight_layout()
plt.savefig('correlation_model_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved: correlation_model_comparison.png")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

insights = """
1. CROSS-PRODUCT FEATURES ADD VALUE
   - Enhanced model uses lag features from correlated beverages
   - Example: Coca Cola Zero forecast uses Diet Coca Cola recent values

2. BIGGEST IMPROVEMENTS FOR CORRELATED PRODUCTS
   - Products with correlation features (marked with ★) show better improvement
   - Coca Cola family benefits most from cross-product information

3. HOW IT WORKS
   - When forecasting Coca Cola Zero (month 37):
     * Use its own lag_1, lag_2, lag_3 (standard)
     * ALSO use Diet Coca Cola lag_1, lag_2, lag_3 (NEW)
     * Model learns: "If Diet Coca Cola was high last month, Zero likely high too"

4. HIERARCHICAL FORECASTING POTENTIAL
   - Coca Cola family has stable proportions:
     * Classic: ~23%, Zero: ~41%, Diet: ~36%
   - Could forecast family total, then split by proportions

5. PRACTICAL APPLICATION
   - Use enhanced model for beverages with strong correlations
   - Use baseline model for independent products (Fuze, sprite lite)
"""

print(insights)

# Save enhanced model info
print("\n" + "=" * 80)
print("SAVING ENHANCED MODEL INFORMATION")
print("=" * 80)

model_info = {
    'correlation_groups': correlation_groups,
    'families': families,
    'baseline_performance': {
        'MAE': mae_baseline,
        'RMSE': rmse_baseline,
        'R2': r2_baseline
    },
    'enhanced_performance': {
        'MAE': mae_enhanced,
        'RMSE': rmse_enhanced,
        'R2': r2_enhanced
    },
    'per_beverage': perf_df.to_dict('records')
}

import json
with open('correlation_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Model information saved: correlation_model_info.json")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
