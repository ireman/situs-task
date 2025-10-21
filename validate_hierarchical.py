"""
Hierarchical Forecasting Validation
Compares baseline vs hierarchical forecasting on holdout data (2020)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("HIERARCHICAL FORECASTING VALIDATION")
print("=" * 80)
print("\nStrategy: Train on 2018-2019, Test on 2020")
print("Compare: Baseline (individual) vs Hierarchical (Coca Cola family)")

# Load data
data = pd.read_excel('monthly_beverage_orders 2018-2020.xlsx')
df = data.rename(columns={'Name': 'beverage', 'Year': 'year', 'Month': 'month', 'Quantity': 'quantity'})
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.sort_values(['beverage', 'date'])

# Split into train (2018-2019) and test (2020)
train_df = df[df['year'] < 2020].copy()
test_df = df[df['year'] == 2020].copy()

print(f"\nTrain data: {len(train_df)} records (2018-2019)")
print(f"Test data: {len(test_df)} records (2020)")

# Coca Cola family definition
coca_cola_family = ['Coca Cola Classic 500ml', 'Coca Cola Zero 500ml', 'Diet Coca Cola 500ml']
other_beverages = [b for b in df['beverage'].unique() if b not in coca_cola_family]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(data, label_encoder=None, fit_encoder=False):
    """Create features for modeling"""
    df = data.copy()

    # Encode beverages
    if fit_encoder:
        label_encoder = LabelEncoder()
        df['beverage_encoded'] = label_encoder.fit_transform(df['beverage'])
    else:
        df['beverage_encoded'] = label_encoder.transform(df['beverage'])

    # Time features
    min_date = df['date'].min()
    df['month_num'] = df['month']
    df['quarter'] = df['date'].dt.quarter
    df['time_idx'] = ((df['date'].dt.year - min_date.year) * 12 +
                      (df['date'].dt.month - min_date.month))
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag and rolling features per beverage
    all_data = []
    for beverage in df['beverage'].unique():
        bev_data = df[df['beverage'] == beverage].copy().sort_values('date')

        # Lags
        for lag in [1, 2, 3, 6, 12]:
            bev_data[f'lag_{lag}'] = bev_data['quantity'].shift(lag)

        # Rolling
        for window in [3, 6, 12]:
            bev_data[f'rolling_mean_{window}'] = (
                bev_data['quantity'].shift(1).rolling(window, min_periods=1).mean()
            )
            bev_data[f'rolling_std_{window}'] = (
                bev_data['quantity'].shift(1).rolling(window, min_periods=1).std()
            )

        all_data.append(bev_data)

    result = pd.concat(all_data, ignore_index=True)

    # Fill NaN
    for col in result.columns:
        if result[col].dtype in ['float64', 'int64'] and col != 'quantity':
            result[col].fillna(result[col].median(), inplace=True)

    return result, label_encoder

print("\nCreating features...")
train_features, label_encoder = create_features(train_df, fit_encoder=True)

# Train model
feature_cols = [
    'beverage_encoded', 'time_idx', 'month_num', 'quarter',
    'month_sin', 'month_cos',
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
    'rolling_std_3', 'rolling_std_6', 'rolling_std_12'
]

X_train = train_features[feature_cols]
y_train = train_features['quantity']

print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ============================================================================
# BASELINE APPROACH: Individual forecasting for all beverages
# ============================================================================

print("\n" + "=" * 80)
print("BASELINE: Individual Forecasting for All Beverages")
print("=" * 80)

baseline_forecasts = []

for beverage in df['beverage'].unique():
    # Get training data for this beverage
    bev_train = train_features[train_features['beverage'] == beverage].copy()

    # Combine train with test dates (to forecast)
    bev_test_dates = test_df[test_df['beverage'] == beverage][['beverage', 'year', 'month', 'date']].copy()

    # Create combined data
    all_dates = pd.concat([
        bev_train[['beverage', 'date', 'year', 'month', 'quantity']],
        bev_test_dates.assign(quantity=np.nan)
    ], ignore_index=True).sort_values('date').reset_index(drop=True)

    hist_end = all_dates['quantity'].notna().sum()

    # Forecast iteratively
    for i in range(hist_end, len(all_dates)):
        current_quantities = all_dates['quantity'].iloc[:i].values
        current_row = all_dates.iloc[i]

        min_date = train_df['date'].min()
        features = {
            'beverage_encoded': label_encoder.transform([beverage])[0],
            'time_idx': ((current_row['year'] - min_date.year) * 12 + (current_row['month'] - min_date.month)),
            'month_num': current_row['month'],
            'quarter': (current_row['month'] - 1) // 3 + 1,
            'month_sin': np.sin(2 * np.pi * current_row['month'] / 12),
            'month_cos': np.cos(2 * np.pi * current_row['month'] / 12),
        }

        for lag in [1, 2, 3, 6, 12]:
            features[f'lag_{lag}'] = current_quantities[i - lag] if i >= lag else np.median(current_quantities)

        for window in [3, 6, 12]:
            if i >= window:
                features[f'rolling_mean_{window}'] = np.mean(current_quantities[i-window:i])
                features[f'rolling_std_{window}'] = np.std(current_quantities[i-window:i])
            else:
                features[f'rolling_mean_{window}'] = np.mean(current_quantities[:i])
                features[f'rolling_std_{window}'] = np.std(current_quantities[:i]) if i > 1 else 0

        X_forecast = np.array([[features[col] for col in feature_cols]])
        prediction = max(0, model.predict(X_forecast)[0])
        all_dates.loc[i, 'quantity'] = prediction

    # Extract 2020 forecasts
    forecasts_2020 = all_dates[all_dates['year'] == 2020].copy()
    baseline_forecasts.append(forecasts_2020)

baseline_df = pd.concat(baseline_forecasts, ignore_index=True)

# ============================================================================
# HIERARCHICAL APPROACH: Hierarchical for Coca Cola, Individual for others
# ============================================================================

print("\n" + "=" * 80)
print("HIERARCHICAL: Coca Cola Family + Individual for Others")
print("=" * 80)

# Calculate Coca Cola family proportions from training data
family_train = train_df[train_df['beverage'].isin(coca_cola_family)]
total_by_beverage = family_train.groupby('beverage')['quantity'].sum()
proportions = total_by_beverage / total_by_beverage.sum()

print(f"\nCoca Cola family proportions (from 2018-2019):")
for bev, prop in proportions.items():
    print(f"  {bev:30s}: {prop:.1%}")

# Forecast Coca Cola family hierarchically
coca_forecasts = []

for beverage in coca_cola_family:
    bev_train = train_features[train_features['beverage'] == beverage].copy()
    bev_test_dates = test_df[test_df['beverage'] == beverage][['beverage', 'year', 'month', 'date']].copy()

    all_dates = pd.concat([
        bev_train[['beverage', 'date', 'year', 'month', 'quantity']],
        bev_test_dates.assign(quantity=np.nan)
    ], ignore_index=True).sort_values('date').reset_index(drop=True)

    hist_end = all_dates['quantity'].notna().sum()

    # Forecast (same as baseline)
    for i in range(hist_end, len(all_dates)):
        current_quantities = all_dates['quantity'].iloc[:i].values
        current_row = all_dates.iloc[i]

        min_date = train_df['date'].min()
        features = {
            'beverage_encoded': label_encoder.transform([beverage])[0],
            'time_idx': ((current_row['year'] - min_date.year) * 12 + (current_row['month'] - min_date.month)),
            'month_num': current_row['month'],
            'quarter': (current_row['month'] - 1) // 3 + 1,
            'month_sin': np.sin(2 * np.pi * current_row['month'] / 12),
            'month_cos': np.cos(2 * np.pi * current_row['month'] / 12),
        }

        for lag in [1, 2, 3, 6, 12]:
            features[f'lag_{lag}'] = current_quantities[i - lag] if i >= lag else np.median(current_quantities)

        for window in [3, 6, 12]:
            if i >= window:
                features[f'rolling_mean_{window}'] = np.mean(current_quantities[i-window:i])
                features[f'rolling_std_{window}'] = np.std(current_quantities[i-window:i])
            else:
                features[f'rolling_mean_{window}'] = np.mean(current_quantities[:i])
                features[f'rolling_std_{window}'] = np.std(current_quantities[:i]) if i > 1 else 0

        X_forecast = np.array([[features[col] for col in feature_cols]])
        prediction = max(0, model.predict(X_forecast)[0])
        all_dates.loc[i, 'quantity'] = prediction

    forecasts_2020 = all_dates[all_dates['year'] == 2020].copy()
    coca_forecasts.append(forecasts_2020)

coca_df = pd.concat(coca_forecasts, ignore_index=True)

# Reconcile using proportions
print("\nReconciling Coca Cola forecasts using proportions...")
monthly_totals = coca_df.groupby(['year', 'month'])['quantity'].transform('sum')

hierarchical_coca = []
for beverage in coca_cola_family:
    bev_mask = coca_df['beverage'] == beverage
    bev_data = coca_df[bev_mask].copy()
    bev_data['quantity'] = monthly_totals[bev_mask] * proportions[beverage]
    hierarchical_coca.append(bev_data)

hierarchical_coca_df = pd.concat(hierarchical_coca, ignore_index=True)

# Other beverages use baseline
hierarchical_df = pd.concat([
    hierarchical_coca_df,
    baseline_df[baseline_df['beverage'].isin(other_beverages)]
], ignore_index=True).sort_values(['beverage', 'date'])

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION ON 2020 HOLDOUT DATA")
print("=" * 80)

# Merge with actuals
baseline_eval = baseline_df.merge(
    test_df[['beverage', 'year', 'month', 'quantity']],
    on=['beverage', 'year', 'month'],
    suffixes=('_pred', '_actual')
)

hierarchical_eval = hierarchical_df.merge(
    test_df[['beverage', 'year', 'month', 'quantity']],
    on=['beverage', 'year', 'month'],
    suffixes=('_pred', '_actual')
)

# Overall metrics
baseline_mae = mean_absolute_error(baseline_eval['quantity_actual'], baseline_eval['quantity_pred'])
baseline_rmse = np.sqrt(mean_squared_error(baseline_eval['quantity_actual'], baseline_eval['quantity_pred']))

hierarchical_mae = mean_absolute_error(hierarchical_eval['quantity_actual'], hierarchical_eval['quantity_pred'])
hierarchical_rmse = np.sqrt(mean_squared_error(hierarchical_eval['quantity_actual'], hierarchical_eval['quantity_pred']))

print("\nOVERALL PERFORMANCE (All Beverages):")
print(f"\nBaseline (Individual forecasting):")
print(f"  MAE:  {baseline_mae:.3f}")
print(f"  RMSE: {baseline_rmse:.3f}")

print(f"\nHierarchical (Coca Cola family + others):")
print(f"  MAE:  {hierarchical_mae:.3f}")
print(f"  RMSE: {hierarchical_rmse:.3f}")

improvement_mae = ((baseline_mae - hierarchical_mae) / baseline_mae) * 100
improvement_rmse = ((baseline_rmse - hierarchical_rmse) / baseline_rmse) * 100

print(f"\nImprovement:")
print(f"  MAE:  {improvement_mae:+.1f}% {'(better)' if improvement_mae > 0 else '(worse)'}")
print(f"  RMSE: {improvement_rmse:+.1f}% {'(better)' if improvement_rmse > 0 else '(worse)'}")

# Per-beverage breakdown
print("\n" + "=" * 80)
print("PER-BEVERAGE PERFORMANCE")
print("=" * 80)

results = []
for beverage in df['beverage'].unique():
    baseline_bev = baseline_eval[baseline_eval['beverage'] == beverage]
    hierarchical_bev = hierarchical_eval[hierarchical_eval['beverage'] == beverage]

    if len(baseline_bev) > 0:
        base_mae = mean_absolute_error(baseline_bev['quantity_actual'], baseline_bev['quantity_pred'])
        hier_mae = mean_absolute_error(hierarchical_bev['quantity_actual'], hierarchical_bev['quantity_pred'])

        improvement = ((base_mae - hier_mae) / base_mae) * 100 if base_mae > 0 else 0

        results.append({
            'Beverage': beverage,
            'Baseline MAE': base_mae,
            'Hierarchical MAE': hier_mae,
            'Improvement %': improvement,
            'Method': 'Hierarchical' if beverage in coca_cola_family else 'Same'
        })

results_df = pd.DataFrame(results).sort_values('Improvement %', ascending=False)
print("\n" + results_df.to_string(index=False))

# Focus on Coca Cola family
print("\n" + "=" * 80)
print("COCA COLA FAMILY ANALYSIS")
print("=" * 80)

coca_baseline = baseline_eval[baseline_eval['beverage'].isin(coca_cola_family)]
coca_hierarchical = hierarchical_eval[hierarchical_eval['beverage'].isin(coca_cola_family)]

coca_base_mae = mean_absolute_error(coca_baseline['quantity_actual'], coca_baseline['quantity_pred'])
coca_hier_mae = mean_absolute_error(coca_hierarchical['quantity_actual'], coca_hierarchical['quantity_pred'])

print(f"\nCoca Cola Family Only:")
print(f"  Baseline MAE:     {coca_base_mae:.3f}")
print(f"  Hierarchical MAE: {coca_hier_mae:.3f}")
print(f"  Improvement:      {((coca_base_mae - coca_hier_mae) / coca_base_mae * 100):+.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Overall comparison
ax = axes[0, 0]
metrics = ['MAE', 'RMSE']
baseline_vals = [baseline_mae, baseline_rmse]
hierarchical_vals = [hierarchical_mae, hierarchical_rmse]

x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, hierarchical_vals, width, label='Hierarchical', alpha=0.8)
ax.set_xlabel('Metric')
ax.set_ylabel('Error')
ax.set_title('Overall Performance Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Per-beverage improvement
ax = axes[0, 1]
colors = ['green' if x > 0 else 'red' for x in results_df['Improvement %']]
ax.barh(results_df['Beverage'], results_df['Improvement %'], color=colors, alpha=0.7)
ax.set_xlabel('Improvement (%)')
ax.set_title('Per-Beverage Improvement with Hierarchical', fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Highlight Coca Cola products
for idx, row in results_df.iterrows():
    if row['Beverage'] in coca_cola_family:
        ax.text(row['Improvement %'] + 0.5, idx, '★', fontsize=12, va='center', color='gold')

# Plot 3: Actual vs Predicted (Coca Cola family)
ax = axes[1, 0]
for beverage in coca_cola_family:
    bev_data = coca_hierarchical[coca_hierarchical['beverage'] == beverage].sort_values('month')
    ax.plot(bev_data['month'], bev_data['quantity_actual'], marker='o', label=f'{beverage[:15]} (Actual)')
    ax.plot(bev_data['month'], bev_data['quantity_pred'], marker='s', linestyle='--', label=f'{beverage[:15]} (Pred)')

ax.set_xlabel('Month (2020)')
ax.set_ylabel('Quantity')
ax.set_title('Coca Cola Family: Actual vs Hierarchical Forecast (2020)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: MAE by beverage
ax = axes[1, 1]
x = np.arange(len(results_df))
width = 0.35

ax.bar(x - width/2, results_df['Baseline MAE'], width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, results_df['Hierarchical MAE'], width, label='Hierarchical', alpha=0.8)
ax.set_xlabel('Beverage')
ax.set_ylabel('MAE')
ax.set_title('MAE Comparison by Beverage', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Beverage'], rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hierarchical_validation_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved: hierarchical_validation_results.png")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if hierarchical_mae < baseline_mae:
    print(f"\n✅ HIERARCHICAL FORECASTING IMPROVES ACCURACY")
    print(f"   MAE reduced by {improvement_mae:.1f}% overall")
    if coca_hier_mae < coca_base_mae:
        coca_improvement = ((coca_base_mae - coca_hier_mae) / coca_base_mae * 100)
        print(f"   Coca Cola family improved by {coca_improvement:.1f}%")
else:
    print(f"\n❌ HIERARCHICAL FORECASTING DOES NOT IMPROVE ACCURACY")
    print(f"   MAE increased by {abs(improvement_mae):.1f}% overall")
    if coca_hier_mae < coca_base_mae:
        coca_improvement = ((coca_base_mae - coca_hier_mae) / coca_base_mae * 100)
        print(f"   But Coca Cola family improved by {coca_improvement:.1f}%")
    else:
        print(f"   Coca Cola family also worse by {abs(((coca_base_mae - coca_hier_mae) / coca_base_mae * 100)):.1f}%")

print("\n" + "=" * 80)
