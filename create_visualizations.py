"""
Create EDA and Model Performance Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
data = pd.read_excel('monthly_beverage_orders 2018-2020.xlsx')
df = data.rename(columns={'Name': 'beverage', 'Year': 'year', 'Month': 'month', 'Quantity': 'quantity'})
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

print("Creating EDA visualizations...")

# ============================================================================
# EDA VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(20, 12))

# 1. Distribution of order quantities
ax1 = plt.subplot(3, 3, 1)
df['quantity'].hist(bins=30, edgecolor='black', alpha=0.7, ax=ax1)
ax1.set_title('Distribution of Order Quantities', fontweight='bold', fontsize=12)
ax1.set_xlabel('Quantity (units)')
ax1.set_ylabel('Frequency')
ax1.axvline(df['quantity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["quantity"].mean():.1f}')
ax1.axvline(df['quantity'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["quantity"].median():.1f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Box plot by beverage
ax2 = plt.subplot(3, 3, 2)
beverage_order = df.groupby('beverage')['quantity'].median().sort_values(ascending=False)
df['beverage_cat'] = pd.Categorical(df['beverage'], categories=beverage_order.index, ordered=True)
df.boxplot(column='quantity', by='beverage_cat', ax=ax2, rot=45)
ax2.set_title('Order Quantity Distribution by Beverage', fontweight='bold', fontsize=12)
ax2.set_xlabel('')
ax2.set_ylabel('Quantity (units)')
plt.suptitle('')
ax2.tick_params(axis='x', labelsize=8)

# 3. Total orders by year
ax3 = plt.subplot(3, 3, 3)
yearly_totals = df.groupby('year')['quantity'].sum()
ax3.bar(yearly_totals.index, yearly_totals.values, alpha=0.7, edgecolor='black')
ax3.set_title('Total Orders by Year', fontweight='bold', fontsize=12)
ax3.set_xlabel('Year')
ax3.set_ylabel('Total Quantity')
ax3.grid(True, alpha=0.3, axis='y')
for i, (year, val) in enumerate(yearly_totals.items()):
    ax3.text(year, val, f'{int(val)}', ha='center', va='bottom', fontsize=10)

# 4. Monthly seasonality (average across all years)
ax4 = plt.subplot(3, 3, 4)
monthly_avg = df.groupby('month')['quantity'].mean()
ax4.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8)
ax4.set_title('Average Monthly Seasonality', fontweight='bold', fontsize=12)
ax4.set_xlabel('Month')
ax4.set_ylabel('Average Quantity')
ax4.set_xticks(range(1, 13))
ax4.grid(True, alpha=0.3)

# 5. Top 5 beverages by total volume
ax5 = plt.subplot(3, 3, 5)
top_beverages = df.groupby('beverage')['quantity'].sum().sort_values(ascending=True).tail(5)
ax5.barh(range(len(top_beverages)), top_beverages.values, alpha=0.7, edgecolor='black')
ax5.set_yticks(range(len(top_beverages)))
ax5.set_yticklabels([name[:25] for name in top_beverages.index], fontsize=9)
ax5.set_title('Top 5 Beverages by Total Volume', fontweight='bold', fontsize=12)
ax5.set_xlabel('Total Quantity (2018-2020)')
ax5.grid(True, alpha=0.3, axis='x')

# 6. Orders over time (total)
ax6 = plt.subplot(3, 3, 6)
time_series = df.groupby('date')['quantity'].sum()
ax6.plot(time_series.index, time_series.values, linewidth=2, marker='o', markersize=4)
ax6.set_title('Total Monthly Orders Over Time', fontweight='bold', fontsize=12)
ax6.set_xlabel('Date')
ax6.set_ylabel('Total Quantity')
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

# 7. Diet vs Regular products
ax7 = plt.subplot(3, 3, 7)
diet_keywords = ['diet', 'zero', 'lite']
df['is_diet'] = df['beverage'].str.lower().apply(
    lambda x: 'Diet/Zero' if any(keyword in x for keyword in diet_keywords) else 'Regular'
)
diet_totals = df.groupby('is_diet')['quantity'].sum()
colors = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = ax7.pie(diet_totals.values, labels=diet_totals.index, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
ax7.set_title('Diet/Zero vs Regular Products\n(Total Volume)', fontweight='bold', fontsize=12)

# 8. Quarterly comparison
ax8 = plt.subplot(3, 3, 8)
df['quarter'] = df['date'].dt.quarter
quarterly_avg = df.groupby('quarter')['quantity'].mean()
ax8.bar(quarterly_avg.index, quarterly_avg.values, alpha=0.7, edgecolor='black', color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
ax8.set_title('Average Orders by Quarter', fontweight='bold', fontsize=12)
ax8.set_xlabel('Quarter')
ax8.set_ylabel('Average Quantity')
ax8.set_xticks([1, 2, 3, 4])
ax8.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax8.grid(True, alpha=0.3, axis='y')

# 9. Volatility (std dev) by beverage
ax9 = plt.subplot(3, 3, 9)
volatility = df.groupby('beverage')['quantity'].std().sort_values(ascending=True).tail(8)
ax9.barh(range(len(volatility)), volatility.values, alpha=0.7, edgecolor='black', color='coral')
ax9.set_yticks(range(len(volatility)))
ax9.set_yticklabels([name[:25] for name in volatility.index], fontsize=9)
ax9.set_title('Top 8 Most Volatile Products', fontweight='bold', fontsize=12)
ax9.set_xlabel('Standard Deviation')
ax9.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ EDA visualizations saved: eda_visualizations.png")

# ============================================================================
# MODEL PERFORMANCE VISUALIZATIONS
# ============================================================================

# Recreate model performance metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Prepare data (simplified version)
df_model = df.copy()
df_model = df_model.sort_values(['beverage', 'date'])

# Create basic features
label_encoder = LabelEncoder()
df_model['beverage_encoded'] = label_encoder.fit_transform(df_model['beverage'])
df_model['month_num'] = df_model['month']
df_model['quarter'] = df_model['date'].dt.quarter
min_date = df_model['date'].min()
df_model['time_idx'] = ((df_model['date'].dt.year - min_date.year) * 12 +
                         (df_model['date'].dt.month - min_date.month))
df_model['month_sin'] = np.sin(2 * np.pi * df_model['month'] / 12)
df_model['month_cos'] = np.cos(2 * np.pi * df_model['month'] / 12)
df_model['is_diet'] = df['is_diet'].map({'Diet/Zero': 1, 'Regular': 0})
df_model['holiday'] = df_model['month'].apply(lambda x: 1 if x in [11, 12, 1] else 0)

# Create lag features
all_data = []
for beverage in df_model['beverage'].unique():
    bev_data = df_model[df_model['beverage'] == beverage].copy()
    for lag in [1, 2, 3]:
        bev_data[f'lag_{lag}'] = bev_data['quantity'].shift(lag)
    for window in [3, 6]:
        bev_data[f'rolling_mean_{window}'] = bev_data['quantity'].shift(1).rolling(window, min_periods=1).mean()
    all_data.append(bev_data)

df_model = pd.concat(all_data, ignore_index=True)
df_model = df_model.fillna(df_model.median(numeric_only=True))

# Train models
feature_cols = ['beverage_encoded', 'time_idx', 'month_num', 'quarter',
                'month_sin', 'month_cos', 'is_diet', 'holiday',
                'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_6']

X = df_model[feature_cols]
y = df_model['quantity']

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X, y)

# Predictions
y_pred_rf = rf_model.predict(X)
y_pred_gb = gb_model.predict(X)

# Metrics
rf_metrics = {
    'MAE': mean_absolute_error(y, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y, y_pred_rf)),
    'R²': r2_score(y, y_pred_rf)
}

gb_metrics = {
    'MAE': mean_absolute_error(y, y_pred_gb),
    'RMSE': np.sqrt(mean_squared_error(y, y_pred_gb)),
    'R²': r2_score(y, y_pred_gb)
}

# Create performance visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model comparison
ax = axes[0, 0]
metrics_names = ['MAE', 'RMSE']
rf_vals = [rf_metrics['MAE'], rf_metrics['RMSE']]
gb_vals = [gb_metrics['MAE'], gb_metrics['RMSE']]

x = np.arange(len(metrics_names))
width = 0.35

ax.bar(x - width/2, rf_vals, width, label='Random Forest', alpha=0.8, color='steelblue')
ax.bar(x + width/2, gb_vals, width, label='Gradient Boosting', alpha=0.8, color='coral')
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(rf_vals):
    ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
for i, v in enumerate(gb_vals):
    ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

# 2. R² comparison
ax = axes[0, 1]
models = ['Random Forest', 'Gradient Boosting']
r2_vals = [rf_metrics['R²'], gb_metrics['R²']]
colors_r2 = ['steelblue', 'coral']

bars = ax.bar(models, r2_vals, alpha=0.8, color=colors_r2, edgecolor='black')
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Model R² Comparison', fontweight='bold', fontsize=14)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, r2_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 3. Actual vs Predicted (Random Forest)
ax = axes[1, 0]
sample_size = min(200, len(y))
indices = np.random.choice(len(y), sample_size, replace=False)

ax.scatter(y.iloc[indices], y_pred_rf[indices], alpha=0.5, s=50, edgecolors='black', linewidths=0.5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Quantity', fontsize=12)
ax.set_ylabel('Predicted Quantity', fontsize=12)
ax.set_title('Random Forest: Actual vs Predicted', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Add R² annotation
ax.text(0.05, 0.95, f'R² = {rf_metrics["R²"]:.4f}\nMAE = {rf_metrics["MAE"]:.2f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Feature importance (Random Forest)
ax = axes[1, 1]
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True).tail(10)

ax.barh(range(len(feature_importance)), feature_importance['importance'].values,
        alpha=0.8, color='steelblue', edgecolor='black')
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['feature'].values, fontsize=10)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 10 Feature Importances (Random Forest)', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Add percentage labels
for i, val in enumerate(feature_importance['importance'].values):
    ax.text(val, i, f' {val*100:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Model performance visualizations saved: model_performance.png")

print("\nAll visualizations created successfully!")
