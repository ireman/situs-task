"""
Analyze why forecasts don't capture data trends
Compare actual data patterns with forecast patterns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load historical data
df = pd.read_excel('monthly_beverage_orders 2018-2020.xlsx')
df = df.rename(columns={'Name': 'beverage', 'Year': 'year', 'Month': 'month', 'Quantity': 'quantity'})
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# Pivot to matrix
beverage_names = sorted(df['beverage'].unique())
quantity_matrix = df.pivot(index='date', columns='beverage', values='quantity')
quantity_matrix = quantity_matrix[beverage_names].fillna(0)

print("="*70)
print("TREND ANALYSIS: Why forecasts don't match actual data patterns")
print("="*70)

# Analyze trends for each beverage
print("\n1. HISTORICAL DATA TRENDS (2018-2020):")
print("-"*70)

for beverage in beverage_names[:5]:  # First 5 beverages
    values = quantity_matrix[beverage].values

    # Calculate trend
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    trend = coeffs[0]

    # Calculate variability
    std_dev = np.std(values)
    cv = (std_dev / np.mean(values)) * 100 if np.mean(values) > 0 else 0

    # Calculate autocorrelation
    if len(values) > 1:
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
    else:
        autocorr = 0

    print(f"\n{beverage}:")
    print(f"  Mean: {np.mean(values):.2f}, Std: {std_dev:.2f}, CV: {cv:.1f}%")
    print(f"  Trend: {trend:.2f} units/month {'↑' if trend > 0 else '↓'}")
    print(f"  Range: {np.min(values):.0f} to {np.max(values):.0f}")
    print(f"  Autocorrelation (lag-1): {autocorr:.3f}")
    print(f"  Last 3 months: {values[-3:]}")

# Load forecasts
print("\n\n2. COMPARING FORECASTS:")
print("-"*70)

try:
    nn_forecasts = pd.read_csv('nn_forecasts_2021_2022.csv')
    rf_forecasts = pd.read_csv('rf_forecasts_2021_2022.csv')

    # Compare first beverage
    bev = beverage_names[0]
    print(f"\n{bev}:")

    hist_values = quantity_matrix[bev].values
    hist_mean = np.mean(hist_values)
    hist_last = hist_values[-1]

    nn_bev = nn_forecasts[nn_forecasts['beverage'] == bev]['quantity'].values
    rf_bev = rf_forecasts[rf_forecasts['beverage'] == bev]['quantity'].values

    print(f"  Historical:")
    print(f"    Mean: {hist_mean:.2f}")
    print(f"    Last value (Dec 2020): {hist_last:.2f}")
    print(f"    Last 6 months: {hist_values[-6:]}")

    print(f"\n  Neural Network Forecast:")
    print(f"    Mean: {np.mean(nn_bev):.2f}")
    print(f"    First value (Jan 2021): {nn_bev[0]:.2f}")
    print(f"    First 6 months: {nn_bev[:6]}")

    print(f"\n  Gradient Boosting Forecast:")
    print(f"    Mean: {np.mean(rf_bev):.2f}")
    print(f"    First value (Jan 2021): {rf_bev[0]:.2f}")
    print(f"    First 6 months: {rf_bev[:6]}")

except Exception as e:
    print(f"Error loading forecasts: {e}")

# Visualize the issue
print("\n\n3. CREATING DETAILED COMPARISON PLOT...")
print("-"*70)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, beverage in enumerate(beverage_names[:6]):
    ax = axes[idx]

    # Historical data
    hist_values = quantity_matrix[beverage].values
    hist_dates = quantity_matrix.index

    # Plot historical
    ax.plot(hist_dates, hist_values, 'o-', linewidth=2, markersize=5,
            label='Historical (2018-2020)', color='#2E86AB')

    # Add trend line
    x = np.arange(len(hist_values))
    coeffs = np.polyfit(x, hist_values, 1)
    trend_line = coeffs[0] * x + coeffs[1]
    ax.plot(hist_dates, trend_line, '--', linewidth=1, alpha=0.5,
            label=f'Trend ({coeffs[0]:.2f}/month)', color='gray')

    # Try to add forecasts
    try:
        nn_bev_data = nn_forecasts[nn_forecasts['beverage'] == beverage]
        forecast_dates = pd.to_datetime(nn_bev_data[['year', 'month']].assign(day=1))
        ax.plot(forecast_dates, nn_bev_data['quantity'].values, 's--',
                linewidth=2, markersize=4, alpha=0.7,
                label='NN Forecast', color='#A23B72')
    except:
        pass

    try:
        rf_bev_data = rf_forecasts[rf_forecasts['beverage'] == beverage]
        forecast_dates = pd.to_datetime(rf_bev_data[['year', 'month']].assign(day=1))
        ax.plot(forecast_dates, rf_bev_data['quantity'].values, '^--',
                linewidth=2, markersize=4, alpha=0.7,
                label='GB Forecast', color='#F18F01')
    except:
        pass

    ax.set_title(f'{beverage}\nMean: {np.mean(hist_values):.1f}, Trend: {coeffs[0]:.2f}/mo',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('trend_analysis_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: trend_analysis_comparison.png")

print("\n\n4. DIAGNOSIS:")
print("-"*70)
print("""
Potential issues with forecasts not capturing trends:

1. STATIC FORECASTS: Using only historical data (no iterative predictions)
   means forecasts are "anchored" to historical patterns and don't
   extrapolate trends forward.

2. SHORT HISTORY: Only 36 months of data makes it hard to distinguish
   trends from seasonal patterns.

3. HIGH VARIABILITY: Some beverages have high coefficient of variation,
   making trend detection difficult.

4. FEATURE ENGINEERING: Current features focus on lags and rolling stats
   but may not capture long-term trends effectively.

5. SEASONALITY vs TREND: Models may be learning seasonal patterns but
   not continuing the underlying trend.

RECOMMENDATIONS:
- Add explicit trend features (time index, linear/polynomial trends)
- Use iterative forecasting to propagate trends forward
- Increase model capacity for trend learning
- Consider decomposition (trend + seasonality + residual)
- Try Prophet-based methods (designed for trend + seasonality)
""")

print("\n" + "="*70)
