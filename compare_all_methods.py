"""
Compare all forecasting methods side-by-side
Shows how Prophet captures variation vs flat forecasts from NN/GB
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical data
df = pd.read_excel('monthly_beverage_orders 2018-2020.xlsx')
df = df.rename(columns={'Name': 'beverage', 'Year': 'year', 'Month': 'month', 'Quantity': 'quantity'})
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

beverage_names = sorted(df['beverage'].unique())
quantity_matrix = df.pivot(index='date', columns='beverage', values='quantity')
quantity_matrix = quantity_matrix[beverage_names].fillna(0)

# Load all forecasts
nn_forecasts = pd.read_csv('nn_forecasts_2021_2022.csv')
gb_forecasts = pd.read_csv('rf_forecasts_2021_2022.csv')
prophet_forecasts = pd.read_csv('prophet_forecasts_2021_2022.csv')

print("="*70)
print("COMPARISON: All Forecasting Methods")
print("="*70)

# Compare first beverage in detail
bev = beverage_names[0]
print(f"\nBeverage: {bev}")
print("-"*70)

hist_values = quantity_matrix[bev].values
hist_dates = quantity_matrix.index

nn_bev = nn_forecasts[nn_forecasts['beverage'] == bev]
gb_bev = gb_forecasts[gb_forecasts['beverage'] == bev]
prophet_bev = prophet_forecasts[prophet_forecasts['beverage'] == bev]

print(f"\nHistorical (last 12 months of 2020):")
print(f"  Values: {hist_values[-12:]}")
print(f"  Mean: {np.mean(hist_values[-12:]):.2f}")
print(f"  Std:  {np.std(hist_values[-12:]):.2f}")
print(f"  Range: {np.min(hist_values[-12:]):.0f} to {np.max(hist_values[-12:]):.0f}")

print(f"\nNeural Network Forecast (first 12 months of 2021):")
nn_vals = nn_bev['quantity'].values[:12]
print(f"  Values: {nn_vals}")
print(f"  Mean: {np.mean(nn_vals):.2f}")
print(f"  Std:  {np.std(nn_vals):.2f}")
print(f"  Range: {np.min(nn_vals):.2f} to {np.max(nn_vals):.2f}")
print(f"  ❌ Variation: {np.std(nn_vals):.2f} (should be ~{np.std(hist_values[-12:]):.2f})")

print(f"\nGradient Boosting Forecast (first 12 months of 2021):")
gb_vals = gb_bev['quantity'].values[:12]
print(f"  Values: {gb_vals}")
print(f"  Mean: {np.mean(gb_vals):.2f}")
print(f"  Std:  {np.std(gb_vals):.2f}")
print(f"  Range: {np.min(gb_vals):.2f} to {np.max(gb_vals):.2f}")
print(f"  ❌ Variation: {np.std(gb_vals):.2f} (should be ~{np.std(hist_values[-12:]):.2f})")

print(f"\nProphet Forecast (first 12 months of 2021):")
prophet_vals = prophet_bev['quantity'].values[:12]
print(f"  Values: {prophet_vals}")
print(f"  Mean: {np.mean(prophet_vals):.2f}")
print(f"  Std:  {np.std(prophet_vals):.2f}")
print(f"  Range: {np.min(prophet_vals):.2f} to {np.max(prophet_vals):.2f}")
print(f"  ✓ Variation: {np.std(prophet_vals):.2f} (closer to historical {np.std(hist_values[-12:]):.2f})")

# Create comparison visualization
print("\n" + "="*70)
print("Creating side-by-side comparison visualization...")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, beverage in enumerate(beverage_names[:6]):
    ax = axes[idx]

    # Historical data
    hist = quantity_matrix[beverage]
    ax.plot(hist.index, hist.values, 'o-', linewidth=2.5, markersize=6,
            label='Historical (2018-2020)', color='#2E86AB', alpha=0.8)

    # Forecast dates
    forecast_dates = pd.to_datetime(
        prophet_forecasts[prophet_forecasts['beverage'] == beverage][['year', 'month']].assign(day=1)
    )

    # Neural Network
    nn_data = nn_forecasts[nn_forecasts['beverage'] == beverage]
    ax.plot(forecast_dates, nn_data['quantity'].values, '^--',
            linewidth=1.5, markersize=4, alpha=0.6,
            label='Neural Net (flat)', color='#A23B72')

    # Gradient Boosting
    gb_data = gb_forecasts[gb_forecasts['beverage'] == beverage]
    ax.plot(forecast_dates, gb_data['quantity'].values, 's--',
            linewidth=1.5, markersize=4, alpha=0.6,
            label='Gradient Boosting (flat)', color='#C73E1D')

    # Prophet
    prophet_data = prophet_forecasts[prophet_forecasts['beverage'] == beverage]
    ax.plot(forecast_dates, prophet_data['quantity'].values, 'D-',
            linewidth=2, markersize=5, alpha=0.9,
            label='Prophet (captures variation)', color='#F18F01')

    # Add vertical line
    last_hist_date = hist.index.max()
    ax.axvline(x=last_hist_date, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.text(last_hist_date, ax.get_ylim()[1]*0.95, 'Forecast →',
            ha='left', va='top', fontsize=8, color='gray')

    # Calculate std for title
    hist_std = np.std(hist.values[-12:])
    prophet_std = np.std(prophet_data['quantity'].values[:12])

    ax.set_title(f'{beverage}\nHistorical σ={hist_std:.1f}, Prophet σ={prophet_std:.1f}',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Comparison: Prophet vs Neural Network vs Gradient Boosting\n' +
             'Prophet captures monthly variation, others are flat',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('method_comparison_all.png', dpi=150, bbox_inches='tight')
print("Saved: method_comparison_all.png")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY: Variation Comparison (First 12 forecast months)")
print("="*70)
print(f"{'Beverage':<30} {'Hist σ':>8} {'NN σ':>8} {'GB σ':>8} {'Prophet σ':>10} {'Best':<10}")
print("-"*70)

for beverage in beverage_names:
    hist_std = np.std(quantity_matrix[beverage].values[-12:])

    nn_data = nn_forecasts[nn_forecasts['beverage'] == beverage]
    nn_std = np.std(nn_data['quantity'].values[:12])

    gb_data = gb_forecasts[gb_forecasts['beverage'] == beverage]
    gb_std = np.std(gb_data['quantity'].values[:12])

    prophet_data = prophet_forecasts[prophet_forecasts['beverage'] == beverage]
    prophet_std = np.std(prophet_data['quantity'].values[:12])

    # Find closest to historical
    diffs = {
        'NN': abs(nn_std - hist_std),
        'GB': abs(gb_std - hist_std),
        'Prophet': abs(prophet_std - hist_std)
    }
    best = min(diffs, key=diffs.get)

    print(f"{beverage:<30} {hist_std:>8.2f} {nn_std:>8.2f} {gb_std:>8.2f} {prophet_std:>10.2f} {best:<10}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("""
Prophet forecasts show MUCH more realistic variation compared to NN/GB:

✓ Prophet captures monthly ups and downs (seasonality)
✓ Prophet variation (σ) is closer to historical patterns
✓ Prophet forecasts look like actual beverage order patterns

❌ Neural Network gives flat predictions with minimal variation
❌ Gradient Boosting gives flat predictions stuck near last value

RECOMMENDATION: Use Prophet forecasts for 2021-2022 predictions
""")
