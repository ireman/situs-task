"""
Complete Guide to Facebook Prophet Parameters
Explains all parameters used in our beverage forecasting models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FACEBOOK PROPHET PARAMETERS - COMPLETE GUIDE")
print("="*80)

print("""

═══════════════════════════════════════════════════════════════════════════
1. SEASONALITY MODE
═══════════════════════════════════════════════════════════════════════════

Parameter: seasonality_mode='additive' or 'multiplicative'

WHAT IT DOES:
Controls how seasonality combines with the trend.

ADDITIVE (default):
  y = trend + seasonality + holidays + error
  - Seasonal fluctuations stay constant over time
  - Example: Sales vary by ±100 units regardless of trend level
  - Best for: Stable seasonal patterns

MULTIPLICATIVE:
  y = trend × (1 + seasonality) × (1 + holidays) + error
  - Seasonal fluctuations scale with the trend
  - Example: Sales vary by ±20% as trend grows
  - Best for: Growing/shrinking seasonal patterns

OUR CHOICE: 'multiplicative'
WHY: Beverage orders show percentage-based seasonality
     (busy months are proportionally busier as business grows)

VISUAL EXAMPLE:
  Additive:        Multiplicative:
  Sales            Sales
    |                |
 150|    /\/\       150|      /\/\/\
 100|   /    \      100|    /        \
  50|  /      \      50|  /            \
    |_________         |________________
       Time               Time
   (constant           (scales with
    amplitude)          trend)


═══════════════════════════════════════════════════════════════════════════
2. SEASONALITY PARAMETERS
═══════════════════════════════════════════════════════════════════════════

Parameters:
- yearly_seasonality: True/False or int (Fourier order)
- weekly_seasonality: True/False or int
- daily_seasonality: True/False or int

WHAT IT DOES:
Enables automatic detection of seasonal patterns at different frequencies.

yearly_seasonality=True:
  - Detects patterns that repeat every year
  - Uses Fourier series with 10 terms (default)
  - Example: "November-December always higher (holidays)"

weekly_seasonality=False:
  - We disabled this (our data is monthly, not daily)

daily_seasonality=False:
  - We disabled this (our data is monthly, not daily)

Fourier Order (if you set to int):
  - Higher = more flexible, can fit complex patterns
  - Lower = smoother, less prone to overfitting
  - yearly_seasonality=10 (default) means 20 parameters (sin+cos for 10 frequencies)

OUR CHOICE:
  yearly_seasonality=True   (detect annual patterns)
  weekly_seasonality=False  (monthly data, not relevant)
  daily_seasonality=False   (monthly data, not relevant)


═══════════════════════════════════════════════════════════════════════════
3. CHANGEPOINT_PRIOR_SCALE
═══════════════════════════════════════════════════════════════════════════

Parameter: changepoint_prior_scale=0.05 (default)

WHAT IT DOES:
Controls how flexible the TREND is (how much it can change direction).

HOW IT WORKS:
- Prophet automatically detects "changepoints" where trend shifts
- This parameter controls the strength of those shifts

VALUES:
  Small (0.001-0.01): Very smooth trend, hard to change direction
  Default (0.05):     Moderate flexibility, balanced
  Large (0.1-0.5):    Very flexible, can change direction frequently

TRADE-OFF:
  Too small → Misses real trend changes, underfits
  Too large → Follows noise, overfits

VISUAL EXAMPLE:

  changepoint_prior_scale = 0.001 (rigid):
  Sales
    |  _____________________ (straight line, misses changes)
    |_____________________

  changepoint_prior_scale = 0.05 (balanced):
  Sales
    |  ___/‾‾‾\_____ (follows major changes)
    |_____________________

  changepoint_prior_scale = 0.5 (flexible):
  Sales
    |  _/\/\/\_ (follows every wiggle, overfits)
    |_____________________

OUR CHOICE: 0.05 (default)
WHY: Good balance for 36 months of data with moderate trend changes


═══════════════════════════════════════════════════════════════════════════
4. SEASONALITY_PRIOR_SCALE
═══════════════════════════════════════════════════════════════════════════

Parameter: seasonality_prior_scale=10.0 (default)

WHAT IT DOES:
Controls the STRENGTH of seasonal patterns.

HOW IT WORKS:
- Regularization parameter for seasonality coefficients
- Larger = stronger seasonality allowed

VALUES:
  Small (0.1-1):  Weak seasonality, smooth predictions
  Default (10):   Moderate seasonality strength
  Large (50+):    Strong seasonality, fits peaks/valleys closely

TRADE-OFF:
  Too small → Misses real seasonal patterns, flat forecasts
  Too large → Overfits seasonal noise

VISUAL EXAMPLE:

  seasonality_prior_scale = 0.1 (weak):
  Sales
    |  ____________ (barely any seasonal variation)
    |_____________________

  seasonality_prior_scale = 10 (balanced):
  Sales
    |  __/\__/\__ (clear seasonal peaks)
    |_____________________

  seasonality_prior_scale = 50 (strong):
  Sales
    |  _/\/\/\/\_ (fits every small fluctuation)
    |_____________________

OUR CHOICE: 10.0 (default)
WHY: Appropriate for visible but not extreme seasonal patterns


═══════════════════════════════════════════════════════════════════════════
5. ADDITIONAL REGRESSORS (Enhanced Model)
═══════════════════════════════════════════════════════════════════════════

Method: model.add_regressor(name, prior_scale=10)

WHAT IT DOES:
Adds extra features beyond time-based patterns.

Parameters:
- name: Feature name (e.g., 'holiday', 'is_diet')
- prior_scale: How much this feature can influence predictions

PRIOR_SCALE values:
  Small (1-5):  Feature has subtle influence
  Medium (10):  Feature has moderate influence
  Large (50+):  Feature has strong influence

OUR REGRESSORS:

1. holiday (prior_scale=10):
   - High-demand months (Nov/Dec/Jan)
   - Medium influence

2. is_diet (prior_scale=5):
   - Diet/zero beverage indicator
   - Lower influence (product characteristic)

3. category_total (prior_scale=10):
   - Other beverages in same category
   - Medium influence (market dynamics)

4. total_sales (prior_scale=5):
   - Overall market indicator
   - Lower influence (general trend)

5. quarter (prior_scale=5):
   - Quarter of year
   - Lower influence (captured by yearly seasonality)

6. cola_indicator (prior_scale=5):
   - Cross-beverage correlation
   - Lower influence


═══════════════════════════════════════════════════════════════════════════
6. OTHER IMPORTANT PARAMETERS (Not Used But Good to Know)
═══════════════════════════════════════════════════════════════════════════

n_changepoints=25:
  - Number of potential changepoints to consider
  - Distributed evenly over first 80% of data
  - More = more flexible trend

changepoint_range=0.8:
  - Proportion of data where changepoints can occur
  - 0.8 = changepoints only in first 80%
  - Prevents overfitting to end of series

mcmc_samples=0:
  - Use MCMC for full Bayesian inference (slow)
  - 0 = use MAP estimation (fast, default)

interval_width=0.80:
  - Width of uncertainty intervals
  - 0.80 = 80% confidence intervals
  - 0.95 = 95% confidence intervals

uncertainty_samples=1000:
  - Number of samples for uncertainty estimation
  - More = smoother intervals, slower


═══════════════════════════════════════════════════════════════════════════
7. SUMMARY: OUR PARAMETER CHOICES
═══════════════════════════════════════════════════════════════════════════

BASIC PROPHET:
  Prophet(
    seasonality_mode='multiplicative',  # Seasonal % grows with trend
    yearly_seasonality=True,           # Detect annual patterns
    weekly_seasonality=False,          # Not relevant for monthly data
    daily_seasonality=False,           # Not relevant for monthly data
    changepoint_prior_scale=0.05,      # Moderate trend flexibility
    seasonality_prior_scale=10.0,      # Moderate seasonality strength
  )

ENHANCED PROPHET (adds 6 regressors):
  Same as above, plus:
    - holiday (prior_scale=10)
    - is_diet (prior_scale=5)
    - category_total (prior_scale=10)
    - total_sales (prior_scale=5)
    - quarter (prior_scale=5)
    - cola_indicator (prior_scale=5)

WHY THESE CHOICES:
✓ Multiplicative seasonality: Sales vary by percentage, not fixed amount
✓ Yearly seasonality: Clear annual patterns (holidays, seasons)
✓ Default changepoint/seasonality priors: Good starting point for 36 months
✓ Additional regressors: Capture business logic (holidays, product types, market)


═══════════════════════════════════════════════════════════════════════════
8. HOW TO TUNE PARAMETERS
═══════════════════════════════════════════════════════════════════════════

IF YOUR FORECASTS ARE:

TOO SMOOTH (missing peaks/valleys):
  → Increase seasonality_prior_scale (10 → 20)
  → Increase changepoint_prior_scale (0.05 → 0.1)

TOO WIGGLY (following noise):
  → Decrease seasonality_prior_scale (10 → 5)
  → Decrease changepoint_prior_scale (0.05 → 0.01)

MISSING TREND CHANGES:
  → Increase changepoint_prior_scale (0.05 → 0.1)
  → Increase n_changepoints (25 → 50)

OVERFITTING TREND:
  → Decrease changepoint_prior_scale (0.05 → 0.01)
  → Decrease changepoint_range (0.8 → 0.6)

SEASONAL PATTERN WRONG:
  → Check seasonality_mode (additive vs multiplicative)
  → Adjust yearly_seasonality Fourier order
  → Add custom seasonality

GENERAL APPROACH:
1. Start with defaults
2. Evaluate with cross-validation
3. Adjust one parameter at a time
4. Use domain knowledge (not just metrics)

""")

print("\n" + "="*80)
print("For more details, see: https://facebook.github.io/prophet/docs/")
print("="*80)
