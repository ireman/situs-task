# Beverage Order Forecasting - Project Documentation

## Data Exploration and Preprocessing

### Dataset Overview
The dataset contains 360 monthly beverage orders from 2018-2020 for 11 products (Coca Cola variants, Sprite variants, Fuze Tea, and fruit juices). Order quantities range from 0-68 units with a mean of 10.2 units per month.

### Exploration Findings
- **High-volume products**: Coca Cola Zero (59% of family), Classic (30%), Zero Sprite
- **Low-volume products**: Grape, Grapefruit, sprite lite (sporadic, 0-5 units/month)
- **Seasonality**: Moderate summer uptick (May-August), holiday patterns (Nov-Jan)
- **Distribution**: Right-skewed (median=3, mean=10.2) with high variance across products
- **Correlation**: Strong within Coca Cola family (r=0.66), moderate for fruit juices (r=0.62)

### Preprocessing Steps

**1. Data Standardization**
- Renamed columns, created datetime index, sorted by beverage and date

**2. Feature Engineering (19 features total)**

*Temporal Features:*
- `time_idx`: Sequential month index (0-35) for linear trends
- `quarter`: Q1-Q4 seasonal patterns
- `month_sin`, `month_cos`: Trigonometric encoding for cyclical seasonality

*Product Features:*
- `is_diet`: Binary indicator for diet/zero-calorie products
- `holiday`: Binary flag for months with major holidays (Nov, Dec, Jan)

*Historical Demand Features (per beverage):*
- **Lag features** (1, 2, 3, 6, 12 months): Capture autoregressive patterns
- **Rolling means** (3, 6, 12 months): Short, medium, long-term trends
- **Rolling standard deviations** (3, 6, 12 months): Demand volatility

---

## Modeling Techniques

### Approach
Ensemble machine learning with iterative multi-step forecasting.

**Model Selection:**
- **Random Forest** (primary): 200 trees, max depth 15
  - Training: MAE=1.45, RMSE=3.08, R²=0.958
  - Selected for better generalization on limited data (36 months)

- **Gradient Boosting** (comparison): 200 estimators, learning rate 0.1
  - Training: MAE=0.21, R²=0.999 (not used - overfitting risk)

### Forecasting Strategy
**Iterative 24-month forecasting (2021-2022):**
1. Forecast month 1 using historical data and features
2. Append prediction to historical data
3. Forecast month 2 using updated history (including month 1 prediction)
4. Repeat for all 24 months

This maintains realistic temporal dependencies and allows lag features to incorporate recent predictions.

### Validation
Tested on 2020 holdout (train 2018-2019, test 2020):
- Individual forecasting performed best
- Hierarchical forecasting tested but 2.8% worse - reverted to baseline
- Lesson: Correlation doesn't automatically improve forecasts

---

## Key Insights and Challenges

### Key Insights

**Feature Importance:**
1. Beverage identity (64.3%) - product type is dominant predictor
2. Rolling mean 6-month (11.0%) - recent trends matter most
3. Rolling mean 3-month (5.6%) - short-term momentum
4. is_diet feature (3.2%) - validates new feature, ranks 6th

**Product Patterns:**
- Coca Cola products dominate (Zero: 40 units/month, Classic: 25 units/month)
- Diet/zero products show distinct patterns captured by `is_diet` feature
- Low-volume products (Grape, Grapefruit) remain sporadic but predictable within historical ranges

**Correlation:**
- Strong: Coca Cola Zero ↔ Diet (r=0.737) - good substitutes
- Moderate: Fruit juices (r=0.62), Coca Cola family overall (r=0.66)
- Weak: Sprite variants (r=-0.05) - independent demand

### Challenges Encountered

**1. Limited Historical Data (36 months)**
- *Solution:* Used Random Forest (works well with smaller datasets), avoided deep learning, focused on interpretable engineered features

**2. Product Heterogeneity (0-68 unit range)**
- *Solution:* Per-beverage lag/rolling features, product encoding, `is_diet` feature for diet/regular distinction

**3. Sparse/Zero Demand Products**
- *Solution:* Lag features model sporadic patterns, rolling averages smooth noise, non-negativity constraints

**4. Hierarchical Forecasting Didn't Help**
- *Finding:* Tested hierarchical approach for Coca Cola family - 2.8% worse on validation
- *Reason:* Forcing fixed proportions (29%/59%/12%) too rigid; individual forecasting with lags already captures correlation
- *Decision:* Reverted to individual forecasting for all products

**5. Long Forecast Horizon (24 months)**
- *Solution:* Iterative approach updates features with recent predictions, conservative Random Forest prevents overfitting, multi-scale features (lag_1, lag_6, lag_12)

---

## Results

**Model Performance:** R²=0.958, MAE=1.45 units
**Deliverables:** 264 predictions (11 beverages × 24 months) for 2021-2022
**Recommendation:** Deploy with monthly monitoring; use for high-volume products (Coca Cola, Sprite), apply buffer stock for low-volume items

The Random Forest model with 19 engineered features provides reliable, interpretable forecasts while avoiding overfitting on limited historical data.
