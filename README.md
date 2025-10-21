# Beverage Order Forecasting Project

A machine learning solution for forecasting monthly beverage orders based on historical sales data (2018-2020).

## Overview

This project forecasts monthly order quantities for 11 different beverage products for the years 2021-2022 using ensemble machine learning techniques with engineered time-series features.

## Project Structure

```
.
├── beverage_forecasting.py                    # Main forecasting script
├── METHODOLOGY_AND_INSIGHTS.md                # Detailed methodology documentation
├── requirements.txt                           # Python dependencies
├── monthly_beverage_orders 2018-2020.xlsx     # Input data
├── beverage_forecasts_2021_2022.csv          # Output: Forecast results
├── beverage_forecasts_visualization.png       # Output: Time series plots
└── beverage_summary_statistics.png            # Output: Summary charts
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the forecasting pipeline:

```bash
python beverage_forecasting.py
```

This will:
1. Load and explore the historical data
2. Perform feature engineering
3. Train Random Forest and Gradient Boosting models
4. Generate forecasts for 2021-2022
5. Create visualizations
6. Save results to CSV

## Output Files

### 1. Forecast Data
**File**: `beverage_forecasts_2021_2022.csv`

Contains predicted monthly quantities for each beverage product:
- Columns: beverage, year, month, quantity
- 264 rows (11 beverages × 24 months)

### 2. Visualizations
**File**: `beverage_forecasts_visualization.png`
- Time series plots for each beverage showing historical data (2018-2020) and forecasts (2021-2022)

**File**: `beverage_summary_statistics.png`
- Total orders by year (historical vs. forecast)
- Average monthly orders by beverage type

## Model Performance

### Random Forest Regressor (used for final forecasts)
- MAE: 1.45 units
- RMSE: 3.08 units
- R²: 0.9577

### Gradient Boosting Regressor (comparison model)
- MAE: 0.21 units
- RMSE: 0.36 units
- R²: 0.9994

## Key Features

### Feature Engineering
- **Temporal features**: month number, quarter, time index
- **Cyclical encoding**: sine/cosine transformations for seasonality
- **Lag features**: 1, 2, 3, 6, and 12-month lags
- **Rolling statistics**: 3, 6, and 12-month moving averages and standard deviations
- **Product encoding**: beverage-specific patterns

### Modeling Approach
- Ensemble machine learning (Random Forest + Gradient Boosting)
- Iterative multi-step ahead forecasting
- Per-beverage feature calculation to maintain product-specific patterns

## Products Forecasted

1. Coca Cola Classic 500ml
2. Coca Cola Zero 500ml
3. Diet Coca Cola 500ml
4. Fanta 500ml
5. Fuze Diet Apricot 500ml
6. Fuze Tea Apricot 500ml
7. Grape 500ml
8. Grapefruit 500ml
9. Sprite 500ml
10. Zero Sprite 500ml
11. sprite lite

## Methodology

For detailed information about the methodology, data exploration, feature engineering, and insights, please see:

**[METHODOLOGY_AND_INSIGHTS.md](METHODOLOGY_AND_INSIGHTS.md)**

This document covers:
- Data exploration and preprocessing
- Feature engineering rationale
- Model selection and training
- Forecasting approach
- Key insights and patterns discovered
- Challenges and solutions
- Recommendations for business use

## Technical Details

### Dependencies
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- openpyxl >= 3.1.0

### Runtime
Approximately 10-15 seconds on standard hardware

### Python Version
Tested with Python 3.11

## Results Summary

### High-Demand Products
- **Coca Cola Zero**: ~35-40 units/month (highest)
- **Coca Cola Classic**: ~20-30 units/month
- **Zero Sprite**: ~20-28 units/month

### Low-Demand Products
- **Grape, Grapefruit, Fanta**: 0-5 units/month (sporadic)
- **sprite lite**: Minimal orders

### Seasonal Patterns
- Slight demand increase in summer months (May-August)
- Some products show year-end uptick (November-December)

## Limitations

- Based solely on historical patterns (no external factors like weather, promotions)
- Assumes stable demand patterns through 2021-2022
- Uncertainty increases for longer forecast horizons
- Limited to 36 months of training data

## Future Enhancements

1. Incorporate external features (temperature, holidays, promotions)
2. Implement probabilistic forecasts with prediction intervals
3. Add online learning capability for model updates
4. Explore hierarchical models for product groups
5. Consider advanced methods (Prophet, LSTM) with more data

## License

This project is for educational and demonstration purposes.

## Author

Developed as part of a data science forecasting challenge.

## Contact

For questions or feedback about this implementation, please refer to the methodology document for technical details.
