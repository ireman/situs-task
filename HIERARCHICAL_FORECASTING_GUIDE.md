# Hierarchical Forecasting Implementation Guide

## What Was Implemented

The main forecasting script (`beverage_forecasting.py`) now uses **hierarchical forecasting** for the Coca Cola product family.

## Why Use Hierarchical Forecasting?

The Coca Cola family (Classic, Zero, Diet) shows strong correlation (r=0.659), meaning their demands move together. Hierarchical forecasting leverages this relationship to create more accurate, coherent forecasts.

## How It Works

### Three-Step Process

**Step 1: Individual Forecasts**
- Forecast each Coca Cola product individually using the standard iterative method
- Coca Cola Classic: Predicts its own demand based on its features
- Coca Cola Zero: Predicts its own demand based on its features
- Diet Coca Cola: Predicts its own demand based on its features

**Step 2: Bottom-Up Aggregation**
- Sum the three individual forecasts to get a family total
- Example: Classic (25) + Zero (40) + Diet (10) = Family Total (75)

**Step 3: Proportion Reconciliation**
- Adjust each product to match historical proportions
- Historical proportions (from 2018-2020 data):
  - Coca Cola Classic: 29.8%
  - Coca Cola Zero: 58.7%
  - Diet Coca Cola: 11.5%
- Redistributed: Classic (22.4), Zero (44.0), Diet (8.6) = Total still 75

### Example

```
BEFORE Reconciliation (individual forecasts):
- Classic: 25 units
- Zero: 40 units
- Diet: 10 units
- Total: 75 units
- Proportions: 33.3% / 53.3% / 13.3% (doesn't match history)

AFTER Reconciliation (hierarchical):
- Total: 75 units (preserved)
- Classic: 75 × 29.8% = 22.4 units
- Zero: 75 × 58.7% = 44.0 units
- Diet: 75 × 11.5% = 8.6 units
- Proportions: 29.8% / 58.7% / 11.5% (matches history ✓)
```

## Results

### Forecast Quality

**Coca Cola Family Totals (2021-2022):**
- Average: ~69 units/month
- Range: 61-79 units/month
- Historical average: ~80 units/month
- **Interpretation**: Forecasts are slightly conservative (86% of historical)

**Individual Products (Jan 2021 example):**
| Product | Forecast | Historical Avg | Proportion |
|---------|----------|----------------|------------|
| Classic | 21.4 units | ~24 units | 29.8% ✓ |
| Zero | 42.0 units | ~47 units | 58.7% ✓ |
| Diet | 8.2 units | ~9 units | 11.5% ✓ |
| **Total** | **71.6 units** | **~80 units** | **100%** |

### Benefits

1. **Coherent Forecasts**: Products maintain realistic proportions
2. **Leverages Correlation**: Uses the strong relationship between family members
3. **Stable**: Family totals are less volatile than individual products
4. **Interpretable**: Easy to explain the 30/59/11 split to stakeholders

## Other Beverages

All non-Coca Cola beverages use **standard individual forecasting**:
- Sprite variants
- Fuze Tea variants
- Fruit juices (Grape, Grapefruit)
- Fanta

These products have weaker correlations, so hierarchical forecasting doesn't help.

## How to Modify

### To Change Which Products Use Hierarchical Forecasting

Edit `beverage_forecasting.py` in the `generate_forecasts()` method:

```python
# Define product families for hierarchical forecasting
coca_cola_family = [
    'Coca Cola Classic 500ml',
    'Coca Cola Zero 500ml',
    'Diet Coca Cola 500ml'
]

# You could add more families here:
# fruit_juice_family = ['Grape 500ml', 'Grapefruit 500ml']
```

### To Add Another Product Family

1. Define the family list (like `coca_cola_family` above)
2. Call `forecast_hierarchical()` for that family:

```python
fruit_forecasts = self.forecast_hierarchical(
    family_name='Fruit Juice Family',
    family_beverages=fruit_juice_family,
    future_dates=future_df
)
```

3. Combine with other forecasts in the final step

### To Disable Hierarchical Forecasting

Remove or comment out the hierarchical forecasting section and forecast all beverages using the standard method.

## Technical Details

### Proportion Calculation

```python
# Historical total by beverage
classic_total = 858 units (over 36 months)
zero_total = 1690 units
diet_total = 330 units
family_total = 2878 units

# Proportions
classic_prop = 858 / 2878 = 0.298 (29.8%)
zero_prop = 1690 / 2878 = 0.587 (58.7%)
diet_prop = 330 / 2878 = 0.115 (11.5%)
```

### Reconciliation Formula

For each month and each beverage in the family:

```
reconciled_quantity = bottom_up_family_total × historical_proportion
```

Example for Coca Cola Classic in January 2021:
```
reconciled = 71.6 × 0.298 = 21.4 units
```

## Comparison with Standard Forecasting

| Approach | Coca Cola Family Total | Pros | Cons |
|----------|------------------------|------|------|
| **Standard** | Varies by product dynamics | Flexible, product-specific | Proportions may drift |
| **Hierarchical** | ~69 units/month avg | Coherent, stable proportions | Slightly less flexible |

Both approaches are valid. Hierarchical is recommended when:
- Products are strongly correlated (r > 0.6)
- Historical proportions are stable
- You want coherent, explainable forecasts

## Files Modified

- `beverage_forecasting.py`: Added `forecast_hierarchical()` method
- `beverage_forecasts_2021_2022.csv`: Updated with hierarchical forecasts
- `beverage_forecasts_visualization.png`: Updated visualizations

## Running the Forecasting

```bash
python beverage_forecasting.py
```

Look for this output section:
```
================================================================================
COCA COLA FAMILY - HIERARCHICAL FORECASTING
================================================================================

  Using HIERARCHICAL FORECASTING (bottom-up with reconciliation) for Coca Cola Family
  Historical proportions:
    Coca Cola Classic 500ml       : 29.8%
    Coca Cola Zero 500ml          : 58.7%
    Diet Coca Cola 500ml          : 11.5%
  Forecasting each product individually...
  Reconciling forecasts to match historical proportions...
  Forecasted 72 records for Coca Cola Family family
```

## Further Reading

- `CORRELATION_INSIGHTS.md` - Why Coca Cola family was chosen for hierarchical forecasting
- `HOW_TO_USE_CORRELATION.md` - Alternative approaches for using correlation
- `METHODOLOGY_AND_INSIGHTS.md` - Overall forecasting methodology

## Questions?

The hierarchical forecasting implementation is fully self-contained in the `forecast_hierarchical()` method. You can easily adapt it for other product families or modify the reconciliation logic.
