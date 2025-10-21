# Hierarchical Forecasting Validation Results

## Question
Does hierarchical forecasting improve forecast accuracy for the Coca Cola family?

## Answer
**No, hierarchical forecasting did NOT improve accuracy in this case.**

## Validation Methodology

### Test Strategy
- **Train**: 2018-2019 data (24 months)
- **Test**: 2020 data (12 months)
- **Comparison**: Baseline (individual) vs Hierarchical (Coca Cola family reconciled)

### Approaches Tested

**Baseline**: Individual forecasting for all beverages
- Each beverage forecasted independently
- No proportion constraints

**Hierarchical**: Coca Cola family reconciliation
- Forecast each Coca Cola product individually
- Sum to get bottom-up total
- Redistribute according to historical proportions (29.4% / 59.3% / 11.4%)

## Results

### Overall Performance on 2020 Holdout Data

| Metric | Baseline | Hierarchical | Change |
|--------|----------|--------------|--------|
| **MAE** | 2.704 | 2.759 | **-2.0% (worse)** |
| **RMSE** | 5.397 | 5.431 | **-0.6% (worse)** |

### Coca Cola Family Only

| Metric | Baseline | Hierarchical | Change |
|--------|----------|--------------|--------|
| **MAE** | 6.580 | 6.763 | **-2.8% (worse)** |

### Per-Beverage Results (Coca Cola Family)

| Beverage | Baseline MAE | Hierarchical MAE | Change |
|----------|--------------|------------------|--------|
| Coca Cola Zero | 10.553 | 10.535 | **+0.17%** (tiny improvement) |
| Coca Cola Classic | 6.723 | 6.970 | **-3.7%** (worse) |
| Diet Coca Cola | 2.462 | 2.784 | **-13.0%** (much worse) |

## Why Didn't Hierarchical Help?

### 1. Proportion Enforcement Can Hurt
The hierarchical approach forces products to maintain fixed proportions (29.4%/59.3%/11.4%), but in 2020:
- Actual proportions may have shifted
- COVID-19 may have changed consumer preferences
- Forcing old proportions introduces error

### 2. Individual Models Already Capture Patterns
The baseline models already have:
- Product-specific lag features
- Product-specific seasonality
- Product-specific trends

Adding proportion constraints removes flexibility that helps adapt to changes.

### 3. Small Sample Size
With only 24 training months (2018-2019), the proportion estimates may not be stable:
- 29.4% vs 29.8% (full dataset)
- Small changes in proportions create forecast errors

### 4. Diet Coca Cola Particularly Hurt
Diet Coca Cola showed -13% worse performance:
- Smallest volume product in family
- Most volatile
- Forcing it to 11.4% when demand might be 8% or 15% adds significant error

## Implications

### For This Dataset
**Recommendation: Use BASELINE (individual forecasting) for all beverages**

Reasons:
- 2% better overall accuracy
- 2.8% better for Coca Cola family specifically
- More flexible to adapt to changing patterns

### When Might Hierarchical Help?

Hierarchical forecasting CAN help when:

1. **Stable proportions**: Product mix doesn't change over time
2. **Longer history**: More data to estimate accurate proportions
3. **Strong business constraints**: You NEED coherent forecasts (e.g., production capacity)
4. **More aggregate forecasting**: Forecasting further ahead where maintaining coherent totals matters more

### For This Use Case
With only 36 months of data and potentially changing consumer preferences (especially 2020 with COVID), the flexibility of individual forecasting outweighs the theoretical benefits of hierarchical reconciliation.

## What About Correlation?

**Yes, correlation exists** (r=0.659 for Coca Cola family), but that doesn't automatically mean hierarchical forecasting improves accuracy.

**Why correlation didn't help here:**
- The baseline model with lag features already captures some correlation implicitly
- Each product's own lags (lag_1, lag_2, lag_3) reflect the general trend
- Forcing proportions adds a constraint that's too rigid for the data

**Alternative ways to use correlation:**
- Cross-product lag features (but showed mixed results)
- Ensemble weighting based on correlation
- Use correlation for business insights, not necessarily forecasting

## Conclusion

### Key Finding
**Hierarchical forecasting made predictions 2% worse overall and 2.8% worse for Coca Cola family.**

### Recommendation
**Continue using individual (baseline) forecasting for all beverages.**

### Lesson Learned
Theoretical relationships (correlation) don't always translate to better forecasts. Always validate with holdout data before deploying a new forecasting method.

### Best Practice
When implementing a new forecasting technique:
1. ✅ Test on holdout data (like we did)
2. ✅ Compare against baseline
3. ✅ Use actual metrics (MAE, RMSE)
4. ✅ Be willing to reject the new method if it doesn't improve
5. ❌ Don't assume correlation = better forecasts

## Files Generated

- `validate_hierarchical.py` - Validation script
- `hierarchical_validation_results.png` - Visual comparison
- `VALIDATION_RESULTS.md` - This document

## To Reproduce

```bash
python validate_hierarchical.py
```

This will:
- Train on 2018-2019
- Test on 2020
- Compare baseline vs hierarchical
- Generate visualization

---

**Bottom Line**: While hierarchical forecasting is a sophisticated technique with theoretical appeal, it did not improve accuracy for this dataset. The baseline individual forecasting approach performs better.
