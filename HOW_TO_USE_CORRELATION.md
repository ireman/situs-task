# How to Use Beverage Correlation in Your Forecasting Model

## Three Practical Approaches

Based on the correlation analysis, here are three ways to incorporate beverage relationships into your forecasts:

---

## 🎯 Approach 1: Cross-Product Lag Features

### What It Is
Add lag features from correlated beverages as predictors.

### How It Works
```python
# Example: Forecasting Coca Cola Zero
Features used:
- Own features: Coca Cola Zero lag_1, lag_2, lag_3
- Cross features: Diet Coca Cola lag_1, lag_2, lag_3  # NEW!
- Cross features: Coca Cola Classic lag_1, lag_2, lag_3  # NEW!
```

### When Forecasting Month 37 (Jan 2021):
- Standard model: Uses only Coca Cola Zero's own history
- Enhanced model: Also looks at what Diet Coca Cola and Classic did recently

### Results from Testing

**Top Feature Importances:**
1. Beverage ID: 65.9%
2. Own rolling mean (6m): 8.0%
3. Own rolling mean (3m): 6.0%
4. Own rolling mean (12m): 5.2%
5. **Corr: Coca Cola mean_3: 2.7%** ← Cross-product feature!

**7 out of top 20 features** were cross-product features from correlated beverages.

### Per-Beverage Improvement

| Beverage | Improvement | Has Correlation Features |
|----------|-------------|--------------------------|
| Grapefruit | +6.7% | ✓ |
| Diet Coca Cola | +1.9% | ✓ |
| Grape | +1.8% | ✓ |
| Coca Cola Classic | -8.0% | ✓ (overfitting) |
| Coca Cola Zero | -17.0% | ✓ (overfitting) |

### Key Insight
Mixed results! Some beverages improved (Grapefruit, Diet), others got worse (Classic, Zero).

**Problem:** High-volume products (Coca Cola Classic, Zero) may have enough data on their own and adding cross-features caused overfitting.

**Solution:** Use cross-product features selectively for low/medium volume products.

---

## 🎯 Approach 2: Hierarchical Forecasting

### What It Is
Forecast product families first, then disaggregate to individual products.

### How It Works

**Step 1:** Forecast total Coca Cola family demand
```
Total Coca Cola (2021-01) = forecast using family-level model
```

**Step 2:** Split by historical proportions
```
- Coca Cola Classic = Total × 29.8%
- Coca Cola Zero = Total × 58.7%
- Diet Coca Cola = Total × 11.5%
```

### Historical Family Proportions

**Coca Cola Family:**
- Classic: 29.8%
- Zero: 58.7%
- Diet: 11.5%

**Fruit Juice Family:**
- Grape: 60.7%
- Grapefruit: 39.3%

### Advantages
1. **More stable:** Family totals are less volatile than individual products
2. **Captures correlation:** Proportions automatically reflect substitution patterns
3. **Simpler:** One model per family instead of per product

### When to Use
- For tightly correlated families (Coca Cola: r=0.659)
- When proportions are stable over time
- For production/inventory planning at family level

### Implementation
```python
# 1. Aggregate to family level
family_total = sum(Classic, Zero, Diet)

# 2. Forecast family total
family_forecast = model.predict(family_features)

# 3. Disaggregate
Classic_forecast = family_forecast × 0.298
Zero_forecast = family_forecast × 0.587
Diet_forecast = family_forecast × 0.115
```

---

## 🎯 Approach 3: Selective Correlation Features

### What It Is
Use cross-product features ONLY when they improve accuracy.

### Strategy by Product Type

#### High-Volume Products (>20 units/month avg)
**Examples:** Coca Cola Zero, Coca Cola Classic, Zero Sprite

**Recommendation:** Use baseline model (own features only)
- **Reason:** Enough data to forecast accurately alone
- **Risk:** Cross-features may cause overfitting

#### Medium-Volume Products (5-20 units/month)
**Examples:** Diet Coca Cola, Fuze Tea, Fanta

**Recommendation:** Test both approaches, use best performer
- **Try:** Cross-features from 1-2 most correlated products
- **Validate:** Compare MAE on holdout period

#### Low-Volume Products (<5 units/month)
**Examples:** Grape, Grapefruit, sprite lite

**Recommendation:** Use cross-features from correlated products
- **Reason:** Limited own history, benefit from borrowing information
- **Best for:** Grapefruit (use Grape), Grape (use Grapefruit)

---

## 📊 Comparison Summary

| Approach | Complexity | Best For | Pros | Cons |
|----------|-----------|----------|------|------|
| **Cross-Product Lags** | High | Low-volume products | Uses correlation directly | Risk of overfitting |
| **Hierarchical** | Medium | Correlated families | Stable, interpretable | Assumes fixed proportions |
| **Selective** | Medium | Mixed portfolio | Balances accuracy/complexity | Requires testing |

---

## 💡 Practical Recommendations

### 1. For Your Current Dataset

**Use Hierarchical Forecasting for:**
- Coca Cola family (Classic, Zero, Diet) - strong correlation (r=0.659)
- Fruit juices (Grape, Grapefruit) - moderate correlation (r=0.616)

**Use Baseline Model for:**
- Sprite variants - weak correlation (r=-0.047)
- Fuze variants - weak correlation (r=0.046)
- Fanta - standalone product

### 2. Quick Win: Hierarchical for Coca Cola

```python
# Forecast Coca Cola family total
coca_total_2021_01 = forecast_family_total()

# Disaggregate
forecasts = {
    'Classic': coca_total_2021_01 * 0.298,
    'Zero': coca_total_2021_01 * 0.587,
    'Diet': coca_total_2021_01 * 0.115
}
```

This is simpler than cross-product features and leverages the strong correlation.

### 3. Advanced: Correlation-Based Ensembling

For each beverage, create an ensemble:
1. **Model A:** Own features only (baseline)
2. **Model B:** Own + correlated beverage features
3. **Final forecast:** Weighted average based on validation performance

Weight Model B higher for low-volume products, Model A higher for high-volume.

---

## 🔬 Testing Methodology

To determine if correlation features help for a specific beverage:

### 1. Train-Test Split
```python
# Use 2018-2019 for training
# Use 2020 for testing
train = data[data['year'] < 2020]
test = data[data['year'] == 2020]
```

### 2. Compare Models
```python
# Baseline: own features only
baseline_forecast = baseline_model.predict(test_features)
baseline_mae = MAE(test_actual, baseline_forecast)

# Enhanced: own + correlated features
enhanced_forecast = enhanced_model.predict(test_features_with_corr)
enhanced_mae = MAE(test_actual, enhanced_forecast)

# Keep the better one
if enhanced_mae < baseline_mae:
    use_correlation_features = True
```

### 3. Per-Beverage Decision
Create a lookup table:
```python
use_correlation = {
    'Coca Cola Zero': False,  # Worse with correlation
    'Diet Coca Cola': True,   # Better with correlation
    'Grapefruit': True,       # Better with correlation
    # ... etc
}
```

---

## 📈 Expected Improvements

Based on testing:

### Products That Benefit from Correlation Features
- **Grapefruit:** +6.7% improvement (use Grape features)
- **Diet Coca Cola:** +1.9% improvement (use Zero, Classic features)
- **Grape:** +1.8% improvement (use Grapefruit features)

### Products That Don't Benefit
- **Coca Cola Zero:** -17% (overfitting)
- **Coca Cola Classic:** -8% (overfitting)
- High-volume products with strong own patterns

---

## 🎓 Key Takeaways

1. **Correlation doesn't always help:** More features ≠ better forecasts
2. **Low-volume products benefit most:** When own history is sparse, borrow from correlated products
3. **High-volume products may suffer:** Overfitting risk when adding cross-features
4. **Hierarchical is safest:** Forecast families, then split - captures correlation without overfitting
5. **Always validate:** Test on holdout data before deploying

---

## 🛠️ Implementation Code

See `correlation_enhanced_forecasting.py` for full implementation of all three approaches.

**Files generated:**
- `correlation_model_comparison.png` - Visual comparison of baseline vs enhanced
- `correlation_model_info.json` - Performance metrics and configuration

**To run:**
```bash
python correlation_enhanced_forecasting.py
```

---

## Next Steps

1. **Try hierarchical forecasting** for Coca Cola family - likely best ROI
2. **Use cross-features selectively** for Grapefruit (use Grape), Grape (use Grapefruit)
3. **Stick with baseline** for high-volume products (Coca Cola Zero, Classic)
4. **Validate on 2021-2022 actuals** when available to confirm approach
