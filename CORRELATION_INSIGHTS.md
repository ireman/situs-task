# Beverage Correlation Analysis - Key Insights

## Overview
This analysis examines how different beverage products' demand patterns correlate with each other over the 2018-2020 period (36 months).

---

## Key Findings

### 1. Highly Correlated Pair (r > 0.7)

**Coca Cola Zero 500ml ↔ Diet Coca Cola 500ml: r = 0.737**
- **Strong positive correlation** - These two products move together
- **Interpretation**: Customers who buy zero-calorie beverages likely substitute between these two
- **Business Implication**: Stock these together; if one is out of stock, the other can serve as a substitute

---

### 2. Moderately Correlated Pairs (0.5 < r < 0.7)

| Beverage 1 | Beverage 2 | Correlation |
|------------|------------|-------------|
| Coca Cola Classic | Coca Cola Zero | 0.669 |
| Grape | Grapefruit | 0.616 |
| Coca Cola Zero | Fuze Tea Apricot | 0.585 |
| Coca Cola Classic | Diet Coca Cola | 0.560 |
| Coca Cola Zero | Fanta | 0.518 |

**Insights:**
- **Coca Cola family shows strong internal correlation** - Classic, Zero, and Diet all move together
- **Fruit juices correlate** - Grape and Grapefruit have similar demand patterns (likely seasonal)
- **Cross-family correlation** - Coca Cola Zero correlates with Fuze Tea and Fanta, suggesting these appeal to similar customer segments

---

### 3. Product Family Correlation Analysis

#### Coca Cola Family (Classic, Zero, Diet)
- **Average within-family correlation: 0.659** ✅ STRONG
- **Finding**: These three products have highly synchronized demand
- **Explanation**: Loyal Coca Cola customers who vary their choice based on preference (regular, zero-calorie, or diet)
- **Recommendation**: Treat as a product bundle for inventory and marketing

#### Fruit Juices (Grape, Grapefruit)
- **Average within-family correlation: 0.616** ✅ STRONG
- **Finding**: Fruit-flavored beverages move together
- **Explanation**: Likely seasonal (summer demand) or customer preference for natural flavors
- **Recommendation**: Stock together, especially during peak seasons

#### Fuze Tea Family (Diet Apricot, Tea Apricot)
- **Average within-family correlation: 0.046** ❌ WEAK
- **Finding**: These two variants have independent demand
- **Explanation**: Different customer segments - diet vs. regular tea drinkers
- **Recommendation**: Treat as separate products, not substitutes

#### Sprite Family (Sprite, Zero Sprite, sprite lite)
- **Average within-family correlation: -0.047** ❌ WEAK/NEGATIVE
- **Finding**: Sprite variants have virtually no correlation (some negative)
- **Explanation**: Different customer preferences; variants don't substitute for each other
- **Note**: Regular "Sprite 500ml" has missing data (NaN), which affects this analysis
- **Recommendation**: Stock independently based on individual demand patterns

---

### 4. Most "Connected" Beverages

Beverages ranked by average correlation with all other products:

1. **Coca Cola Zero** (avg |r| = 0.431) - Most interconnected
2. **Coca Cola Classic** (avg |r| = 0.357)
3. **Diet Coca Cola** (avg |r| = 0.339)
4. **Fanta** (avg |r| = 0.311)
5. **Grape** (avg |r| = 0.288)

**Interpretation:**
- Coca Cola products are "hub" products whose demand reflects overall beverage trends
- Lower-volume products (sprite lite, Fuze Diet Apricot) have weaker correlations - more independent demand

---

## Business Implications

### Inventory Management
1. **Bundle correlated products** - Stock Coca Cola family together
2. **Substitution strategy** - If Coca Cola Zero is out, promote Diet Coca Cola (r=0.737)
3. **Independent stocking** - Sprite variants need separate inventory planning

### Marketing & Promotions
1. **Cross-promote** - Promote Coca Cola Zero with Diet Coca Cola
2. **Seasonal campaigns** - Bundle fruit juices (Grape + Grapefruit) for summer
3. **Family promotions** - "Coca Cola Variety Pack" would make sense given high correlation

### Demand Forecasting
1. **Use correlated products as features** - When forecasting Coca Cola Zero, include Diet Coca Cola demand
2. **Independent forecasting for uncorrelated products** - Fuze Tea variants need separate models
3. **Leading indicators** - Coca Cola Zero demand may predict other products

---

## Statistical Notes

### What Correlation Means
- **r = 1.0**: Perfect positive correlation (products always move together)
- **r = 0.7-1.0**: Strong correlation (usually move together)
- **r = 0.5-0.7**: Moderate correlation (tend to move together)
- **r = 0.3-0.5**: Weak correlation (some relationship)
- **r = 0.0**: No correlation (independent)
- **r < 0.0**: Negative correlation (move in opposite directions)

### Limitations
1. **Correlation ≠ Causation** - High correlation doesn't mean one product drives the other
2. **Sample size** - Only 36 months of data; longer periods would be more robust
3. **Missing data** - "Sprite 500ml" has missing values affecting its correlations
4. **External factors** - Correlations may be driven by external factors (season, promotions) not captured here

---

## Visualizations

- **beverage_correlation_analysis.png**: Full correlation matrix heatmap
- **beverage_correlation_timeseries.png**: Time series plots of highly correlated pairs

---

## Recommendations for Model Improvement

Based on correlation findings, the forecasting model could be enhanced:

1. **Add cross-product features**: Include lagged values of correlated products
   - Example: When forecasting Coca Cola Zero, add Diet Coca Cola lag features

2. **Hierarchical forecasting**: Forecast Coca Cola family total, then disaggregate
   - More accurate for strongly correlated product families

3. **Clustering**: Group products by correlation for cluster-based forecasting
   - Cluster 1: Coca Cola family (high correlation)
   - Cluster 2: Fruit juices (moderate correlation)
   - Cluster 3: Independent products (low correlation)

4. **Transfer learning**: Use demand patterns from correlated products to improve forecasts for low-volume items

---

## Conclusion

The correlation analysis reveals **strong relationships within product families**, particularly:
- **Coca Cola variants** (r = 0.56 to 0.74) - customers substitute within the brand
- **Fruit juices** (r = 0.62) - similar seasonal/customer preferences

These insights can improve:
- Inventory planning (bundle correlated products)
- Marketing strategy (cross-promote correlated items)
- Demand forecasting (use correlated products as predictive features)

The analysis also identifies **independent products** (Fuze Tea variants, Sprite variants) that require separate treatment.
