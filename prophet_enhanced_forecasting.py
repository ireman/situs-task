"""
Enhanced Facebook Prophet with Additional Features
Adds: holiday effects, is_diet, beverage category, cross-beverage correlations
Compares enhanced model vs basic Prophet model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet


class EnhancedProphetForecaster:
    """Enhanced Prophet with additional regressors for better accuracy"""

    def __init__(self, excel_path='monthly_beverage_orders 2018-2020.xlsx'):
        self.excel_path = excel_path
        self.df = None
        self.beverage_names = None
        self.models_basic = {}
        self.models_enhanced = {}
        self.forecasts_basic = {}
        self.forecasts_enhanced = {}

    def load_and_preprocess(self):
        """Load data and prepare in Prophet format"""
        print("Loading data...")
        df = pd.read_excel(self.excel_path)

        # Rename columns
        df = df.rename(columns={
            'Name': 'beverage',
            'Year': 'year',
            'Month': 'month',
            'Quantity': 'quantity'
        })

        # Create date column
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

        # Get unique beverages
        self.beverage_names = sorted(df['beverage'].unique())
        print(f"Found {len(self.beverage_names)} beverages: {self.beverage_names}")

        # Pivot to matrix format
        quantity_matrix = df.pivot(index='date', columns='beverage', values='quantity')
        quantity_matrix = quantity_matrix[self.beverage_names].fillna(0)

        self.df = quantity_matrix
        print(f"Data shape: {self.df.shape} (months x beverages)")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

        return self.df

    def get_beverage_category(self, beverage):
        """Categorize beverage by type"""
        bev_lower = beverage.lower()
        if 'coca cola' in bev_lower or 'diet coca' in bev_lower:
            return 'cola'
        elif 'sprite' in bev_lower:
            return 'sprite'
        elif 'fuze' in bev_lower:
            return 'fuze'
        elif 'fanta' in bev_lower:
            return 'fanta'
        elif 'grape' in bev_lower:
            return 'grape'
        elif 'grapefruit' in bev_lower:
            return 'grapefruit'
        else:
            return 'other'

    def is_diet_beverage(self, beverage):
        """Check if beverage is diet/zero/light"""
        diet_keywords = ['diet', 'zero', 'light', 'lite']
        return 1 if any(kw in beverage.lower() for kw in diet_keywords) else 0

    def prepare_beverage_data_basic(self, beverage):
        """Prepare basic Prophet data (just ds, y)"""
        data = pd.DataFrame({
            'ds': self.df.index,
            'y': self.df[beverage].values
        })
        return data

    def prepare_beverage_data_enhanced(self, beverage):
        """
        Prepare enhanced Prophet data with additional regressors:
        - holiday: High-demand months (Nov, Dec, Jan)
        - is_diet: Diet/zero product indicator
        - category_total: Total sales for beverage category
        - total_sales: Total sales across all beverages
        """
        data = pd.DataFrame({
            'ds': self.df.index,
            'y': self.df[beverage].values
        })

        # 1. Holiday indicator (high-demand months)
        data['holiday'] = data['ds'].dt.month.isin([11, 12, 1]).astype(int)

        # 2. Is diet beverage
        data['is_diet'] = self.is_diet_beverage(beverage)

        # 3. Category total (sum of OTHER beverages in same category, excluding target)
        category = self.get_beverage_category(beverage)
        category_beverages = [b for b in self.beverage_names if self.get_beverage_category(b) == category and b != beverage]
        if category_beverages:
            data['category_total'] = self.df[category_beverages].sum(axis=1).values
        else:
            data['category_total'] = 0

        # 4. Total sales (all OTHER beverages, excluding target)
        other_beverages = [b for b in self.beverage_names if b != beverage]
        data['total_sales'] = self.df[other_beverages].sum(axis=1).values

        # 5. Quarter indicator
        data['quarter'] = data['ds'].dt.quarter

        # 6. Cross-beverage correlation (cola products correlation)
        # Use main cola products as indicator
        cola_products = ['Coca Cola Classic 500ml', 'Coca Cola Zero 500ml', 'Diet Coca Cola 500ml']
        available_colas = [c for c in cola_products if c in self.beverage_names and c != beverage]
        if available_colas:
            data['cola_indicator'] = self.df[available_colas].mean(axis=1).values
        else:
            data['cola_indicator'] = 0

        return data

    def train_models(self, seasonality_mode='multiplicative'):
        """Train both basic and enhanced Prophet models"""
        print("\n" + "="*70)
        print("Training BOTH Basic and Enhanced Prophet Models")
        print("="*70)

        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)

        for beverage in self.beverage_names:
            print(f"\n--- {beverage} ---")

            # BASIC MODEL
            print("  Training BASIC model (no features)...")
            data_basic = self.prepare_beverage_data_basic(beverage)

            model_basic = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )
            model_basic.fit(data_basic)
            self.models_basic[beverage] = model_basic

            # ENHANCED MODEL
            print("  Training ENHANCED model (with features)...")
            data_enhanced = self.prepare_beverage_data_enhanced(beverage)

            model_enhanced = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )

            # Add regressors
            model_enhanced.add_regressor('holiday', prior_scale=10)
            model_enhanced.add_regressor('is_diet', prior_scale=5)
            model_enhanced.add_regressor('category_total', prior_scale=10)
            model_enhanced.add_regressor('total_sales', prior_scale=5)
            model_enhanced.add_regressor('quarter', prior_scale=5)
            model_enhanced.add_regressor('cola_indicator', prior_scale=5)

            model_enhanced.fit(data_enhanced)
            self.models_enhanced[beverage] = model_enhanced

            print("  ✓ Both models trained")

        print("\n" + "="*70)
        print(f"Training complete: {len(self.models_basic)} basic + {len(self.models_enhanced)} enhanced models")
        print("="*70)

    def calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = actual != 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    def evaluate_models(self):
        """Evaluate both basic and enhanced models on historical data"""
        print("\n" + "="*70)
        print("EVALUATION: Basic vs Enhanced Prophet Models")
        print("="*70)

        results = []

        for beverage in self.beverage_names:
            # Get actual values
            actual = self.df[beverage].values

            # BASIC MODEL predictions
            model_basic = self.models_basic[beverage]
            data_basic = self.prepare_beverage_data_basic(beverage)
            forecast_basic = model_basic.predict(data_basic)
            pred_basic = forecast_basic['yhat'].values

            mae_basic = np.mean(np.abs(actual - pred_basic))
            rmse_basic = np.sqrt(np.mean((actual - pred_basic)**2))
            mape_basic = self.calculate_mape(actual, pred_basic)

            # ENHANCED MODEL predictions
            model_enhanced = self.models_enhanced[beverage]
            data_enhanced = self.prepare_beverage_data_enhanced(beverage)
            forecast_enhanced = model_enhanced.predict(data_enhanced)
            pred_enhanced = forecast_enhanced['yhat'].values

            mae_enhanced = np.mean(np.abs(actual - pred_enhanced))
            rmse_enhanced = np.sqrt(np.mean((actual - pred_enhanced)**2))
            mape_enhanced = self.calculate_mape(actual, pred_enhanced)

            # Calculate improvement
            mae_improvement = ((mae_basic - mae_enhanced) / mae_basic) * 100
            rmse_improvement = ((rmse_basic - rmse_enhanced) / rmse_basic) * 100
            mape_improvement = ((mape_basic - mape_enhanced) / mape_basic) * 100

            results.append({
                'beverage': beverage,
                'basic_mae': mae_basic,
                'enhanced_mae': mae_enhanced,
                'basic_rmse': rmse_basic,
                'enhanced_rmse': rmse_enhanced,
                'basic_mape': mape_basic,
                'enhanced_mape': mape_enhanced,
                'mae_improvement_%': mae_improvement,
                'rmse_improvement_%': rmse_improvement,
                'mape_improvement_%': mape_improvement,
            })

            winner_mae = '✓ Enhanced' if mae_enhanced < mae_basic else 'Basic'
            winner_mape = '✓ Enhanced' if mape_enhanced < mape_basic else 'Basic'

            print(f"\n{beverage}:")
            print(f"  Basic:    MAE={mae_basic:.2f}, RMSE={rmse_basic:.2f}, MAPE={mape_basic:.1f}%")
            print(f"  Enhanced: MAE={mae_enhanced:.2f}, RMSE={rmse_enhanced:.2f}, MAPE={mape_enhanced:.1f}%")
            print(f"  Winner (MAE): {winner_mae}, Winner (MAPE): {winner_mape}")
            print(f"  Improvement: MAE {mae_improvement:+.1f}%, RMSE {rmse_improvement:+.1f}%, MAPE {mape_improvement:+.1f}%")

        results_df = pd.DataFrame(results)

        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        print(f"\nAverage Metrics:")
        print(f"  Basic Prophet:")
        print(f"    MAE:  {results_df['basic_mae'].mean():.2f}")
        print(f"    RMSE: {results_df['basic_rmse'].mean():.2f}")
        print(f"    MAPE: {results_df['basic_mape'].mean():.1f}%")
        print(f"\n  Enhanced Prophet:")
        print(f"    MAE:  {results_df['enhanced_mae'].mean():.2f}")
        print(f"    RMSE: {results_df['enhanced_rmse'].mean():.2f}")
        print(f"    MAPE: {results_df['enhanced_mape'].mean():.1f}%")
        print(f"\n  Average Improvement:")
        print(f"    MAE:  {results_df['mae_improvement_%'].mean():+.1f}%")
        print(f"    RMSE: {results_df['rmse_improvement_%'].mean():+.1f}%")
        print(f"    MAPE: {results_df['mape_improvement_%'].mean():+.1f}%")

        # Count wins
        mae_wins = (results_df['enhanced_mae'] < results_df['basic_mae']).sum()
        mape_wins = (results_df['enhanced_mape'] < results_df['basic_mape']).sum()
        print(f"\n  Enhanced wins on: {mae_wins}/{len(results_df)} beverages (MAE)")
        print(f"  Enhanced wins on: {mape_wins}/{len(results_df)} beverages (MAPE)")

        return results_df

    def generate_forecasts(self, start_year=2021, end_year=2022):
        """Generate forecasts using both models"""
        print(f"\nGenerating forecasts for {start_year}-{end_year}...")

        num_months = (end_year - start_year + 1) * 12

        all_forecasts_basic = []
        all_forecasts_enhanced = []

        for beverage in self.beverage_names:
            # BASIC forecast
            model_basic = self.models_basic[beverage]
            future_basic = model_basic.make_future_dataframe(periods=num_months, freq='MS')
            forecast_basic = model_basic.predict(future_basic)

            forecast_basic['beverage'] = beverage
            forecast_basic['year'] = forecast_basic['ds'].dt.year
            forecast_basic['month'] = forecast_basic['ds'].dt.month

            future_basic_df = forecast_basic[
                (forecast_basic['year'] >= start_year) &
                (forecast_basic['year'] <= end_year)
            ].copy()
            future_basic_df['quantity'] = future_basic_df['yhat'].clip(lower=0)
            all_forecasts_basic.append(future_basic_df[['beverage', 'year', 'month', 'quantity']])

            # ENHANCED forecast
            model_enhanced = self.models_enhanced[beverage]

            # Create future dataframe with regressors
            data_enhanced = self.prepare_beverage_data_enhanced(beverage)
            future_enhanced = model_enhanced.make_future_dataframe(periods=num_months, freq='MS')

            # Add regressors for future dates
            future_enhanced['holiday'] = future_enhanced['ds'].dt.month.isin([11, 12, 1]).astype(int)
            future_enhanced['is_diet'] = self.is_diet_beverage(beverage)
            future_enhanced['quarter'] = future_enhanced['ds'].dt.quarter

            # For future dates, use last known values for cross-beverage features
            last_category_total = data_enhanced['category_total'].iloc[-1]
            last_total_sales = data_enhanced['total_sales'].iloc[-1]
            last_cola_indicator = data_enhanced['cola_indicator'].iloc[-1]

            future_enhanced['category_total'] = last_category_total
            future_enhanced['total_sales'] = last_total_sales
            future_enhanced['cola_indicator'] = last_cola_indicator

            forecast_enhanced = model_enhanced.predict(future_enhanced)

            forecast_enhanced['beverage'] = beverage
            forecast_enhanced['year'] = forecast_enhanced['ds'].dt.year
            forecast_enhanced['month'] = forecast_enhanced['ds'].dt.month

            future_enhanced_df = forecast_enhanced[
                (forecast_enhanced['year'] >= start_year) &
                (forecast_enhanced['year'] <= end_year)
            ].copy()
            future_enhanced_df['quantity'] = future_enhanced_df['yhat'].clip(lower=0)
            all_forecasts_enhanced.append(future_enhanced_df[['beverage', 'year', 'month', 'quantity']])

            print(f"{beverage:30s} - Generated {len(future_basic_df)} forecasts (both models)")

        result_basic = pd.concat(all_forecasts_basic, ignore_index=True).sort_values(['beverage', 'year', 'month'])
        result_enhanced = pd.concat(all_forecasts_enhanced, ignore_index=True).sort_values(['beverage', 'year', 'month'])

        print(f"\nTotal forecasts: {len(result_basic)} (basic) + {len(result_enhanced)} (enhanced)")

        return result_basic, result_enhanced

    def visualize_comparison(self, forecasts_basic, forecasts_enhanced):
        """Compare basic vs enhanced forecasts visually"""
        print("\nCreating comparison visualization...")

        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, beverage in enumerate(self.beverage_names):
            ax = axes[idx]

            # Historical
            hist = self.df[beverage]
            ax.plot(hist.index, hist.values, 'o-', linewidth=2.5, markersize=6,
                   label='Historical', color='#2E86AB', alpha=0.8)

            # Forecast dates
            forecast_dates = pd.to_datetime(
                forecasts_basic[forecasts_basic['beverage'] == beverage][['year', 'month']].assign(day=1)
            )

            # Basic forecast
            basic_data = forecasts_basic[forecasts_basic['beverage'] == beverage]
            ax.plot(forecast_dates, basic_data['quantity'].values, 's--',
                   linewidth=2, markersize=4, alpha=0.7,
                   label='Basic Prophet', color='#F18F01')

            # Enhanced forecast
            enhanced_data = forecasts_enhanced[forecasts_enhanced['beverage'] == beverage]
            ax.plot(forecast_dates, enhanced_data['quantity'].values, 'D-',
                   linewidth=2, markersize=5, alpha=0.9,
                   label='Enhanced Prophet', color='#06A77D')

            # Vertical line
            last_hist_date = hist.index.max()
            ax.axvline(x=last_hist_date, color='gray', linestyle=':', alpha=0.5, linewidth=2)

            ax.set_title(beverage, fontsize=10, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('Comparison: Basic Prophet vs Enhanced Prophet (with features)',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('prophet_basic_vs_enhanced.png', dpi=150, bbox_inches='tight')
        print("Saved: prophet_basic_vs_enhanced.png")
        plt.close()


def main():
    """Main execution"""
    print("="*70)
    print("Enhanced Facebook Prophet with Additional Features")
    print("="*70)

    forecaster = EnhancedProphetForecaster('monthly_beverage_orders 2018-2020.xlsx')

    # Load data
    forecaster.load_and_preprocess()

    # Train both models
    forecaster.train_models(seasonality_mode='multiplicative')

    # Evaluate and compare
    eval_results = forecaster.evaluate_models()

    # Save evaluation results
    eval_results.to_csv('prophet_model_comparison.csv', index=False)
    print("\nEvaluation results saved to: prophet_model_comparison.csv")

    # Generate forecasts
    forecasts_basic, forecasts_enhanced = forecaster.generate_forecasts(2021, 2022)

    # Save forecasts
    forecasts_basic.to_csv('prophet_basic_forecasts_2021_2022.csv', index=False)
    forecasts_enhanced.to_csv('prophet_enhanced_forecasts_2021_2022.csv', index=False)
    print("\nForecasts saved:")
    print("  - prophet_basic_forecasts_2021_2022.csv")
    print("  - prophet_enhanced_forecasts_2021_2022.csv")

    # Visualize comparison
    forecaster.visualize_comparison(forecasts_basic, forecasts_enhanced)

    print("\n" + "="*70)
    print("Enhanced Prophet Implementation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
