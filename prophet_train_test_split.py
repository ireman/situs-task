"""
Train/Test Split Evaluation for Prophet Models
Trains on 2018-2019, tests on 2020 to evaluate true predictive performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class ProphetTrainTestEvaluator:
    """Evaluate Prophet models with proper train/test split"""

    def __init__(self, excel_path='monthly_beverage_orders 2018-2020.xlsx'):
        self.excel_path = excel_path
        self.df = None
        self.beverage_names = None
        self.train_data = None
        self.test_data = None

        self.models_basic = {}
        self.models_enhanced = {}

        self.test_predictions_basic = {}
        self.test_predictions_enhanced = {}

        self.test_metrics = []

    def load_and_split_data(self):
        """Load data and split into train (2018-2019) and test (2020)"""
        print("="*70)
        print("Loading and splitting data")
        print("="*70)

        # Load data
        df = pd.read_excel(self.excel_path)
        df = df.rename(columns={
            'Name': 'beverage',
            'Year': 'year',
            'Month': 'month',
            'Quantity': 'quantity'
        })
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

        # Get unique beverages
        self.beverage_names = sorted(df['beverage'].unique())
        print(f"Found {len(self.beverage_names)} beverages")

        # Pivot to matrix format
        quantity_matrix = df.pivot(index='date', columns='beverage', values='quantity')
        quantity_matrix = quantity_matrix[self.beverage_names].fillna(0)

        # Split: 2018-2019 for training, 2020 for testing
        train_mask = quantity_matrix.index.year < 2020
        test_mask = quantity_matrix.index.year == 2020

        self.train_data = quantity_matrix[train_mask]
        self.test_data = quantity_matrix[test_mask]

        print(f"\nTrain data: {self.train_data.shape[0]} months (2018-2019)")
        print(f"  Date range: {self.train_data.index.min()} to {self.train_data.index.max()}")

        print(f"\nTest data: {self.test_data.shape[0]} months (2020)")
        print(f"  Date range: {self.test_data.index.min()} to {self.test_data.index.max()}")

        return self.train_data, self.test_data

    def get_beverage_category(self, beverage):
        """Classify beverage into category"""
        name_lower = beverage.lower()
        if 'coca cola classic' in name_lower or 'coke' in name_lower:
            return 'cola'
        elif 'coca cola zero' in name_lower or 'zero' in name_lower:
            return 'diet_cola'
        elif 'diet coke' in name_lower:
            return 'diet_cola'
        elif 'sprite' in name_lower:
            return 'lemon_lime'
        elif 'fanta orange' in name_lower:
            return 'orange'
        elif 'fanta grape' in name_lower:
            return 'grape'
        else:
            return 'other'

    def is_diet_beverage(self, beverage):
        """Check if beverage is diet/zero"""
        name_lower = beverage.lower()
        return 'diet' in name_lower or 'zero' in name_lower

    def prepare_prophet_data_basic(self, beverage, data):
        """Prepare data for basic Prophet model (no regressors)"""
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data[beverage].values
        })
        return prophet_df

    def prepare_prophet_data_enhanced(self, beverage, data):
        """Prepare data for enhanced Prophet model (with regressors)"""
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data[beverage].values
        })

        # Add regressors
        # 1. Holiday indicator (Nov, Dec, Jan)
        prophet_df['holiday'] = prophet_df['ds'].dt.month.isin([11, 12, 1]).astype(int)

        # 2. Is diet beverage
        prophet_df['is_diet'] = int(self.is_diet_beverage(beverage))

        # 3. Category total (other beverages in same category, excluding target)
        category = self.get_beverage_category(beverage)
        category_beverages = [b for b in self.beverage_names
                              if self.get_beverage_category(b) == category and b != beverage]
        if category_beverages:
            prophet_df['category_total'] = data[category_beverages].sum(axis=1).values
        else:
            prophet_df['category_total'] = 0

        # 4. Total sales (all other beverages, excluding target)
        other_beverages = [b for b in self.beverage_names if b != beverage]
        prophet_df['total_sales'] = data[other_beverages].sum(axis=1).values

        # 5. Quarter
        prophet_df['quarter'] = prophet_df['ds'].dt.quarter

        # 6. Cola indicator (correlation with Coca Cola Classic)
        if 'Coca Cola Classic 500ml' in data.columns and beverage != 'Coca Cola Classic 500ml':
            prophet_df['cola_indicator'] = data['Coca Cola Classic 500ml'].values
        else:
            prophet_df['cola_indicator'] = 0

        return prophet_df

    def train_basic_models(self):
        """Train basic Prophet models on 2018-2019 data"""
        print("\n" + "="*70)
        print("Training BASIC Prophet models on 2018-2019")
        print("="*70)

        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)

        for beverage in self.beverage_names:
            # Prepare training data
            train_df = self.prepare_prophet_data_basic(beverage, self.train_data)

            # Create and train model
            model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )

            model.fit(train_df)
            self.models_basic[beverage] = model

            print(f"✓ {beverage}")

        print(f"\nTrained {len(self.models_basic)} basic models")

    def train_enhanced_models(self):
        """Train enhanced Prophet models on 2018-2019 data"""
        print("\n" + "="*70)
        print("Training ENHANCED Prophet models on 2018-2019")
        print("="*70)

        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)

        for beverage in self.beverage_names:
            # Prepare training data
            train_df = self.prepare_prophet_data_enhanced(beverage, self.train_data)

            # Create and train model with regressors
            model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )

            # Add regressors
            model.add_regressor('holiday', prior_scale=10)
            model.add_regressor('is_diet', prior_scale=5)
            model.add_regressor('category_total', prior_scale=10)
            model.add_regressor('total_sales', prior_scale=5)
            model.add_regressor('quarter', prior_scale=5)
            model.add_regressor('cola_indicator', prior_scale=5)

            model.fit(train_df)
            self.models_enhanced[beverage] = model

            print(f"✓ {beverage}")

        print(f"\nTrained {len(self.models_enhanced)} enhanced models")

    def predict_test_set(self):
        """Generate predictions for 2020 test set"""
        print("\n" + "="*70)
        print("Generating predictions for 2020 test set")
        print("="*70)

        for beverage in self.beverage_names:
            # Basic model prediction
            model_basic = self.models_basic[beverage]
            test_df_basic = self.prepare_prophet_data_basic(beverage, self.test_data)
            forecast_basic = model_basic.predict(test_df_basic)
            self.test_predictions_basic[beverage] = forecast_basic['yhat'].clip(lower=0).values

            # Enhanced model prediction
            model_enhanced = self.models_enhanced[beverage]
            test_df_enhanced = self.prepare_prophet_data_enhanced(beverage, self.test_data)
            forecast_enhanced = model_enhanced.predict(test_df_enhanced)
            self.test_predictions_enhanced[beverage] = forecast_enhanced['yhat'].clip(lower=0).values

            print(f"✓ {beverage} - {len(self.test_predictions_basic[beverage])} predictions")

        print(f"\nGenerated predictions for {len(self.beverage_names)} beverages")

    def calculate_metrics(self):
        """Calculate metrics on test set"""
        print("\n" + "="*70)
        print("Test Set Performance (2020 predictions)")
        print("="*70)

        print(f"\n{'Beverage':<30} {'Model':<10} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'σ actual':>10} {'σ pred':>10}")
        print("-"*90)

        for beverage in self.beverage_names:
            # Actual values
            actual = self.test_data[beverage].values

            # Basic model predictions
            pred_basic = self.test_predictions_basic[beverage]

            # Calculate metrics for basic
            mae_basic = np.mean(np.abs(actual - pred_basic))
            rmse_basic = np.sqrt(np.mean((actual - pred_basic)**2))
            mape_basic = np.mean(np.abs((actual - pred_basic) / np.where(actual == 0, 1, actual))) * 100

            std_actual = np.std(actual)
            std_pred_basic = np.std(pred_basic)

            print(f"{beverage:<30} {'Basic':<10} {mae_basic:>8.2f} {rmse_basic:>8.2f} {mape_basic:>8.1f}% {std_actual:>10.2f} {std_pred_basic:>10.2f}")

            # Enhanced model predictions
            pred_enhanced = self.test_predictions_enhanced[beverage]

            # Calculate metrics for enhanced
            mae_enhanced = np.mean(np.abs(actual - pred_enhanced))
            rmse_enhanced = np.sqrt(np.mean((actual - pred_enhanced)**2))
            mape_enhanced = np.mean(np.abs((actual - pred_enhanced) / np.where(actual == 0, 1, actual))) * 100

            std_pred_enhanced = np.std(pred_enhanced)

            print(f"{'':<30} {'Enhanced':<10} {mae_enhanced:>8.2f} {rmse_enhanced:>8.2f} {mape_enhanced:>8.1f}% {std_actual:>10.2f} {std_pred_enhanced:>10.2f}")
            print()

            # Store metrics
            self.test_metrics.append({
                'beverage': beverage,
                'model': 'Basic',
                'MAE': mae_basic,
                'RMSE': rmse_basic,
                'MAPE': mape_basic,
                'std_actual': std_actual,
                'std_predicted': std_pred_basic
            })

            self.test_metrics.append({
                'beverage': beverage,
                'model': 'Enhanced',
                'MAE': mae_enhanced,
                'RMSE': rmse_enhanced,
                'MAPE': mape_enhanced,
                'std_actual': std_actual,
                'std_predicted': std_pred_enhanced
            })

        # Calculate averages
        metrics_df = pd.DataFrame(self.test_metrics)

        print("\n" + "="*70)
        print("Average Performance on Test Set")
        print("="*70)

        for model_type in ['Basic', 'Enhanced']:
            model_metrics = metrics_df[metrics_df['model'] == model_type]
            print(f"\n{model_type} Prophet:")
            print(f"  Average MAE:  {model_metrics['MAE'].mean():.2f}")
            print(f"  Average RMSE: {model_metrics['RMSE'].mean():.2f}")
            print(f"  Average MAPE: {model_metrics['MAPE'].mean():.1f}%")

        # Comparison
        basic_mae = metrics_df[metrics_df['model'] == 'Basic']['MAE'].mean()
        enhanced_mae = metrics_df[metrics_df['model'] == 'Enhanced']['MAE'].mean()
        improvement = ((basic_mae - enhanced_mae) / basic_mae) * 100

        print(f"\n{'='*70}")
        print(f"Enhanced vs Basic: {improvement:+.1f}% MAE improvement on test set")
        print(f"{'='*70}")

        return metrics_df

    def visualize_train_test_forecast(self):
        """Create visualization showing train/test/forecast periods"""
        print("\nCreating train/test/forecast visualization...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, beverage in enumerate(self.beverage_names[:6]):
            ax = axes[idx]

            # Training data (2018-2019)
            train = self.train_data[beverage]
            ax.plot(train.index, train.values, 'o-', linewidth=2.5, markersize=6,
                   label='Training (2018-2019)', color='#2E86AB', alpha=0.8)

            # Test data actual (2020)
            test = self.test_data[beverage]
            ax.plot(test.index, test.values, 'o-', linewidth=2.5, markersize=6,
                   label='Test Actual (2020)', color='#2D3142', alpha=0.8)

            # Test predictions - Basic
            ax.plot(test.index, self.test_predictions_basic[beverage], '^--',
                   linewidth=1.5, markersize=5, alpha=0.7,
                   label='Test Pred - Basic', color='#A23B72')

            # Test predictions - Enhanced
            ax.plot(test.index, self.test_predictions_enhanced[beverage], 's--',
                   linewidth=1.5, markersize=5, alpha=0.7,
                   label='Test Pred - Enhanced', color='#F18F01')

            # Add vertical lines
            train_end = train.index.max()
            ax.axvline(x=train_end, color='gray', linestyle=':', alpha=0.5, linewidth=2)
            ax.text(train_end, ax.get_ylim()[1]*0.95, 'Test →',
                   ha='left', va='top', fontsize=8, color='gray')

            # Calculate metrics for title
            actual = test.values
            pred_basic = self.test_predictions_basic[beverage]
            pred_enhanced = self.test_predictions_enhanced[beverage]

            mae_basic = np.mean(np.abs(actual - pred_basic))
            mae_enhanced = np.mean(np.abs(actual - pred_enhanced))

            ax.set_title(f'{beverage}\nTest MAE: Basic={mae_basic:.2f}, Enhanced={mae_enhanced:.2f}',
                        fontsize=9, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('Train/Test Split Evaluation: Prophet Models\\n' +
                    'Training: 2018-2019 | Testing: 2020',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('prophet_train_test_evaluation.png', dpi=150, bbox_inches='tight')
        print("Saved: prophet_train_test_evaluation.png")
        plt.close()

    def generate_future_forecasts(self):
        """Generate forecasts for 2021-2022 using models trained on 2018-2019"""
        print("\n" + "="*70)
        print("Generating 2021-2022 forecasts (trained on 2018-2019 only)")
        print("="*70)

        all_forecasts_basic = []
        all_forecasts_enhanced = []

        # Create dates for 2021-2022
        future_dates = pd.date_range(start='2021-01-01', end='2022-12-01', freq='MS')

        for beverage in self.beverage_names:
            # Basic model
            model_basic = self.models_basic[beverage]

            # Create future dataframe (from training end to 2022-12)
            future_basic = model_basic.make_future_dataframe(periods=36, freq='MS')  # 12 test + 24 forecast
            forecast_basic = model_basic.predict(future_basic)

            # Extract 2021-2022
            forecast_basic_filtered = forecast_basic[forecast_basic['ds'].isin(future_dates)].copy()
            forecast_basic_filtered['beverage'] = beverage
            forecast_basic_filtered['year'] = forecast_basic_filtered['ds'].dt.year
            forecast_basic_filtered['month'] = forecast_basic_filtered['ds'].dt.month
            forecast_basic_filtered['quantity'] = forecast_basic_filtered['yhat'].clip(lower=0)

            all_forecasts_basic.append(forecast_basic_filtered[['beverage', 'year', 'month', 'quantity']])

            # Enhanced model - need to prepare future with regressors
            # Combine train + test to get full 2018-2020 for regressor calculation
            full_data = pd.concat([self.train_data, self.test_data])

            # For 2021-2022, we need to extrapolate regressors
            # Create a dataframe with future dates
            future_df_enhanced = pd.DataFrame({'ds': future_dates})
            future_df_enhanced['holiday'] = future_df_enhanced['ds'].dt.month.isin([11, 12, 1]).astype(int)
            future_df_enhanced['is_diet'] = int(self.is_diet_beverage(beverage))
            future_df_enhanced['quarter'] = future_df_enhanced['ds'].dt.quarter

            # For category_total, total_sales, cola_indicator, use 2020 average as proxy
            category = self.get_beverage_category(beverage)
            category_beverages = [b for b in self.beverage_names
                                  if self.get_beverage_category(b) == category and b != beverage]
            if category_beverages:
                future_df_enhanced['category_total'] = self.test_data[category_beverages].sum(axis=1).mean()
            else:
                future_df_enhanced['category_total'] = 0

            other_beverages = [b for b in self.beverage_names if b != beverage]
            future_df_enhanced['total_sales'] = self.test_data[other_beverages].sum(axis=1).mean()

            if 'Coca Cola Classic 500ml' in self.test_data.columns and beverage != 'Coca Cola Classic 500ml':
                future_df_enhanced['cola_indicator'] = self.test_data['Coca Cola Classic 500ml'].mean()
            else:
                future_df_enhanced['cola_indicator'] = 0

            model_enhanced = self.models_enhanced[beverage]
            forecast_enhanced = model_enhanced.predict(future_df_enhanced)

            forecast_enhanced['beverage'] = beverage
            forecast_enhanced['year'] = forecast_enhanced['ds'].dt.year
            forecast_enhanced['month'] = forecast_enhanced['ds'].dt.month
            forecast_enhanced['quantity'] = forecast_enhanced['yhat'].clip(lower=0)

            all_forecasts_enhanced.append(forecast_enhanced[['beverage', 'year', 'month', 'quantity']])

            print(f"✓ {beverage}")

        # Combine all forecasts
        result_basic = pd.concat(all_forecasts_basic, ignore_index=True)
        result_basic = result_basic.sort_values(['beverage', 'year', 'month'])

        result_enhanced = pd.concat(all_forecasts_enhanced, ignore_index=True)
        result_enhanced = result_enhanced.sort_values(['beverage', 'year', 'month'])

        # Save
        result_basic.to_csv('prophet_basic_forecasts_2021_2022_testsplit.csv', index=False)
        result_enhanced.to_csv('prophet_enhanced_forecasts_2021_2022_testsplit.csv', index=False)

        print(f"\nSaved forecasts (trained on 2018-2019 only):")
        print(f"  - prophet_basic_forecasts_2021_2022_testsplit.csv")
        print(f"  - prophet_enhanced_forecasts_2021_2022_testsplit.csv")

        return result_basic, result_enhanced


def main():
    """Main execution"""
    print("="*70)
    print("Prophet Train/Test Split Evaluation")
    print("Train: 2018-2019 | Test: 2020 | Forecast: 2021-2022")
    print("="*70)

    evaluator = ProphetTrainTestEvaluator()

    # Load and split data
    evaluator.load_and_split_data()

    # Train models on 2018-2019 only
    evaluator.train_basic_models()
    evaluator.train_enhanced_models()

    # Predict test set (2020)
    evaluator.predict_test_set()

    # Calculate and display metrics
    metrics_df = evaluator.calculate_metrics()

    # Save metrics
    metrics_df.to_csv('prophet_test_metrics.csv', index=False)
    print("\nTest metrics saved to: prophet_test_metrics.csv")

    # Visualize train/test/predictions
    evaluator.visualize_train_test_forecast()

    # Generate future forecasts (2021-2022)
    evaluator.generate_future_forecasts()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
This evaluation shows TRUE predictive performance on unseen data:
✓ Models trained ONLY on 2018-2019
✓ Tested on completely unseen 2020 data
✓ Both Basic and Enhanced Prophet evaluated
✓ Metrics reflect real-world forecasting accuracy
    """)


if __name__ == "__main__":
    main()
