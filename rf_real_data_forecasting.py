"""
Random Forest Beverage Forecasting - Using Only Real Data
Similar to neural network approach: predict future months using only historical data
No iterative predictions - each forecast uses only actual historical data (2018-2020)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class RealDataForecaster:
    """Forecasts using only real historical data, no iterative predictions."""

    def __init__(self, excel_path='monthly_beverage_orders 2018-2020.xlsx'):
        self.excel_path = excel_path
        self.df = None
        self.beverage_names = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.diet_keywords = ['diet', 'zero', 'light', 'lite']

    def load_and_preprocess(self):
        """Load data and reshape into matrix format (months x beverages)"""
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

        # Pivot to create matrix: rows=months, columns=beverages
        quantity_matrix = df.pivot(index='date', columns='beverage', values='quantity')
        quantity_matrix = quantity_matrix[self.beverage_names]  # Ensure consistent order

        # Fill NaN values with 0 (no orders for that beverage in that month)
        quantity_matrix = quantity_matrix.fillna(0)

        self.df = quantity_matrix
        print(f"Data shape: {self.df.shape} (months x beverages)")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

        return self.df

    def create_temporal_features(self, dates):
        """Create temporal features for given dates"""
        temporal_features = pd.DataFrame({
            'month': dates.month,
            'quarter': dates.quarter,
            'holiday': dates.month.isin([11, 12, 1]).astype(int),
            'month_sin': np.sin(2 * np.pi * dates.month / 12),
            'month_cos': np.cos(2 * np.pi * dates.month / 12),
        }, index=dates)
        return temporal_features

    def create_beverage_features(self):
        """Create beverage-specific features (is_diet)"""
        is_diet = []
        for bev in self.beverage_names:
            is_diet_flag = 1 if any(kw in bev.lower() for kw in self.diet_keywords) else 0
            is_diet.append(is_diet_flag)
        return np.array(is_diet)

    def prepare_training_data(self, window_size=24):
        """
        Prepare training data using rolling window approach.
        Each sample uses a fixed historical window to predict the next month.
        """
        print(f"\nPreparing training data with rolling window approach...")

        dates = self.df.index
        all_quantities = self.df.values
        all_temporal = self.create_temporal_features(dates).values
        beverage_features = self.create_beverage_features()

        # Encode beverage names
        self.label_encoder.fit(self.beverage_names)

        X_train_list = []
        y_train_list = []

        # Create samples using rolling window
        for i in range(len(dates) - window_size):
            hist_start = i
            hist_end = i + window_size
            target_idx = i + window_size

            # Historical window (last 24 months)
            hist_quantities = all_quantities[hist_start:hist_end]  # Shape: (24, 11)
            hist_temporal = all_temporal[hist_start:hist_end]  # Shape: (24, 5)

            # Target month
            target_quantities = all_quantities[target_idx]  # Shape: (11,)
            target_temporal = all_temporal[target_idx]  # Shape: (5,)

            # For each beverage, create a sample
            for bev_idx, beverage in enumerate(self.beverage_names):
                # Features:
                # - Historical quantities for this beverage: lag features
                hist_bev_quantities = hist_quantities[:, bev_idx]

                # Compute statistics from historical window
                features = {
                    'beverage_encoded': self.label_encoder.transform([beverage])[0],

                    # Target month temporal features
                    'target_month': target_temporal[0],
                    'target_quarter': target_temporal[1],
                    'target_holiday': target_temporal[2],
                    'target_month_sin': target_temporal[3],
                    'target_month_cos': target_temporal[4],

                    # Beverage features
                    'is_diet': beverage_features[bev_idx],

                    # Lag features from historical window
                    'lag_1': hist_bev_quantities[-1],
                    'lag_2': hist_bev_quantities[-2] if len(hist_bev_quantities) >= 2 else hist_bev_quantities[-1],
                    'lag_3': hist_bev_quantities[-3] if len(hist_bev_quantities) >= 3 else hist_bev_quantities[-1],
                    'lag_6': hist_bev_quantities[-6] if len(hist_bev_quantities) >= 6 else np.mean(hist_bev_quantities),
                    'lag_12': hist_bev_quantities[-12] if len(hist_bev_quantities) >= 12 else np.mean(hist_bev_quantities),

                    # Rolling statistics from historical window
                    'rolling_mean_3': np.mean(hist_bev_quantities[-3:]),
                    'rolling_mean_6': np.mean(hist_bev_quantities[-6:]) if len(hist_bev_quantities) >= 6 else np.mean(hist_bev_quantities),
                    'rolling_mean_12': np.mean(hist_bev_quantities[-12:]) if len(hist_bev_quantities) >= 12 else np.mean(hist_bev_quantities),
                    'rolling_std_3': np.std(hist_bev_quantities[-3:]),
                    'rolling_std_6': np.std(hist_bev_quantities[-6:]) if len(hist_bev_quantities) >= 6 else np.std(hist_bev_quantities),
                    'rolling_std_12': np.std(hist_bev_quantities[-12:]) if len(hist_bev_quantities) >= 12 else np.std(hist_bev_quantities),

                    # Overall statistics
                    'hist_mean': np.mean(hist_bev_quantities),
                    'hist_std': np.std(hist_bev_quantities),
                    'hist_min': np.min(hist_bev_quantities),
                    'hist_max': np.max(hist_bev_quantities),
                    'hist_trend': (hist_bev_quantities[-1] - hist_bev_quantities[0]) / len(hist_bev_quantities) if hist_bev_quantities[0] > 0 else 0,
                }

                X_train_list.append(list(features.values()))
                y_train_list.append(target_quantities[bev_idx])

        self.X_train = np.array(X_train_list)
        self.y_train = np.array(y_train_list)
        self.feature_names = list(features.keys())

        print(f"Created {len(self.X_train)} training samples")
        print(f"Features: {len(self.feature_names)}")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")

        return self.X_train, self.y_train

    def train_model(self, model_type='random_forest'):
        """Train Random Forest or Gradient Boosting model"""
        print(f"\nTraining {model_type} model...")

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.fit(self.X_train, self.y_train)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 most important features:")
        print(feature_importance.head(10))

        return self.model

    def evaluate_model(self):
        """Evaluate model on training data"""
        print("\nEvaluating model on training data...")

        y_pred = self.model.predict(self.X_train)

        mae = mean_absolute_error(self.y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_train, y_pred))
        r2 = r2_score(self.y_train, y_pred)

        print(f"Training Metrics:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.4f}")

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def generate_forecasts(self, start_year=2021, end_year=2022):
        """
        Generate forecasts for 2021-2022 using only historical data (2018-2020).
        Each prediction uses the full historical window, no iterative predictions.
        """
        print(f"\nGenerating forecasts for {start_year}-{end_year}...")

        # Use all historical data as the window
        hist_dates = self.df.index
        hist_quantities = self.df.values  # Shape: (36, 11)
        hist_temporal = self.create_temporal_features(hist_dates).values
        beverage_features = self.create_beverage_features()

        # Create future dates
        future_dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                future_dates.append(pd.Timestamp(year=year, month=month, day=1))

        forecast_results = []

        # Predict each future month using only historical data
        for target_date in future_dates:
            target_temporal = self.create_temporal_features(pd.DatetimeIndex([target_date])).values[0]

            # For each beverage
            for bev_idx, beverage in enumerate(self.beverage_names):
                # Use full historical window (36 months or last 24 months)
                window_size = min(24, len(hist_quantities))
                hist_bev_quantities = hist_quantities[-window_size:, bev_idx]

                # Build features
                features = {
                    'beverage_encoded': self.label_encoder.transform([beverage])[0],

                    # Target month temporal features
                    'target_month': target_temporal[0],
                    'target_quarter': target_temporal[1],
                    'target_holiday': target_temporal[2],
                    'target_month_sin': target_temporal[3],
                    'target_month_cos': target_temporal[4],

                    # Beverage features
                    'is_diet': beverage_features[bev_idx],

                    # Lag features from historical window
                    'lag_1': hist_bev_quantities[-1],
                    'lag_2': hist_bev_quantities[-2] if len(hist_bev_quantities) >= 2 else hist_bev_quantities[-1],
                    'lag_3': hist_bev_quantities[-3] if len(hist_bev_quantities) >= 3 else hist_bev_quantities[-1],
                    'lag_6': hist_bev_quantities[-6] if len(hist_bev_quantities) >= 6 else np.mean(hist_bev_quantities),
                    'lag_12': hist_bev_quantities[-12] if len(hist_bev_quantities) >= 12 else np.mean(hist_bev_quantities),

                    # Rolling statistics from historical window
                    'rolling_mean_3': np.mean(hist_bev_quantities[-3:]),
                    'rolling_mean_6': np.mean(hist_bev_quantities[-6:]) if len(hist_bev_quantities) >= 6 else np.mean(hist_bev_quantities),
                    'rolling_mean_12': np.mean(hist_bev_quantities[-12:]) if len(hist_bev_quantities) >= 12 else np.mean(hist_bev_quantities),
                    'rolling_std_3': np.std(hist_bev_quantities[-3:]),
                    'rolling_std_6': np.std(hist_bev_quantities[-6:]) if len(hist_bev_quantities) >= 6 else np.std(hist_bev_quantities),
                    'rolling_std_12': np.std(hist_bev_quantities[-12:]) if len(hist_bev_quantities) >= 12 else np.std(hist_bev_quantities),

                    # Overall statistics
                    'hist_mean': np.mean(hist_bev_quantities),
                    'hist_std': np.std(hist_bev_quantities),
                    'hist_min': np.min(hist_bev_quantities),
                    'hist_max': np.max(hist_bev_quantities),
                    'hist_trend': (hist_bev_quantities[-1] - hist_bev_quantities[0]) / len(hist_bev_quantities) if hist_bev_quantities[0] > 0 else 0,
                }

                # Predict
                X_forecast = np.array([list(features.values())])
                prediction = max(0, self.model.predict(X_forecast)[0])

                forecast_results.append({
                    'beverage': beverage,
                    'year': target_date.year,
                    'month': target_date.month,
                    'quantity': prediction
                })

        forecast_df = pd.DataFrame(forecast_results)
        print(f"Generated {len(forecast_df)} forecasts ({len(self.beverage_names)} beverages × 24 months)")

        return forecast_df

    def visualize_forecasts(self, forecasts):
        """Create visualization comparing historical data with forecasts"""
        print("\nCreating forecast visualization...")

        # Prepare historical data
        hist_data = []
        for date in self.df.index:
            for bev in self.beverage_names:
                hist_data.append({
                    'beverage': bev,
                    'year': date.year,
                    'month': date.month,
                    'quantity': self.df.loc[date, bev],
                    'type': 'Historical'
                })
        hist_df = pd.DataFrame(hist_data)

        # Add type to forecasts
        forecast_df = forecasts.copy()
        forecast_df['type'] = 'Forecast'

        # Combine
        combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df[['year', 'month']].assign(day=1))

        # Plot
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, beverage in enumerate(self.beverage_names):
            ax = axes[idx]
            data = combined_df[combined_df['beverage'] == beverage]

            # Historical
            hist = data[data['type'] == 'Historical']
            ax.plot(hist['date'], hist['quantity'], 'o-', label='Historical', linewidth=2, markersize=4)

            # Forecast
            forecast = data[data['type'] == 'Forecast']
            ax.plot(forecast['date'], forecast['quantity'], 's--', label='Forecast',
                   linewidth=2, markersize=4, alpha=0.7)

            ax.set_title(beverage, fontsize=10, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('rf_forecasts_visualization.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to: rf_forecasts_visualization.png")
        plt.close()


def main():
    """Main execution"""
    print("=" * 70)
    print("Random Forest - Beverage Order Forecasting (Real Data Only)")
    print("=" * 70)

    # Initialize forecaster
    forecaster = RealDataForecaster('monthly_beverage_orders 2018-2020.xlsx')

    # Load and preprocess
    forecaster.load_and_preprocess()

    # Prepare training data
    forecaster.prepare_training_data(window_size=24)

    # Train model (can use 'random_forest' or 'gradient_boosting')
    forecaster.train_model(model_type='gradient_boosting')

    # Evaluate
    metrics = forecaster.evaluate_model()

    # Generate forecasts
    forecasts = forecaster.generate_forecasts(start_year=2021, end_year=2022)

    # Save forecasts
    forecasts.to_csv('rf_forecasts_2021_2022.csv', index=False)
    print(f"\nForecasts saved to: rf_forecasts_2021_2022.csv")

    # Visualize
    forecaster.visualize_forecasts(forecasts)

    print("\n" + "=" * 70)
    print("Forecasting complete!")
    print(f"Model: Gradient Boosting")
    print(f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
