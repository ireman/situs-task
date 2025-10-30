"""
Facebook Prophet Beverage Forecasting
Uses Prophet to capture trends and seasonality in beverage order data
Prophet automatically handles trend + seasonal decomposition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet


class ProphetForecaster:
    """Beverage forecasting using Facebook Prophet"""

    def __init__(self, excel_path='monthly_beverage_orders 2018-2020.xlsx'):
        self.excel_path = excel_path
        self.df = None
        self.beverage_names = None
        self.models = {}
        self.forecasts = {}

    def load_and_preprocess(self):
        """Load data and prepare in Prophet format (ds, y columns)"""
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

    def prepare_beverage_data(self, beverage):
        """
        Prepare data for a single beverage in Prophet format.
        Prophet requires columns: 'ds' (datetime) and 'y' (values)
        """
        data = pd.DataFrame({
            'ds': self.df.index,
            'y': self.df[beverage].values
        })
        return data

    def train_models(self,
                     seasonality_mode='multiplicative',
                     changepoint_prior_scale=0.05,
                     seasonality_prior_scale=10.0):
        """
        Train Prophet models for each beverage.

        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Controls trend flexibility (default 0.05)
            seasonality_prior_scale: Controls seasonality strength (default 10.0)
        """
        print("\n" + "="*70)
        print("Training Facebook Prophet models for each beverage")
        print("="*70)
        print(f"Configuration:")
        print(f"  Seasonality mode: {seasonality_mode}")
        print(f"  Changepoint prior scale: {changepoint_prior_scale}")
        print(f"  Seasonality prior scale: {seasonality_prior_scale}")

        for beverage in self.beverage_names:
            print(f"\n--- Training model for: {beverage} ---")

            # Prepare data
            data = self.prepare_beverage_data(beverage)
            print(f"Training samples: {len(data)}")
            print(f"Date range: {data['ds'].min()} to {data['ds'].max()}")
            print(f"Value range: {data['y'].min():.2f} to {data['y'].max():.2f}")
            print(f"Mean: {data['y'].mean():.2f}, Std: {data['y'].std():.2f}")

            # Create Prophet model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=True,      # Capture yearly patterns
                weekly_seasonality=False,     # Disable (monthly data)
                daily_seasonality=False,      # Disable (monthly data)
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
            )

            # Fit model (suppress output)
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            model.fit(data)

            # Store model
            self.models[beverage] = model
            print(f"✓ Model trained successfully")

        print("\n" + "="*70)
        print(f"Training complete for {len(self.models)} beverages")
        print("="*70)

    def generate_forecasts(self, start_year=2021, end_year=2022):
        """
        Generate forecasts for all beverages for 2021-2022.
        """
        print(f"\nGenerating forecasts for {start_year}-{end_year}...")

        all_forecasts = []

        for beverage in self.beverage_names:
            model = self.models[beverage]

            # Calculate number of periods to forecast
            num_months = (end_year - start_year + 1) * 12

            # Create future dataframe
            future = model.make_future_dataframe(periods=num_months, freq='MS')

            # Make predictions
            forecast = model.predict(future)

            # Extract only future predictions (2021-2022)
            forecast['beverage'] = beverage
            forecast['year'] = forecast['ds'].dt.year
            forecast['month'] = forecast['ds'].dt.month

            # Filter to forecast period
            future_forecast = forecast[
                (forecast['year'] >= start_year) &
                (forecast['year'] <= end_year)
            ].copy()

            # Use yhat as the prediction (main forecast), clip to non-negative
            future_forecast['quantity'] = future_forecast['yhat'].clip(lower=0)

            all_forecasts.append(future_forecast[['beverage', 'year', 'month', 'quantity']])

            print(f"{beverage:30s} - Generated {len(future_forecast)} forecasts")

        # Combine all forecasts
        result = pd.concat(all_forecasts, ignore_index=True)
        result = result.sort_values(['beverage', 'year', 'month'])

        print(f"\nTotal forecasts generated: {len(result)}")
        return result

    def evaluate_on_historical(self):
        """
        Evaluate models on historical data (in-sample performance)
        """
        print("\n" + "="*70)
        print("Evaluating models on historical data")
        print("="*70)

        results = []

        for beverage in self.beverage_names:
            model = self.models[beverage]
            data = self.prepare_beverage_data(beverage)

            # Predict on historical data
            forecast = model.predict(data)

            # Calculate metrics
            actual = data['y'].values
            predicted = forecast['yhat'].values

            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted)**2))

            # Avoid division by zero
            mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100

            results.append({
                'beverage': beverage,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            })

            print(f"{beverage:30s} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")

        results_df = pd.DataFrame(results)

        print(f"\n{'Average across all beverages':30s} - MAE: {results_df['MAE'].mean():.2f}, "
              f"RMSE: {results_df['RMSE'].mean():.2f}, MAPE: {results_df['MAPE'].mean():.1f}%")

        return results_df

    def visualize_forecasts(self, forecasts):
        """Create visualization comparing historical data with forecasts"""
        print("\nCreating forecast visualization...")

        # Prepare historical data
        hist_data = []
        for date in self.df.index:
            for bev in self.beverage_names:
                hist_data.append({
                    'beverage': bev,
                    'date': date,
                    'quantity': self.df.loc[date, bev],
                    'type': 'Historical'
                })
        hist_df = pd.DataFrame(hist_data)

        # Prepare forecast data
        forecast_df = forecasts.copy()
        forecast_df['date'] = pd.to_datetime(forecast_df[['year', 'month']].assign(day=1))
        forecast_df['type'] = 'Forecast'

        # Combine
        combined_df = pd.concat([
            hist_df[['beverage', 'date', 'quantity', 'type']],
            forecast_df[['beverage', 'date', 'quantity', 'type']]
        ], ignore_index=True)

        # Plot
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, beverage in enumerate(self.beverage_names):
            ax = axes[idx]
            data = combined_df[combined_df['beverage'] == beverage]

            # Historical
            hist = data[data['type'] == 'Historical']
            ax.plot(hist['date'], hist['quantity'], 'o-', label='Historical (2018-2020)',
                   linewidth=2, markersize=5, color='#2E86AB')

            # Forecast
            forecast = data[data['type'] == 'Forecast']
            ax.plot(forecast['date'], forecast['quantity'], 's--', label='Prophet Forecast',
                   linewidth=2, markersize=4, alpha=0.8, color='#F18F01')

            # Add vertical line to separate historical and forecast
            last_hist_date = hist['date'].max()
            ax.axvline(x=last_hist_date, color='gray', linestyle=':', alpha=0.5, linewidth=1)

            ax.set_title(beverage, fontsize=10, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('prophet_forecasts_visualization.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to: prophet_forecasts_visualization.png")
        plt.close()

    def plot_components(self, beverage):
        """Plot Prophet components (trend, seasonality) for a specific beverage"""
        print(f"\nPlotting components for: {beverage}")

        model = self.models[beverage]
        data = self.prepare_beverage_data(beverage)

        # Create future dataframe including historical
        future = model.make_future_dataframe(periods=24, freq='MS')
        forecast = model.predict(future)

        # Plot components
        fig = model.plot_components(forecast)
        plt.savefig(f'prophet_components_{beverage.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Components saved to: prophet_components_{beverage.replace(' ', '_')}.png")


def main():
    """Main execution"""
    print("="*70)
    print("Facebook Prophet - Beverage Order Forecasting")
    print("Captures trend + seasonality patterns automatically")
    print("="*70)

    # Initialize forecaster
    forecaster = ProphetForecaster('monthly_beverage_orders 2018-2020.xlsx')

    # Load and preprocess
    forecaster.load_and_preprocess()

    # Train models for all beverages
    forecaster.train_models(
        seasonality_mode='multiplicative',  # Better for varying amplitude
        changepoint_prior_scale=0.05,       # Default flexibility
        seasonality_prior_scale=10.0        # Default seasonality strength
    )

    # Evaluate on historical data
    eval_results = forecaster.evaluate_on_historical()

    # Generate forecasts for 2021-2022
    forecasts = forecaster.generate_forecasts(start_year=2021, end_year=2022)

    # Save forecasts
    forecasts.to_csv('prophet_forecasts_2021_2022.csv', index=False)
    print(f"\nForecasts saved to: prophet_forecasts_2021_2022.csv")

    # Visualize
    forecaster.visualize_forecasts(forecasts)

    # Plot components for first beverage as example
    forecaster.plot_components(forecaster.beverage_names[0])

    print("\n" + "="*70)
    print("Prophet Forecasting Complete!")
    print("Prophet captures trends and seasonal patterns automatically")
    print("="*70)


if __name__ == "__main__":
    main()
