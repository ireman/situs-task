"""
Beverage Order Forecasting Model
Author: Claude
Date: 2025-10-21

This script loads historical beverage order data (2018-2020) and forecasts
monthly order quantities for each beverage for 2021-2022.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BeverageForecaster:
    """
    A class to handle beverage order forecasting using multiple approaches.
    """

    def __init__(self, filepath):
        """Initialize the forecaster with data filepath."""
        self.filepath = filepath
        self.data = None
        self.processed_data = None
        self.models = {}
        self.predictions = {}
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """Load and inspect the Excel data."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load the Excel file
        self.data = pd.read_excel(self.filepath)

        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nColumn names: {list(self.data.columns)}")
        print(f"\nFirst few rows:")
        print(self.data.head(10))
        print(f"\nData types:")
        print(self.data.dtypes)
        print(f"\nMissing values:")
        print(self.data.isnull().sum())
        print(f"\nBasic statistics:")
        print(self.data.describe())

        return self.data

    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)

        # Identify beverage column and quantity column
        # Common patterns: 'beverage', 'product', 'item', 'quantity', 'orders', 'amount'
        print(f"\nUnique values per column:")
        for col in self.data.columns:
            n_unique = self.data[col].nunique()
            print(f"  {col}: {n_unique} unique values")
            if n_unique < 50:  # Show unique values for categorical columns
                print(f"    Values: {sorted(self.data[col].unique())[:10]}")

        # Analyze beverage types
        beverage_col = self._identify_beverage_column()
        if beverage_col:
            print(f"\nBeverage types found: {self.data[beverage_col].nunique()}")
            print(f"Beverages: {sorted(self.data[beverage_col].unique())}")

        # Analyze time range
        year_col = self._identify_year_column()
        month_col = self._identify_month_column()

        if year_col and month_col:
            print(f"\nTime range: {self.data[year_col].min()}-{self.data[year_col].max()}")
            print(f"Months covered: {sorted(self.data[month_col].unique())}")

        return self.data

    def _identify_beverage_column(self):
        """Identify the column containing beverage names."""
        # First, check for exact match "Name"
        if 'Name' in self.data.columns:
            return 'Name'

        possible_names = ['beverage', 'product', 'item', 'drink', 'name']
        for col in self.data.columns:
            if any(name in col.lower() for name in possible_names):
                return col
        # If not found by name, look for object/string column with moderate unique values
        # Exclude columns like 'Unnamed: 0' which are likely indices
        for col in self.data.columns:
            if (self.data[col].dtype == 'object' and
                5 <= self.data[col].nunique() <= 100 and
                'unnamed' not in col.lower()):
                return col
        return None

    def _identify_year_column(self):
        """Identify the column containing year information."""
        for col in self.data.columns:
            if 'year' in col.lower():
                return col
        return None

    def _identify_month_column(self):
        """Identify the column containing month information."""
        for col in self.data.columns:
            if 'month' in col.lower():
                return col
        return None

    def _identify_quantity_column(self):
        """Identify the column containing order quantities."""
        possible_names = ['quantity', 'amount', 'order', 'count', 'total', 'qty']
        for col in self.data.columns:
            if any(name in col.lower() for name in possible_names):
                if self.data[col].dtype in ['int64', 'float64']:
                    return col
        # If not found by name, look for numeric column
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[-1]  # Return last numeric column
        return None

    def preprocess_data(self):
        """Preprocess and feature engineer the data."""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING & FEATURE ENGINEERING")
        print("=" * 80)

        # Identify key columns
        beverage_col = self._identify_beverage_column()
        year_col = self._identify_year_column()
        month_col = self._identify_month_column()
        qty_col = self._identify_quantity_column()

        print(f"\nIdentified columns:")
        print(f"  Beverage: {beverage_col}")
        print(f"  Year: {year_col}")
        print(f"  Month: {month_col}")
        print(f"  Quantity: {qty_col}")

        # Create a working copy
        df = self.data.copy()

        # Standardize column names
        df = df.rename(columns={
            beverage_col: 'beverage',
            year_col: 'year',
            month_col: 'month',
            qty_col: 'quantity'
        })

        # Create datetime column
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

        # Sort by date and beverage
        df = df.sort_values(['beverage', 'date'])

        # Feature engineering
        print("\nCreating time-based features...")

        # Temporal features
        df['month_num'] = df['month']
        df['quarter'] = df['date'].dt.quarter
        df['year_month'] = df['year'] * 100 + df['month']

        # Create sequential time index (months since start)
        min_date = df['date'].min()
        df['time_idx'] = ((df['date'].dt.year - min_date.year) * 12 +
                          (df['date'].dt.month - min_date.month))

        # Cyclical encoding for month (to capture seasonality)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Encode beverage names
        df['beverage_encoded'] = self.label_encoder.fit_transform(df['beverage'])

        # Lag features for each beverage
        print("Creating lag features...")
        lag_features = []
        for beverage in df['beverage'].unique():
            beverage_mask = df['beverage'] == beverage
            beverage_data = df[beverage_mask].copy()

            # Create lags (1, 2, 3, 6, 12 months)
            for lag in [1, 2, 3, 6, 12]:
                col_name = f'lag_{lag}'
                beverage_data[col_name] = beverage_data['quantity'].shift(lag)

            # Rolling statistics
            for window in [3, 6, 12]:
                beverage_data[f'rolling_mean_{window}'] = (
                    beverage_data['quantity'].shift(1).rolling(window, min_periods=1).mean()
                )
                beverage_data[f'rolling_std_{window}'] = (
                    beverage_data['quantity'].shift(1).rolling(window, min_periods=1).std()
                )

            lag_features.append(beverage_data)

        df = pd.concat(lag_features, ignore_index=True)

        # Fill NaN values for lag features with median
        lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        for col in lag_cols:
            df[col].fillna(df[col].median(), inplace=True)

        self.processed_data = df

        print(f"\nProcessed data shape: {df.shape}")
        print(f"Features created: {df.columns.tolist()}")
        print(f"\nSample of processed data:")
        print(df.head())

        return df

    def train_models(self):
        """Train multiple forecasting models."""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        # Prepare training data (all 2018-2020 data)
        train_data = self.processed_data.copy()

        # Features for modeling
        feature_cols = [
            'beverage_encoded', 'time_idx', 'month_num', 'quarter',
            'month_sin', 'month_cos',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
            'rolling_std_3', 'rolling_std_6', 'rolling_std_12'
        ]

        X_train = train_data[feature_cols]
        y_train = train_data['quantity']

        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Target variable statistics:")
        print(f"  Mean: {y_train.mean():.2f}")
        print(f"  Std: {y_train.std():.2f}")
        print(f"  Min: {y_train.min():.2f}")
        print(f"  Max: {y_train.max():.2f}")

        # Train Random Forest
        print("\nTraining Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        # Train Gradient Boosting
        print("Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model

        # Evaluate on training data
        print("\nModel Performance on Training Data:")
        for name, model in self.models.items():
            y_pred = model.predict(X_train)
            mae = mean_absolute_error(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            r2 = r2_score(y_train, y_pred)

            print(f"\n{name.upper()}:")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")

        # Feature importance
        print("\nTop 10 Feature Importances (Random Forest):")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10))

        return self.models

    def generate_forecasts(self):
        """Generate forecasts for 2021-2022."""
        print("\n" + "=" * 80)
        print("GENERATING FORECASTS FOR 2021-2022")
        print("=" * 80)

        # Get unique beverages
        beverages = self.processed_data['beverage'].unique()

        # Prepare future dates (2021-2022)
        future_dates = []
        for year in [2021, 2022]:
            for month in range(1, 13):
                future_dates.append({'year': year, 'month': month})

        future_df = pd.DataFrame(future_dates)

        # Create full forecast dataframe with all beverage-date combinations
        forecast_data = []

        for beverage in beverages:
            print(f"\nForecasting for: {beverage}")

            # Get historical data for this beverage
            hist_data = self.processed_data[
                self.processed_data['beverage'] == beverage
            ].copy()

            # Create future records for this beverage
            for idx, row in future_df.iterrows():
                future_record = {
                    'beverage': beverage,
                    'year': row['year'],
                    'month': row['month']
                }

                # Create date
                future_record['date'] = pd.to_datetime(
                    f"{row['year']}-{row['month']:02d}-01"
                )

                # Calculate time_idx
                min_date = self.processed_data['date'].min()
                future_record['time_idx'] = (
                    (row['year'] - min_date.year) * 12 +
                    (row['month'] - min_date.month)
                )

                # Temporal features
                future_record['month_num'] = row['month']
                future_record['quarter'] = (row['month'] - 1) // 3 + 1
                future_record['month_sin'] = np.sin(2 * np.pi * row['month'] / 12)
                future_record['month_cos'] = np.cos(2 * np.pi * row['month'] / 12)

                # Encode beverage
                future_record['beverage_encoded'] = self.label_encoder.transform([beverage])[0]

                forecast_data.append(future_record)

        forecast_df = pd.DataFrame(forecast_data)

        # Now we need to iteratively forecast and update lag features
        print("\nIterative forecasting with lag updates...")

        # Combine historical and future data
        all_data = pd.concat([
            self.processed_data[['beverage', 'date', 'year', 'month', 'quantity']],
            forecast_df[['beverage', 'date', 'year', 'month']]
        ], ignore_index=True).sort_values(['beverage', 'date'])

        # Use Random Forest for final predictions (typically performs well)
        model = self.models['random_forest']

        # Iterate through future dates
        for beverage in beverages:
            beverage_mask = all_data['beverage'] == beverage
            beverage_data = all_data[beverage_mask].copy().reset_index(drop=True)

            # Find where historical data ends
            hist_end_idx = beverage_data['quantity'].notna().sum()

            # Forecast month by month
            for i in range(hist_end_idx, len(beverage_data)):
                # Calculate lag features based on available data
                current_quantities = beverage_data['quantity'].iloc[:i].values

                # Create features for current forecast point
                current_row = beverage_data.iloc[i]

                features = {
                    'beverage_encoded': self.label_encoder.transform([current_row['beverage']])[0],
                    'time_idx': (
                        (current_row['year'] - self.processed_data['date'].min().year) * 12 +
                        (current_row['month'] - self.processed_data['date'].min().month)
                    ),
                    'month_num': current_row['month'],
                    'quarter': (current_row['month'] - 1) // 3 + 1,
                    'month_sin': np.sin(2 * np.pi * current_row['month'] / 12),
                    'month_cos': np.cos(2 * np.pi * current_row['month'] / 12),
                }

                # Lag features
                for lag in [1, 2, 3, 6, 12]:
                    if i >= lag:
                        features[f'lag_{lag}'] = current_quantities[i - lag]
                    else:
                        features[f'lag_{lag}'] = np.median(current_quantities)

                # Rolling statistics
                for window in [3, 6, 12]:
                    if i >= window:
                        features[f'rolling_mean_{window}'] = np.mean(current_quantities[i-window:i])
                        features[f'rolling_std_{window}'] = np.std(current_quantities[i-window:i])
                    else:
                        features[f'rolling_mean_{window}'] = np.mean(current_quantities[:i])
                        features[f'rolling_std_{window}'] = np.std(current_quantities[:i]) if i > 1 else 0

                # Create feature vector in correct order
                feature_cols = [
                    'beverage_encoded', 'time_idx', 'month_num', 'quarter',
                    'month_sin', 'month_cos',
                    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                    'rolling_std_3', 'rolling_std_6', 'rolling_std_12'
                ]

                X_forecast = np.array([[features[col] for col in feature_cols]])

                # Predict
                prediction = model.predict(X_forecast)[0]

                # Ensure non-negative predictions
                prediction = max(0, prediction)

                # Update quantity in the dataframe
                beverage_data.loc[i, 'quantity'] = prediction

            # Update all_data with forecasted values
            all_data.loc[beverage_mask, 'quantity'] = beverage_data['quantity'].values

        # Extract forecasts for 2021-2022
        forecasts = all_data[all_data['year'].isin([2021, 2022])].copy()
        forecasts = forecasts.sort_values(['beverage', 'date'])

        self.predictions['forecasts'] = forecasts

        print(f"\nGenerated forecasts for {len(beverages)} beverages")
        print(f"Total forecast records: {len(forecasts)}")
        print(f"\nSample forecasts:")
        print(forecasts.head(10))

        # Save forecasts to CSV
        output_file = 'beverage_forecasts_2021_2022.csv'
        forecasts[['beverage', 'year', 'month', 'quantity']].to_csv(
            output_file, index=False
        )
        print(f"\nForecasts saved to: {output_file}")

        return forecasts

    def create_visualizations(self):
        """Create visualizations for historical data and forecasts."""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        # Combine historical and forecast data
        historical = self.processed_data[['beverage', 'date', 'year', 'month', 'quantity']].copy()
        historical['type'] = 'Historical'

        forecasts = self.predictions['forecasts'][['beverage', 'date', 'year', 'month', 'quantity']].copy()
        forecasts['type'] = 'Forecast'

        combined = pd.concat([historical, forecasts], ignore_index=True)
        combined = combined.sort_values(['beverage', 'date'])

        # Get unique beverages
        beverages = sorted(combined['beverage'].unique())

        # Create visualizations
        n_beverages = len(beverages)
        n_cols = 3
        n_rows = (n_beverages + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_beverages > 1 else [axes]

        for idx, beverage in enumerate(beverages):
            ax = axes[idx]

            # Filter data for this beverage
            beverage_data = combined[combined['beverage'] == beverage]

            # Split by type
            hist = beverage_data[beverage_data['type'] == 'Historical']
            forecast = beverage_data[beverage_data['type'] == 'Forecast']

            # Plot
            ax.plot(hist['date'], hist['quantity'],
                   marker='o', linewidth=2, markersize=4,
                   label='Historical (2018-2020)', color='steelblue')

            if len(forecast) > 0:
                ax.plot(forecast['date'], forecast['quantity'],
                       marker='s', linewidth=2, markersize=4,
                       label='Forecast (2021-2022)',
                       color='coral', linestyle='--')

            ax.set_title(f'{beverage}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Quantity', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for idx in range(n_beverages, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig('beverage_forecasts_visualization.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved: beverage_forecasts_visualization.png")

        # Create summary statistics plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Total orders by year
        yearly_totals = combined.groupby(['year', 'type'])['quantity'].sum().reset_index()

        for data_type in ['Historical', 'Forecast']:
            data = yearly_totals[yearly_totals['type'] == data_type]
            color = 'steelblue' if data_type == 'Historical' else 'coral'
            axes[0].bar(data['year'], data['quantity'],
                       label=data_type, alpha=0.7, color=color)

        axes[0].set_title('Total Orders by Year', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Total Quantity', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Average monthly orders by beverage
        beverage_avg = combined.groupby(['beverage', 'type'])['quantity'].mean().reset_index()

        beverages = sorted(beverage_avg['beverage'].unique())
        x = np.arange(len(beverages))
        width = 0.35

        hist_avg = beverage_avg[beverage_avg['type'] == 'Historical']['quantity'].values
        forecast_avg = beverage_avg[beverage_avg['type'] == 'Forecast']['quantity'].values

        axes[1].bar(x - width/2, hist_avg, width, label='Historical', color='steelblue', alpha=0.7)
        axes[1].bar(x + width/2, forecast_avg, width, label='Forecast', color='coral', alpha=0.7)

        axes[1].set_title('Average Monthly Orders by Beverage', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Beverage', fontsize=12)
        axes[1].set_ylabel('Average Quantity', fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(beverages, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('beverage_summary_statistics.png', dpi=300, bbox_inches='tight')
        print("Summary statistics saved: beverage_summary_statistics.png")

        plt.close('all')

        return True


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("BEVERAGE ORDER FORECASTING SYSTEM")
    print("=" * 80)

    # Initialize forecaster
    forecaster = BeverageForecaster('monthly_beverage_orders 2018-2020.xlsx')

    # Step 1: Load data
    forecaster.load_data()

    # Step 2: Explore data
    forecaster.explore_data()

    # Step 3: Preprocess and feature engineering
    forecaster.preprocess_data()

    # Step 4: Train models
    forecaster.train_models()

    # Step 5: Generate forecasts
    forecaster.generate_forecasts()

    # Step 6: Create visualizations
    forecaster.create_visualizations()

    print("\n" + "=" * 80)
    print("FORECASTING COMPLETE!")
    print("=" * 80)
    print("\nOutputs generated:")
    print("  1. beverage_forecasts_2021_2022.csv - Forecast data")
    print("  2. beverage_forecasts_visualization.png - Individual beverage trends")
    print("  3. beverage_summary_statistics.png - Summary statistics")


if __name__ == "__main__":
    main()
