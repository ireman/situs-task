"""
Beverage Order Forecasting Model
Forecasts monthly order quantities for beverages for 2021-2022 based on historical data (2018-2020)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class BeverageForecaster:
    """Forecasts beverage orders using Random Forest with engineered features."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.processed_data = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_and_preprocess(self):
        """Load and preprocess data with feature engineering."""
        # Load data
        self.data = pd.read_excel(self.filepath)
        df = self.data.rename(columns={
            'Name': 'beverage',
            'Year': 'year',
            'Month': 'month',
            'Quantity': 'quantity'
        })

        # Create datetime and sort
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values(['beverage', 'date'])

        # Time features
        min_date = df['date'].min()
        df['month_num'] = df['month']
        df['quarter'] = df['date'].dt.quarter
        df['time_idx'] = ((df['date'].dt.year - min_date.year) * 12 +
                          (df['date'].dt.month - min_date.month))
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Product features
        diet_keywords = ['diet', 'zero', 'lite']
        df['is_diet'] = df['beverage'].str.lower().apply(
            lambda x: 1 if any(keyword in x for keyword in diet_keywords) else 0
        )
        df['holiday'] = df['month'].apply(lambda x: 1 if x in [11, 12, 1] else 0)
        df['beverage_encoded'] = self.label_encoder.fit_transform(df['beverage'])

        # Lag and rolling features per beverage
        lag_features = []
        for beverage in df['beverage'].unique():
            beverage_data = df[df['beverage'] == beverage].copy()

            # Lags: 1, 2, 3, 6, 12 months
            for lag in [1, 2, 3, 6, 12]:
                beverage_data[f'lag_{lag}'] = beverage_data['quantity'].shift(lag)

            # Rolling statistics: 3, 6, 12 month windows
            for window in [3, 6, 12]:
                beverage_data[f'rolling_mean_{window}'] = (
                    beverage_data['quantity'].shift(1).rolling(window, min_periods=1).mean()
                )
                beverage_data[f'rolling_std_{window}'] = (
                    beverage_data['quantity'].shift(1).rolling(window, min_periods=1).std()
                )

            lag_features.append(beverage_data)

        df = pd.concat(lag_features, ignore_index=True)

        # Fill NaN values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64'] and col != 'quantity':
                df[col].fillna(df[col].median(), inplace=True)

        self.processed_data = df
        return df

    def train_model(self):
        """Train Random Forest model."""
        feature_cols = [
            'beverage_encoded', 'time_idx', 'month_num', 'quarter',
            'month_sin', 'month_cos', 'is_diet', 'holiday',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
            'rolling_std_3', 'rolling_std_6', 'rolling_std_12'
        ]

        X_train = self.processed_data[feature_cols]
        y_train = self.processed_data['quantity']

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        return self.model

    def generate_forecasts(self, start_year=2021, end_year=2022):
        """Generate forecasts for specified years using iterative approach."""
        # Prepare future dates
        future_dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                future_dates.append({'year': year, 'month': month})
        future_df = pd.DataFrame(future_dates)

        # Forecast each beverage
        all_forecasts = []
        beverages = self.processed_data['beverage'].unique()

        for beverage in beverages:
            # Combine historical and future
            hist_data = self.processed_data[
                self.processed_data['beverage'] == beverage
            ][['beverage', 'date', 'year', 'month', 'quantity']].copy()

            future_data = future_df.copy()
            future_data['beverage'] = beverage
            future_data['date'] = pd.to_datetime(
                future_data[['year', 'month']].assign(day=1)
            )
            future_data['quantity'] = np.nan

            all_dates = pd.concat([hist_data, future_data], ignore_index=True)
            all_dates = all_dates.sort_values('date').reset_index(drop=True)

            # Iterative forecasting
            hist_end = all_dates['quantity'].notna().sum()

            for i in range(hist_end, len(all_dates)):
                current_quantities = all_dates['quantity'].iloc[:i].values
                current_row = all_dates.iloc[i]

                # Build features
                min_date = self.processed_data['date'].min()
                diet_keywords = ['diet', 'zero', 'lite']
                is_diet = 1 if any(kw in beverage.lower() for kw in diet_keywords) else 0

                features = {
                    'beverage_encoded': self.label_encoder.transform([beverage])[0],
                    'time_idx': ((current_row['year'] - min_date.year) * 12 +
                                (current_row['month'] - min_date.month)),
                    'month_num': current_row['month'],
                    'quarter': (current_row['month'] - 1) // 3 + 1,
                    'month_sin': np.sin(2 * np.pi * current_row['month'] / 12),
                    'month_cos': np.cos(2 * np.pi * current_row['month'] / 12),
                    'is_diet': is_diet,
                    'holiday': 1 if current_row['month'] in [11, 12, 1] else 0,
                }

                # Lag features
                for lag in [1, 2, 3, 6, 12]:
                    if i >= lag:
                        features[f'lag_{lag}'] = current_quantities[i - lag]
                    else:
                        features[f'lag_{lag}'] = np.median(current_quantities)

                # Rolling features
                for window in [3, 6, 12]:
                    if i >= window:
                        features[f'rolling_mean_{window}'] = np.mean(current_quantities[i-window:i])
                        features[f'rolling_std_{window}'] = np.std(current_quantities[i-window:i])
                    else:
                        features[f'rolling_mean_{window}'] = np.mean(current_quantities[:i])
                        features[f'rolling_std_{window}'] = np.std(current_quantities[:i]) if i > 1 else 0

                # Predict
                feature_cols = [
                    'beverage_encoded', 'time_idx', 'month_num', 'quarter',
                    'month_sin', 'month_cos', 'is_diet', 'holiday',
                    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                    'rolling_std_3', 'rolling_std_6', 'rolling_std_12'
                ]
                X_forecast = np.array([[features[col] for col in feature_cols]])
                prediction = max(0, self.model.predict(X_forecast)[0])

                all_dates.loc[i, 'quantity'] = prediction

            # Extract forecasts
            forecasts = all_dates[all_dates['year'].isin(range(start_year, end_year + 1))].copy()
            all_forecasts.append(forecasts)

        result = pd.concat(all_forecasts, ignore_index=True)
        result = result.sort_values(['beverage', 'date'])
        return result


def main():
    """Main execution."""
    # Initialize and run forecaster
    forecaster = BeverageForecaster('monthly_beverage_orders 2018-2020.xlsx')

    # Load and preprocess data
    forecaster.load_and_preprocess()

    # Train model
    forecaster.train_model()

    # Generate forecasts for 2021-2022
    forecasts = forecaster.generate_forecasts(start_year=2021, end_year=2022)

    # Save results
    forecasts[['beverage', 'year', 'month', 'quantity']].to_csv(
        'beverage_forecasts_2021_2022.csv',
        index=False
    )

    print(f"Forecasts generated for {forecasts['beverage'].nunique()} beverages")
    print(f"Total forecast records: {len(forecasts)}")
    print(f"Forecasts saved to: beverage_forecasts_2021_2022.csv")


if __name__ == "__main__":
    main()
