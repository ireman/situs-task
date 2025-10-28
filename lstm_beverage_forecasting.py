"""
LSTM-based Beverage Order Forecasting
Treats each month as a vector of 11 beverages (matrix approach)
Uses LSTM to predict future months based on historical patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class LSTMBeverageForecaster:
    def __init__(self, excel_path='monthly_beverage_orders 2018-2020.xlsx'):
        self.excel_path = excel_path
        self.df = None
        self.beverage_names = None
        self.scaler_quantities = MinMaxScaler(feature_range=(0, 1))
        self.scaler_temporal = MinMaxScaler(feature_range=(0, 1))
        self.model = None

        # Diet keywords for is_diet feature
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

        # Get unique beverages (should be 11)
        self.beverage_names = sorted(df['beverage'].unique())
        print(f"Found {len(self.beverage_names)} beverages: {self.beverage_names}")

        # Pivot to create matrix: rows=months, columns=beverages
        quantity_matrix = df.pivot(index='date', columns='beverage', values='quantity')
        quantity_matrix = quantity_matrix[self.beverage_names]  # Ensure consistent order

        self.df = quantity_matrix
        print(f"Data shape: {self.df.shape} (months x beverages)")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

        return self.df

    def create_temporal_features(self, dates):
        """Create temporal features for given dates"""
        features = pd.DataFrame({
            'month': dates.month,
            'quarter': dates.quarter,
            'holiday': dates.month.isin([11, 12, 1]).astype(int),
            'month_sin': np.sin(2 * np.pi * dates.month / 12),
            'month_cos': np.cos(2 * np.pi * dates.month / 12),
        }, index=dates)

        return features

    def create_beverage_features(self):
        """Create per-beverage static features (is_diet)"""
        is_diet = []
        for bev in self.beverage_names:
            is_diet_flag = 1 if any(kw in bev.lower() for kw in self.diet_keywords) else 0
            is_diet.append(is_diet_flag)

        return np.array(is_diet)

    def prepare_training_data(self, window_size=24):
        """
        Prepare training data using rolling window approach:
        - Use sliding window of `window_size` months to predict next month
        - Creates multiple samples from 2018-2020 data
        - Each sample uses different historical period
        """
        print("\nPreparing training data with rolling window approach...")

        # Get beverage features (static)
        beverage_features = self.create_beverage_features()  # Shape: (11,)

        # Get all data
        all_dates = self.df.index
        all_quantities = self.df.values  # Shape: (36, 11)
        all_temporal = self.create_temporal_features(all_dates).values  # Shape: (36, 5)

        # Fit scalers on all data
        self.scaler_quantities.fit(all_quantities)
        self.scaler_temporal.fit(all_temporal)

        # Create rolling window samples
        X_train_list = []
        y_train_list = []
        sample_dates = []

        # Create samples using rolling window
        # For 36 months with window_size=24, we can create 36-24=12 samples
        for i in range(len(all_dates) - window_size):
            # Get window of historical data
            hist_start = i
            hist_end = i + window_size
            target_idx = i + window_size

            hist_quantities = all_quantities[hist_start:hist_end]  # Shape: (24, 11)
            hist_temporal = all_temporal[hist_start:hist_end]      # Shape: (24, 5)
            target_quantities = all_quantities[target_idx]          # Shape: (11,)
            target_date = all_dates[target_idx]
            target_temporal = all_temporal[target_idx]              # Shape: (5,)

            # Normalize
            hist_q_scaled = self.scaler_quantities.transform(hist_quantities)
            hist_t_scaled = self.scaler_temporal.transform(hist_temporal)
            target_q_scaled = self.scaler_quantities.transform(target_quantities.reshape(1, -1))[0]
            target_t_scaled = self.scaler_temporal.transform(target_temporal.reshape(1, -1))[0]

            # Combine features for each historical month
            # [11 quantities, 5 temporal (hist), 11 is_diet] = 27 features per month
            hist_combined = []
            for j in range(len(hist_q_scaled)):
                month_features = np.concatenate([
                    hist_q_scaled[j],      # 11 quantities
                    hist_t_scaled[j],      # 5 temporal (historical)
                    beverage_features      # 11 is_diet flags
                ])
                hist_combined.append(month_features)

            X_sample = np.array(hist_combined)  # Shape: (24, 27)

            # Broadcast target temporal features across all timesteps
            target_t_broadcast = np.tile(target_t_scaled, (window_size, 1))  # Shape: (24, 5)
            X_sample = np.concatenate([X_sample, target_t_broadcast], axis=1)  # Shape: (24, 32)

            X_train_list.append(X_sample)
            y_train_list.append(target_q_scaled)
            sample_dates.append(target_date)

        X_train = np.array(X_train_list)  # Shape: (12, 24, 32)
        y_train = np.array(y_train_list)  # Shape: (12, 11)

        print(f"Created {len(X_train_list)} training samples using rolling window")
        print(f"Window size: {window_size} months")
        print(f"X_train shape: {X_train.shape} (samples, timesteps, features)")
        print(f"y_train shape: {y_train.shape} (samples, beverages)")
        print(f"Training date range: {sample_dates[0]} to {sample_dates[-1]}")

        # Store for later use
        self.X_train = X_train
        self.y_train = y_train
        self.beverage_features = beverage_features
        self.test_dates = sample_dates

        return X_train, y_train

    def build_model(self, lstm_units=32, dropout_rate=0.3):
        """Build LSTM model with gradient clipping"""
        print("\nBuilding LSTM model...")

        # Input: (timesteps=24, features=32)
        # 32 features = 11 quantities + 5 temporal (hist) + 11 is_diet + 5 temporal (target)
        input_layer = layers.Input(shape=(24, 32))

        # LSTM layers with reduced units to prevent overfitting
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)(input_layer)
        x = layers.LayerNormalization()(x)  # Add normalization
        x = layers.LSTM(lstm_units // 2, return_sequences=False, dropout=dropout_rate)(x)
        x = layers.LayerNormalization()(x)  # Add normalization

        # Dense layers to predict 1 month × 11 beverages = 11 values
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(11)(x)  # 11 outputs (one per beverage)

        output_layer = x

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        # Use Adam with gradient clipping and lower learning rate
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,
            clipnorm=1.0  # Gradient clipping to prevent explosion
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        print(model.summary())

        return model

    def train_model(self, epochs=300, batch_size=4, verbose=1):
        """Train the LSTM model"""
        print("\nTraining model...")

        # Early stopping to prevent overfitting
        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )

        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=15,
            min_lr=0.00001,
            verbose=1
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        self.history = history
        print("Training completed!")

        return history

    def evaluate_on_2020(self):
        """Evaluate model performance on 2020 predictions"""
        print("\nEvaluating on 2020...")

        # Predict 2020 (we have 12 samples, one for each month)
        y_pred_scaled = self.model.predict(self.X_train, verbose=0)  # Shape: (12, 11)

        # Inverse transform to get actual quantities
        y_pred = self.scaler_quantities.inverse_transform(y_pred_scaled)
        y_true = self.scaler_quantities.inverse_transform(self.y_train)

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"2020 Evaluation Metrics:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")

        # Per-beverage metrics
        print("\nPer-beverage MAE:")
        for i, bev in enumerate(self.beverage_names):
            bev_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            print(f"  {bev}: {bev_mae:.2f}")

        # Store predictions
        self.predictions_2020 = pd.DataFrame(
            y_pred,
            index=self.test_dates,
            columns=self.beverage_names
        )

        return mae, rmse, r2

    def prepare_inference_data(self):
        """
        Prepare inference data to predict 2021-2022:
        - Use 2019-2020 (last 24 months) as historical input
        - Predict each month of 2021-2022 separately (24 predictions)
        - Each prediction uses same historical data but different temporal features
        """
        print("\nPreparing inference data for 2021-2022...")

        # Get last 24 months (2019-2020)
        inference_dates = self.df.index[self.df.index.year.isin([2019, 2020])]
        X_inference_quantities = self.df.loc[inference_dates].values  # Shape: (24, 11)
        X_inference_temporal = self.create_temporal_features(inference_dates).values  # Shape: (24, 5)

        # Normalize
        X_inference_quantities_scaled = self.scaler_quantities.transform(X_inference_quantities)
        X_inference_temporal_scaled = self.scaler_temporal.transform(X_inference_temporal)

        # Get beverage features
        beverage_features = self.beverage_features

        # Create future dates (2021-2022)
        future_start = pd.Timestamp('2021-01-01')
        future_dates = pd.date_range(start=future_start, periods=24, freq='MS')

        # For each future month, create a separate input with appropriate temporal features
        X_inference_list = []

        for future_date in future_dates:
            # Create temporal features for this specific future month
            future_temporal = self.create_temporal_features(pd.DatetimeIndex([future_date])).values[0]
            future_temporal_scaled = self.scaler_temporal.transform(future_temporal.reshape(1, -1))[0]

            # Combine features for each historical month
            # [11 quantities, 5 temporal (hist), 11 is_diet] = 27 features
            month_features_list = []
            for i in range(len(X_inference_quantities_scaled)):
                month_features = np.concatenate([
                    X_inference_quantities_scaled[i],  # 11 quantities
                    X_inference_temporal_scaled[i],    # 5 temporal (historical)
                    beverage_features                  # 11 is_diet flags
                ])
                month_features_list.append(month_features)

            X_sample = np.array(month_features_list)  # Shape: (24, 27)

            # Broadcast target temporal features across all timesteps
            target_t_broadcast = np.tile(future_temporal_scaled, (24, 1))  # Shape: (24, 5)
            X_sample = np.concatenate([X_sample, target_t_broadcast], axis=1)  # Shape: (24, 32)

            X_inference_list.append(X_sample)

        X_inference = np.array(X_inference_list)  # Shape: (24, 24, 32)

        print(f"X_inference shape: {X_inference.shape} (future_months, timesteps, features)")

        self.X_inference = X_inference
        self.future_dates = future_dates

        return X_inference, future_dates

    def generate_forecasts(self):
        """Generate forecasts for 2021-2022"""
        print("\nGenerating forecasts for 2021-2022...")

        # Predict for each future month (24 predictions, one per month)
        y_pred_scaled = self.model.predict(self.X_inference, verbose=0)  # Shape: (24, 11)

        # Inverse transform
        y_pred = self.scaler_quantities.inverse_transform(y_pred_scaled)

        # Create DataFrame
        forecasts_df = pd.DataFrame(
            y_pred,
            index=self.future_dates,
            columns=self.beverage_names
        )

        print(f"Generated forecasts for {len(forecasts_df)} months")
        print(f"Date range: {forecasts_df.index.min()} to {forecasts_df.index.max()}")

        self.forecasts_2021_2022 = forecasts_df

        return forecasts_df

    def save_forecasts(self, output_path='lstm_forecasts_2021_2022.csv'):
        """Save forecasts to CSV"""
        # Reshape to long format
        forecasts_long = []
        for date in self.forecasts_2021_2022.index:
            for beverage in self.beverage_names:
                quantity = self.forecasts_2021_2022.loc[date, beverage]
                forecasts_long.append({
                    'beverage': beverage,
                    'year': date.year,
                    'month': date.month,
                    'quantity': round(quantity, 2)
                })

        forecasts_df = pd.DataFrame(forecasts_long)
        forecasts_df.to_csv(output_path, index=False)
        print(f"\nForecasts saved to {output_path}")

        return forecasts_df

    def visualize_results(self):
        """Create visualizations"""
        print("\nCreating visualizations...")

        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        # Plot each beverage
        for idx, beverage in enumerate(self.beverage_names):
            ax = axes[idx]

            # Historical data (2018-2020)
            historical = self.df[beverage]
            ax.plot(historical.index, historical.values, 'o-', label='Historical', linewidth=2)

            # 2020 predictions
            pred_2020 = self.predictions_2020[beverage]
            ax.plot(pred_2020.index, pred_2020.values, 's-', label='2020 Prediction', linewidth=2, alpha=0.7)

            # 2021-2022 forecasts
            forecast = self.forecasts_2021_2022[beverage]
            ax.plot(forecast.index, forecast.values, '^-', label='2021-2022 Forecast', linewidth=2, alpha=0.7)

            ax.set_title(beverage, fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

        # Remove extra subplot
        fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig('lstm_forecasts_visualization.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to lstm_forecasts_visualization.png")
        plt.show()

        # Training history
        if hasattr(self, 'history'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(self.history.history['loss'])
            ax1.set_title('Model Loss During Training')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (MSE)')
            ax1.grid(True, alpha=0.3)

            ax2.plot(self.history.history['mae'])
            ax2.set_title('Model MAE During Training')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
            print("Training history saved to lstm_training_history.png")
            plt.show()


def main():
    """Main execution function"""
    print("=" * 70)
    print("LSTM-based Beverage Order Forecasting")
    print("=" * 70)

    # Initialize forecaster
    forecaster = LSTMBeverageForecaster()

    # Load and preprocess data
    forecaster.load_and_preprocess()

    # Prepare training data (2018-2019 → 2020)
    forecaster.prepare_training_data()

    # Build model
    forecaster.build_model(lstm_units=32, dropout_rate=0.3)

    # Train model
    forecaster.train_model(epochs=300, batch_size=4, verbose=1)

    # Evaluate on 2020
    forecaster.evaluate_on_2020()

    # Prepare inference data and generate forecasts for 2021-2022
    forecaster.prepare_inference_data()
    forecaster.generate_forecasts()

    # Save forecasts
    forecaster.save_forecasts()

    # Visualize results
    forecaster.visualize_results()

    print("\n" + "=" * 70)
    print("LSTM Forecasting Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
