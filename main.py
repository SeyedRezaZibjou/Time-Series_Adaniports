"""
Main execution pipeline for Adani Ports Time Series Forecasting.
Orchestrates data preparation, ARIMA modeling, RNN modeling, and evaluation.
"""

from pathlib import Path
import pandas as pd
from src.data_preparation import DataPreparation
from src.evaluate import ModelEvaluation
from src.modeling import Modeling


def main():
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    data_path = Path("data/ADANIPORTS.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    print("[INFO] Loading data...")
    try:
        df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        df = df.sort_index()
        raw_data = df["VWAP"].copy()
        print('The dataset was loaded successfully...')
    except Exception as e:
        print(f"[ERROR] Failed to read or parse dataset: {e}")
        return

    # ---------------------------------------------------------
    # 2. Data Preparation Pipeline
    # ---------------------------------------------------------
    print("[INFO] Preprocessing data...")
    preparation_instance = (DataPreparation(data=raw_data)
                            .adjust_stock_split(split_date='2010-09-23', ratio=5)
                            .clipping()
                            .apply_boxcox()
                            .apply_seasonal_diff(period=1))

    prepared_data = preparation_instance.result

    # ---------------------------------------------------------
    # 3. Bulding, fitting and evaluating ARIMA model
    # ---------------------------------------------------------
    print("\n--- Running ARIMA Pipeline ---")
    try:
        arima_instance = Modeling(
            data=prepared_data, prep=preparation_instance)
        arima_instance.train_test_split(model_type='ARIMA')

        arima_order = (0, 0, 1)
        print(f"[INFO] Fitting ARIMA{arima_order}...")
        arima_instance.arima_model(order=arima_order)
        arima_preds = arima_instance.forecast(model_type='ARIMA')

        # Converting the predicted values back to the original scale.
        arima_preds_original = arima_instance.inverse_transform(arima_preds)

        # Restoring the testing data to the original scale
        arima_test_original = arima_instance.inverse_transform(
            arima_instance.test_data)

        # Evaluate ARIMA
        arima_evaluate = ModelEvaluation()
        arima_evaluate.evaluate(y_true=arima_test_original,
                                y_pred=arima_preds_original,
                                model_type='ARIMA')
    except Exception as e:
        print(f"[ERROR] ARIMA pipeline failed: {e}")

    # ---------------------------------------------------------
    # 4. Bulding, fitting and evaluating Simple RNN model
    # ---------------------------------------------------------
    print("\n--- Running RNN Pipeline ---")
    try:
        rnn_instance = Modeling(data=prepared_data, prep=preparation_instance)
        rnn_instance.train_test_split(model_type='RNN', step=10)

        print("[INFO] Fitting Simple RNN...")

        """
        The RNN model hyperparameters have been initialized 
        with the following default values ​​in build_fit_rnn_model() method:
        
        units=64,
        loss='mean_squared_error',
        optimizer='rmsprop',
        epochs=100,
        batch_size=16,
        validation_split=0.1

        """
        rnn_instance.build_fit_rnn_model()

        rnn_preds = rnn_instance.forecast(model_type='RNN')

        # Converting the predicted values back to the original scale.
        rnn_preds_original = rnn_instance.inverse_transform(rnn_preds)

        # Restoring the aligned test segment to the original scale.
        rnn_test_original = rnn_instance.inverse_transform(
            rnn_instance.test_data[rnn_instance.rnn_step:])

        # evaluate RNN
        rnn_evaluate = ModelEvaluation()
        rnn_evaluate.evaluate(y_true=rnn_test_original,
                              y_pred=rnn_preds_original,
                              model_type='RNN',
                              rnn_step=rnn_instance.rnn_step)
    except Exception as e:
        print(f"[ERROR] RNN pipeline failed: {e}")

    print("\n[SUCCESS] Pipeline executed successfully.")


if __name__ == "__main__":
    main()
