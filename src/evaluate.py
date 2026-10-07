import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython.display import display


class ModelEvaluation:

    def plot_validation(self,
                        train,
                        test,
                        forecast,
                        forecast_label='forecast',
                        title='Time Series Validation'
                        ):

        # Ploting training data, actual test data, and model forecasts
        # for visual comparison.

        colors = {'train': '#4A4E69', 'test': '#2A9D8F', 'forecast': '#E76F51'}
        plt.figure(figsize=(14, 7))

        plt.plot(train.index, train, label='Train (Original)',
                 color=colors['train'], alpha=0.4, linewidth=1.5)

        plt.plot(test.index, test, label='Test (Actual)',
                 color=colors['test'], linewidth=2.5)

        plt.plot(forecast.index, forecast, label=forecast_label,
                 color=colors['forecast'], linestyle='--', linewidth=2.5)

        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=12, frameon=True, shadow=True)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

        return self

# *************************************************************************************

    def evaluate(self, y_true, y_pred, model_type, rnn_step=None):

        # Evaluate forecast accuracy using MAE and RMSE.

        model_type = model_type.upper().strip()

        if model_type == 'ARIMA':
            actual = y_true

        elif model_type == 'RNN':
            if rnn_step is None:
                raise ValueError(
                    "RNN step is not set. Call train_test_split(model_type='RNN', step=...) first.")
            actual = y_true[rnn_step:]
        else:
            raise ValueError("model_type must be either 'ARIMA' or 'RNN'.")

        # The alignment step ensures that actual and predicted values
        # are compared only on shared indices.

        act, predicted = actual.align(y_pred, join='inner')

        mae = mean_absolute_error(act, predicted)
        rmse = np.sqrt(mean_squared_error(act, predicted))

        report = {
            'Model Type': model_type,
            'MAE': mae,
            'RMSE': rmse}

        return display(pd.DataFrame([report]).T)
