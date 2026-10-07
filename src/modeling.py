import os
import datetime
import itertools
import numpy as np
import pandas as pd
from IPython.display import display

# Classical Time Series
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from scipy.special import inv_boxcox

# Deep Learning (RNN)
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class Modeling:

    def __init__(self, data, prep=None):

        # Storing a copy of the input data to avoid modifying the original series.

        self.data = data.copy()

        # Keeping a reference to the preprocessing object for inverse transformations

        self.prep = prep

        # Holding train/test splits for classical time series modeling.

        self.train_data = None
        self.test_data = None

        # Holding windowed train/test sets for RNN models.

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Storing fitted ARIMA and RNN models separately.

        self.fitted_model = None
        self.rnn_fitted_model = None

        # Storing training history and the RNN window size.

        self.history = None
        self.rnn_step = None

# *************************************************************************************

    def __make_windows(self, data, step):

        # Convert a 1D time series into overlapping input-output windows
        # for sequence learning models such as RNN.

        X, y = [], []

        values = np.array(data).reshape(-1, 1)

        for i in range(len(values) - step):
            X.append(values[i:i+step])
            y.append(values[i+step])

        return np.array(X), np.array(y)

# *************************************************************************************

    def train_test_split(self, model_type, train_size=0.8, step=None):

        # Split the data into training and testing sets.
        # If the model is RNN, also generate sliding windows.

        model_type = model_type.upper().strip()

        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")

        split_idx = int(len(self.data)*train_size)

        self.train_data = self.data[: split_idx]
        self.test_data = self.data[split_idx:]

        if model_type == 'ARIMA':

            # For ARIMA, only the raw train/test split is needed.
            report = {'Model Type': 'ARIMA',
                      'Sample Size': len(self.data),
                      'Train Size': len(self.train_data),
                      'Test Size': len(self.test_data)
                      }

            display(pd.DataFrame([report]).T)

        elif model_type == 'RNN':

            # For RNN, build sequential windows from train/test data.
            if not isinstance(step, int) or step <= 0:
                raise ValueError(
                    "step must be a positive integer when model_type='RNN'")

            self.rnn_step = step

            self.X_train, self.y_train = self.__make_windows(
                self.train_data, step)
            self.X_test, self.y_test = self.__make_windows(
                self.test_data, step)

            report = {'Model Type': 'RNN',
                      'Step': step,
                      'X_train shape': (self.X_train.shape),
                      'y_train shape': (self.y_train.shape),
                      'X_test shape':  (self.X_test.shape),
                      'y_test shape':  (self.y_test.shape)
                      }

            display(pd.DataFrame([report]).T)

        else:
            raise ValueError("model_type must be either 'ARIMA' or 'RNN'.")

        return self

# *************************************************************************************

    def build_fit_rnn_model(self,
                            units=64,
                            loss='mean_squared_error',
                            optimizer='rmsprop',
                            epochs=100,
                            batch_size=16,
                            validation_split=0.1):

        # Building and training a simple RNN model for forecasting.
        # Includes TensorBoard logging, early stopping, and learning-rate reduction.

        log_dir = os.path.join(
            "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   restore_best_weights=True
                                   )

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=2,
                                      min_lr=0.00001
                                      )
        my_callbacks = [early_stop, reduce_lr]

        # Creating a simple single-layer RNN followed by a dense output layer.
        self.rnn_fitted_model = Sequential()
        self.rnn_fitted_model.add(SimpleRNN(units=units,
                                            input_shape=(
                                                self.X_train.shape[1], self.X_train.shape[2]),
                                            activation="tanh"
                                            )
                                  )

        self.rnn_fitted_model.add(Dense(1))

        # Compiling the model with the selected loss and optimizer.
        self.rnn_fitted_model.compile(loss=loss, optimizer=optimizer)

        self.rnn_fitted_model.summary()

        # Training the model and keep the training history.
        history = self.rnn_fitted_model.fit(self.X_train,
                                            self.y_train,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_split=validation_split,
                                            verbose=2,
                                            callbacks=my_callbacks
                                            )

        return self

# *************************************************************************************

    def arima_model(self, order):

        # Fiting an ARIMA model with the specified (p, d, q) order.

        if any(i < 0 for i in order):
            raise ValueError("All parameters must be greater than or equal 0")

        arima = ARIMA(self.train_data.values, order=order)

        self.fitted_model = arima.fit()

        display(self.fitted_model.summary())

        return self.fitted_model

# *************************************************************************************

    def auto_arima(self, start_p=0, start_q=0, max_p=5, max_q=5):

        # Automatically search for a suitable ARIMA configuration
        # within the specified p and q ranges

        auto_model = pm.auto_arima(y=self.train_data,
                                   d=0,
                                   start_p=start_p,
                                   start_q=start_q,
                                   max_p=max_p,
                                   max_q=max_q,
                                   seasonal=False,
                                   trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True
                                   )

        display(auto_model.summary())

        return auto_model

# *************************************************************************************

    def grid_search_arima(self, max_p=5, max_q=5, d=0):

        # Perform a manual grid search over ARIMA hyperparameters
        # and rank the fitted models using AIC.

        if (max_p or max_q) <= 0:
            raise ValueError(" max_p and max_q must be greater than 0")

        p = range(0, max_p + 1)
        q = range(0, max_q + 1)

        combinations = list(itertools.product(p, [d], q))

        results = []

        for order in combinations:

            try:
                model = ARIMA(self.train_data, order=order).fit()

                results.append({
                    "order": order,
                    "AIC": model.aic,
                    "BIC": model.bic
                })

            except:
                continue

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values("AIC")

        display(results_df.head())

        best_order = results_df.iloc[0]["order"]

        print("Best order:", best_order)

        return best_order, results_df

# *************************************************************************************

    def forecast(self, model_type, steps=None):

        # Generating forecasts using the fitted ARIMA or RNN model.

        model_type = model_type.upper().strip()

        if model_type == 'ARIMA':

            if self.fitted_model is None:
                raise ValueError(
                    "Model is not fitted yet. Call arima_model() first.")
            if steps is None:
                steps = len(self.test_data)

            arima_preds = self.fitted_model.forecast(steps=steps)
            forecast = pd.Series(np.asarray(arima_preds),
                                 index=self.test_data.index[:steps],
                                 name=' forecast')

        elif model_type == 'RNN':
            if self.rnn_fitted_model is None:
                raise ValueError(
                    "RNN model is not fitted yet. Call build_fit_rnn_model() first.")
            if self.X_test is None:
                raise ValueError(
                    "Test windows are not created. Call train_test_split(model_type='RNN', step=...) first.")

            rnn_preds = self.rnn_fitted_model.predict(self.X_test, verbose=0)
            forecast = pd.Series(rnn_preds.ravel(),
                                 index=self.test_data.index[self.rnn_step:],
                                 name=' forecast')

        else:
            raise ValueError("model_type must be either 'ARIMA' or 'RNN'.")

        return forecast

# *************************************************************************************

    def inverse_transform(self, transformed_data):

        # Reverse preprocessing steps such as seasonal differencing
        # and Box-Cox transformation to bring forecasts back to original scale.

        period = self.prep.period
        restored = transformed_data.copy()

        if period > 0:
            anchor = self.prep.data_before_diff.shift(
                period).loc[transformed_data.index]
            restored = anchor.add(transformed_data, fill_value=0)

        if self.prep.boxcox_lambda is not None:
            restored = inv_boxcox(restored, self.prep.boxcox_lambda)

        return restored
