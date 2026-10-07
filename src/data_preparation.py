
# 1. Core Scientific & Data Manipulation
import numpy as np
import pandas as pd
from scipy import stats

# 2. Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 3. Statistical
from statsmodels.tsa.stattools import adfuller

# 4. Interactive Display
from IPython.display import display

# 5. Project-Specific Modules
from src.data_explore import DataExplorer


class DataPreparation(DataExplorer):

    def __init__(self, data):

        # Initialize the parent class with a copy of the input data
        # to avoid modifying the original dataset directly.

        super().__init__(data.copy())

        # Store transformation-related parameters for later reference.

        self.boxcox_lambda = None
        self.data_before_diff = None
        self.scaler = None
        self.period = None

# *************************************************************************************

    @property
    def result(self):

        # Return a copy of the processed data.

        return self.data.copy()

# *************************************************************************************

    def clipping(self, threshold=1.5):
        # Cap extreme values using the IQR rule.

        q1 = self.data.quantile(0.25)
        q3 = self.data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        self.data = self.data.clip(lower_bound, upper_bound)

        return self

# *************************************************************************************

    def normalization(self, method='minmax'):

        # Scale the data using either Min-Max scaling or Standard scaling.

        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError("method must be either 'minmax' or 'standard'.")
        scaled_data = self.scaler.fit_transform(
            self.data.values.reshape(-1, 1))

        # Convert the scaled output back to a pandas Series
        # while preserving the original index.

        self.data = pd.Series(scaled_data.flatten(), index=self.data.index)
        return self

# *************************************************************************************

    def log_transform(self):
        # Apply logarithmic transformation to reduce skewness
        # and stabilize variance.

        self.data = np.log(self.data)

        return self

# *************************************************************************************

    def apply_boxcox(self):

        # Apply Box-Cox transformation and store the optimal lambda value.

        fitted_data, fitted_lambda = stats.boxcox(self.data)

        self.boxcox_lambda = fitted_lambda

        print('-' * 50)
        print(f"{' ':5} Optimal Lambda for Box-Cox: {fitted_lambda:.4f}")
        print('-' * 50)

        self.data = pd.Series(fitted_data, index=self.data.index)

        return self

# *************************************************************************************

    def apply_seasonal_diff(self, period):

        # Apply seasonal differencing to remove repeating seasonal patterns.

        print(f"Applying Seasonal Differencing (Periods={period})...")

        if period < 1:
            raise ValueError("period must be >= 1")

        self.period = period

        # Keep a copy of the original data before differencing.

        self.data_before_diff = self.data.copy()

        # Subtract each observation from its seasonal lag value.

        self.data = self.data - self.data.shift(periods=period)
        self.data.dropna(inplace=True)

        return self

# *************************************************************************************

    def ADF(self):

        # Run the Augmented Dickey-Fuller test to check stationarity.

        result = adfuller(self.data)
        result_dict = pd.Series(
            result[0:4], index=['ADF Statistic', 'p-value', 'n_lags', 'n_obs'])

        # Add critical values to the result table.

        for key, value in result[4].items():
            result_dict[f'Critical Value ({key})'] = value

        df_result = pd.DataFrame(result_dict, columns=['Value'])

        styled_df = df_result.style.set_caption(
            f'ADF Test Results - {self.caption}')
        display(styled_df)

        p_value = result[1]
        adf_stat = result[0]
        critical_5 = result[4]['5%']

        # Print an interpretable conclusion based on the test result.

        print("\nConclusion:")
        if p_value < 0.05 and adf_stat < critical_5:
            print(f"Since the P-value ({p_value:.4f}) is less than 0.05 and the ADF Statistic ({adf_stat:.4f}) "
                  f" \n is less than the Critical Value (5%: {critical_5:.4f}), the series is Stationary.")
        else:
            print(f"Since the P-value ({p_value:.4f}) is greater than 0.05 or the ADF Statistic ({adf_stat:.4f}) "
                  f" \n is greater than the Critical Value (5%: {critical_5:.4f}), the series is Not Stationary.")
        return self

# *************************************************************************************

    def adjust_stock_split(self, split_date, ratio=5):

        # Adjust historical values before the stock split date
        # so the series remains consistent over time.

        split_date = pd.to_datetime(split_date)

        self.data.loc[self.data.index < split_date] = (
            self.data.loc[self.data.index < split_date] / ratio)

        # Show a summary of the stock split adjustment.

        result_df = pd.DataFrame({
            'Event': ['Stock Split Adjustment'],
            'Split Ratio': [f'{ratio}-for-1'],
            'Split Date': [split_date.date()],
            'Adjustment': [f'VWAP before {split_date.date()} were divided by {ratio}']
        })

        display(result_df)

        return self
