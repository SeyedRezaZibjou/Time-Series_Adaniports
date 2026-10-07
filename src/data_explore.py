
# Essential Libraries

# 1. Visualization & Interactive Display
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# 2. Statistical & Time Series Modeling
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 3. Core Scientific & Data Manipulation
import pandas as pd


class DataExplorer:

    def __init__(self, data):

        # Save the original data and define a caption to use in the chart title.

        self.data = data
        self.caption = None
# *************************************************************************************

    def set_caption(self, caption):

        # Specify data title/label to display in reports and charts

        self.caption = caption
        return self

# *************************************************************************************

    def plot_time_series(self, ax=None):

        # If no axis is given, a new shape is created

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
            show = True
        else:
            show = False

        # Plotting time series data

        ax.plot(self.data, label='VWAP ', alpha=0.5)
        ax.set_title(f'Time Series Plot: {self.caption}')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{self.caption}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Only display the plot when it is created inside this method

        if show:
            plt.tight_layout()
            plt.show()

        return self

# *************************************************************************************

    def plot_box_hist(self):

        # Create two charts to examine the data distribution:
        # One is a boxplot and the other is a histogram

        fig, axes = plt.subplots(2, 1, figsize=(14, 5))

        axes[0].boxplot(self.data, vert=False)
        axes[0].set_title(f'Boxplot of {self.caption}')
        axes[0].set_xlabel(f'{self.caption}')
        axes[0].grid(True, alpha=0.3)

        sns.histplot(self.data,
                     bins=30,
                     kde=True,
                     ax=axes[1],
                     alpha=0.7
                     )
        axes[1].set_title(f'Histogram of {self.caption}')
        axes[1].set_xlabel(f'{self.caption}')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return self

# *************************************************************************************

    def outlier_report(self, threshold=1.5):

        # Calculating the first and third quartiles

        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)

        # Calculate the interquartile range

        IQR = Q3 - Q1

        # Determining the allowable interval based on the IQR method

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Extracting outliers

        outliers = self.data[
            (self.data < lower_bound) |
            (self.data > upper_bound)
        ]

        outlier_count = len(outliers)

        total_count = len(self.data)

        outlier_percentage = (outlier_count / total_count) * 100

        # Create an outlier report table

        report = pd.DataFrame({
            'Metric': [
                'Threshold',
                'Q1',
                'Q3',
                'IQR',
                'Lower Bound',
                'Upper Bound',
                'Total Samples',
                'Outlier Count',
                'Outlier Percentage (%)'
            ],
            'Value': [
                threshold,
                Q1,
                Q3,
                IQR,
                lower_bound,
                upper_bound,
                total_count,
                outlier_count,
                round(outlier_percentage, 2)
            ]
        })

        # Display the report in a styled way

        styled_report = report.style.set_caption(
            f'Outliers Report - {self.caption}')
        display(styled_report)

        # Save outlier data for later use

        self.outliers = outliers

        return self

# *************************************************************************************

    def plot_rolling_mean_std(self, window=30):

        # Calculate the moving average and standard deviation

        rolmean = self.data.rolling(window=window).mean()
        rolstd = self.data.rolling(window=window).std()

        # Plot the original data with rolling mean and rolling std

        orig = plt.plot(self.data,  label=f'{self.caption}')
        mean = plt.plot(rolmean,  label='Rolling Mean')
        std = plt.plot(rolstd,  label='Rolling Std')
        plt.xlabel('Date')
        plt.ylabel(f'{self.caption}')
        plt.legend()
        plt.title(f'Rolling Mean & Standard Deviation - {self.caption}')
        plt.tight_layout()
        plt.show()

        return self

# *************************************************************************************

    def plot_decomposition(self, model='additive', period=252):

        # Decomposition of time series into trend / seasonal / residual components

        decomposition = seasonal_decompose(
            self.data, model=model, period=period)

        fig = decomposition.plot()
        fig.set_size_inches(14, 10)
        plt.suptitle(
            f'Decomposition of {self.caption} ({model.capitalize()} model)', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        return self

# *************************************************************************************

    def check_missing_values(self):

        # Count the number of missing values

        missing_count = self.data.isna().sum()

        print(f"Missing values in {self.caption} data: {missing_count}")

        return self

# *************************************************************************************

    def ACF_and_PACF(self):

        # Draw ACF and PACF charts to examine time dependence

        fig, axes = plt.subplots(2, 1, figsize=(17, 10))

        plot_acf(self.data, lags=30, ax=axes[0])
        axes[0].set_title(f'Autocorrelation Function (ACF) - {self.caption}')

        plot_pacf(self.data, lags=30, ax=axes[1])
        axes[1].set_title(
            f'Partial Autocorrelation Function (PACF) - {self.caption}')

        plt.tight_layout()

        plt.show()

        return self
