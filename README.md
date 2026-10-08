[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-orange.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-SimpleRNN%20%7C%20Keras-red.svg)](https://keras.io/api/layers/recurrent_layers/simple_rnn/)
[![Time Series](https://img.shields.io/badge/Model-ARIMA%20(AutoARIMA)-green.svg)](https://alkaline-ml.com/pmdarima/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ADANIPORTS Time Series Forecasting

A complete end-to-end **Time Series Analysis and Forecasting** project for **Adani Ports and Special Economic Zone Ltd. (`ADANIPORTS`)**.

The project focuses on analyzing and forecasting the **Volume Weighted Average Price (VWAP)** using both classical statistical time-series models and deep learning techniques.

The project is implemented in two different architectural approaches:

- **Modular Architecture** – functionality is separated into reusable Python modules.
- **Object-Oriented Programming (OOP)** – the complete workflow is implemented using dedicated classes.

## Project Overview

> **Author:** Seyed Reza Zibjou <br>
> **Asset:** Adani Ports and Special Economic Zone Ltd (`ADANIPORTS.csv`) <br>
> **Time Horizon:** November 27, 2007 – April 30, 2021 (3,322 trading days) <br>
> **Target Feature:** Volume Weighted Average Price (`VWAP`)

This project demonstrates a complete time-series forecasting workflow, starting from raw historical stock-market data and progressing through:

1. Data loading and understanding
2. Exploratory Data Analysis (EDA)
3. Stock split adjustment
4. Outlier detection and treatment
5. Data transformation
6. Stationarity analysis
7. Autocorrelation analysis
8. Time-series decomposition
9. Train/Test splitting
10. ARIMA modeling
11. Simple RNN modeling
12. Hyperparameter/model selection
13. Forecast generation
14. Inverse transformation
15. Model validation
16. Performance evaluation

The main objective is to build a reusable and structured forecasting pipeline rather than simply fitting a single model.

## Project Structure
```text

TIME-SERIES_ADANI_PORTS/
│
├── data/
│   └── ADANIPORTS.csv
│
├── notebooks/
│   ├── adaniports_based_on_modules.ipynb
│   └── adaniports_full_code_oop.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_explore.py
│   ├── data_preparation.py
│   ├── evaluate.py
│   └── modeling.py
│
├── .gitignore
├── main.py
├── README.md
├── README-fa.md
└── requirements.txt
```
## Architecture

The project follows a modular design in which different stages of the machine-learning/time-series pipeline are separated into independent components.

```text
                    ┌─────────────────────┐
                    │   ADANIPORTS.csv    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Exploration  │
                    │    DataExplorer     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Preparation   │
                    │  DataPreparation    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Modeling      │
                    │       Modeling      │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
           ┌─────────────┐          ┌─────────────┐
           │    ARIMA    │          │ Simple RNN  │
           └──────┬──────┘          └──────┬──────┘
                  │                        │
                  └────────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │     Evaluation      │
                    │  MAE / RMSE / Plots │
                    └─────────────────────┘
```
## 1. Data Understanding & Exploratory Data Analysis

The `DataExplorer` class is responsible for understanding the structure and statistical behavior of the time series.

The exploratory analysis includes:

- Time-series visualization
- Rolling mean
- Rolling standard deviation
- Box plot
- Histogram
- Outlier detection
- Time-series decomposition
- Autocorrelation Function (ACF)
- Partial Autocorrelation Function (PACF)
- Missing-value checking

Example workflow:

```python
explore_instance = (
    DataExplorer(Data)
    .set_caption('Original VWAP')
    .plot_time_series()
    .plot_rolling_mean_std()
    .plot_box_hist()
    .outlier_report()
    .plot_decomposition(model='additive')
    .ACF_and_PACF()
)
```
The purpose of this stage is to understand:

- Trend
- Variability
- Distribution
- Potential outliers
- Autocorrelation
- Seasonal/repeating behavior
- Stationarity characteristics

## 2. Data Preparation

The `DataPreparation` class handles the preprocessing stage.

Available preprocessing operations include:

- Stock split adjustment
- Outlier clipping
- Min-Max normalization
- Standardization
- Log transformation
- Box-Cox transformation
- Seasonal differencing
- Augmented Dickey-Fuller (ADF) test
- Time-series decomposition

The class extends DataExplorer, allowing exploratory and preprocessing operations to be combined in a pipeline.

### Stock Split Adjustment

The dataset contains a stock split event for Adani Ports.

Stock Split Information 

| Field | Value |
|---|---|
| Company | Adani Ports |
| Symbol | `ADANIPORTS` |
| Series | `EQ` |
| Previous Face Value | Rs. 10 |
| New Face Value | Rs. 2 |
| Split Ratio | 5:1 |
| Ex-Date | 23-Sep-2010 |
| Record Date | 24-Sep-2010 |

For time-series analysis, historical VWAP values before the split date are adjusted to maintain consistency across the complete time series.

The preprocessing pipeline uses:

```python
.adjust_stock_split(
    split_date='2010-09-23',
    ratio=5
)
```
## 3. Outlier Treatment

The project uses the Interquartile Range (IQR) method to detect and handle extreme values.

The default clipping threshold is:

```python
1.5 × IQR
```
The method calculates:
```python
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```
Extreme values are then clipped to the calculated boundaries.

Example:
```python
preparation_instance.clipping()
```
## 4. Data Transformation

Several transformation techniques are implemented.

### Min-Max Normalization
```python
normalization(method='minmax')
```
### Standardization
```python
normalization(method='standard')
```
### Log Transformation
```python
log_transform()
```
### Box-Cox Transformation
```python
apply_boxcox()
```
The Box-Cox transformation is used to help stabilize variance and improve the statistical properties of the series.

The optimal lambda parameter is estimated automatically.

## 5. Stationarity Analysis

Stationarity is an important requirement for classical time-series models such as ARIMA.

The project uses the:

Augmented Dickey-Fuller (ADF) Test
```python
ADF()
```
The method reports:

- ADF Statistic
- p-value
- Number of lags
- Number of observations
- Critical values

The implemented interpretation uses a significance level of:
```python
α = 0.05
```

A stationary series is generally indicated when:
```python
p-value < 0.05
```
and the ADF statistic is below the corresponding critical value.

## 6. Differencing

The project supports seasonal differencing through:
```python
apply_seasonal_diff(period=1)
```
Differencing is applied to reduce non-stationary behavior and remove temporal dependency patterns that may interfere with forecasting.

 ## 7. ACF & PACF Analysis

The project calculates and visualizes:

ACF(Autocorrelation Function)

and 

PACF(Partial Autocorrelation Function) 

These plots help analyze the temporal dependency structure of the series and assist in selecting appropriate parameters for ARIMA models.

## 8. Modeling

The project compares two different forecasting approaches:

- ARIMA
- Simple RNN

This provides a comparison between a classical statistical model and a neural-network-based model.

# 8.1 ARIMA

The project uses the ARIMA model:

```python
ARIMA(p, d, q)
```

where:

- `p` = autoregressive order 
- `d` = degree of differencing 
- `q` = moving-average order 

The modeling class provides:

-  Manual ARIMA fitting 
-  AutoARIMA 
-  Grid Search 
-  Forecasting 
-  Inverse transformation 
-  Validation plots 
-  MAE calculation 
-  RMSE calculation 

---

## AutoARIMA

The project automatically searches for suitable ARIMA parameters.

Example:

```python
auto_model = arima_instance.auto_arima(
    start_p=0,
    start_q=0,
    max_p=5,
    max_q=5
)
```

---

## ARIMA Grid Search

A manual grid search is also implemented:

```python
best_order, results_grid_search = (
    arima_instance.grid_search_arima(
        max_p=5,
        max_q=5
    )
)
```

The candidate models are compared using information criteria such as:

-  AIC 
-  BIC 

---

# 🏆 ARIMA Model Selection

Two important candidates obtained during model selection were:

```
ARIMA(0, 0, 1)
ARIMA(5, 0, 0)
```

Their comparison is summarized below:

| Criterion            | ARIMA(0,0,1)  | ARIMA(5,0,0) |
| -------------------- | ------------- | ------------ |
| Model Type           | MA(1)         | AR(5)        |
| Number of Parameters | 1             | 5            |
| AIC                  | -8320.414     | -8320.859    |
| BIC                  | **-8302.761** | -8279.667    |
| Ljung-Box p-value    | 0.99          | 1.00         |
| Residuals            | White Noise   | White Noise  |
| Complexity           | Very Low      | Moderate     |

Although the ARIMA(5,0,0) model has a slightly lower AIC, the ARIMA(0,0,1) model provides a substantially better BIC while using significantly fewer parameters.

The `ma.L1` parameter in ARIMA(0,0,1) is also statistically significant.

Therefore, based on **parsimony, BIC, parameter significance, and model complexity**, the project selects:

```
ARIMA(0,0,1)
```

as the preferred ARIMA configuration.

---

# 8.2 Simple RNN

The second forecasting approach is a **Simple Recurrent Neural Network (RNN)**.

The RNN receives sequential windows of historical observations and attempts to predict the next value.

The project creates sliding windows using:

```python
step=10
```

Conceptually:

```
[t1, t2, t3, ..., t10] → t11
[t2, t3, t4, ..., t11] → t12
[t3, t4, t5, ..., t12] → t13
...
```

---

## RNN Architecture

The implemented network consists of:

```
Input Sequence
      │
      ▼
SimpleRNN
      │
      ▼
Dense(1)
      │
      ▼
Forecast
```

Default training configuration:

```
units=64
loss='mean_squared_error'
optimizer='rmsprop'
epochs=100
batch_size=16
validation_split=0.1
```

The model also uses:

-  Early Stopping 
-  Learning Rate Reduction 

These callbacks help control overfitting and improve training behavior.

---

# 9. Train/Test Split

The default train/test split is:

```
80% Training
20% Testing
```

For example:

```python
train_test_split(
    model_type='ARIMA',
    train_size=0.8
)
```

For the RNN:

```python
train_test_split(
    model_type='RNN',
    train_size=0.8,
    step=10
)
```

The chronological order of the time series is preserved during splitting.

---

# 10. Inverse Transformation

Because the original series is transformed during preprocessing, the model predictions must be converted back to the original scale before final evaluation.

The project provides:

```python
inverse_transform()
```

This allows forecasts to be compared with the original VWAP values.

The workflow is therefore:

```
Raw VWAP
   │
   ▼
Stock Split Adjustment
   │
   ▼
Outlier Treatment
   │
   ▼
Box-Cox Transformation
   │
   ▼
Differencing
   │
   ▼
Model
   │
   ▼
Forecast
   │
   ▼
Inverse Transformation
   │
   ▼
Original Scale
```

---

# 11. Model Evaluation

The project evaluates forecasts using two main metrics.

## MAE

**Mean Absolute Error**

```
MAE = mean(|y_true - y_pred|)
```

Lower values indicate better performance.

---

## RMSE

**Root Mean Squared Error**

```
RMSE = sqrt(mean((y_true - y_pred)²))
```

RMSE gives greater weight to larger prediction errors.

Lower values indicate better performance.

---

# 12. Model Results

The results below are from the modular implementation notebook.

| Model            | MAE        | RMSE       |
| ---------------- | ---------- | ---------- |
| **ARIMA(0,0,1)** | **4.9400** | **7.2838** |
| Simple RNN       | 4.9976     | 7.3602     |

Based on these results:

-  ARIMA achieved a lower MAE. 
-  ARIMA achieved a lower RMSE. 
-  The difference between the two models is relatively small. 
-  For this particular experiment, **ARIMA(0,0,1)** slightly outperformed the Simple RNN. 

Therefore, the current experimental results favor the ARIMA model.

> These metrics represent the results obtained from the current notebook configuration and should not be interpreted as a universal statement that ARIMA is always superior to RNN for this dataset or financial forecasting in general.

---

# 13. Validation

The project provides validation plots containing:

-  Training data 
-  Actual test data 
-  Forecasted values 

For example:

```python
arima_instance.plot_validation(
    train=arima_train_original,
    test=arima_test_original,
    forecast=arima_preds_original,
    forecast_label='ARIMA Forecast'
)
```

and:

```python
rnn_instance.plot_validation(
    train=rnn_train_orginal,
    test=rnn_test_orginal,
    forecast=rnn_preds_original,
    forecast_label='RNN Forecast'
)
```

These plots make it possible to visually inspect how closely the forecasts follow the actual time series.

---

# 14. Main Modules

The `src` directory contains the main reusable components.

---

## `data_explore.py`

Responsible for exploratory analysis.

Main class:

```
DataExplorer
```

Responsibilities include:

-  Time-series plotting 
-  Rolling statistics 
-  Distribution analysis 
-  Outlier analysis 
-  Decomposition 
-  ACF/PACF 
-  Missing-value analysis 

---

## `data_preparation.py`

Responsible for preparing the time series before modeling.

Main class:

```
DataPreparation
```

Responsibilities include:

-  Stock split adjustment 
-  Outlier clipping 
-  Normalization 
-  Standardization 
-  Log transformation 
-  Box-Cox transformation 
-  Differencing 
-  ADF testing 
-  Data recovery/inverse transformations 

---

## `modeling.py`

Responsible for model training and forecasting.

Main class:

```
Modeling
```

Supported models:

```
ARIMA
RNN
```

Responsibilities include:

-  Train/test splitting 
-  RNN window generation 
-  ARIMA fitting 
-  AutoARIMA 
-  ARIMA grid search 
-  RNN training 
-  Forecast generation 
-  Inverse transformations 
-  Validation plots 
-  Evaluation 

---

## `evaluate.py`

Provides model evaluation functionality.

The project uses:

```
MAE
RMSE
```

to quantify forecasting performance.

---

# 15. Notebooks

Two notebooks are included.

## `adaniports_based_on_modules.ipynb`

This notebook demonstrates the project using the reusable modules located in:

```
src/
```

The workflow is therefore:

```
Notebook
   │
   ├── data_explore.py
   ├── data_preparation.py
   ├── modeling.py
   └── evaluate.py
```

This is the recommended notebook for demonstrating the modular architecture.

---

## `adaniports_full_code_oop.ipynb`

This notebook contains the complete implementation using an object-oriented approach directly inside the notebook.

It demonstrates the same general workflow while keeping the class implementations inside the notebook.

The main classes are:

```
DataExplorer
DataPreparation
Modeling
```

---

# 16. Installation

Clone the repository:

```
git clone https://github.com/SeyedRezaZibjou/Time-Series_Adaniports.git
```

Move into the project directory:

```
cd TIME-SERIES_ADANI_PORTS
```

Create a virtual environment:

### Windows

```
python -m venv .venv
```

Activate it:

```
.venv\Scripts\activate
```

### Linux / macOS

```
python3 -m venv .venv
```

Activate it:

```
source .venv/bin/activate
```

Install the required dependencies:

```
pip install -r requirements.txt
```

---

# 17. Running the Project

The project contains an end-to-end execution script:

```
main.py
```

Run it with:

```
python main.py
```

Alternatively, the analysis can be executed interactively through Jupyter Notebook.

Start Jupyter:

```
jupyter notebook
```

Then open:

```
notebooks/adaniports_based_on_modules.ipynb
```

---

# 18. Technology Stack

The project is implemented primarily in Python.

Main libraries include:

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| Python           | Core programming language        |
| NumPy            | Numerical computation            |
| Pandas           | Data manipulation                |
| SciPy            | Statistical transformations      |
| Matplotlib       | Visualization                    |
| Seaborn          | Statistical visualization        |
| Scikit-learn     | Scaling and evaluation metrics   |
| Statsmodels      | Statistical time-series analysis |
| pmdarima         | AutoARIMA                        |
| Keras            | Deep learning                    |
| TensorFlow       | RNN backend                      |
| Jupyter Notebook | Interactive experimentation      |

---

# 19. Key Concepts Demonstrated

This project demonstrates practical implementation of several important concepts in Data Science and Machine Learning:

### Time Series Analysis

-  Trend analysis 
-  Seasonality 
-  Autocorrelation 
-  Stationarity 
-  Differencing 
-  Time-series decomposition 

### Data Preprocessing

-  Outlier detection 
-  IQR clipping 
-  Scaling 
-  Log transformation 
-  Box-Cox transformation 
-  Corporate-action adjustment 

### Statistical Modeling

-  ARIMA 
-  AutoARIMA 
-  Grid Search 
-  AIC 
-  BIC 
-  Ljung-Box test 
-  ADF test 

### Deep Learning

-  Recurrent Neural Networks 
-  Sequential data 
-  Sliding windows 
-  Early stopping 
-  Learning-rate reduction 

### Model Evaluation

-  MAE 
-  RMSE 
-  Forecast visualization 
-  Train/Test validation 

---

# 20. End-to-End Workflow

The complete pipeline can be summarized as:

```
                    Raw Dataset
                         │
                         ▼
                Data Understanding
                         │
                         ▼
                       EDA
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    Distribution                  Time Series
      Analysis                     Analysis
          │                             │
          └──────────────┬──────────────┘
                         ▼
                 Stock Split Adjustment
                         │
                         ▼
                  Outlier Treatment
                         │
                         ▼
                 Box-Cox Transformation
                         │
                         ▼
                  Differencing / ADF
                         │
                         ▼
                  ACF / PACF Analysis
                         │
                         ▼
                  Train / Test Split
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
          ARIMA                    Simple RNN
             │                       │
             ▼                       ▼
          Forecast                 Forecast
             │                       │
             └───────────┬───────────┘
                         ▼
                 Inverse Transform
                         │
                         ▼
                     Evaluation
                         │
                         ▼
                    MAE / RMSE
                         │
                         ▼
                  Model Comparison
```

# 21. Project Goals

The primary goals of this project are:

-  Build a complete time-series forecasting pipeline. 
-  Separate data exploration, preparation, modeling, and evaluation. 
-  Demonstrate reusable Python classes. 
-  Compare classical statistical forecasting with deep learning. 
-  Handle real-world data issues such as stock splits and outliers. 
-  Apply appropriate statistical transformations before modeling. 
-  Evaluate different forecasting approaches using consistent metrics. 
-  Create a project structure suitable for further development and experimentation. 


# 22. Possible Future Improvements

Several improvements can be considered for future versions of the project:

-  Add additional forecasting models such as: 
  -  SARIMA 
  -  SARIMAX 
  -  Prophet 
  -  LSTM 
  -  GRU 
  -  XGBoost-based forecasting 
-  Implement systematic hyperparameter optimization for the RNN. 
-  Add walk-forward validation. 
-  Introduce rolling/expanding-window backtesting. 
-  Compare additional error metrics such as: 
  -  MAPE 
  -  SMAPE 
  -  MASE 
-  Add prediction intervals and uncertainty estimation. 
-  Add automated experiment tracking. 
-  Improve the model comparison framework. 
-  Add visualization dashboards. 
-  Add unit tests for individual modules. 
-  Add CI/CD and automated testing. 

---

# Disclaimer

This project is intended for **educational, research, and software-engineering purposes**.

The forecasts generated by the models should not be considered financial advice or a recommendation to buy, sell, or hold any financial asset.

Historical market behavior does not guarantee future performance.

---

# Author

**Seyed Reza Zibjou**

Data Science / Machine Learning / Time Series Forecasting

---

# License
```
MIT License
```

# Acknowledgement

This project was developed as a practical implementation of an end-to-end time-series forecasting workflow, combining:

```
Data Analysis
      +
Data Preparation
      +
Statistical Modeling
      +
Deep Learning
      +
Model Evaluation
      +
Software Architecture
```

The project emphasizes not only forecasting accuracy, but also **clean architecture, reusable components, reproducibility, and practical data-science workflow design**.

