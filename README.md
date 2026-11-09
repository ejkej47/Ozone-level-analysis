# Ground-Level Ozone Analysis

**Exploring weather patterns, detecting high-ozone days, and forecasting monthly temperature with Python.**

A data science course project analyzing meteorological observations from Houston–Galveston–Brazoria, Texas (1998–2004). The analysis combines exploratory visualization, statistical testing, dimensionality reduction, classification, and time-series forecasting in a single Jupyter notebook.

The central challenge is class imbalance: only **73 of 2,536 observations (2.9%)** are labeled as ozone days. The project examines how different models trade off detecting these rare events against generating false alarms.

**[Explore the notebook](Ozone_level_analysis.ipynb)** · [Dataset documentation](ozone/onehr.names) · [Run locally](#run-locally)

## Project at a glance

| Area | Implementation |
| --- | --- |
| Data preparation | Numeric conversion, missing-value interpolation, feature scaling |
| Exploratory analysis | Distributions, weather relationships, correlation heatmaps, class comparisons |
| Statistical analysis | Shapiro–Wilk, Mann–Whitney U, Cohen's d with approximate confidence intervals |
| Dimensionality reduction | PCA on wind and temperature-related variables |
| Classification | Logistic regression, linear SVM, decision tree, perceptron, Keras neural network |
| Clustering | K-means comparison with observed ozone labels |
| Time series | Seasonal decomposition, ADF, ACF/PACF, SARIMA temperature forecasting |

**Tools:** Python · pandas · NumPy · Matplotlib · Seaborn · SciPy · scikit-learn · TensorFlow/Keras · statsmodels · pmdarima

## Dataset

The repository includes the **Ozone Level Detection** dataset and its original attribute documentation. The notebook uses the **one-hour peak** variant; the eight-hour variant is included for reference.

- **Observations:** 2,536 daily records with 72 numeric predictors, a date, and a binary target.
- **Features:** hourly and aggregate wind speed and temperature, relative humidity, pressure, precipitation, and other atmospheric measurements.
- **Target:** `Class = 1` for an ozone day and `Class = 0` for a normal day. The target is a label, rather than a continuous ozone concentration.
- **Class distribution:** 2,463 normal days and 73 ozone days.

See [onehr.names](ozone/onehr.names) for the original source credits, attribute definitions, and related publications.

## Key findings

- **Temperature and wind distinguish the two classes.** The notebook's Mann–Whitney tests find differences in temperature and wind PCA scores between normal and ozone days. The reported effect sizes are approximately `0.81` and `−0.93`, respectively. These are observational associations.
- **Class weighting exposes a recall–precision tradeoff.** Linear SVM detects 18 of the 22 ozone days in the test set, but also produces 207 false positives. Logistic regression achieves a similar F1 score.
- **A deeper model does not automatically improve results.** The two-hidden-layer neural network does not outperform the linear models on positive-class F1 in the saved run.
- **Monthly temperature supports seasonal forecasting.** A SARIMA model trained on 1998–2003 achieves an RMSE of approximately **1.79 °C** on 2004 monthly mean temperatures.

## Classification results

The classifiers use four features: average temperature (`T_AV`), average wind speed (`WSR_AV`), relative humidity at 850 hPa (`RH85`), and sea-level pressure (`SLP`). Evaluation uses a **stratified 70/30 train–test split** with `random_state=2019`: 1,775 training observations and 761 test observations, including 22 ozone days. Class balancing is applied through class weights.

The values below come from the notebook's saved outputs. Precision, recall, and F1 refer to the **ozone-day class**.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7385 | 0.0806 | 0.7727 | 0.1459 |
| Linear SVM | 0.7227 | 0.0800 | 0.8182 | 0.1457 |
| Neural network (2 × 16 ReLU units) | 0.7438 | 0.0739 | 0.6818 | 0.1333 |
| Decision tree (max depth 5) | 0.7740 | 0.0739 | 0.5909 | 0.1313 |
| Perceptron | 0.7254 | 0.0691 | 0.6818 | 0.1255 |

For context, predicting every test observation as a normal day would yield **97.1% accuracy but zero ozone-day recall**. This illustrates why the analysis considers class-specific metrics and confusion matrices alongside accuracy.

## Temperature forecasting

The time-series section aggregates daily average temperature into monthly means, examines seasonality and stationarity, and uses `auto_arima` to select a seasonal model.

| Setting | Value |
| --- | --- |
| Forecast target | Monthly mean temperature |
| Training period | 1998–2003 |
| Test period | 2004 |
| Selected model | SARIMA(0, 1, 1)(0, 1, 1, 12) |
| Test RMSE | 1.7932 °C |

This is a separate temperature forecasting experiment; ozone-day detection is handled by the classifiers above.

## Run locally

Clone or download this repository, then open a terminal in its root directory.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies and open the notebook:

```bash
python -m pip install -r requirements.txt
python -m notebook Ozone_level_analysis.ipynb
```

Run the cells from top to bottom using the environment's Python kernel. The dataset is included, and the notebook reads it using the relative path `ozone/onehr.data`. Notebook commentary and some plot labels are in Serbian; saved figures and model outputs can also be viewed directly on GitHub.

## Repository structure

```text
.
├── Ozone_level_analysis.ipynb   # Analysis, visualizations, models, and saved results
├── requirements.txt           # Python dependencies
├── README.md
└── ozone/
    ├── onehr.data              # One-hour peak dataset used in the notebook
    ├── onehr.names             # Source credits and attribute documentation
    ├── eighthr.data            # Eight-hour peak dataset
    └── eighthr.names           # Eight-hour dataset documentation
```

## Limitations and next steps

These results are exploratory. The classification test set contains only 22 positive examples, and missing values are interpolated before splitting the data, which can introduce information leakage. The random classification split also does not measure performance on strictly future observations.

Useful extensions include preprocessing within each training fold, chronological validation, precision–recall curves, threshold tuning, and evaluation of additional meteorological features. A seasonal-naive baseline would provide further context for the temperature forecast. Dependency versions are currently unpinned, and neural-network training is not seeded, so rerun results may vary.
