# Sales Forecasting & MLOps Platform

A reproducible machine-learning engineering project for forecasting retail unit sales by store and product family. **Status: data preparation and exploratory analysis complete; models and deployment are in development.** No forecasting accuracy is claimed yet.

## Objective

Create a configurable forecasting workflow for the [Store Sales - Time Series Forecasting dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), which describes Corporación Favorita retail sales in Ecuador. Forecast length will be supplied as a configuration option in the modeling pipeline rather than fixed in data preparation.

**Target:** `sales` (unit sales, not revenue).  
**Series identity:** `store_nbr` + `family`.  
**Available sources:** historical sales, promotions, store metadata; other competition files can be incorporated as the project develops.

## Architecture

```text
Kaggle CSVs
    |
    v
Data validation -> Store metadata join -> Calendar and history features
    |                                    |
    v                                    v
Data quality report                   Feature table (Parquet)
    |                                    |
    v                                    v
EDA & anomaly review              Modeling & backtesting [planned]
                                         |
                                         v
                                 MLflow tracking [planned]
                                         |
                                         v
                                 FastAPI service [planned]
                                         |
                                         v
                              Dashboard & deployment [planned]
```

## Environment

- Python 3.13
- Pandas, NumPy, PyArrow, Matplotlib, Pytest
- Planned: scikit-learn, MLflow, FastAPI, Docker and monitoring tools

## Setup (Windows / PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the competition dataset, then place `train.csv` and `stores.csv` in `data/raw/`. Do not commit the original Kaggle CSVs to Git.

```powershell
# Prepare features and a quality report
.\.venv\Scripts\python.exe -m src.data_pipeline

# Generate EDA figures
.\.venv\Scripts\python.exe -m src.eda

# Run tests
.\.venv\Scripts\python.exe -m pytest -q
```

Outputs:

- `data/processed/features.parquet` — historical, engineered features (not tracked in Git)
- `data/processed/data_quality_report.json` — validation summary
- `reports/figures/` — charts for daily sales, top product families and weekday patterns
- `notebooks/01_eda.ipynb` — interactive exploration and anomaly checks

## Data quality and findings

The dataset inspected locally contains **3,000,888 records**, **54 stores**, **33 product families**, and dates from **2013-01-01** through **2017-08-15**. There were no null target sales and no duplicated combinations of date, store and family.

The initial exploration found:

- Overall unit sales generally grew between 2013 and 2017, with substantial periodic variation.
- `GROCERY I` has the highest cumulative unit sales among the product families examined.
- Average unit sales per record are higher on Saturday and Sunday than on most weekdays.
- The source has no records on December 25 in 2013, 2014, 2015 and 2016; missing dates **must not be silently treated as zero sales**.
- On 2013-01-01, **1,765 out of 1,782** store-family records had zero sales. Holiday-related activity is a hypothesis, not a confirmed explanation.

## Feature engineering and leakage precautions

- Features include calendar fields, historical lags and a trailing mean, computed using exact calendar dates for each store-family series.
- Gaps in the source timeline remain missing rather than being shifted into incorrect dates.
- **Important:** Historical features may use observed values that would not be available at prediction time. Multi-step model evaluation will therefore require a forecast-origin-safe backtesting process.
- Lag lookback lengths are **historical features**, not a hard-coded forecast length.

## Tests

Existing unit tests cover calendar-aligned lags, store isolation, missing-date behavior, and duplicate-key validation. The initial local run completed **3 passing tests**. A passing test suite is not a substitute for leakage-safe backtesting or end-to-end production validation.

## Repository hygiene

The `.gitignore` keeps virtual environments, raw competition data and processed datasets out of Git. Review `git status` before committing. EDA notebook, summary figures, code and documentation can be tracked.

## Data attribution

Data: Kaggle *Store Sales - Time Series Forecasting* competition, Corporación Favorita. Observe the competition's terms and licensing rules when using or sharing files.
