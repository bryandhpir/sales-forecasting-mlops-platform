# Gradient Boosting: Training and Backtesting

## Goal

Benchmark a supervised forecasting model against the existing seasonal naive baseline. The forecast horizon is supplied via CLI and is not hard-coded in the training pipeline.

## Inputs

Run `src.data_pipeline` before training. Input table: `data/processed/features.parquet`.

**Important:** The full feature table contains historical lag columns for every observed date. Passing future target-row lag columns directly to a model would leak observed values from the evaluation interval. This trainer deliberately ignores target-row lag columns and instead builds each input from the calendar and a snapshot **at the forecast cutoff**.

Features include:

- Store and product family identifiers
- Calendar features for the target date
- Horizon step relative to the forecast cutoff
- Same-weekday historical reference (restricted to dates available by cutoff)
- Last observed sales at cutoff, and historical lag/trailing average at cutoff

The model uses `HistGradientBoostingRegressor` with a log-transformed target `log1p(sales)`. Outputs are converted back to unit sales and clipped to be nonnegative.

## Setup

From repository root on Windows (Python 3.13):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Backtesting

Use the SAME horizon and evaluation origins as your existing baseline to compare fairly:

```powershell
.\.venv\Scripts\python.exe -m src.train_model --horizon 7 --origins 3 --train-origins 12
```

`--horizon`: configurable number of future calendar dates per origin.  
`--origins`: number of non-overlapping evaluation intervals.  
`--train-origins`: number of earlier non-overlapping training intervals.  
`--max-iter`: training iteration limit (default: 100).

The train windows end before the earliest evaluation target. Each origin constructs its features from information available at that origin. The model is fitted once on earlier historical windows; later evaluation intervals remain unseen labels. Do not train on the evaluation targets.

## Outputs

`data/processed/gradient_boosting_backtest.parquet` — predictions and observed targets for evaluation.  
`data/processed/gradient_boosting_metrics.json` — per-origin and overall MAE/RMSLE, coverage, and baseline comparison when the existing baseline report matches.

Both output files are ignored by `.gitignore` and should not be committed. Document verified results in README after local execution.

## Reading the comparison

- Negative `mae_delta_model_minus_baseline`: ML has lower MAE.
- Negative `rmsle_delta_model_minus_baseline`: ML has lower RMSLE.
- Coverage reports scored observations out of observed target rows. It does not measure probabilistic interval coverage or ensure the missing calendar dates were handled.

The model is an experimental baseline challenger. It is NOT a production deployment, and it does not yet incorporate known-ahead promotional plans, holiday events, or proper multi-fold tuning. Future inference needs an origin snapshot and future date grid.
