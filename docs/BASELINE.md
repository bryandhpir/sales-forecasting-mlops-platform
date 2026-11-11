# Seasonal Naive Baseline & Backtesting

## Goal

Establish a reproducible benchmark for forecasting store–product-family unit sales. This is a non-trained baseline, not a deployed forecasting service.

## Technique

At each historical cutoff, predict each observed future target using the corresponding day within the most recent **7-calendar-day cycle** available at the cutoff. If the relevant reference row is missing, keep `prediction` as missing and exclude it from error scoring. Do not interpret missing source dates as zero sales.

The script scores multiple **non-overlapping** evaluation intervals. The forecast horizon is supplied via command line, not fixed in the project.

## Commands (Windows PowerShell, project root)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.backtesting --horizon 7 --origins 3
```

The `7` in this example is a user-chosen horizon for this run; you can replace it with another positive integer.

## Generated files

- `data/processed/baseline_metrics.json` — metrics for each cutoff and in aggregate
- `data/processed/baseline_backtest.parquet` — observed targets, predictions, cutoff and reference dates

The project's `.gitignore` already excludes `data/processed/*`.

## Metrics

- **MAE:** average absolute difference in sales units; lower is better.
- **RMSLE:** root-mean-square error on `log1p(sales)`; lower is better.
- **Coverage:** fraction of observed target rows for which a past reference was available. It does not measure coverage of completely absent source dates.

## Leakage prevention

- At each cutoff, reference sales must be observed **on or before** the cutoff.
- Real target `sales` inside the evaluation interval are only used for scoring, not prediction.
- A later cutoff may use observations from earlier evaluation intervals, because these would already be known in a realistic rolling-origin simulation.
- For fair model comparison, keep the same evaluation windows and eligible target rows across models.

## Limitations

- This initial benchmark does not use holidays, promotions or store metadata.
- Reference data missing on a calendar date lead to missing predictions, not silent zero-filling.
- There is no model training in this module, and there is no live future forecast endpoint yet.
