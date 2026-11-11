
import pandas as pd
from scipy.stats import ks_2samp


MONITORED_FEATURES = [
    "sales_lag_7",
    "sales_rolling_mean_7",
]


def compare_windows(
    data,
    reference_start,
    reference_end,
    current_start,
    current_end,
    features=MONITORED_FEATURES,
    threshold=0.20,
):
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    required = {"date", *features}

    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    dates = pd.to_datetime(data["date"])

    ref_start = pd.Timestamp(reference_start)
    ref_end = pd.Timestamp(reference_end)
    cur_start = pd.Timestamp(current_start)
    cur_end = pd.Timestamp(current_end)

    if not (ref_start <= ref_end < cur_start <= cur_end):
        raise ValueError("Invalid or overlapping periods")

    reference = data.loc[
        dates.between(ref_start, ref_end)
    ]

    current = data.loc[
        dates.between(cur_start, cur_end)
    ]

    if reference.empty or current.empty:
        raise ValueError("Monitoring periods contain no data")

    results = {}

    for feature in features:
        ref_values = pd.to_numeric(
            reference[feature], errors="raise"
        )

        cur_values = pd.to_numeric(
            current[feature], errors="raise"
        )

        ref_missing = float(ref_values.isna().mean())
        cur_missing = float(cur_values.isna().mean())

        ref_values = ref_values.dropna()
        cur_values = cur_values.dropna()

        if min(len(ref_values), len(cur_values)) < 30:
            raise ValueError(
                f"Insufficient observations for {feature}"
            )

        statistic = float(
            ks_2samp(
                ref_values,
                cur_values,
                method="asymp",
            ).statistic
        )

        results[feature] = {
            "ks_statistic": round(statistic, 4),
            "drift_detected": bool(statistic >= threshold),
            "reference_missing_rate": round(ref_missing, 4),
            "current_missing_rate": round(cur_missing, 4),
        }

    return {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "threshold": threshold,
        "features": results,
    }
