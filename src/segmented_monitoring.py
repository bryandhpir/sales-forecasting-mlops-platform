
import argparse
from pathlib import Path

import pandas as pd

from src.monitoring import (
    MONITORED_FEATURES,
    compare_windows,
)


def compare_segments(
    data,
    reference_start,
    reference_end,
    current_start,
    current_end,
    threshold=0.20,
):
    required = {
        "date", "store_nbr", "family",
        *MONITORED_FEATURES,
    }

    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    start_ref = pd.Timestamp(reference_start)
    end_ref = pd.Timestamp(reference_end)
    start_cur = pd.Timestamp(current_start)
    end_cur = pd.Timestamp(current_end)

    if not start_ref <= end_ref < start_cur <= end_cur:
        raise ValueError("Invalid monitoring periods")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    frame = frame.loc[
        frame["date"].between(start_ref, end_ref)
        | frame["date"].between(start_cur, end_cur)
    ]

    records = []
    skipped_segments = 0

    # Xử lý riêng từng cặp cửa hàng - sản phẩm
    groups = frame.groupby(
        ["store_nbr", "family"],
        observed=True,
    )

    for (store, family), segment in groups:
        ref = segment.loc[
            segment["date"].between(start_ref, end_ref)
        ]

        cur = segment.loc[
            segment["date"].between(start_cur, end_cur)
        ]

        # compare_windows yêu cầu ít nhất 30
        # giá trị hợp lệ cho mỗi feature ở mỗi giai đoạn
        sufficient = all(
            ref[feature].count() >= 30
            and cur[feature].count() >= 30
            for feature in MONITORED_FEATURES
        )

        if not sufficient:
            skipped_segments += 1
            continue

        report = compare_windows(
            data=segment,
            reference_start=reference_start,
            reference_end=reference_end,
            current_start=current_start,
            current_end=current_end,
            threshold=threshold,
        )


        for feature, metrics in report["features"].items():
            ref_median = float(ref[feature].median())
            cur_median = float(cur[feature].median())

            median_delta = cur_median - ref_median

            # Không chia cho 0 khi median tham chiếu bằng 0
            median_change_pct = (
                100 * median_delta / abs(ref_median)
                if abs(ref_median) > 1e-9
                else None
            )

            records.append({
                "store_nbr": int(store),
                "family": str(family),
                "feature": feature,
                **metrics,
                "reference_median": round(ref_median, 2),
                "current_median": round(cur_median, 2),
                "median_delta": round(median_delta, 2),
                "median_change_pct": (
                    round(median_change_pct, 2)
                    if median_change_pct is not None
                    else None
                ),
            })


    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError("No segments have sufficient data")

    result = result.sort_values(
        ["drift_detected", "ks_statistic"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result, skipped_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-start", required=True)
    parser.add_argument("--reference-end", required=True)
    parser.add_argument("--current-start", required=True)
    parser.add_argument("--current-end", required=True)
    parser.add_argument("--threshold", type=float, default=0.20)

    args = parser.parse_args()

    data = pd.read_parquet(
        "data/processed/features.parquet",
        columns=[
            "date", "store_nbr", "family",
            *MONITORED_FEATURES,
        ],
    )

    result, skipped = compare_segments(
        data,
        reference_start=args.reference_start,
        reference_end=args.reference_end,
        current_start=args.current_start,
        current_end=args.current_end,
        threshold=args.threshold,
    )

    output = Path(
        "reports/monitoring/segmented_drift.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    flagged = result.loc[
        result["drift_detected"],
        ["store_nbr", "family"],
    ].drop_duplicates()

    print("Segmented report saved:", output)
    print("Feature comparisons:", len(result))
    print("Flagged store-family pairs:", len(flagged))
    print("Skipped segments:", skipped)
    print("\nTop results:")
    
    display_columns = [
        "store_nbr",
        "family",
        "feature",
        "ks_statistic",
        "reference_median",
        "current_median",
        "median_change_pct",
    ]

    print(
        result[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nMost frequently flagged families:")

    flagged_by_family = result.loc[
        result["drift_detected"],
        ["store_nbr", "family"],
    ].drop_duplicates()

    print(
        flagged_by_family["family"]
        .value_counts()
        .head(10)
        .to_string()
    )



if __name__ == "__main__":
    main()
