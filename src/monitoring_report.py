
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.monitoring import compare_windows, MONITORED_FEATURES


def main():
    parser = argparse.ArgumentParser(
        description="Generate historical feature drift report"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.parquet"),
    )
    parser.add_argument("--reference-start", required=True)
    parser.add_argument("--reference-end", required=True)
    parser.add_argument("--current-start", required=True)
    parser.add_argument("--current-end", required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/monitoring/feature_drift.json"),
    )

    args = parser.parse_args()

    data = pd.read_parquet(
        args.input,
        columns=["date", *MONITORED_FEATURES],
    )

    report = compare_windows(
        data=data,
        reference_start=args.reference_start,
        reference_end=args.reference_end,
        current_start=args.current_start,
        current_end=args.current_end,
        threshold=args.threshold,
    )

    report["generated_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    report["reference_period"] = {
        "start": args.reference_start,
        "end": args.reference_end,
    }
    report["current_period"] = {
        "start": args.current_start,
        "end": args.current_end,
    }

    report["drift_detected"] = any(
        item["drift_detected"]
        for item in report["features"].values()
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Monitoring report saved:", args.output)
    print("Drift detected:", report["drift_detected"])


if __name__ == "__main__":
    main()
