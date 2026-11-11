
import json
from datetime import datetime, timezone
from pathlib import Path


def log_experiment(
    name,
    parameters,
    features,
    metrics,
    cutoffs,
    log_path
):
    # Tạo bản ghi thử nghiệm
    record = {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "HistGradientBoostingRegressor",
        "parameters": parameters,
        "features": list(features),
        "validation_cutoffs": [
            str(c.date()) for c in cutoffs
        ],
        "metrics": {
            "mae": float(metrics["mae"]),
            "rmsle": float(metrics["rmsle"]),
            "coverage": float(metrics["coverage"])
        }
    }

    # Tạo thư mục lưu trữ
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ghi thêm dữ liệu vào JSONL
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

    return record
