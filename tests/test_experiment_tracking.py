
import json
import pandas as pd

from src.experiment_tracking import log_experiment


def test_log_experiment(tmp_path):

    # Đường dẫn file thử nghiệm
    log_path = tmp_path / "experiments.jsonl"

    # Gọi hàm cần kiểm thử
    record = log_experiment(
        name="test_model",
        parameters={"max_iter": 100},
        features=["store_nbr", "day_of_week"],
        metrics={
            "mae": 50.0,
            "rmsle": 0.3,
            "coverage": 1.0
        },
        cutoffs=[pd.Timestamp("2017-07-04")],
        log_path=log_path
    )

    # Kiểm tra file đã được tạo
    assert log_path.exists()

    # Đọc lại file
    with log_path.open("r", encoding="utf-8") as file:
        saved = json.loads(file.readline())

    # Kiểm tra nội dung
    assert saved["name"] == "test_model"
    assert saved["parameters"]["max_iter"] == 100
    assert saved["metrics"]["mae"] == 50.0
    assert saved["validation_cutoffs"] == ["2017-07-04"]

    # Kiểm tra giá trị trả về
    assert record["name"] == saved["name"]
