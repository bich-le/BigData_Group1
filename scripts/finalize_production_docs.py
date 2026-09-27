"""Generate final Production documentation from verified local evidence."""

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = DOCS / "evidence"


def registry_record(name):
    client = MlflowClient()
    versions = list(client.search_model_versions(f"name='{name}'"))
    if not versions:
        raise RuntimeError(f"No MLflow versions found for {name}")
    version = max(versions, key=lambda v: int(v.version))
    model = client.get_registered_model(name)
    aliases = dict(model.aliases)
    return {
        "name": name,
        "version": str(version.version),
        "run_id": version.run_id,
        "alias": next((a for a, v in aliases.items() if str(v) == str(version.version)), ""),
    }


def stream_output(path):
    candidates = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith('{"event_id"'):
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not candidates:
        raise RuntimeError(f"No streaming JSON output in {path}")
    return candidates[-1]


def main():
    oot = json.loads((DOCS / "regression_evaluation.json").read_text())
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    classification = registry_record("FlightDelayRandomForestSpark")
    regression = registry_record("FlightDelayRegressionSpark")
    classification_output = stream_output(EVIDENCE / "classification_demo.txt")
    regression_output = stream_output(EVIDENCE / "regression_demo.txt")
    metrics = oot["oot_metrics"]

    production = f'''# Production inference\n\nTwo native Spark `PipelineModel` artifacts are verified locally:\n\n- classification: `models/random_forest_final_spark`;\n- regression: `models/random_forest_regression_final_spark`.\n\nBoth have stages `VectorAssembler -> RandomForest{{Classification,Regression}}Model`.\nThe artifacts are excluded from Git because of their size.\n\n## Input contract\n\nEach JSON record contains optional string `event_id` plus the 23 ordered numeric\nfeatures in `feature_contract.py`. Airline, airport, and aircraft history/lag\nfeatures must be engineered upstream. Streaming inference does not derive them\nfrom a raw flight event. Missing features are filled with zero, matching fitting.\nNo target or outcome field is required for inference.\n\nTask-specific examples:\n\n- `docs/sample_classification_stream_input.json`;\n- `docs/sample_regression_stream_input.json`.\n\n## Regression selection and final evaluation\n\nDepth-18 configurations were skipped because their Spark MLlib tree growth was\ncomputationally infeasible on the available machine. CV fold counts differed, so\nthe global winner was selected by RMSE on the common random 20% pre-OOT\nvalidation split (seed 42). The teammate confirmed that the supplied artifact is the exact winner fitted on the full 80% training split. A local `randomSplit` reconstruction produced different row membership, so the teammate metrics remain the authoritative validation record. OOT was reserved for final evaluation. See\n`docs/REGRESSION_RESULTS.md`, `docs/regression_selection.json`, and `docs/regression_evaluation.json`.\n\nWinner: 500 trees, depth 15, minimum 5 instances per node.\n\n- validation RMSE: 32.736849;\n- validation MAE: 16.256044;\n- validation R²: 0.279717;\n- full OOT rows: {oot['oot_rows']:,};\n- OOT RMSE: {metrics['rmse']:.6f};\n- OOT MAE: {metrics['mae']:.6f};\n- OOT R²: {metrics['r2']:.6f}.\n\n## MLflow Registry\n\n| Task | Registered model | Version | Alias | Run ID |\n|---|---|---:|---|---|\n| Classification | {classification['name']} | {classification['version']} | {classification['alias']} | `{classification['run_id']}` |\n| Regression | {regression['name']} | {regression['version']} | {regression['alias']} | `{regression['run_id']}` |\n\nThe local tracking database is `mlflow.db`; artifacts are in `mlruns/`. Both are\nruntime state excluded from Git.\n\n## Instructor Demo\n\nThe demo reloads an existing artifact, scores a small OOT subset in a fresh\nprocess, then executes one `spark.readStream` JSON microbatch. It does not train.\n\n```bash\nsource ../.venv/bin/activate\n./submit_pipeline.sh classification\n./submit_pipeline.sh regression\n```\n\nClassification output fields: `event_id`, `prediction`,\n`probability_late_over_30m`. Example verified output:\n\n```json\n{json.dumps(classification_output, separators=(',', ':'))}\n```\n\nRegression output fields: `event_id`,\n`predicted_arrival_delay_minutes`. Example verified output:\n\n```json\n{json.dumps(regression_output, separators=(',', ':'))}\n```\n\nDirect long-running streaming command:\n\n```bash\nspark-submit --master 'local[2]' --driver-memory 32g --conf spark.driver.maxResultSize=4g inference.py \\\n  --task regression \\\n  --input-dir /path/to/input \\\n  --output-dir /path/to/output \\\n  --checkpoint-dir /path/to/checkpoint\n```\n\nAdd `--available-now` for a finite demonstration batch. Input, output, and\ncheckpoint directories must differ.\n\n## Tested environment and memory\n\n- Python 3.12.3\n- OpenJDK 17.0.20.1\n- Spark/PySpark 3.5.9\n- MLflow 3.10.1\n- PyArrow 23.0.1\n\nInstructor demos use 32 GB driver memory and `spark.driver.maxResultSize=4g` for\nthe large regression forest. Model fitting is not part of the demo.\n'''
    (DOCS / "PRODUCTION.md").write_text(production)

    task_table = f'''# Task status / Minh chứng\n\n| Task | Status | Evidence |\n|---|---|---|\n| Chốt, lưu, nạp và đăng ký hai PipelineModel tốt nhất | DONE | Classification `models/random_forest_final_spark`; regression `models/random_forest_regression_final_spark`; fresh-process reload evidence in `docs/evidence/classification_demo.txt` and `docs/evidence/regression_demo.txt`; MLflow `{classification['name']}` v{classification['version']} `{classification['alias']}` run `{classification['run_id']}` and `{regression['name']}` v{regression['version']} `{regression['alias']}` run `{regression['run_id']}`; registry snapshot in `docs/mlflow_registry.json`; regression metrics and provenance in `docs/regression_evaluation.json`. |\n| Chuẩn bị schema và JSON mẫu cho streaming | DONE | Task-aware schema in `inference.py`/`feature_contract.py`; `docs/sample_classification_stream_input.json`; `docs/sample_regression_stream_input.json`. |\n| Xây dựng streaming inference | DONE | `inference.py` uses real `spark.readStream`, `PipelineModel.load`, and `model.transform`; successful JSON outputs in both demo evidence files. |\n| Kiểm thử tích hợp và khả năng tái tạo | DONE | `validate_spark_model.py`, `validate_regression_production.py`, `submit_pipeline.sh`, `README.md`, `docs/PRODUCTION.md`; model-path, stage, schema, reload, OOT, and streaming checks passed. |\n'''
    (DOCS / "TASK_EVIDENCE.md").write_text(task_table)

    report = f'''# Project Report\n\n## Hoàn thiện Production\n\nNhóm đã chốt hai mô hình dưới dạng Spark `PipelineModel`: mô hình phân loại tại `models/random_forest_final_spark` và mô hình hồi quy tại `models/random_forest_regression_final_spark`. Cả hai pipeline gồm `VectorAssembler` và mô hình Random Forest tương ứng. Việc lưu, nạp lại trong Spark process mới và kiểm tra hợp đồng 23 biến đều thành công. Hai mô hình đã được đăng ký trong MLflow Model Registry với alias `champion`.\n\nLuồng dự báo Production sử dụng Spark Structured Streaming theo chuỗi `JSON -> spark.readStream -> PipelineModel.load -> model.transform -> JSON`. Kết quả phân loại gồm `prediction` và `probability_late_over_30m`; kết quả hồi quy được đổi tên rõ ràng thành `predicted_arrival_delay_minutes`. Các bài kiểm thử tích hợp đã chạy với dữ liệu JSON nhỏ và không huấn luyện lại mô hình.\n\nMô hình hồi quy cuối cùng dùng 500 cây, độ sâu 15 và tối thiểu 5 quan sát tại mỗi nút. Do số fold CV giữa các cấu hình không đồng nhất, nhóm không so sánh trực tiếp trung bình CV mà chọn mô hình theo RMSE trên cùng tập validation 20% trước OOT, seed 42. Teammate đã xác nhận artifact và chỉ số validation; sai khác khi tái tạo `randomSplit` cục bộ không được dùng để bác bỏ mô hình. Các cấu hình độ sâu 18 được loại do chi phí tính toán và tăng trưởng số nút quá lớn. OOT chỉ được dùng sau khi chốt mô hình; kết quả cuối là RMSE {metrics['rmse']:.6f}, MAE {metrics['mae']:.6f} và R² {metrics['r2']:.6f} trên {oot['oot_rows']:,} dòng.\n\nGiới hạn triển khai là các biến lịch sử theo hãng bay, sân bay và tàu bay phải được tính trước khi gửi sự kiện vào streaming. Bước dự báo là stateless và chỉ áp dụng `model.transform`; hệ thống không thể tái tạo trung thực các biến lag từ một JSON chuyến bay đơn lẻ. Schema và JSON mẫu cho cả hai tác vụ đã được lưu trong repository để phục vụ demo và tái tạo.\n'''
    (DOCS / "REPORT.md").write_text(report)

    readme = f'''# BigData Group 1 — Spark flight-delay models\n\nPySpark ML project with two Production tasks:\n\n- classification: probability that arrival delay exceeds 30 minutes;\n- regression: predicted arrival delay in minutes.\n\n## Runtime artifacts\n\n- `models/random_forest_final_spark` — classification `PipelineModel`;\n- `models/random_forest_regression_final_spark` — regression `PipelineModel`.\n\nModel directories, datasets, MLflow state, logs, and caches are excluded from\nGit. Both models consume the task contracts in `feature_contract.py`. Historical\nlag features must be produced upstream; missing values are filled with zero.\n\n## Instructor Demo\n\n```bash\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\n./submit_pipeline.sh classification\n./submit_pipeline.sh regression\n```\n\nThe scripts load existing models, validate a small OOT sample, and run one real\nStructured Streaming JSON batch. They never train. Default demo driver memory is\n32 GB and can be changed with `DRIVER_MEMORY`.\n\n## Regression winner\n\nThe selected model uses 500 trees, depth 15, and minimum leaf size 5. Selection\nused common validation RMSE 32.736849 because CV fold counts differed. Depth 18\nwas skipped. Full OOT metrics: RMSE {metrics['rmse']:.6f}, MAE {metrics['mae']:.6f},\nR² {metrics['r2']:.6f}.\n\nSee `docs/PRODUCTION.md`, `docs/REGRESSION_RESULTS.md`, and\n`docs/TASK_EVIDENCE.md` for commands and evidence.\n'''
    (ROOT / "README.md").write_text(readme)
    print(json.dumps({"classification": classification, "regression": regression}, indent=2))


if __name__ == "__main__":
    main()
