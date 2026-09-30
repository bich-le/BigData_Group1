# MLflow: kiểm tra phần Sương Mai và demo khi thuyết trình

## Những gì đã có và giới hạn cần nói đúng

- `mllib_pipeline.py` đã log tham số preprocessing, metrics PCA, config, PCA plot và fitted preprocessing model. `docs/assets/pca_variance.png` đã có trong Git.
- `scripts/log_tournament_mlflow.py` đưa **các kết quả model tournament đã ghi trong repo** vào MLflow UI: 3 mô hình phân loại, 4 họ mô hình hồi quy và 12 cấu hình Random Forest hồi quy. Mỗi run được gắn `evidence.kind=reported_result_import` và `training.executed=false`; script không huấn luyện lại các mô hình.
- `scripts/register_mlflow.py` nhận model Spark đã fit, log hyperparameters, metrics được cung cấp, feature importances **đọc từ fitted model**, plot tương ứng, PCA plot, evaluation artifacts; liên kết tới run tournament và đưa phiên bản model vào `Staging`. Chỉ dùng `--promote-production` khi có đủ metrics OOT; lúc đó script chuyển sang `Production` và đặt alias `champion`.
- Hai thư mục model cuối cùng, dữ liệu OOT, `mlflow.db` và `mlruns/` của các run cũ đều bị Git ignore và không có trong repo. Vì vậy hiện không thể nạp lại hai run registry cũ chỉ từ GitHub, tính AUC-ROC thật, hay xác nhận các model production đã đăng ký trong máy của người khác.
- `docs/assets/feature_importance.png` cũ được tạo từ các số **hard-coded đại diện** trong `scripts/benchmark_results.py`. Trong run import, ảnh này nằm dưới `plots/representative_only`. Không trình bày nó như feature importance trích xuất từ model. Khi có model thật, script đăng ký sẽ tự tạo plot và metrics từ `estimator.featureImportances`.

## Demo MLflow UI từ dữ liệu hiện có trong Git

### Cách mở nhanh trên máy Mac của Mai

Mở file `MO_DEMO_MLFLOW.command` ở thư mục gốc bằng cách nháy đúp. File sẽ dùng môi trường Python đã chuẩn bị trên máy này, tạo dữ liệu demo nếu chưa có, bật MLflow và mở trình duyệt tại `http://127.0.0.1:5000`. Giữ cửa sổ Terminal hiện ra trong lúc thuyết trình. Nếu tải file trực tiếp từ GitHub và macOS không cho nháy đúp, mở Terminal trong thư mục project rồi chạy `zsh MO_DEMO_MLFLOW.command`.

Trong UI: chọn **Model training** → `FlightDelay-Tournament-Reported` → **Runs**. Mở `classification tournament (reported)` hoặc `regression tournament (reported)`, rồi chọn **Metrics**, **Params** và **Artifacts**. Ở Artifacts có `plots/pca_variance.png` và biểu đồ so sánh tournament. Mở run con `Random Forest` để chỉ vào các metrics của một candidate. Có thể nói: “MLflow lưu tham số, số đo và biểu đồ để nhóm so sánh mô hình; màn hình này tái hiện các kết quả đã được nhóm báo cáo.”

**Không giới thiệu run demo này như một lần train model mới**, và không nhận run cũ `f8a471b2996246ff9bda929853f5cf0f` là đã được kiểm chứng trên máy này.

### Cách chạy bằng lệnh

Chạy từ thư mục gốc repository sau khi cài `requirements.txt`:

```bash
python scripts/log_tournament_mlflow.py
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Mở `http://127.0.0.1:5000`, chọn **Model training** ở thanh bên trái rồi vào experiment `FlightDelay-Tournament-Reported` → **Runs**. Hai parent runs `classification tournament (reported)` và `regression tournament (reported)` chứa biểu đồ PCA, biểu đồ so sánh mô hình và các file nguồn. Các child runs chứa params/metrics của từng candidate và từng cấu hình RF. Các cấu hình bị skip/incomplete được gắn status tương ứng, không có metric giả. Luồng này đã được kiểm tra trên MLflow UI 3.10.1: 21 runs hiển thị, gồm 2 parent runs, 7 candidate runs và 12 cấu hình RF.

Khi giải thích trước lớp: đây là **UI demo từ kết quả nhóm đã báo cáo**, không phải bằng chứng các mô hình đã được train lại trên máy demo. Run preprocessing riêng đã được tái chạy và xác minh cục bộ với ID `d6751ddbeaf24f1c9f29bb13c0fb5501`; tracking store của lần chạy đó không được commit vào repo vì `mlruns/` là dữ liệu runtime.

## Kết nối model thực vào MLflow Registry

Khi nhận được thư mục model của Huy, chạy script đăng ký với tracking backend mà nhóm dùng. Ví dụ hồi quy:

```bash
python scripts/register_mlflow.py \
  --task regression \
  --model models/random_forest_regression_final_spark \
  --training-rows 3739186 \
  --metrics-file docs/evaluations/regression_evaluation.json \
  --tournament-run-id <REGRESSION_PARENT_RUN_ID>
```

Lệnh trên đăng ký phiên bản ở `Staging`. Kiểm tra model và nguồn metrics OOT trước khi chạy thêm `--promote-production`. Với phân loại, file `classification_tournament.json` chỉ có Gini, KS, precision, recall theo báo cáo. Cần tính hoặc cung cấp `oot_auc_roc` từ dự đoán trên OOT thật mới được promote theo yêu cầu đề bài. Không suy ra AUC từ số Gini nếu chưa xác nhận cách tính Gini trong kết quả gốc.

## Đối chiếu yêu cầu

| Yêu cầu | Tình trạng xác minh |
| --- | --- |
| PCA(k=10) và cumulative plot | Có trong code và `docs/assets/pca_variance.png` |
| MLflow preprocessing params, metrics, plot, artifacts | Đã chạy lại độc lập trên mẫu giả lập; run mới có model artifact nạp lại được |
| MLflow gắn với model tournament | Có script import các kết quả đã báo cáo; chạy được UI và child runs; chưa có rerun huấn luyện đầy đủ từ dữ liệu thật |
| Feature importance thực từ fitted model | Code đăng ký đã bổ sung; chưa thực thi vì model bị Git ignore và không có ở đây |
| AUC-ROC, registry Staging → Production của model thật | Chưa thể xác minh hoàn chỉnh nếu thiếu model, dự đoán OOT và tracking backend gốc |
