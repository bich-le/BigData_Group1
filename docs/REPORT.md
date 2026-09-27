# Project Report

## Hoàn thiện Production

Nhóm đã chốt hai mô hình dưới dạng Spark `PipelineModel`: mô hình phân loại tại `models/random_forest_final_spark` và mô hình hồi quy tại `models/random_forest_regression_final_spark`. Cả hai pipeline gồm `VectorAssembler` và mô hình Random Forest tương ứng. Việc lưu, nạp lại trong Spark process mới và kiểm tra hợp đồng 23 biến đều thành công. Hai mô hình đã được đăng ký trong MLflow Model Registry với alias `champion`.

Luồng dự báo Production sử dụng Spark Structured Streaming theo chuỗi `JSON -> spark.readStream -> PipelineModel.load -> model.transform -> JSON`. Kết quả phân loại gồm `prediction` và `probability_late_over_30m`; kết quả hồi quy được đổi tên rõ ràng thành `predicted_arrival_delay_minutes`. Các bài kiểm thử tích hợp đã chạy với dữ liệu JSON nhỏ và không huấn luyện lại mô hình.

Mô hình hồi quy cuối cùng dùng 500 cây, độ sâu 15 và tối thiểu 5 quan sát tại mỗi nút. Do số fold CV giữa các cấu hình không đồng nhất, nhóm không so sánh trực tiếp trung bình CV mà chọn mô hình theo RMSE trên cùng tập validation 20% trước OOT, seed 42. Teammate đã xác nhận artifact và chỉ số validation; sai khác khi tái tạo `randomSplit` cục bộ không được dùng để bác bỏ mô hình. Các cấu hình độ sâu 18 được loại do chi phí tính toán và tăng trưởng số nút quá lớn. OOT chỉ được dùng sau khi chốt mô hình; kết quả cuối là RMSE 35.051807, MAE 17.101296 và R² 0.267770 trên 932,084 dòng.

Giới hạn triển khai là các biến lịch sử theo hãng bay, sân bay và tàu bay phải được tính trước khi gửi sự kiện vào streaming. Bước dự báo là stateless và chỉ áp dụng `model.transform`; hệ thống không thể tái tạo trung thực các biến lag từ một JSON chuyến bay đơn lẻ. Schema và JSON mẫu cho cả hai tác vụ đã được lưu trong repository để phục vụ demo và tái tạo.
