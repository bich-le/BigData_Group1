# Feature Dictionary — Custom & Feature Engineering
## US Flight Delays (USDOT) — Regression: `ARRIVAL_DELAY` | Classification: `ARRIVAL_DELAY > 30`

> Model dự đoán ngay sau khi máy bay rời cổng,
> trước khi taxi-out hoàn tất (theo giải thích của Huy: "khởi hành" = bắt
> đầu taxi-out). `DEPARTURE_DELAY` **hợp lệ** làm feature. Ranh giới
> leakage nằm ngay sau `DEPARTURE_DELAY`, trước `TAXI_OUT`.

---

## 1. Phương hướng xây dựng feature

Feature được tổ chức theo **5 cấp độ** (đề xuất từ modeling team review),
thay vì rời rạc như thiết kế ban đầu:

| Cấp độ | Composite key | Bản chất |
|---|---|---|
| Airline-level | `(AIRLINE, NGAY_DL, SCHEDULED_DEPARTURE)` | Lịch sử vận hành của hãng, tính đến thời điểm chuyến đang xét |
| Airport-level | `(ORIGIN_AIRPORT, NGAY_DL, SCHEDULED_DEPARTURE)` | Kết hợp lịch bay (schedule) + lịch sử delay tại sân bay |
| Airplane-level | `(TAIL_NUMBER, NGAY_DL, SCHEDULED_DEPARTURE)` | Lịch sử vận hành của chính máy bay đó |
| Flight-level | `(AIRLINE, FLIGHT_NUMBER, NGAY_DL, SCHEDULED_DEPARTURE)` | Đặc điểm riêng của chính chuyến đang xét, dùng `DEPARTURE_DELAY` |
| Temporal-level | Không cần entity | Đặc điểm thời gian (cuối tuần, khung giờ) |

**2 nguyên tắc leak-safety áp dụng xuyên suốt:**

1. **Feature LOOKBACK** (thống kê lịch sử `ARRIVAL_DELAY` của **các chuyến
   khác**, không phải chuyến đang xét): cắt cửa sổ **theo timestamp thật**
   (`NGAY_DL` + `SCHEDULED_DEPARTURE`), KHÔNG chỉ theo ranh giới ngày — 2
   chuyến cùng ngày nhưng khác giờ, chuyến sớm hơn KHÔNG được thấy dữ liệu
   của chuyến muộn hơn dù cùng ngày `t`. Implement bởi `derive_lookback_agg`
   (Section 5, `custom_transformers.py`), dùng `Window.rangeBetween` trên
   epoch-seconds.
2. **Feature SCHEDULE-BASED** (đếm số chuyến theo lịch, không phải
   outcome): an toàn dùng dữ liệu **cả ngày** kể cả "sau" thời điểm quan
   sát, vì lịch bay đã setting sẵn từ trước (áp dụng cho Airport-level và
   Airplane-level schedule-count). Implement bởi `derive_schedule_count`.

**Kỳ phái sinh:** ngày hiện tại là `t`. Kỳ 1 ngày = `{t}`; kỳ 3 ngày =
`{t-2,t-1,t}`; kỳ 7 ngày = `{t-6,...,t}`. Về code, thỏa mãn tự nhiên bằng
cách cắt cửa sổ theo epoch-seconds với `rangeBetween(-lookback_seconds, -1)`
— không cần xử lý "bao gồm ngày t" như 1 case riêng.

### 1.1 Chiến lược chia train/test — ĐÃ ĐỔI sang time-based split

`05_build_modeling_dataset.py` (file local, không push git) hiện chia
train/test theo **thời gian**, không phải random 80/20 như bản trước:

- Tính `split_date` = percentile 80% của `flight_timestamp` trên toàn bộ base.
- **Train** = mọi chuyến có `flight_timestamp < split_date` (80% thời gian đầu).
- **Test** = mọi chuyến có `flight_timestamp >= split_date` (20% thời gian cuối).

**Lý do đổi:** dù về mặt leak-safety, feature LOOKBACK đã được tính đúng
theo timestamp thật *trước khi* split (nên random split hay time split đều
không gây leakage kỹ thuật) — nhưng **time-based split đánh giá thực tế
hơn** cho bài toán có yếu tố thời gian: random split có thể vô tình đặt 1
chuyến tháng 1 vào test trong khi các chuyến tháng 1 tương tự (cùng
disruption, cùng điều kiện thời tiết) nằm ở train → model "nhìn thấy" ngữ
cảnh gần giống qua tập train, khiến test performance lạc quan giả tạo.
Time-based split (train = quá khứ, test = tương lai) mô phỏng đúng tình
huống production: dự đoán chuyến bay CHƯA XẢY RA dựa trên lịch sử đã biết.

⚠️ **Hệ quả cần lưu ý:** vì `AIRLINE`/`ORIGIN_AIRPORT` không phân bố đều
theo thời gian, tỷ lệ train/test thực tế theo SỐ DÒNG có thể lệch khỏi đúng
80/20 (script in ra `Train ratio`/`Test ratio` thực tế để kiểm tra) — khác
với random split luôn đảm bảo đúng tỷ lệ theo dòng.


---

## 2. Bảng trường gốc (raw fields)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `YEAR`, `MONTH`, `DAY` | Ngày thực hiện chuyến bay | Dùng trực tiếp + input `NGAY_DL`, composite key | ✅ Yes | Biết trước, không leakage |
| `DAY_OF_WEEK` | Thứ trong tuần (1=Thứ 2,...,7=CN, đã xác nhận ISO 8601) | Input cho `is_weekend` (Temporal-level) | ✅ Yes | Biết trước |
| `AIRLINE` | Mã IATA hãng bay | Dùng trực tiếp (categorical) + partition key Airline-level | ✅ Yes | Biết trước |
| `FLIGHT_NUMBER` | Số hiệu chuyến bay | Dùng trực tiếp (categorical, cardinality cao) | ⚠️ Cân nhắc | Chưa đánh giá kỹ — cardinality cao, cần xem xét encode nếu giữ |
| `TAIL_NUMBER` | Mã đuôi máy bay | Dùng trực tiếp + partition key Airplane-level | ✅ Yes | Đã quyết định giữ (mục 5.1); missing 0%, cardinality vừa phải (4,896) |
| `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT` | Mã sân bay đi/đến | Dùng trực tiếp + partition key Airport-level + input Haversine | ✅ Yes | Biết trước |
| `SCHEDULED_DEPARTURE` | Giờ khởi hành theo lịch (HHMM) | Input `scheduled_hour`, `flight_timestamp`, composite key | ✅ Yes | Biết trước |
| `DEPARTURE_TIME` | Giờ khởi hành thực tế | Loại bỏ | ❌ No | Leakage — actual time, biết SAU thời điểm dự đoán |
| `DEPARTURE_DELAY` | Số phút chênh lệch khởi hành | Input `flight_departure_delay_per_km`, `flight_departure_delay_vs_airline_avg_rt` (Flight-level) | ✅ **Yes** | Hợp lệ theo Phương án A — đã biết tại thời điểm dự đoán |
| `TAXI_OUT` | Phút từ rời cổng đến cất cánh | Loại bỏ | ❌ No | Leakage — ranh giới nằm ngay trước cột này |
| `WHEELS_OFF` | Thời điểm cất cánh thực tế | Loại bỏ | ❌ No | Actual time, post-taxi-out |
| `SCHEDULED_TIME` | Thời lượng bay theo lịch | Input `OutlierIQRTruncator` (giữ code, không dùng) | ✅ Yes (raw) | Biết trước; bản clip không dùng (xem mục 4) |
| `ELAPSED_TIME`, `AIR_TIME` | Thời gian bay thực tế | Loại bỏ | ❌ No | Biết sau chuyến bay |
| `DISTANCE` | Khoảng cách route gốc (dặm) | Input Haversine fallback (`route_distance_km`) | ✅ Yes (gián tiếp qua `distance_feature_km`) | Biết trước |
| `WHEELS_ON`, `TAXI_IN`, `ARRIVAL_TIME` | Thời điểm hạ cánh/vào cổng/đến thực tế | Loại bỏ | ❌ No | Post-outcome |
| `SCHEDULED_ARRIVAL` | Giờ đến theo lịch | Dùng trực tiếp nếu cần | ✅ Yes | Biết trước khi khởi hành |
| `ARRIVAL_DELAY` | Số phút chênh lệch đến | **TARGET** | — | Regression target; nguồn cho mọi feature LOOKBACK (của các chuyến KHÁC) |
| `DIVERTED`, `CANCELLED` | Cờ hủy/chuyển hướng | Lọc eligibility, không làm feature | ❌ No | Eligibility only |
| `CANCELLATION_REASON` | Mã lý do hủy | Loại bỏ | ❌ No | Chỉ có sau khi hủy |
| `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY` | Thành phần delay theo nguyên nhân | Loại bỏ | ❌ No | Cộng dồn nên chính `ARRIVAL_DELAY` |

---

## 3. Section 1 — Khoảng cách bay (CALC)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `haversine_distance_km` | Khoảng cách bay thực tế (great-circle) | `pandas_udf` Haversine từ lat/long; fallback tháng 10 (mã DOT numeric) dùng `DISTANCE` gốc, không tìm mapping thật (chi phí/lợi ích không tương xứng) | ❌ No *(trung gian)* | Chỉ là bước trung gian cho `distance_feature_km` |
| `route_distance_km` | `DISTANCE` gốc quy đổi km | `DISTANCE × 1.60934` | ❌ No *(trung gian)* | Fallback trung gian |
| `has_haversine_distance` | Có tính được Haversine hay không | Boolean flag | ✅ Yes | Minh bạch nguồn gốc; có thể mang tín hiệu mùa vụ (lệch tập trung tháng 10) |
| `distance_feature_km` | Khoảng cách feature cuối cùng | `coalesce(haversine, route_distance_km)` | ✅ Yes | Feature khoảng cách chính thức |

---

## 4. Section 2 — Outlier IQR (CLIP_IQR / FLAG_IQR)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `distance_feature_km_clipped`, `scheduled_time_clipped` | Giá trị đã cắt outlier | `OutlierIQRTruncator`: clip ngoài `[Q1-1.5×IQR, Q3+1.5×IQR]`, bounds fit trên train/fold | ❌ **No** | Tree-based model tự robust với outlier; clip làm mất thông tin thật (route dài thật, không phải noise), khó vận hành khi gặp giá trị mới ngoài range lúc train. **Vẫn giữ code** (đáp ứng Task 1 rubric, 15 điểm) |
| `distance_feature_km_is_outlier`, `scheduled_time_is_outlier` | Cờ đánh dấu quan sát bất thường | Binary flag dựa trên `[Q1, Q3]` thô (KHÔNG nhân multiplier) | ✅ **Yes** | Thay thế cho clip — giữ nguyên giá trị gốc, chỉ thêm tín hiệu nhị phân |

---

## 5. Section 3 — Target Encoding (MEAN_ENC)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `airline_encoded`, `origin_airport_encoded`, `destination_airport_encoded`, `tail_number_encoded` | Mean `ARRIVAL_DELAY` đã smooth theo category | `TargetEncoder`: m-estimate smoothing, fit trên train/fold, unseen category → fallback global mean | ❌ **No** | Aggregate TĨNH trên toàn bộ train (không theo thời gian) — dù fit-on-train-only đã tránh leak giữa train/test, vẫn không đúng bản chất "chỉ biết quá khứ tại thời điểm dự đoán". **Giữ code** (đáp ứng rubric), thay thế bằng nhóm LAG ở Section 5 |

---

## 6. Section 4 — Feature cơ bản (EXTRACT / COUNT)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `scheduled_hour` | Giờ trong ngày (0-23) | Trích từ `SCHEDULED_DEPARTURE`, xử lý `"2400"` → giờ 0 | ✅ Yes | Bắt pattern giờ cao điểm |
| `route_congestion_count` | Số chuyến cùng sân bay/ngày/giờ | `groupBy` + join ngược | ✅ Yes | Lịch bay, không phải outcome — an toàn (⚠️ thiếu `YEAR`, xem mục 1) |

---

## 7. Section 5 — 5 cấp độ

### 7.1 Airline-level (`build_airline_level_features`)

> Mỗi feature dưới đây sinh ra **3 cột** theo 3 cửa sổ mặc định
> (`lookback_windows=(1,3,7)`): suffix `_lag1d`, `_lag3d`, `_lag7d`.

| Tên biến (mẫu, có 3 cửa sổ) | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `airline_flights_count_lag{1,3,7}d` | Số chuyến của hãng trong cửa sổ lookback | `derive_lookback_agg`, COUNT, cắt theo timestamp | ✅ Yes | Dùng `ARRIVAL_DELAY` lịch sử của chuyến KHÁC |
| `airline_late_flights_count_lag{1,3,7}d` | Số chuyến trễ (`ARRIVAL_DELAY>30`) | `derive_lookback_agg`, SUM(CASE) | ✅ Yes | Ngưỡng đã chốt = 30 |
| `airline_total_delay_minutes_lag{1,3,7}d` | Tổng phút trễ của hãng | `derive_lookback_agg`, SUM | ✅ Yes | — |
| `airline_distinct_tails_late_lag{1,3,7}d` | Số máy bay khác nhau bị trễ | `derive_lookback_agg`, approx_count_distinct(CASE) | ✅ Yes | — |
| `airline_late_flight_ratio_lag{1,3,7}d` | Tỷ lệ chuyến trễ | `safe_divide(late_count, total_count)` | ✅ Yes | — |
| `airline_delay_per_tail_lag{1,3,7}d` | Phút trễ / số máy bay bị trễ | `safe_divide(total_delay, distinct_tails)` | ✅ Yes | — |

### 7.2 Airport-level (`build_airport_level_features`)

> **CẬP NHẬT**: kỳ phái sinh đổi sang **GIỜ** (mặc định `lookback_windows_hours=(3,6,12)`)
> thay vì ngày — disruption sân bay (thời tiết, ATC) là hiện tượng ngắn hạn.
> Suffix đổi từ `_lag{n}d` sang `_lag{n}h`.

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `airport_scheduled_departures_same_day` | Số chuyến đi từ sân bay trong ngày | `derive_schedule_count` — **an toàn dùng cả ngày** (lịch bay), KHÔNG cần multi-window | ✅ Yes | Schedule-based, không phải outcome |
| `airport_scheduled_arrivals_same_day` | Số chuyến đến sân bay trong ngày | `derive_schedule_count` theo `DESTINATION_AIRPORT` | ✅ Yes | — |
| `airport_late_flights_count_lag{3,6,12}h` | Số chuyến trễ (`ARRIVAL_DELAY>30`) tại sân bay | `derive_lookback_agg`, cắt theo timestamp, cửa sổ tính bằng giờ | ✅ Yes | Ngưỡng đã chốt = 30 |
| `airport_avg_delay_lag{3,6,12}h` | Trung bình phút trễ tại sân bay (lookback) | `derive_lookback_agg`, AVG, cửa sổ tính bằng giờ | ✅ Yes | — |

### 7.3 Airplane-level (`build_airplane_level_features`)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `airplane_flights_scheduled_same_day` | Số chuyến máy bay phải bay trong ngày | `derive_schedule_count` — an toàn dùng cả ngày, KHÔNG cần multi-window | ✅ Yes | Schedule-based |
| `airplane_total_delay_minutes_lag{1,3,7}d` | Tổng phút trễ lịch sử của chính máy bay | `derive_lookback_agg`, SUM, cắt theo timestamp | ✅ Yes | Dùng `ARRIVAL_DELAY` lịch sử |
| `airplane_delay_per_km_lag{1,3,7}d` | Phút trễ lịch sử / km chuyến hiện tại | `safe_divide(total_delay_lag, distance_feature_km)` | ✅ Yes | — |
| **`airplane_prev_flight_arrival_delay`** *(mới)* | `ARRIVAL_DELAY` của **đúng 1 chuyến liền trước** (không phải aggregate nhiều chuyến) của cùng máy bay | `F.lag(ARRIVAL_DELAY, 1)` theo `Window.partitionBy(TAIL_NUMBER).orderBy(timestamp)` | ✅ Yes | Tín hiệu "vấn đề tức thời" của máy bay, khác nhóm `_lag{n}d` (tổng hợp nhiều chuyến) |
| **`airplane_prev_flight_delay_vs_current_schedule`** *(mới)* | `airplane_prev_flight_arrival_delay` trừ `SCHEDULED_ARRIVAL` (quy đổi phút) của chuyến hiện tại | `airplane_prev_flight_arrival_delay - _hhmm_to_minutes(SCHEDULED_ARRIVAL)` | ⚠️ **Cần xác nhận** | ⚠️ **2 đại lượng khác bản chất** (duration vs. time-of-day) — implement đúng theo yêu cầu, nhưng ý nghĩa nghiệp vụ cần xác nhận lại trước khi dùng chính thức trong training |

### 7.4 Flight-level (`build_flight_level_features`)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `flight_departure_delay_per_km` | `DEPARTURE_DELAY` của chính chuyến / km | `safe_divide(DEPARTURE_DELAY, distance_feature_km)` | ✅ **Yes** | Hợp lệ theo Phương án A đã chốt |
| `flight_departure_delay_vs_airline_avg_rt7d` | Tỷ lệ delay chuyến này / delay TB 7 ngày gần đây của hãng | `safe_divide(DEPARTURE_DELAY, avg_airline_delay_lag7d)`, dùng rule chia đặc biệt; tham số `airline_avg_window` chọn cửa sổ so sánh (mặc định 7 ngày) |  **Yes** | Hợp lệ theo Phương án A đã chốt.  Cả tử và mẫu đều có thể âm — xem lưu ý bên dưới |
| **`flight_computed_scheduled_duration_minutes`** *(mới)* | `SCHEDULED_ARRIVAL - SCHEDULED_DEPARTURE`, tính bằng phút | `_hhmm_to_minutes(SCHEDULED_ARRIVAL) - _hhmm_to_minutes(SCHEDULED_DEPARTURE)`, xử lý qua đêm (+1440 nếu âm) | ⚠️ **Cân nhắc** | **KHÔNG áp dụng kỳ phái sinh** (thông tin mặc định của chuyến).  Hiệu giờ đồng hồ ĐỊA PHƯƠNG — với chuyến xuyên múi giờ sẽ KHÁC elapsed time thực (cột `SCHEDULED_TIME` gốc đã tính đúng, không lệch múi giờ) |
| *(không tạo cột mới)* Số km đi | `distance_feature_km` đã có sẵn từ Section 1 | — | ✅ Yes | Không cần derive lại |
| **`flight_scheduled_minutes_per_km`** *(mới)* | Số phút bay theo lịch / km — proxy tốc độ bay theo lịch | `safe_divide(flight_computed_scheduled_duration_minutes, distance_feature_km)` |  **Cân nhắc** | **KHÔNG áp dụng kỳ phái sinh**; phụ thuộc độ tin cậy của `flight_computed_scheduled_duration_minutes` ở trên |


### 7.5 Temporal-level (`build_temporal_level_features`)

| Tên biến | Ý nghĩa | Cách xử lý | Dùng model? | Lý do |
|---|---|---|---|---|
| `is_weekend` | Cuối tuần hay không | `DAY_OF_WEEK.isin([6,7])`, đã xác nhận encoding ISO 8601 (1=Thứ 2) | ✅ Yes | Biết trước |
| `time_of_day_bucket` | Khung giờ (morning/afternoon/evening/night) | Bucket từ `scheduled_hour` | ✅ Yes | — |

---

## 8. Quyết định phụ trợ

### 8.1 `TAIL_NUMBER`
**GIỮ** làm candidate feature. Profiling (`01_tail_number_review.py`): missing
0.00%, cardinality 4,896 (0.0857% số dòng), TB 1,167 chuyến/tail, percentile
10% vẫn đạt 148 chuyến/tail → không phải long-tail. 337/4,896 tail number
(~6.9%) xuất hiện ở nhiều hãng — khả năng do thuê ướt máy bay (wet lease),
không phải lỗi dữ liệu.

### 8.2 DOT→IATA mapping (tháng 10)
Đã thử tìm nguồn mapping thật (BTS `L_AIRPORT_ID.csv` + `L_AIRPORT.csv`), bị
chặn do trang BTS TranStats chỉ cho tải qua form thủ công. **QUYẾT ĐỊNH**:
dùng fallback `DISTANCE` cho 482,878 dòng tháng 10 (8.45% eligible dataset)
— chi phí/lợi ích không tương xứng để tiếp tục tìm.

### 8.3 Patch tọa độ 3 sân bay thiếu
`ECP`, `PBG`, `UST` — đã tra cứu thủ công (đối chiếu Wikipedia/FAA + Great
Circle Mapper), patch trong `MANUAL_AIRPORT_COORDS` (`custom_transformers.py`).

---

## 9. File liên quan

| File | Vai trò |
|---|---|
| `custom_transformers.py` | Toàn bộ 5 Section: Haversine, Outlier+flag, TargetEncoder (không dùng training), feature cơ bản, 5 cấp độ |
| `tests/test_custom_transformers.py` | 30 test case, cover cả 5 Section |
