# Custom Features — Quick Guide

## 1. Tổng quan

Mục tiêu của Custom Features là tạo thêm các biến có ý nghĩa hơn từ dữ liệu flight delay, giúp Prediction Model có thêm thông tin để dự đoán `ARRIVAL_DELAY`.

| Nhóm | Feature | Ý nghĩa |
|---|---|---|
| Spatial | `distance_feature_km` | Khoảng cách giữa sân bay đi và đến |
| Spatial | `has_haversine_distance` | Có dùng tọa độ để tính distance hay dùng fallback |
| Outlier | `distance_feature_km_clipped` | Distance sau khi giới hạn outlier bằng IQR |
| Outlier | `scheduled_time_clipped` | Scheduled flight time sau khi giới hạn outlier |
| Target Encoding | `*_target_encoded` | Mức delay lịch sử của từng category |
| Schedule | `scheduled_hour` | Giờ khởi hành dự kiến |
| Schedule | `route_congestion_count` | Số chuyến bay scheduled cùng sân bay/thời điểm |
| Historical (đề xuất) | `*_delay_lag` | Delay của các chuyến trước đó |

---

## 2. Các feature hiện tại

### A. Haversine Distance

**Feature:** `distance_feature_km`

Tính khoảng cách giữa Origin và Destination bằng latitude/longitude.

- Dùng công thức Haversine.
- Nếu không có tọa độ, dùng `DISTANCE × 1.60934` làm fallback.
- `has_haversine_distance` cho biết có dùng Haversine hay không.

**Ý nghĩa:** Chuyến bay càng dài có thể có đặc điểm vận hành và delay khác nhau.

### B. IQR Outlier

**Features:**
- `distance_feature_km_clipped`
- `scheduled_time_clipped`

```text
IQR = Q3 - Q1
Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR
```

Giá trị vượt `[Lower, Upper]` được đưa về biên gần nhất.

**Ý nghĩa:** Giảm ảnh hưởng của giá trị cực đoan mà không xóa observation.

**Không áp dụng cho:** `ARRIVAL_DELAY` (target) và `DEPARTURE_DELAY` (post-departure leakage).

### C. Target Encoding

Áp dụng cho:

```text
AIRLINE
ORIGIN_AIRPORT
DESTINATION_AIRPORT
TAIL_NUMBER
```

Chuyển categorical → numerical dựa trên historical `ARRIVAL_DELAY`.

```text
encoded =
(n_category × category_mean + m × global_mean)
/
(n_category + m)
```

**Ý nghĩa:** Cho model biết airline/airport/aircraft có xu hướng delay như thế nào.

**Smoothing:** Category ít dữ liệu được kéo về global mean để giảm overfitting.

**Leakage:** Encoder phải fit trên training/fold. Code hiện tại tránh leakage giữa train/test/CV folds, nhưng chưa phải internal OOF target encoding.

### D. Schedule Features

#### `scheduled_hour`

```text
0830 → 8
1745 → 17
```

`2400` được chuẩn hóa thành hour `0`.

**Ý nghĩa:** Delay có thể khác nhau theo thời điểm trong ngày.

#### `route_congestion_count`

Đếm số chuyến bay scheduled theo:

```text
Origin Airport
+ Month
+ Day
+ Scheduled Hour
```

**Ý nghĩa:** Đại diện cho mức độ bận của lịch bay tại sân bay/thời điểm đó.

**Lưu ý:** Nếu prediction contract chỉ cho phép thông tin có trước một thời điểm cụ thể, cần xác định feature này có được phép biết toàn bộ schedule cùng giờ hay không.

---

