# Data Dictionary

Dictionary mô tả **typed raw schema** được tạo bởi `scripts/prepare_data.py`.

Inference contract hiện tại: model dự đoán **trước khi chuyến bay thực sự khởi hành**.

> `Candidate feature` chỉ là đánh giá sơ bộ. Feature set cuối cùng thuộc module Feature/Pipeline.

## Flights

| Column | Type | Meaning | Role / status | Pre-departure? | Notes |
|---|---|---|---|---|---|
| `YEAR` | integer | Năm thực hiện chuyến bay | Candidate feature | Yes | Dataset hiện tại chỉ có năm 2015 |
| `MONTH` | integer | Tháng thực hiện chuyến bay | Candidate feature / partition key | Yes | Domain 1–12; flights Parquet partition theo cột này |
| `DAY` | integer | Ngày trong tháng | Candidate feature | Yes | Domain 1–31 |
| `DAY_OF_WEEK` | integer | Thứ trong tuần | Candidate feature | Yes | Domain 1–7 |
| `AIRLINE` | string | Mã IATA của hãng hàng không khai thác chuyến bay | Candidate feature | Yes | Join `airlines.IATA_CODE`; coverage 100% |
| `FLIGHT_NUMBER` | string | Số hiệu chuyến bay do hãng khai thác | Candidate feature | Yes | Identifier, cardinality tương đối cao |
| `TAIL_NUMBER` | string | Số đăng ký của máy bay cụ thể thực hiện chuyến bay | **Requires review** | Need verification | High cardinality; cần xác minh availability trước departure |
| `ORIGIN_AIRPORT` | string | Mã sân bay khởi hành | Candidate feature | Yes | Có IATA 3 ký tự và DOT numeric 5 chữ số |
| `DESTINATION_AIRPORT` | string | Mã sân bay đến | Candidate feature | Yes | Có IATA 3 ký tự và DOT numeric 5 chữ số |
| `SCHEDULED_DEPARTURE` | string | Giờ khởi hành theo lịch, dạng HHMM | Candidate feature | Yes | `2400` được xem là convention hợp lệ |
| `DEPARTURE_TIME` | string | Giờ khởi hành thực tế, dạng HHMM | Leakage | No | Chỉ biết sau khi chuyến bay bắt đầu khởi hành |
| `DEPARTURE_DELAY` | integer | Số phút chênh lệch giữa giờ khởi hành thực tế và giờ theo lịch | Leakage | No | Giá trị dương = trễ; giá trị âm = khởi hành sớm |
| `TAXI_OUT` | integer | Số phút từ lúc rời cổng đến lúc cất cánh | Leakage | No | Post-departure |
| `WHEELS_OFF` | string | Thời điểm máy bay rời mặt đất, dạng HHMM | Leakage | No | Actual time |
| `SCHEDULED_TIME` | integer | Tổng thời lượng chuyến bay theo lịch, tính bằng phút | Candidate feature | Yes | Không phải elapsed time thực tế |
| `ELAPSED_TIME` | integer | Tổng thời gian thực tế từ cổng đi đến cổng đến | Leakage | No | Chỉ biết sau hành trình |
| `AIR_TIME` | integer | Thời gian thực tế máy bay ở trên không, tính bằng phút | Leakage | No | Chỉ biết sau chuyến bay |
| `DISTANCE` | integer | Khoảng cách tuyến bay giữa sân bay đi và đến | Candidate feature | Yes | Raw route distance; chưa phải Haversine feature |
| `WHEELS_ON` | string | Thời điểm máy bay chạm đất, dạng HHMM | Leakage | No | Actual arrival-stage time |
| `TAXI_IN` | integer | Số phút từ lúc chạm đất đến lúc vào cổng | Leakage | No | Post-arrival |
| `SCHEDULED_ARRIVAL` | string | Giờ đến theo lịch, dạng HHMM | Candidate feature | Yes | Biết trước khi khởi hành |
| `ARRIVAL_TIME` | string | Giờ đến thực tế, dạng HHMM | Leakage | No | Post-outcome |
| `ARRIVAL_DELAY` | integer | Số phút chênh lệch giữa giờ đến thực tế và giờ đến theo lịch | **Target** | No | Regression target; classification label dự kiến là `> 30`; số âm nghĩa là đến sớm |
| `DIVERTED` | integer | Cờ cho biết chuyến bay có bị chuyển hướng hay không | Eligibility only | No | Domain 0/1; không dùng làm feature |
| `CANCELLED` | integer | Cờ cho biết chuyến bay có bị hủy hay không | Eligibility only | No | Domain 0/1; không dùng làm feature |
| `CANCELLATION_REASON` | string | Mã nguyên nhân hủy chuyến | Leakage | No | Chỉ có sau quyết định hủy |
| `AIR_SYSTEM_DELAY` | integer | Số phút trễ được quy cho hệ thống hàng không | Post-outcome leakage | No | Missing có ý nghĩa nghiệp vụ; không tự fill 0 |
| `SECURITY_DELAY` | integer | Số phút trễ do nguyên nhân an ninh | Post-outcome leakage | No | Missing có ý nghĩa nghiệp vụ; không tự fill 0 |
| `AIRLINE_DELAY` | integer | Số phút trễ do nguyên nhân thuộc hãng hàng không | Post-outcome leakage | No | Missing có ý nghĩa nghiệp vụ; không tự fill 0 |
| `LATE_AIRCRAFT_DELAY` | integer | Số phút trễ do máy bay từ chặng trước đến muộn | Post-outcome leakage | No | Missing có ý nghĩa nghiệp vụ; không tự fill 0 |
| `WEATHER_DELAY` | integer | Số phút trễ do thời tiết | Post-outcome leakage | No | Missing có ý nghĩa nghiệp vụ; không tự fill 0 |

### Target eligibility

Dòng hợp lệ cho cả regression và classification:

```text
ARRIVAL_DELAY IS NOT NULL
AND CANCELLED = 0
AND DIVERTED = 0
```

Classification label dự kiến:

```text
label = 1 if ARRIVAL_DELAY > 30
label = 0 otherwise
```

Typed raw **chưa có** cột label được persist.

## Airports

| Column | Type | Meaning | Role | Notes |
|---|---|---|---|---|
| `IATA_CODE` | string | Mã IATA 3 ký tự của sân bay | Lookup key | Unique, non-null; không match trực tiếp DOT numeric code |
| `AIRPORT` | string | Tên đầy đủ của sân bay | Lookup attribute | Thuộc tính tham chiếu |
| `CITY` | string | Thành phố nơi sân bay đặt | Lookup attribute | Thuộc tính tham chiếu |
| `STATE` | string | Bang hoặc vùng hành chính của sân bay | Lookup attribute | Thuộc tính tham chiếu |
| `COUNTRY` | string | Quốc gia của sân bay | Lookup attribute | Thuộc tính tham chiếu |
| `LATITUDE` | double | Vĩ độ địa lý của sân bay | Lookup attribute | 3 rows thiếu đồng thời với longitude |
| `LONGITUDE` | double | Kinh độ địa lý của sân bay | Lookup attribute | 3 rows thiếu đồng thời với latitude |

## Airlines

| Column | Type | Meaning | Role | Notes |
|---|---|---|---|---|
| `IATA_CODE` | string | Mã IATA của hãng hàng không | Lookup key | Unique, non-null; join với `flights.AIRLINE` |
| `AIRLINE` | string | Tên đầy đủ của hãng hàng không | Lookup attribute | Thuộc tính tham chiếu |

## Known data notes

- `ARRIVAL_DELAY` null: 105,071 rows.
- 105,071 target-null rows tương ứng 89,884 cancelled + 15,187 diverted flights.
- Delay-cause columns missing khoảng 81.725% và chỉ có giá trị khi `ARRIVAL_DELAY >= 15`.
- Direct airport IATA lookup coverage khoảng 91.645%.
- Tháng 10 có 486,165 rows dùng airport code numeric/DOT 5 chữ số ở cả origin và destination.
- `ARRIVAL_DELAY` có giá trị âm hợp lệ: chuyến bay đến sớm.
