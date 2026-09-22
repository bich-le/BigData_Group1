"""
custom_transformers.py

Custom PySpark Transformer classes & PyArrow-accelerated pandas_udf
cho phase Custom & Feature Engineering.

Deliverable chính thức theo cấu trúc nộp bài (Task 1, 15/100 điểm).

Nội dung file này sẽ được bổ sung dần theo tracker:
  [x] Section 1 — Haversine distance (pandas_udf) + fallback DOT/IATA
  [x] Section 2 — OutlierIQRTruncator (+ binary outlier flag)
  [x] Section 3 — Leakage-safe Target Encoder (giữ code, không dùng training)
  [x] Section 4 — Feature cơ bản (giờ, route congestion)
  [x] Section 5 — 5 cấp độ (Airline/Airport/Airplane/Flight/Temporal-level)
"""

import numpy as np
import pandas as pd
import json
from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType

# =============================================================================
# SECTION 1 — Haversine distance feature (pandas_udf, Arrow-accelerated)
# =============================================================================
#
# Bối cảnh (đã xác nhận bằng khảo sát thực tế trên toàn bộ eligible dataset):
#   - Tháng 1-9, 11, 12: ORIGIN_AIRPORT/DESTINATION_AIRPORT dùng mã IATA 3 ký tự,
#     join 100% được với bảng airports.
#   - Tháng 10 (482,878 dòng, 8.45% eligible dataset): dùng mã DOT numeric 5 chữ
#     số ở CẢ hai đầu, KHÔNG join được với airports.IATA_CODE => không có
#     lat/long => không tính được Haversine.
#
# Chiến lược: tính Haversine khi có đủ tọa độ; nếu thiếu (chủ yếu là tháng 10),
# fallback dùng cột DISTANCE gốc đã có sẵn trong flights, và đánh dấu bằng cờ
# has_haversine_distance để minh bạch nguồn gốc feature.
#
# QUYẾT ĐỊNH VỀ DOT->IATA MAPPING THẬT (tháng 10):
#   Đã thử tìm nguồn mapping chính thức (BTS Aviation Support Tables,
#   L_AIRPORT_ID.csv + L_AIRPORT.csv, join 2 bước qua tên sân bay). Bị chặn
#   vì trang BTS TranStats chỉ cho tải qua form thủ công, không có link ổn
#   định để tái tạo cho teammate; chỉ tìm được 1/2 file cần thiết qua mirror
#   công khai. QUYẾT ĐỊNH: dừng tìm mapping thật, dùng fallback DISTANCE làm
#   giải pháp chính thức -- chi phí tìm/tải file ngoài không tương xứng với
#   lợi ích (chỉ cải thiện ~8.45% của 1 feature phụ trợ đã có fallback tốt).


# Patch thủ công cho 3 sân bay thiếu lat/long trong airports_df (đã tra cứu
# và đối chiếu chéo nhiều nguồn — Wikipedia/FAA, Great Circle Mapper).
# Đây là fix RẺ và DỨT ĐIỂM (chỉ 3 sân bay cố định), khác hẳn vấn đề DOT/IATA
# ở trên (toàn bộ 1 tháng, không thể patch thủ công từng dòng).
# DOT numeric -> IATA mapping theo notebook 06.ipynb.
MANUAL_DOT_IATA_MAPPING = {
    "11193": "CVG", "11278": "DCA", "12016": "GUM",
    "12264": "IAD", "14222": "PPG", "14489": "RDM", "15016": "STL",
}

MANUAL_AIRPORT_COORDS = {
    # IATA_CODE: (LATITUDE, LONGITUDE)  -- nguồn: Wikipedia/FAA, đối chiếu Great Circle Mapper
    "ECP": (30.3417, -85.7973),    # Northwest Florida Beaches International Airport
    "PBG": (44.65083, -73.46806),  # Plattsburgh International Airport
    "UST": (29.95925, -81.33972),  # Northeast Florida Regional Airport (St. Augustine)
}


def _session_of(df):
    """Lấy SparkSession từ DataFrame, tương thích cả PySpark < 3.3 (chưa có
    thuộc tính DataFrame.sparkSession)."""
    session = getattr(df, "sparkSession", None)
    if session is not None:
        return session
    sql_ctx = getattr(df, "sql_ctx", None)
    if sql_ctx is not None:
        return sql_ctx.sparkSession
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _normalize_airport_descriptions(df):
    """Chuẩn hoá bảng tra cứu DOT (Code, Description) -> (AIRPORT_CODE, Description)."""
    if df is None:
        return None
    return df.select(
        F.col("Code").cast("string").alias("AIRPORT_CODE"),
        F.col("Description").cast("string").alias("Description"),
    ).filter(F.col("AIRPORT_CODE").isNotNull())


def build_airport_code_mapping(
    flights_df, airports_df, airport_ref_df=None, airport_id_df=None, manual_mapping=None
):
    """
    Dựng bảng AIRPORT_CODE -> IATA_CODE (+ toạ độ) theo logic notebook 06.ipynb.

    Thứ tự ưu tiên (mỗi cấp chỉ xử lý các code CHƯA được cấp trước giải quyết,
    nên kết quả luôn đúng 1 dòng / 1 AIRPORT_CODE, không cần dropDuplicates cuối):
      0. direct  : code vốn đã là IATA -> giữ nguyên
      1. level 1 : STATE + CITY + tên sân bay (từ đầu khớp)
      2. level 2 : STATE + CITY
      3. level 3 : STATE + tên sân bay
      4. manual  : mapping thủ công (MANUAL_DOT_IATA_MAPPING), chỉ áp dụng cho
                   code còn sót lại VÀ có mặt trong flights_df

    manual_mapping=None -> dùng MANUAL_DOT_IATA_MAPPING mặc định.
    manual_mapping={}   -> KHÔNG dùng mapping thủ công nào (khác với None).
    """
    spark = _session_of(flights_df)
    manual_mapping = MANUAL_DOT_IATA_MAPPING if manual_mapping is None else manual_mapping

    origin = flights_df.select(
        F.col("ORIGIN_AIRPORT").cast("string").alias("AIRPORT_CODE")
    ).filter(F.col("AIRPORT_CODE").isNotNull())
    dest = flights_df.select(
        F.col("DESTINATION_AIRPORT").cast("string").alias("AIRPORT_CODE")
    ).filter(F.col("AIRPORT_CODE").isNotNull())
    codes = origin.unionByName(dest).distinct()

    airports = (
        airports_df.select(
            F.col("IATA_CODE").cast("string").alias("IATA_CODE"),
            "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE",
        )
        .filter(F.col("IATA_CODE").isNotNull())
        .withColumn(
            "AIRPORT_FIRST_WORD",
            F.regexp_extract(F.trim(F.col("AIRPORT")), r"^([A-Za-z]+)", 1),
        )
        .dropDuplicates(["IATA_CODE"])
    )

    OUT_COLS = ["AIRPORT_CODE", "IATA_CODE", "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE"]

    direct = (
        codes.alias("c")
        .join(airports.alias("a"), F.col("c.AIRPORT_CODE") == F.col("a.IATA_CODE"), "inner")
        .select(F.col("c.AIRPORT_CODE"), *[F.col(f"a.{c}") for c in OUT_COLS[1:]])
    )

    non_iata = codes.alias("c").join(
        airports.alias("a"), F.col("c.AIRPORT_CODE") == F.col("a.IATA_CODE"), "left_anti"
    )

    # Bảng mô tả DOT: ưu tiên airport_id_df, bổ sung airport_ref_df cho code còn thiếu.
    id_desc = _normalize_airport_descriptions(airport_id_df)
    ref_desc = _normalize_airport_descriptions(airport_ref_df)
    if id_desc is None:
        descriptions = ref_desc
    elif ref_desc is None:
        descriptions = id_desc
    else:
        extra = ref_desc.join(id_desc.select("AIRPORT_CODE"), "AIRPORT_CODE", "left_anti")
        descriptions = id_desc.unionByName(extra)

    if descriptions is None:
        dot = non_iata.select(
            "AIRPORT_CODE",
            F.lit(None).cast("string").alias("Description"),
            F.lit("").alias("DOT_FIRST_WORD"),
            F.lit("").alias("STATE_CODE"),
            F.lit("").alias("AFTER_COLON_FIRST_WORD"),
        )
    else:
        descriptions = descriptions.dropDuplicates(["AIRPORT_CODE"])
        dot = (
            non_iata.join(descriptions, "AIRPORT_CODE", "left")
            # LƯU Ý: các regex dưới đây PHẢI là raw-string 1 lớp backslash
            # (r"\s" trong raw-string = literal "\" + "s", KHÔNG phải whitespace
            # -- đây chính là bug khiến STATE_CODE/AFTER_COLON luôn rỗng).
            .withColumn("DOT_FIRST_WORD", F.trim(F.regexp_extract(F.col("Description"), r"^([^,]+)", 1)))
            .withColumn("STATE_CODE", F.regexp_extract(F.col("Description"), r",\s*([A-Za-z]{2})\s*:", 1))
            .withColumn("AFTER_COLON_FIRST_WORD", F.regexp_extract(F.col("Description"), r":\s*([A-Za-z]+)", 1))
        )

    def match(src, cond):
        return (
            src.alias("u")
            .join(airports.alias("a"), cond, "inner")
            .select(
                F.col("u.AIRPORT_CODE"),
                F.col("a.IATA_CODE"), F.col("a.AIRPORT"), F.col("a.CITY"),
                F.col("a.STATE"), F.col("a.LATITUDE"), F.col("a.LONGITUDE"),
            )
            .dropDuplicates(["AIRPORT_CODE"])
        )

    same_state = F.upper(F.col("u.STATE_CODE")) == F.upper(F.col("a.STATE"))
    same_city = F.lower(F.col("u.DOT_FIRST_WORD")) == F.lower(F.col("a.CITY"))
    same_airport = F.lower(F.col("u.AFTER_COLON_FIRST_WORD")) == F.lower(F.col("a.AIRPORT_FIRST_WORD"))

    m1 = match(dot, same_state & same_city & same_airport)
    resolved_1 = [r["AIRPORT_CODE"] for r in m1.select("AIRPORT_CODE").distinct().collect()]
    u1 = dot.filter(~F.col("AIRPORT_CODE").isin(resolved_1)) if resolved_1 else dot

    m2 = match(u1, same_state & same_city)
    resolved_2 = [r["AIRPORT_CODE"] for r in m2.select("AIRPORT_CODE").distinct().collect()]
    u2 = u1.filter(~F.col("AIRPORT_CODE").isin(resolved_2)) if resolved_2 else u1

    m3 = match(u2, same_state & same_airport)
    resolved_3 = [r["AIRPORT_CODE"] for r in m3.select("AIRPORT_CODE").distinct().collect()]
    u3 = u2.filter(~F.col("AIRPORT_CODE").isin(resolved_3)) if resolved_3 else u2

    # LƯU Ý (bản vá quan trọng): 3 dòng "loại trừ" ở trên TRƯỚC ĐÂY dùng
    # left_anti join giữa 1 DataFrame và chính derivative của nó
    # (u1 = dot.join(m1...,"left_anti")) -- đây là self-join xuyên qua nhiều
    # tầng cùng chia sẻ 1 lineage gốc (dot), lại cộng thêm việc tái dùng
    # cùng chuỗi alias "u"/"a" ở nhiều nhánh join lồng nhau (4 lần gọi
    # match() đều alias("a") trên CHÍNH 1 DataFrame airports). Đây là pattern
    # nhạy cảm với phiên bản Spark/JVM: có thể chạy đúng ở bản Spark này
    # nhưng ném Py4JJavaError lúc collect() ở bản khác (ghi nhận thực tế:
    # test_mapping_manual_fallback_is_used fail trên máy khác, pass ở đây).
    # Số mã sân bay (không phải số dòng chuyến bay) luôn nhỏ (vài trăm/vài
    # nghìn), nên collect() về driver rồi lọc bằng isin() là AN TOÀN và loại
    # bỏ hoàn toàn rủi ro self-join, bất kể phiên bản Spark.

    manual_schema = "AIRPORT_CODE string, IATA_CODE string"
    manual_rows = [(str(k), str(v)) for k, v in (manual_mapping or {}).items()]
    manual_df = spark.createDataFrame(manual_rows, manual_schema)

    # Chỉ giữ mã THỰC SỰ chưa được cấp trước (level 1/2/3) giải quyết --
    # cũng dùng isin() với danh sách đã collect thay vì join lại với u3,
    # cùng lý do như trên.
    unresolved_codes = [r["AIRPORT_CODE"] for r in u3.select("AIRPORT_CODE").distinct().collect()]
    manual_candidates = (
        manual_df.filter(F.col("AIRPORT_CODE").isin(unresolved_codes))
        if unresolved_codes
        else manual_df.limit(0)
    )

    # Join với airports KHÔNG dùng alias string ("a"/"m") -- đổi tên cột
    # bằng prefix "__" duy nhất, cùng pattern đã dùng để sửa bug ambiguous
    # column ở add_haversine_distance_feature (Section 1). manual_candidates
    # và airports_for_manual không chia sẻ lineage nào với nhau nên đây
    # KHÔNG phải self-join -- hoàn toàn an toàn.
    airports_for_manual = airports.select(
        F.col("IATA_CODE").alias("__manual_iata"),
        F.col("AIRPORT").alias("__manual_airport"),
        F.col("CITY").alias("__manual_city"),
        F.col("STATE").alias("__manual_state"),
        F.col("LATITUDE").alias("__manual_lat"),
        F.col("LONGITUDE").alias("__manual_lon"),
    )
    manual = (
        manual_candidates
        .join(
            airports_for_manual,
            F.col("IATA_CODE") == F.col("__manual_iata"),
            "left",
        )
        .select(
            F.col("AIRPORT_CODE"),
            F.col("IATA_CODE"),
            F.col("__manual_airport").alias("AIRPORT"),
            F.col("__manual_city").alias("CITY"),
            F.col("__manual_state").alias("STATE"),
            F.col("__manual_lat").alias("LATITUDE"),
            F.col("__manual_lon").alias("LONGITUDE"),
        )
    )

    return (
        direct.select(OUT_COLS)
        .unionByName(m1.select(OUT_COLS))
        .unionByName(m2.select(OUT_COLS))
        .unionByName(m3.select(OUT_COLS))
        .unionByName(manual.select(OUT_COLS))
    )


def patch_missing_airport_coordinates(airports_df):
    """
    Điền lat/long cho các sân bay thiếu tọa độ trong bảng airports gốc,
    dùng MANUAL_AIRPORT_COORDS đã tra cứu thủ công. Không ảnh hưởng các
    sân bay khác đã có sẵn tọa độ.
    """
    result = airports_df
    for iata_code, (lat, lon) in MANUAL_AIRPORT_COORDS.items():
        result = result.withColumn(
            "LATITUDE",
            F.when(
                (F.col("IATA_CODE") == iata_code) & F.col("LATITUDE").isNull(),
                F.lit(lat),
            ).otherwise(F.col("LATITUDE")),
        ).withColumn(
            "LONGITUDE",
            F.when(
                (F.col("IATA_CODE") == iata_code) & F.col("LONGITUDE").isNull(),
                F.lit(lon),
            ).otherwise(F.col("LONGITUDE")),
        )
    return result


@F.pandas_udf(DoubleType())
def haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """
    Tính khoảng cách Haversine (km) giữa hai điểm tọa độ, vectorized bằng
    NumPy trên từng Arrow batch — không loop từng dòng.

    Trả về NaN nếu bất kỳ tọa độ nào null hoặc nằm ngoài phạm vi hợp lệ
    (lat trong [-90, 90], lon trong [-180, 180]).
    """
    R = 6371.0088  # bán kính Trái Đất trung bình, km

    lat1_arr = lat1.to_numpy(dtype="float64", na_value=np.nan)
    lon1_arr = lon1.to_numpy(dtype="float64", na_value=np.nan)
    lat2_arr = lat2.to_numpy(dtype="float64", na_value=np.nan)
    lon2_arr = lon2.to_numpy(dtype="float64", na_value=np.nan)

    # Đánh dấu tọa độ không hợp lệ (ngoài domain) thành NaN trước khi tính
    def invalid_lat(a):
        return (a < -90) | (a > 90)

    def invalid_lon(a):
        return (a < -180) | (a > 180)

    invalid_mask = (
        invalid_lat(lat1_arr)
        | invalid_lat(lat2_arr)
        | invalid_lon(lon1_arr)
        | invalid_lon(lon2_arr)
    )

    lat1_rad = np.radians(lat1_arr)
    lat2_rad = np.radians(lat2_arr)
    dlat = np.radians(lat2_arr - lat1_arr)
    dlon = np.radians(lon2_arr - lon1_arr)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distance = R * c

    distance = np.where(invalid_mask, np.nan, distance)
    return pd.Series(distance)


def add_haversine_distance_feature(flights_df, airports_df, airport_mapping_df=None):
    """
    Join tọa độ sân bay đi/đến, tính Haversine, fallback về DISTANCE gốc khi
    thiếu tọa độ (chủ yếu là các dòng dùng mã DOT numeric — vd. MONTH=10).

    Input:
        flights_df   : DataFrame eligible flights (đã lọc theo eligibility filter)
        airports_df  : DataFrame airports (IATA_CODE, LATITUDE, LONGITUDE, ...)

    Output:
        DataFrame gốc + 3 cột mới:
            - haversine_distance_km : khoảng cách tính được (NULL nếu fallback)
            - route_distance_km     : DISTANCE gốc quy đổi ra km để đồng nhất đơn vị
                                       (DISTANCE gốc trong dataset là dặm -> *1.60934)
            - has_haversine_distance: boolean, True nếu tính được Haversine
    """
    airports_df = patch_missing_airport_coordinates(airports_df)

    # Tên cột phụ trợ đặt tiền tố "__" và DUY NHẤT cho từng vế join -- nếu
    # đặt trùng tên với cột gốc (ORIGIN_AIRPORT/DESTINATION_AIRPORT) thì Spark
    # báo AMBIGUOUS_REFERENCE ngay khi join, đây là lỗi cũ của hàm này.
    origin_coords = airports_df.select(
        F.col("IATA_CODE").cast("string").alias("__origin_iata_ref"),
        F.col("LATITUDE").cast("double").alias("origin_lat"),
        F.col("LONGITUDE").cast("double").alias("origin_lon"),
    ).dropDuplicates(["__origin_iata_ref"])
    dest_coords = airports_df.select(
        F.col("IATA_CODE").cast("string").alias("__dest_iata_ref"),
        F.col("LATITUDE").cast("double").alias("dest_lat"),
        F.col("LONGITUDE").cast("double").alias("dest_lon"),
    ).dropDuplicates(["__dest_iata_ref"])

    flights_for_join = (
        flights_df
        .withColumn("__origin_code", F.col("ORIGIN_AIRPORT").cast("string"))
        .withColumn("__destination_code", F.col("DESTINATION_AIRPORT").cast("string"))
    )

    if airport_mapping_df is None:
        flights_for_join = (
            flights_for_join
            .withColumn("__origin_iata", F.col("__origin_code"))
            .withColumn("__destination_iata", F.col("__destination_code"))
        )
    else:
        origin_map = airport_mapping_df.select(
            F.col("AIRPORT_CODE").cast("string").alias("__om_code"),
            F.col("IATA_CODE").cast("string").alias("__om_iata"),
        ).dropDuplicates(["__om_code"])
        dest_map = airport_mapping_df.select(
            F.col("AIRPORT_CODE").cast("string").alias("__dm_code"),
            F.col("IATA_CODE").cast("string").alias("__dm_iata"),
        ).dropDuplicates(["__dm_code"])

        flights_for_join = (
            flights_for_join
            .join(origin_map, F.col("__origin_code") == F.col("__om_code"), "left")
            .withColumn("__origin_iata", F.coalesce(F.col("__om_iata"), F.col("__origin_code")))
            .drop("__om_code", "__om_iata")
            .join(dest_map, F.col("__destination_code") == F.col("__dm_code"), "left")
            .withColumn("__destination_iata", F.coalesce(F.col("__dm_iata"), F.col("__destination_code")))
            .drop("__dm_code", "__dm_iata")
        )

    joined = (
        flights_for_join
        .join(origin_coords, F.col("__origin_iata") == F.col("__origin_iata_ref"), "left")
        .join(dest_coords, F.col("__destination_iata") == F.col("__dest_iata_ref"), "left")
        .drop(
            "__origin_code", "__destination_code",
            "__origin_iata", "__destination_iata",
            "__origin_iata_ref", "__dest_iata_ref",
        )
    )

    joined = joined.withColumn(
        "haversine_distance_km",
        haversine_km(
            F.col("origin_lat"), F.col("origin_lon"), F.col("dest_lat"), F.col("dest_lon")
        ),
    )

    MILES_TO_KM = 1.60934
    joined = joined.withColumn(
        "route_distance_km", F.col("DISTANCE") * F.lit(MILES_TO_KM)
    )

    joined = joined.withColumn(
        "has_haversine_distance", F.col("haversine_distance_km").isNotNull()
    )

    # Feature cuối cùng dùng cho model: Haversine khi có, fallback route_distance_km
    joined = joined.withColumn(
        "distance_feature_km",
        F.coalesce(F.col("haversine_distance_km"), F.col("route_distance_km")),
    )

    return joined.drop("origin_lat", "origin_lon", "dest_lat", "dest_lon")


# -----------------------------------------------------------------------------
# Quick sanity check khi chạy trực tiếp file này (không phải unit test chính
# thức — unit test đầy đủ nằm ở tests/test_custom_transformers.py)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    from pyspark.sql import SparkSession

    # BẮT BUỘC set TRƯỚC khi tạo SparkSession: pandas_udf (haversine_km) cần
    # spawn Python worker process riêng cho từng partition. Không set biến
    # này, Spark mặc định gọi lệnh "python" trên PATH -- trên Windows, nếu
    # không có bản Python thật đứng trước App Execution Alias, hệ thống chặn
    # lệnh và in "Python was not found; run without arguments to install..."
    # thay vì chạy worker thật. sys.executable luôn là đường dẫn tuyệt đối
    # tới đúng bản Python đang chạy script này, nên luôn đúng trên mọi OS.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.appName("custom_transformers_sanity_check")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )

    flights = spark.read.parquet("data/parquet/flights")
    airports = spark.read.parquet("data/parquet/airports")

    eligible = flights.filter(
        (F.col("ARRIVAL_DELAY").isNotNull())
        & (F.col("CANCELLED") == 0)
        & (F.col("DIVERTED") == 0)
    )

    result = add_haversine_distance_feature(eligible, airports)

    print("Phân bố has_haversine_distance theo tháng:")
    result.groupBy("MONTH", "has_haversine_distance").count().orderBy(
        "MONTH", "has_haversine_distance"
    ).show(24)

    print("So sánh haversine_distance_km vs DISTANCE gốc (mẫu 10 dòng có Haversine):")
    result.filter(F.col("has_haversine_distance")).select(
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "DISTANCE",
        "route_distance_km",
        "haversine_distance_km",
    ).show(10, truncate=False)


# =============================================================================
# SECTION 2 — OutlierIQRTruncator (Estimator/Model pair)
# =============================================================================
#
# LƯU Ý QUAN TRỌNG (đã cập nhật sau review của modeling team):
#   - Class này KHÔNG hardcode giả định về việc cột nào được phép dùng làm
#     feature (VD DEPARTURE_DELAY có hợp lệ hay không hiện đang được nhóm
#     tranh luận lại về inference contract) -- quyết định "cột nào đưa vào"
#     thuộc về feature_dictionary.md, không phải trách nhiệm của transformer
#     generic này. Transformer chỉ nhận inputCols theo tham số và xử lý
#     đúng thống kê, bất kể ý nghĩa nghiệp vụ của cột đó.
#   - KHÔNG áp dụng lên ARRIVAL_DELAY (target): clip trực tiếp label sẽ làm
#     sai lệch định nghĩa bài toán regression -- đây là ràng buộc CỐ ĐỊNH,
#     không phụ thuộc vào tranh luận inference contract.
#
# QUYẾT ĐỊNH MỚI (review modeling team, 2026):
#   - Giá trị đã CLIP (distance_feature_km_clipped, scheduled_time_clipped)
#     VẪN được tính (đáp ứng yêu cầu đề bài, Task 1 rubric) nhưng KHÔNG đưa
#     vào feature set training -- lý do: (1) tree-based model (RF/GBT) đã
#     tự robust với outlier, clip không giúp ích; (2) clip làm mất thông tin
#     thật (route dài thật không phải noise), khó vận hành thực tế.
#   - Thay vào đó, thêm BINARY FLAG (outlierFlagCols) đánh dấu quan sát nằm
#     ngoài [Q1, Q3] (KHÔNG nhân IQR multiplier) -- flag này MỚI là feature
#     được đưa vào training, giữ nguyên giá trị gốc, chỉ thêm tín hiệu
#     "quan sát này có bất thường hay không".
#
# Thiết kế Estimator/Model (không subclass Transformer trực tiếp như mô tả
# ngắn gọn trong đề) vì: bounds (Q1, Q3, IQR) BẮT BUỘC chỉ được tính trên
# tập train/fold. Nếu gộp chung fit+transform trong 1 Transformer duy nhất,
# rất dễ vô tình tính lại bounds trên tập test/toàn bộ dataset -> leakage.
# Estimator/Model là pattern chuẩn của Spark ML để tránh đúng lỗi này
# (giống hệt cách StandardScaler/StringIndexer hoạt động trong MLlib).


class HasIQRMultiplier(Params):
    iqrMultiplier = Param(
        Params._dummy(),
        "iqrMultiplier",
        "Hệ số nhân IQR để xác định outlier bounds (mặc định 1.5)",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self):
        super(HasIQRMultiplier, self).__init__()
        self._setDefault(iqrMultiplier=1.5)

    def getIqrMultiplier(self):
        return self.getOrDefault(self.iqrMultiplier)


class HasOutlierFlagCols(Params):
    """Danh sách tên cột binary flag (song song với outputCols) đánh dấu
    quan sát nằm ngoài [Q1, Q3] -- KHÔNG dùng bound đã nhân iqrMultiplier.
    Để trống (danh sách rỗng) nếu không cần flag cho cột đó."""

    outlierFlagCols = Param(
        Params._dummy(),
        "outlierFlagCols",
        "Danh sách tên cột flag binary (song song với outputCols); dùng '' "
        "cho vị trí không cần flag",
        typeConverter=TypeConverters.toListString,
    )

    def getOutlierFlagCols(self):
        return self.getOrDefault(self.outlierFlagCols)


class HasBoundsMap(Params):
    """Lưu toàn bộ IQR bounds dạng JSON string trong 1 Param — bắt buộc để
    tương thích với DefaultParamsWritable/Readable (Spark ML chỉ serialize
    được Param, không serialize được thuộc tính Python tùy ý như
    self.bounds = {...}). Sửa lại theo đúng pattern đã dùng cho
    TargetEncoderModel ở Section 3, sau khi phát hiện lỗi persistence."""

    boundsMapJson = Param(
        Params._dummy(),
        "boundsMapJson",
        "JSON string chứa toàn bộ IQR bounds đã fit",
        typeConverter=TypeConverters.toString,
    )

    def getBoundsMapJson(self):
        return self.getOrDefault(self.boundsMapJson)


class OutlierIQRTruncator(
    Estimator,
    HasInputCols,
    HasOutputCols,
    HasIQRMultiplier,
    HasOutlierFlagCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Estimator: tính Q1, Q3 bằng approxQuantile (phân tán, KHÔNG collect toàn
    bộ dữ liệu về Driver) trên dataset được truyền vào fit(). Trả về
    OutlierIQRTruncatorModel chứa bounds cố định để tái sử dụng.

    fit() CHỈ được gọi trên tập train/fold. Trong CrossValidator, Spark ML
    tự động gọi lại fit() cho từng fold nếu transformer nằm trong Pipeline
    -- đây chính là lý do thiết kế Estimator/Model, không phải chi tiết cần
    tự quản lý thủ công.
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, iqrMultiplier=1.5, outlierFlagCols=None):
        super(OutlierIQRTruncator, self).__init__()
        self._setDefault(iqrMultiplier=1.5, outlierFlagCols=[])
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, iqrMultiplier=1.5, outlierFlagCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()
        multiplier = self.getIqrMultiplier()
        flag_cols = self.getOutlierFlagCols() or []

        if len(input_cols) != len(output_cols):
            raise ValueError("inputCols và outputCols phải cùng độ dài")
        if flag_cols and len(flag_cols) != len(input_cols):
            raise ValueError("outlierFlagCols (nếu dùng) phải cùng độ dài với inputCols")

        bounds = {}
        for idx, (in_col, out_col) in enumerate(zip(input_cols, output_cols)):
            q1, q3 = dataset.approxQuantile(in_col, [0.25, 0.75], relativeError=0.01)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            flag_col = flag_cols[idx] if flag_cols else ""
            bounds[out_col] = {
                "input_col": in_col,
                "q1": q1,
                "q3": q3,
                "lower": lower,
                "upper": upper,
                "flag_col": flag_col,
            }

        bounds_json = json.dumps(bounds)
        return OutlierIQRTruncatorModel(
            inputCols=input_cols, outputCols=output_cols, boundsMapJson=bounds_json
        )


class OutlierIQRTruncatorModel(
    Model, HasInputCols, HasOutputCols, HasBoundsMap, DefaultParamsReadable, DefaultParamsWritable
):
    """
    Model (Transformer thực sự dùng trong Pipeline.transform): áp dụng bounds
    đã tính từ tập train để clip giá trị ngoài [lower, upper] trên bất kỳ
    dataset nào (train, validation, test, streaming) — bounds KHÔNG bao giờ
    được tính lại ở bước này, đảm bảo leak-proof.

    Bounds lưu qua Param JSON (boundsMapJson), không phải thuộc tính Python
    thường -- đảm bảo model.save()/load() hoạt động đúng.

    Ngoài cột đã CLIP (output_col), nếu bounds có ghi flag_col (đặt qua
    outlierFlagCols khi khởi tạo Estimator), Model sẽ tự thêm cột binary
    flag = 1 nếu giá trị gốc nằm ngoài [Q1, Q3] (KHÔNG nhân multiplier),
    = 0 nếu trong khoảng. Flag này là feature ĐƯỢC dùng cho training; cột
    đã clip chỉ giữ lại để đáp ứng yêu cầu đề bài, KHÔNG đưa vào training
    (xem quyết định ở đầu Section 2 và feature_dictionary.md).
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, boundsMapJson=None):
        super(OutlierIQRTruncatorModel, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        bounds = json.loads(self.getBoundsMapJson())
        result = dataset
        for out_col, info in bounds.items():
            in_col = info["input_col"]
            lower = info["lower"]
            upper = info["upper"]
            result = result.withColumn(
                out_col,
                F.when(F.col(in_col) < F.lit(lower), F.lit(lower))
                .when(F.col(in_col) > F.lit(upper), F.lit(upper))
                .otherwise(F.col(in_col)),
            )
            flag_col = info.get("flag_col", "")
            if flag_col:
                q1, q3 = info["q1"], info["q3"]
                result = result.withColumn(
                    flag_col,
                    F.when(
                        (F.col(in_col) < F.lit(q1)) | (F.col(in_col) > F.lit(q3)), F.lit(1)
                    ).otherwise(F.lit(0)),
                )
        return result

    def bounds_summary(self) -> pd.DataFrame:
        """Tóm tắt bounds dạng bảng nhỏ — dùng để log vào MLflow/REPORT.md,
        hoặc trình bày trực tiếp khi demo (đúng yêu cầu 'trực tiếp, dễ hình dung')."""
        bounds = json.loads(self.getBoundsMapJson())
        rows = [
            {
                "output_col": out_col,
                "input_col": info["input_col"],
                "q1": info["q1"],
                "q3": info["q3"],
                "lower_bound": info["lower"],
                "upper_bound": info["upper"],
                "flag_col": info.get("flag_col", ""),
            }
            for out_col, info in bounds.items()
        ]
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Sanity check cho Section 2
# -----------------------------------------------------------------------------
def _section2_sanity_check(spark):
    flights = spark.read.parquet("data/parquet/flights")
    airports = spark.read.parquet("data/parquet/airports")

    eligible = flights.filter(
        (F.col("ARRIVAL_DELAY").isNotNull())
        & (F.col("CANCELLED") == 0)
        & (F.col("DIVERTED") == 0)
    )
    with_distance = add_haversine_distance_feature(eligible, airports)

    # Giả lập split train/test đơn giản để CHỨNG MINH fit chỉ dùng train
    train_df, test_df = with_distance.randomSplit([0.8, 0.2], seed=42)

    truncator = OutlierIQRTruncator(
        inputCols=["distance_feature_km", "SCHEDULED_TIME"],
        outputCols=["distance_feature_km_clipped", "scheduled_time_clipped"],
        iqrMultiplier=1.5,
    )
    model = truncator.fit(train_df)  # fit CHỈ trên train

    print("Bounds tính được (chỉ từ train):")
    print(model.bounds_summary().to_string(index=False))

    train_clipped = model.transform(train_df)
    test_clipped = model.transform(test_df)  # dùng lại bounds của train, không fit lại

    for out_col in ["distance_feature_km_clipped", "scheduled_time_clipped"]:
        n_clipped_train = train_clipped.filter(
            F.col(out_col) != F.col(
                dict(
                    distance_feature_km_clipped="distance_feature_km",
                    scheduled_time_clipped="SCHEDULED_TIME",
                )[out_col]
            )
        ).count()
        print(f"Số dòng bị clip trong train ở {out_col}: {n_clipped_train:,}")

    return train_clipped, test_clipped


if __name__ == "__main__" and False:
    # Bật lại nếu muốn chạy riêng Section 2 sanity check bằng:
    #   python -c "from custom_transformers import _section2_sanity_check; ..."
    pass


# =============================================================================
# SECTION 3 — Leakage-safe Target Encoder (Estimator/Model pair)
# =============================================================================
#
# Áp dụng cho 4 cột cardinality cao: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT,
# TAIL_NUMBER (đã quyết định giữ TAIL_NUMBER — xem feature_dictionary.md).
#
# THIẾT KẾ QUAN TRỌNG #1 — Tránh leakage giữa các fold của CrossValidator:
#   Giống OutlierIQRTruncator, dùng Estimator/Model pair. Khi đặt trong 1
#   Pipeline và chạy qua CrossValidator, Spark ML TỰ ĐỘNG gọi lại _fit()
#   cho từng fold train riêng biệt -- đây là lý do bắt buộc phải tách
#   Estimator/Model, không được gộp fit+transform trong 1 Transformer.
#
# THIẾT KẾ QUAN TRỌNG #2 — Smoothing cho category tần suất thấp:
#   Nếu chỉ dùng mean thô của target theo từng category, những category
#   xuất hiện rất ít lần trong tập fit (vd 1 tail number chỉ bay 2 chuyến)
#   sẽ cho encoding value cực đoan, dễ overfit. Dùng công thức m-estimate:
#
#       encoded = (n_category * mean_category + m * mean_global)
#                 / (n_category + m)
#
#   m càng lớn thì càng "kéo" encoding của category ít dữ liệu về gần
#   mean_global hơn -- category có nhiều dữ liệu (n_category lớn) gần như
#   không bị ảnh hưởng bởi smoothing.
#
# THIẾT KẾ QUAN TRỌNG #3 — Serialization đúng cách (khác với lỗi ở Section 2):
#   Toàn bộ encoding map được lưu dưới dạng 1 JSON string trong 1 Param
#   (encodingMapJson), KHÔNG lưu như thuộc tính Python thường. Điều này
#   đảm bảo model.save()/load() hoạt động đúng khi tích hợp vào PipelineModel
#   ở Production phase.
#
# LƯU Ý VỀ GIỚI HẠN: đây là target encoding "fit-on-train-fold" tiêu chuẩn
# (tránh leakage GIỮA các fold train/test) -- không phải kỹ thuật nested/
# out-of-fold encoding NGAY TRONG tập train (kỹ thuật nâng cao hơn, dùng
# thêm 1 lớp K-fold nội bộ để mỗi dòng train không dùng chính target của nó
# để tính encoding của chính nó). Với quy mô đồ án này, fit-on-train-fold
# là mức đủ để giải quyết đúng rủi ro leakage được đề cập trong đề bài;
# ghi rõ giới hạn này trong REPORT.md để thể hiện hiểu bản chất.


class HasTargetCol(Params):
    targetCol = Param(
        Params._dummy(),
        "targetCol",
        "Tên cột target dùng để tính mean encoding",
        typeConverter=TypeConverters.toString,
    )

    def getTargetCol(self):
        return self.getOrDefault(self.targetCol)


class HasSmoothing(Params):
    smoothing = Param(
        Params._dummy(),
        "smoothing",
        "Hệ số smoothing (m-estimate) — category càng ít dữ liệu càng bị "
        "kéo về gần global mean",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self):
        super(HasSmoothing, self).__init__()
        self._setDefault(smoothing=10.0)

    def getSmoothing(self):
        return self.getOrDefault(self.smoothing)


class HasEncodingMap(Params):
    """Lưu toàn bộ encoding map dạng JSON string trong 1 Param — bắt buộc để
    tương thích với DefaultParamsWritable/Readable (Spark ML chỉ serialize
    được Param, không serialize được thuộc tính Python tùy ý)."""

    encodingMapJson = Param(
        Params._dummy(),
        "encodingMapJson",
        "JSON string chứa toàn bộ encoding map đã fit",
        typeConverter=TypeConverters.toString,
    )

    def getEncodingMapJson(self):
        return self.getOrDefault(self.encodingMapJson)


class TargetEncoder(
    Estimator,
    HasInputCols,
    HasOutputCols,
    HasTargetCol,
    HasSmoothing,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Estimator: với mỗi cột trong inputCols, tính mean(target) theo từng
    category trên dataset được fit (train/fold), áp dụng m-estimate
    smoothing, và lưu toàn bộ map vào TargetEncoderModel.

    fit() CHỈ được gọi trên tập train/fold — giống nguyên tắc của
    OutlierIQRTruncator.
    """

    @keyword_only
    def __init__(
        self, inputCols=None, outputCols=None, targetCol=None, smoothing=10.0
    ):
        super(TargetEncoder, self).__init__()
        self._setDefault(smoothing=10.0)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(
        self, inputCols=None, outputCols=None, targetCol=None, smoothing=10.0
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()
        target_col = self.getTargetCol()
        smoothing = self.getSmoothing()

        if len(input_cols) != len(output_cols):
            raise ValueError("inputCols và outputCols phải cùng độ dài")

        columns_map = {}
        for in_col, out_col in zip(input_cols, output_cols):
            global_mean_row = dataset.agg(F.mean(F.col(target_col))).collect()[0]
            global_mean = float(global_mean_row[0])

            stats = (
                dataset.groupBy(in_col)
                .agg(
                    F.mean(F.col(target_col)).alias("cat_mean"),
                    F.count(F.col(target_col)).alias("cat_count"),
                )
                .withColumn(
                    "smoothed",
                    (
                        F.col("cat_count") * F.col("cat_mean")
                        + F.lit(smoothing) * F.lit(global_mean)
                    )
                    / (F.col("cat_count") + F.lit(smoothing)),
                )
            )

            # Cardinality của 4 cột này đều ở mức nghìn (tối đa ~4,896 cho
            # TAIL_NUMBER) -> an toàn để collect về Driver, lưu thành dict.
            category_map = {
                str(row[in_col]): float(row["smoothed"])
                for row in stats.select(in_col, "smoothed").collect()
                if row[in_col] is not None
            }

            columns_map[out_col] = {
                "input_col": in_col,
                "global_mean": global_mean,
                "map": category_map,
            }

        encoding_json = json.dumps(columns_map)
        return TargetEncoderModel(
            inputCols=input_cols, outputCols=output_cols, encodingMapJson=encoding_json
        )


class TargetEncoderModel(
    Model, HasInputCols, HasOutputCols, HasEncodingMap, DefaultParamsReadable, DefaultParamsWritable
):
    """
    Model: áp dụng encoding map đã fit từ train để join vào bất kỳ dataset
    nào (train, validation, test, streaming). Category không xuất hiện
    trong tập fit (unseen category) -> fallback về global_mean của cột đó,
    KHÔNG lỗi, KHÔNG null.
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, encodingMapJson=None):
        super(TargetEncoderModel, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        spark = _session_of(dataset)
        columns_map = json.loads(self.getEncodingMapJson())

        result = dataset
        for out_col, info in columns_map.items():
            in_col = info["input_col"]
            global_mean = info["global_mean"]
            category_map = info["map"]

            if len(category_map) == 0:
                result = result.withColumn(out_col, F.lit(global_mean))
                continue

            mapping_rows = [(cat, val) for cat, val in category_map.items()]
            mapping_df = spark.createDataFrame(mapping_rows, [in_col, out_col])

            # Cast in_col của dataset chính sang string để join khớp kiểu
            # với key trong category_map (đã ép str khi fit).
            result = result.withColumn(f"__{in_col}_str", F.col(in_col).cast("string"))
            mapping_df = mapping_df.withColumnRenamed(in_col, f"__{in_col}_str")

            result = result.join(F.broadcast(mapping_df), on=f"__{in_col}_str", how="left")
            result = result.withColumn(
                out_col, F.coalesce(F.col(out_col), F.lit(global_mean))
            ).drop(f"__{in_col}_str")

        return result

    def encoding_summary(self, output_col: str, top_n: int = 5) -> pd.DataFrame:
        """Tóm tắt vài category tiêu biểu — dùng để demo trực tiếp
        ('trực tiếp, dễ hình dung' theo yêu cầu giáo viên): cho xem cụ thể
        1 vài sân bay/hãng bay có encoding value bao nhiêu."""
        columns_map = json.loads(self.getEncodingMapJson())
        info = columns_map[output_col]
        rows = sorted(info["map"].items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        df = pd.DataFrame(rows, columns=["category", "encoded_value"])
        df["global_mean"] = info["global_mean"]
        return df


# =============================================================================
# SECTION 4 (bổ sung, không thuộc 3 hạng mục chính của rubric Task 1)
# Feature cơ bản khác: giờ trong ngày, route congestion
# =============================================================================
#
# 2 feature này KHÔNG cần Estimator/Model như Section 2-3, vì:
#   - Không dùng target (ARRIVAL_DELAY) để tính -> không có rủi ro
#     target-leakage kiểu CrossValidator (không cần fit riêng theo fold).
#   - Là thông tin thuộc về lịch bay, biết trước departure -> hợp lệ theo
#     inference contract đã chốt trong data_dictionary.md.
#   - DAY_OF_WEEK đã có sẵn trong dataset gốc, KHÔNG cần tự tạo lại.


def add_scheduled_hour_feature(
    flights_df, time_col: str = "SCHEDULED_DEPARTURE", output_col: str = "scheduled_hour"
):
    """
    Trích giờ trong ngày (0-23) từ cột thời gian dạng HHMM (string), theo
    đúng convention đã ghi trong data_dictionary.md: "2400" là giá trị hợp
    lệ, quy về giờ 0 (nửa đêm) thay vì lỗi parse hoặc NULL.
    """
    padded = F.lpad(F.col(time_col), 4, "0")
    hour_int = F.substring(padded, 1, 2).cast("int")
    normalized_hour = F.when(hour_int == 24, F.lit(0)).otherwise(hour_int)
    return flights_df.withColumn(output_col, normalized_hour)


def add_route_congestion_feature(
    flights_df,
    airport_col: str = "ORIGIN_AIRPORT",
    hour_col: str = "scheduled_hour",
    output_col: str = "route_congestion_count",
    year_col: str = "YEAR",
):
    """
    Đếm số chuyến bay được lên lịch tại cùng sân bay đi, cùng NĂM, cùng ngày,
    cùng khung giờ (scheduled_hour) -- proxy cho mức độ tắc nghẽn lịch bay.
    Đây là con số tính từ chính lịch bay (không phải outcome thực tế), nên
    hợp lệ để tính trên toàn bộ dataset mà không cần tách train/fold riêng
    -- khác về bản chất với OutlierIQRTruncator/TargetEncoder (2 cái đó
    học từ phân phối của TARGET, nên bắt buộc chỉ fit trên train).

    LƯU Ý (đã sửa, xem feature_dictionary.md mục "Vấn đề còn treo" #1):
    group key BẮT BUỘC có year_col. Thiếu YEAR sẽ gộp nhầm ngày 15/1/2015
    với ngày 15/1/2016 (và mọi năm khác trùng MONTH/DAY/giờ) thành 1 nhóm
    -- vô hại với dataset 1 năm nhưng SAI ngay khi mở rộng nhiều năm.
    year_col=None để tắt hành vi này (khôi phục behaviour cũ, không khuyến
    khích) nếu chắc chắn dataset chỉ có đúng 1 năm.

    yêu cầu: flights_df đã có cột `hour_col` (chạy add_scheduled_hour_feature
    trước khi gọi hàm này).
    """
    window_cols = [airport_col, "MONTH", "DAY", hour_col]
    if year_col is not None:
        window_cols = [airport_col, year_col, "MONTH", "DAY", hour_col]
    congestion = flights_df.groupBy(*window_cols).agg(
        F.count(F.lit(1)).alias(output_col)
    )
    return flights_df.join(congestion, on=window_cols, how="left")


# =============================================================================
# SECTION 5 — Feature theo 5 cấp độ (Airline / Airport / Airplane / Flight /
# Temporal-level), theo đề xuất review của modeling team.
# =============================================================================
#
# BỐI CẢNH: modeling team đề xuất tổ chức lại feature theo 5 nhóm thay vì
# rời rạc như Section 1-4. Migrate 2 hàm lag từ feature_derivation_functions.py
# (file KHÔNG được push git theo xác nhận của nhóm) vào đây để không bị mất.
#
# NGUYÊN TẮC LEAK-SAFE CHUNG:
#   - Feature LOOKBACK (lịch sử ARRIVAL_DELAY/DEPARTURE_DELAY của CÁC CHUYẾN
#     KHÁC, không phải chuyến đang xét): cắt theo TIMESTAMP THẬT
#     (NGAY_DL + SCHEDULED_DEPARTURE), KHÔNG chỉ theo ranh giới ngày -- tránh
#     leak nội bộ ngày t (chuyến 8h sáng không được dùng dữ liệu chuyến 8h tối
#     cùng ngày, dù "kỳ 1 ngày" có bao gồm ngày t).
#   - Feature SCHEDULE-BASED (đếm số chuyến theo lịch, không dùng outcome):
#     an toàn dùng dữ liệu CẢ NGÀY (kể cả "sau" thời điểm quan sát), vì lịch
#     bay đã được setting sẵn từ trước -- đây là điểm modeling team lưu ý
#     riêng cho Airport-level.
#   - Feature dùng DEPARTURE_DELAY của CHÍNH chuyến đang xét (Flight-level):
#     CHỈ hợp lệ theo kịch bản "CÓ DÙNG" trong feature_dictionary.md. Nếu
#     nhóm chốt theo kịch bản "KHÔNG DÙNG", phải bỏ các feature này (đã
#     tách riêng thành build_flight_level_features() để dễ loại bỏ).
#
# Kỳ phái sinh (đã chốt wording với nhóm): ngày hiện tại là t; kỳ 1 ngày =
# {t}; kỳ 3 ngày = {t-2,t-1,t}; kỳ 7 ngày = {t-6,...,t}. Về mặt code, điều
# này được thỏa mãn TỰ NHIÊN bằng cách cắt cửa sổ theo epoch-SECONDS (không
# phải epoch-day) và luôn dùng rangeBetween(-lookback_seconds, -1): mọi
# dòng có timestamp SỚM HƠN dòng hiện tại trong vòng N ngày, kể cả cùng
# ngày t, đều được tính -- không cần xử lý "bao gồm ngày t" như 1 case riêng.


def safe_divide(numerator_col, denominator_col):
    """
    Phép chia có thay giá trị đặc biệt (theo đúng rule bảng "Rule Phái Sinh"
    mẫu credit risk mà modeling team gửi, áp dụng nhất quán toàn dự án):
      (1) Nếu tử HOẶC mẫu là NULL -> NULL
      (2) Elif tử == 0 VÀ mẫu == 0 -> 0
      (3) Elif mẫu == 0 VÀ tử > 0 -> 99999.9999
      (4) Elif mẫu == 0 VÀ tử < 0 -> -99999.9999
      (5) Else -> tử / mẫu (phép chia thông thường)
    """
    return (
        F.when(numerator_col.isNull() | denominator_col.isNull(), F.lit(None))
        .when((numerator_col == 0) & (denominator_col == 0), F.lit(0.0))
        .when((denominator_col == 0) & (numerator_col > 0), F.lit(99999.9999))
        .when((denominator_col == 0) & (numerator_col < 0), F.lit(-99999.9999))
        .otherwise(numerator_col / denominator_col)
    )


def add_flight_timestamp_column(
    flights_df,
    date_cols=("YEAR", "MONTH", "DAY"),
    time_col: str = "SCHEDULED_DEPARTURE",
    output_col: str = "flight_timestamp",
):
    """
    Dựng timestamp THẬT (không chỉ ngày) từ YEAR/MONTH/DAY + SCHEDULED_DEPARTURE
    (HHMM), cộng thêm cột epoch-seconds để dùng cho Window.rangeBetween --
    đây là điều kiện bắt buộc để cắt cửa sổ lookback theo timestamp thay vì
    chỉ theo ranh giới ngày (tránh leak nội bộ ngày t).

    Xử lý convention "2400" (data_dictionary.md ghi là hợp lệ) -> quy về
    00:00 NGÀY HÔM SAU (khác với add_scheduled_hour_feature ở Section 4, nơi
    "2400" quy về giờ 0 CÙNG ngày để phục vụ mục đích khác -- ở đây cần đúng
    thứ tự thời gian tuyệt đối nên phải cộng thêm 1 ngày).
    """
    year_col, month_col, day_col = date_cols
    padded_time = F.lpad(F.col(time_col), 4, "0")
    hour = F.substring(padded_time, 1, 2).cast("int")
    minute = F.substring(padded_time, 3, 2).cast("int")

    base_date = F.to_date(F.concat_ws("-", F.col(year_col), F.col(month_col), F.col(day_col)))
    is_2400 = hour == 24

    date_part = F.when(is_2400, F.date_add(base_date, 1)).otherwise(base_date)
    hour_part = F.when(is_2400, F.lit(0)).otherwise(hour)

    timestamp_col = F.to_timestamp(
        F.concat_ws(
            " ",
            F.date_format(date_part, "yyyy-MM-dd"),
            F.format_string("%02d:%02d:00", hour_part, minute),
        )
    )
    df = flights_df.withColumn(output_col, timestamp_col)
    return df.withColumn(f"_{output_col}_epoch", F.col(output_col).cast("long"))


def _hhmm_to_minutes(col):
    """Quy đổi cột dạng HHMM (string) sang số phút tính từ 0h -- dùng cho
    các phép tính số học cần cả 2 vế cùng đơn vị (VD Flight-level, hoặc so
    sánh với ARRIVAL_DELAY tính bằng phút). Xử lý convention '2400' -> 0."""
    padded = F.lpad(col, 4, "0")
    hour = F.substring(padded, 1, 2).cast("int")
    minute = F.substring(padded, 3, 2).cast("int")
    hour = F.when(hour == 24, F.lit(0)).otherwise(hour)
    return hour * 60 + minute


def derive_lookback_agg(
    df,
    partition_cols: list,
    output_col: str,
    lookback_days: float,
    agg_fn,
    target_col: str,
    timestamp_epoch_col: str = "_flight_timestamp_epoch",
):
    """
    Thống kê rolling theo composite key (partition_cols), cắt cửa sổ THEO
    TIMESTAMP THẬT (không phải theo ngày): chỉ dùng các dòng có timestamp
    NHỎ HƠN (strictly before) dòng hiện tại, trong vòng lookback_days ngày
    gần nhất -- point-in-time correct tuyệt đối, kể cả trong cùng ngày t.

    partition_cols : list các cột định nghĩa entity (VD ["AIRLINE"],
                      ["ORIGIN_AIRPORT"], ["TAIL_NUMBER"])
    lookback_days   : số ngày look-back (có thể lẻ, VD 0.5 cho 12 giờ)
    agg_fn          : F.avg, F.max, F.min, F.sum, hoặc lambda c: F.count(c)
    target_col      : cột dùng để tính thống kê (VD ARRIVAL_DELAY, DEPARTURE_DELAY)

    Yêu cầu: df đã có cột timestamp_epoch_col (chạy add_flight_timestamp_column
    trước khi gọi hàm này).
    """
    lookback_seconds = int(lookback_days * 86400)
    window = (
        Window.partitionBy(*partition_cols)
        .orderBy(timestamp_epoch_col)
        .rangeBetween(-lookback_seconds, -1)
    )
    return df.withColumn(output_col, agg_fn(F.col(target_col)).over(window))


def derive_schedule_count(df, group_cols: list, output_col: str):
    """
    Đếm số bản ghi theo composite key thuộc về LỊCH BAY (không phải outcome)
    -- an toàn dùng dữ liệu CẢ NGÀY, kể cả "sau" thời điểm quan sát, vì lịch
    đã setting sẵn từ trước (áp dụng cho Airport-level theo đúng lưu ý của
    modeling team). Khác về bản chất với derive_lookback_agg (không cắt theo
    timestamp, không cần point-in-time).
    """
    counts = df.groupBy(*group_cols).agg(F.count(F.lit(1)).alias(output_col))
    return df.join(counts, on=group_cols, how="left")


# -----------------------------------------------------------------------------
# 5.1 — Airline-level: composite key (AIRLINE, NGAY_DL, SCHEDULED_DEPARTURE)
# Entity lookback thực chất là AIRLINE; composite key mô tả điểm "as-of" (mốc
# thời gian cắt cửa sổ) chính là timestamp của dòng hiện tại.
#
# CẬP NHẬT (chốt với modeling team):
#   - Ngưỡng "late": ARRIVAL_DELAY > 30 (khớp label_severe_delay đã có sẵn
#     trong dataset, thay cho ngưỡng >0 tạm dùng trước đó).
#   - Hỗ trợ NHIỀU cửa sổ lookback cùng lúc (1/3/7 ngày theo đúng kỳ phái
#     sinh đã chốt), mỗi cửa sổ có suffix riêng trong tên cột (VD
#     "airline_late_flights_count_lag1d", "..._lag7d") -- sửa đúng vấn đề
#     "ghi đè cột cũ" đã phát hiện trước đó.
# -----------------------------------------------------------------------------
def _window_suffix(lookback_days) -> str:
    """VD 1 -> '_lag1d', 3 -> '_lag3d', 0.5 -> '_lag0.5d'"""
    n = int(lookback_days) if float(lookback_days).is_integer() else lookback_days
    return f"_lag{n}d"


LATE_ARRIVAL_THRESHOLD_MINUTES = 30  # đã chốt với modeling team, khớp label_severe_delay


def build_airline_level_features(df, lookback_windows=(1, 3, 7), late_threshold=LATE_ARRIVAL_THRESHOLD_MINUTES):
    """
    lookback_windows : tuple/list các cửa sổ look-back (ngày), mỗi cửa sổ sinh
                        ra 1 bộ cột riêng, phân biệt bằng suffix (VD _lag1d).
    late_threshold    : ngưỡng ARRIVAL_DELAY để tính là "late" (mặc định 30,
                         đã chốt với modeling team).
    """
    for w in lookback_windows:
        suf = _window_suffix(w)
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_late_flights_count{suf}", w,
            agg_fn=lambda c: F.sum(F.when(c > late_threshold, 1).otherwise(0)), target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_total_delay_minutes{suf}", w,
            agg_fn=F.sum, target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_flights_count{suf}", w,
            agg_fn=lambda c: F.count(c), target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["AIRLINE"], f"airline_distinct_tails_late{suf}", w,
            agg_fn=lambda c: F.approx_count_distinct(F.when(c > late_threshold, F.col("TAIL_NUMBER"))),
            target_col="ARRIVAL_DELAY",
        )
        df = df.withColumn(
            f"airline_late_flight_ratio{suf}",
            safe_divide(F.col(f"airline_late_flights_count{suf}"), F.col(f"airline_flights_count{suf}")),
        )
        df = df.withColumn(
            f"airline_delay_per_tail{suf}",
            safe_divide(F.col(f"airline_total_delay_minutes{suf}"), F.col(f"airline_distinct_tails_late{suf}")),
        )
    return df


# -----------------------------------------------------------------------------
# 5.2 — Airport-level: composite key (ORIGIN_AIRPORT, NGAY_DL, SCHEDULED_DEPARTURE)
# Có 2 loại: (a) schedule-based (an toàn dùng cả ngày, không cần multi-window
# vì luôn là "trong ngày"), (b) lookback (delay lịch sử).
#
# CẬP NHẬT: kỳ phái sinh đổi sang GIỜ thay vì ngày (theo yêu cầu modeling
# team) -- disruption ở cấp sân bay (thời tiết, ATC) thường là hiện tượng
# ngắn hạn trong vài giờ, không cần nhìn xa như cấp Airline/Airplane.
# -----------------------------------------------------------------------------
def _window_suffix_hours(hours) -> str:
    """VD 3 -> '_lag3h', 12 -> '_lag12h'"""
    n = int(hours) if float(hours).is_integer() else hours
    return f"_lag{n}h"


def build_airport_level_features(df, lookback_windows_hours=(3, 6, 12), late_threshold=LATE_ARRIVAL_THRESHOLD_MINUTES):
    """
    lookback_windows_hours : tuple các cửa sổ look-back tính bằng GIỜ (mặc
                              định 3/6/12h). Quy đổi sang lookback_days =
                              hours/24 khi gọi derive_lookback_agg (hàm nền
                              tảng vẫn tính theo ngày, chỉ khác đơn vị input).
    """
    # (a) Schedule-based -- an toàn dùng cả ngày vì là lịch bay, không phải outcome
    df = derive_schedule_count(df, ["ORIGIN_AIRPORT", "YEAR", "MONTH", "DAY"], "airport_scheduled_departures_same_day")
    df = derive_schedule_count(df, ["DESTINATION_AIRPORT", "YEAR", "MONTH", "DAY"], "airport_scheduled_arrivals_same_day")

    # (b) Lookback -- lịch sử delay tại sân bay, PHẢI cắt theo timestamp, kỳ tính bằng giờ
    for h in lookback_windows_hours:
        suf = _window_suffix_hours(h)
        lookback_days = h / 24.0
        df = derive_lookback_agg(
            df, ["ORIGIN_AIRPORT"], f"airport_late_flights_count{suf}", lookback_days,
            agg_fn=lambda c: F.sum(F.when(c > late_threshold, 1).otherwise(0)), target_col="ARRIVAL_DELAY",
        )
        df = derive_lookback_agg(
            df, ["ORIGIN_AIRPORT"], f"airport_avg_delay{suf}", lookback_days,
            agg_fn=F.avg, target_col="ARRIVAL_DELAY",
        )
    return df


# -----------------------------------------------------------------------------
# 5.3 — Airplane-level: composite key (TAIL_NUMBER, NGAY_DL, SCHEDULED_DEPARTURE)
# -----------------------------------------------------------------------------
def build_airplane_level_features(df, lookback_windows=(1, 3, 7)):
    # Số chuyến máy bay PHẢI thực hiện trong ngày -- schedule-based, an toàn
    # dùng cả ngày (máy bay được gán lịch bay trước, không phải outcome).
    # KHÔNG cần multi-window vì luôn là "trong ngày hiện tại".
    df = derive_schedule_count(df, ["TAIL_NUMBER", "YEAR", "MONTH", "DAY"], "airplane_flights_scheduled_same_day")

    # Lookback (aggregate nhiều chuyến) -- lịch sử delay của chính máy bay đó
    for w in lookback_windows:
        suf = _window_suffix(w)
        df = derive_lookback_agg(
            df, ["TAIL_NUMBER"], f"airplane_total_delay_minutes{suf}", w,
            agg_fn=F.sum, target_col="ARRIVAL_DELAY",
        )
        df = df.withColumn(
            f"airplane_delay_per_km{suf}",
            safe_divide(F.col(f"airplane_total_delay_minutes{suf}"), F.col("distance_feature_km")),
        )

    # -------------------------------------------------------------------
    # MỚI (theo yêu cầu modeling team): CHỈ 1 chuyến liền trước (không phải
    # aggregate nhiều chuyến như ở trên) -- dùng F.lag(1), không phải
    # rangeBetween. Vẫn leak-safe: lag(1) theo timestamp thực chỉ lấy đúng
    # 1 dòng NGAY TRƯỚC, không bao giờ nhìn thấy dòng hiện tại hay tương lai.
    # -------------------------------------------------------------------
    prev_flight_window = Window.partitionBy("TAIL_NUMBER").orderBy("_flight_timestamp_epoch")
    df = df.withColumn(
        "airplane_prev_flight_arrival_delay",
        F.lag(F.col("ARRIVAL_DELAY"), 1).over(prev_flight_window),
    )

    # ⚠️ LƯU Ý ĐƠN VỊ: ARRIVAL_DELAY (phút, khoảng thời gian trễ) trừ cho
    # SCHEDULED_ARRIVAL (quy đổi số phút từ 0h, mốc giờ trong ngày) -- 2 đại
    # lượng khác bản chất (duration vs. time-of-day). Implement ĐÚNG theo
    # yêu cầu modeling team; CẦN XÁC NHẬN lại ý nghĩa nghiệp vụ trước khi
    # đưa vào training chính thức.
    current_scheduled_arrival_minutes = _hhmm_to_minutes(F.col("SCHEDULED_ARRIVAL"))
    df = df.withColumn(
        "airplane_prev_flight_delay_vs_current_schedule",
        F.col("airplane_prev_flight_arrival_delay") - current_scheduled_arrival_minutes,
    )

    return df


# -----------------------------------------------------------------------------
# 5.4 — Flight-level: composite key (AIRLINE, FLIGHT_NUMBER, NGAY_DL, SCHEDULED_DEPARTURE)
# Dùng DEPARTURE_DELAY của CHÍNH chuyến đang xét -- hợp lệ theo Phương án A
# đã chốt.
#
# CẬP NHẬT: tham số airline_avg_window chỉ định RÕ dùng cửa sổ lookback nào
# của Airline-level (vì giờ Airline-level có nhiều cửa sổ, phải chọn 1 cụ
# thể để so sánh, không còn tên cột cố định duy nhất như trước).
# -----------------------------------------------------------------------------
def build_flight_level_features(df, airline_avg_window=7):
    df = df.withColumn(
        "flight_departure_delay_per_km",
        safe_divide(F.col("DEPARTURE_DELAY"), F.col("distance_feature_km")),
    )
    suf = _window_suffix(airline_avg_window)
    total_col, count_col = f"airline_total_delay_minutes{suf}", f"airline_flights_count{suf}"
    if total_col in df.columns and count_col in df.columns:
        avg_airline_delay = safe_divide(F.col(total_col), F.col(count_col))
        df = df.withColumn(
            f"flight_departure_delay_vs_airline_avg_rt{suf}",
            safe_divide(F.col("DEPARTURE_DELAY"), avg_airline_delay),
        )

    # -------------------------------------------------------------------
    # MỚI (theo yêu cầu modeling team): thông tin MẶC ĐỊNH của chính chuyến
    # đang xét -- KHÔNG áp dụng kỳ phái sinh (không phải lookback/lag), vì
    # đây là đặc điểm cố định của bản thân chuyến bay, biết ngay tại thời
    # điểm dự đoán.
    # -------------------------------------------------------------------
    dep_minutes = _hhmm_to_minutes(F.col("SCHEDULED_DEPARTURE"))
    arr_minutes = _hhmm_to_minutes(F.col("SCHEDULED_ARRIVAL"))
    raw_diff = arr_minutes - dep_minutes
    # Chuyến qua đêm (giờ đến "nhỏ hơn" giờ đi theo đồng hồ) -> cộng thêm 1 ngày
    computed_duration = F.when(raw_diff < 0, raw_diff + 1440).otherwise(raw_diff)
    df = df.withColumn("flight_computed_scheduled_duration_minutes", computed_duration)

    # ⚠️ LƯU Ý QUAN TRỌNG: đây là hiệu số giờ đồng hồ ĐỊA PHƯƠNG tại 2 sân
    # bay (SCHEDULED_ARRIVAL/SCHEDULED_DEPARTURE), KHÔNG phải elapsed time
    # thực. Với chuyến bay XUYÊN MÚI GIỜ (VD LAX -> JFK), hiệu số này SẼ
    # KHÁC với thời lượng bay thực tế (cột SCHEDULED_TIME gốc trong dataset
    # đã tính đúng elapsed time, không bị lệch múi giờ). Cân nhắc dùng
    # SCHEDULED_TIME gốc thay vì cột này nếu cần độ chính xác elapsed time
    # thực; giữ cột này lại vì được yêu cầu tường minh, có thể hữu ích như
    # 1 proxy khác (chênh lệch múi giờ giữa 2 sân bay) chứ không phải elapsed
    # time đúng nghĩa.

    # "so km di": da co san tu Section 1 (distance_feature_km), khong tao lai.
    df = df.withColumn(
        "flight_scheduled_minutes_per_km",
        safe_divide(F.col("flight_computed_scheduled_duration_minutes"), F.col("distance_feature_km")),
    )

    return df


# -----------------------------------------------------------------------------
# 5.5 — Temporal-level: không cần composite key/entity, chỉ dùng ngày/giờ
# -----------------------------------------------------------------------------
def build_temporal_level_features(df, weekend_days=(6, 7)):
    """
    weekend_days: giá trị DAY_OF_WEEK coi là cuối tuần. ĐÃ XÁC NHẬN với nhóm:
    DAY_OF_WEEK dùng encoding ISO 8601 (1=Thứ Hai, ..., 7=Chủ Nhật), nên
    mặc định (6, 7) = Thứ Bảy/Chủ Nhật là chính xác.
    """
    df = df.withColumn("is_weekend", F.col("DAY_OF_WEEK").isin(list(weekend_days)).cast("int"))
    df = df.withColumn(
        "time_of_day_bucket",
        F.when(F.col("scheduled_hour").between(5, 11), F.lit("morning"))
        .when(F.col("scheduled_hour").between(12, 16), F.lit("afternoon"))
        .when(F.col("scheduled_hour").between(17, 20), F.lit("evening"))
        .otherwise(F.lit("night")),
    )
    return df