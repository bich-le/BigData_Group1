"""
custom_transformers.py

Custom PySpark Transformer classes & PyArrow-accelerated pandas_udf
cho phase Custom & Feature Engineering.
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
from pyspark.sql.types import DoubleType

import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

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
MANUAL_AIRPORT_COORDS = {
    # IATA_CODE: (LATITUDE, LONGITUDE)  -- nguồn: Wikipedia/FAA, đối chiếu Great Circle Mapper
    "ECP": (30.3417, -85.7973),    # Northwest Florida Beaches International Airport
    "PBG": (44.65083, -73.46806),  # Plattsburgh International Airport
    "UST": (29.95925, -81.33972),  # Northeast Florida Regional Airport (St. Augustine)
}


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


def add_haversine_distance_feature(flights_df, airports_df):
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

    origin_coords = airports_df.select(
        F.col("IATA_CODE").alias("ORIGIN_AIRPORT"),
        F.col("LATITUDE").alias("origin_lat"),
        F.col("LONGITUDE").alias("origin_lon"),
    )
    dest_coords = airports_df.select(
        F.col("IATA_CODE").alias("DESTINATION_AIRPORT"),
        F.col("LATITUDE").alias("dest_lat"),
        F.col("LONGITUDE").alias("dest_lon"),
    )

    joined = (
        flights_df.join(origin_coords, on="ORIGIN_AIRPORT", how="left")
        .join(dest_coords, on="DESTINATION_AIRPORT", how="left")
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
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("custom_transformers_sanity_check").getOrCreate()

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
# LƯU Ý QUAN TRỌNG (đã điều chỉnh so với kế hoạch ban đầu):
#   - KHÔNG áp dụng lên DEPARTURE_DELAY: đây là cột leakage (post-departure),
#     đã bị loại khỏi feature set theo data_dictionary.md.
#   - KHÔNG áp dụng lên ARRIVAL_DELAY: đây là TARGET của bài toán regression.
#     Clip trực tiếp label sẽ làm sai lệch định nghĩa bài toán (model sẽ học
#     để dự đoán một target đã bị bóp méo, không phải ARRIVAL_DELAY thật).
#   - Áp dụng lên feature pre-departure có khả năng lệch/outlier:
#     distance_feature_km (Section 1), SCHEDULED_TIME.
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
    def __init__(self, inputCols=None, outputCols=None, iqrMultiplier=1.5):
        super(OutlierIQRTruncator, self).__init__()
        self._setDefault(iqrMultiplier=1.5)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, iqrMultiplier=1.5):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()
        multiplier = self.getIqrMultiplier()

        if len(input_cols) != len(output_cols):
            raise ValueError("inputCols và outputCols phải cùng độ dài")

        bounds = {}
        for in_col, out_col in zip(input_cols, output_cols):
            q1, q3 = dataset.approxQuantile(in_col, [0.25, 0.75], relativeError=0.01)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            bounds[out_col] = {"input_col": in_col, "lower": lower, "upper": upper}

        bounds_json = json.dumps(bounds)
        return OutlierIQRTruncatorModel(inputCols=input_cols, outputCols=output_cols, boundsMapJson=bounds_json)


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
        return result

    def bounds_summary(self) -> pd.DataFrame:
        """Tóm tắt bounds dạng bảng nhỏ — dùng để log vào MLflow/REPORT.md,
        hoặc trình bày trực tiếp khi demo (đúng yêu cầu 'trực tiếp, dễ hình dung')."""
        bounds = json.loads(self.getBoundsMapJson())
        rows = [
            {
                "output_col": out_col,
                "input_col": info["input_col"],
                "lower_bound": info["lower"],
                "upper_bound": info["upper"],
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
        spark = dataset.sparkSession
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
):
    """
    Đếm số chuyến bay được lên lịch tại cùng sân bay đi, cùng ngày, cùng
    khung giờ (scheduled_hour) -- proxy cho mức độ tắc nghẽn lịch bay.
    Đây là con số tính từ chính lịch bay (không phải outcome thực tế), nên
    hợp lệ để tính trên toàn bộ dataset mà không cần tách train/fold riêng
    -- khác về bản chất với OutlierIQRTruncator/TargetEncoder (2 cái đó
    học từ phân phối của TARGET, nên bắt buộc chỉ fit trên train).

    yêu cầu: flights_df đã có cột `hour_col` (chạy add_scheduled_hour_feature
    trước khi gọi hàm này).
    """
    window_cols = [airport_col, "MONTH", "DAY", hour_col]
    congestion = flights_df.groupBy(*window_cols).agg(
        F.count(F.lit(1)).alias(output_col)
    )
    return flights_df.join(congestion, on=window_cols, how="left")