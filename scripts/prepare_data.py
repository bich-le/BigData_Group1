"""Validate raw flight-delay CSV files and publish typed Parquet datasets."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pyspark
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T


FLIGHTS_COLUMNS = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
    "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT",
    "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "WHEELS_ON", "TAXI_IN", "SCHEDULED_ARRIVAL", "ARRIVAL_TIME",
    "ARRIVAL_DELAY", "DIVERTED", "CANCELLED", "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
AIRPORTS_COLUMNS = [
    "IATA_CODE", "AIRPORT", "CITY", "STATE", "COUNTRY", "LATITUDE", "LONGITUDE"
]
AIRLINES_COLUMNS = ["IATA_CODE", "AIRLINE"]

FLIGHTS_STRING_COLUMNS = {
    "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "WHEELS_OFF",
    "WHEELS_ON", "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "CANCELLATION_REASON",
}
FLIGHTS_INTEGER_COLUMNS = set(FLIGHTS_COLUMNS) - FLIGHTS_STRING_COLUMNS
AIRPORTS_DOUBLE_COLUMNS = {"LATITUDE", "LONGITUDE"}

EXPECTED_COUNTS = {"flights": 5_819_079, "airports": 322, "airlines": 14}
EXPECTED_ARRIVAL_NONNULL = 5_714_008
EXPECTED_ARRIVAL_NULL = 105_071
EXPECTED_MONTH_10_DOT_ROWS = 486_165


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--shuffle-partitions", default=12, type=int)
    args = parser.parse_args()
    if args.shuffle_partitions < 1:
        parser.error("--shuffle-partitions must be positive")
    return args


def timed(label: str):
    class Timer:
        def __enter__(self):
            self.started = time.perf_counter()
            print(f"TIMING_START|{label}", flush=True)
            return self

        def __exit__(self, exc_type, exc, traceback):
            elapsed = time.perf_counter() - self.started
            print(f"TIMING_END|{label}|seconds={elapsed:.3f}", flush=True)

    return Timer()


def read_csv(spark: SparkSession, path: Path) -> DataFrame:
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "FAILFAST")
        .option("encoding", "UTF-8")
        .load(str(path))
    )


def require_raw_schema(df: DataFrame, expected: list[str], name: str) -> None:
    if df.columns != expected:
        raise ValueError(f"{name}: columns/order differ: actual={df.columns}, expected={expected}")
    non_strings = [field.simpleString() for field in df.schema.fields if not isinstance(field.dataType, T.StringType)]
    if non_strings:
        raise ValueError(f"{name}: raw columns are not all string: {non_strings}")
    print(f"RAW_SCHEMA_OK|{name}|columns={len(expected)}", flush=True)


def non_missing(column: str):
    return F.col(column).isNotNull() & (F.trim(F.col(column)) != "")


def checked_typed(df: DataFrame, name: str, expected_count: int) -> tuple[DataFrame, int]:
    if name == "flights":
        cast_types = {column: "int" for column in FLIGHTS_INTEGER_COLUMNS}
    elif name == "airports":
        cast_types = {column: "double" for column in AIRPORTS_DOUBLE_COLUMNS}
    else:
        cast_types = {}

    aggregations = [F.count(F.lit(1)).alias("row_count")]
    for column, target in cast_types.items():
        aggregations.append(
            F.sum(F.when(non_missing(column) & F.expr(f"try_cast(`{column}` as {target})").isNull(), 1).otherwise(0))
            .cast("long")
            .alias(f"invalid__{column}")
        )
    stats = df.agg(*aggregations).first().asDict()
    row_count = int(stats.pop("row_count"))
    if row_count != expected_count:
        raise ValueError(f"{name}: row_count={row_count}, expected={expected_count}")
    invalid = {key.removeprefix("invalid__"): value for key, value in stats.items() if value != 0}
    if invalid:
        raise ValueError(f"{name}: non-missing values would become null after cast: {invalid}")

    expressions = []
    for column in df.columns:
        if column in FLIGHTS_INTEGER_COLUMNS and name == "flights":
            expressions.append(F.col(column).cast(T.IntegerType()).alias(column))
        elif column in AIRPORTS_DOUBLE_COLUMNS and name == "airports":
            expressions.append(F.col(column).cast(T.DoubleType()).alias(column))
        else:
            expressions.append(F.col(column))
    typed = df.select(*expressions)
    if typed.columns != df.columns:
        raise AssertionError(f"{name}: typed column order changed")
    print(f"PREWRITE_CAST_OK|{name}|rows={row_count}", flush=True)
    return typed, row_count


def validate_prewrite_domains(flights: DataFrame, airports: DataFrame) -> None:
    domains = {
        "YEAR": {2015},
        "MONTH": set(range(1, 13)),
        "DAY": set(range(1, 32)),
        "DAY_OF_WEEK": set(range(1, 8)),
        "DIVERTED": {0, 1},
        "CANCELLED": {0, 1},
    }
    domain_row = flights.agg(
        *(F.collect_set(column).alias(column) for column in domains)
    ).first().asDict()
    for column, expected in domains.items():
        actual = set(domain_row[column])
        if column in {"MONTH", "DAY", "DAY_OF_WEEK"}:
            if not actual or not actual.issubset(expected):
                raise ValueError(f"{column}: values outside domain: {sorted(actual - expected)}")
        elif actual != expected:
            raise ValueError(f"{column}: actual={sorted(actual)}, expected={sorted(expected)}")
        print(f"PREWRITE_DOMAIN_OK|{column}|values={sorted(actual)}", flush=True)

    coordinate_stats = airports.agg(
        F.sum(F.when(F.col("LATITUDE").isNull() & F.col("LONGITUDE").isNull(), 1).otherwise(0)).alias("both_null"),
        F.sum(F.when(F.col("LATITUDE").isNull() != F.col("LONGITUDE").isNull(), 1).otherwise(0)).alias("one_null"),
    ).first()
    if coordinate_stats["both_null"] != 3 or coordinate_stats["one_null"] != 0:
        raise ValueError(f"airports: coordinate null invariant failed: {coordinate_stats.asDict()}")
    print("PREWRITE_COORDINATES_OK|both_null=3|one_null=0", flush=True)


def expected_types(name: str) -> dict[str, type[T.DataType]]:
    if name == "flights":
        return {
            column: (T.StringType if column in FLIGHTS_STRING_COLUMNS else T.IntegerType)
            for column in FLIGHTS_COLUMNS
        }
    if name == "airports":
        return {
            column: (T.DoubleType if column in AIRPORTS_DOUBLE_COLUMNS else T.StringType)
            for column in AIRPORTS_COLUMNS
        }
    return {column: T.StringType for column in AIRLINES_COLUMNS}


def validate_schema(df: DataFrame, name: str, raw_order: list[str]) -> None:
    allowed_orders = [raw_order]
    if name == "flights":
        allowed_orders.append([column for column in raw_order if column != "MONTH"] + ["MONTH"])
    if df.columns not in allowed_orders:
        raise ValueError(f"{name}: unexpected read-back column order: {df.columns}")
    expected = expected_types(name)
    mismatches = {
        field.name: (field.dataType.simpleString(), expected[field.name]().simpleString())
        for field in df.schema.fields
        if not isinstance(field.dataType, expected[field.name])
    }
    if mismatches:
        raise ValueError(f"{name}: read-back type mismatch: {mismatches}")
    print(f"READBACK_SCHEMA_OK|{name}|columns={df.columns}", flush=True)


def validate_lookup(df: DataFrame, name: str, expected_count: int) -> None:
    key = "IATA_CODE"
    stats = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct(key).alias("distinct_keys"),
        F.sum(F.when(F.col(key).isNull() | (F.trim(F.col(key)) == ""), 1).otherwise(0)).alias("missing_keys"),
    ).first().asDict()
    if stats != {"rows": expected_count, "distinct_keys": expected_count, "missing_keys": 0}:
        raise ValueError(f"{name}: lookup invariant failed: {stats}")
    if name == "airports":
        coords = df.agg(
            F.sum(F.when(F.col("LATITUDE").isNull() & F.col("LONGITUDE").isNull(), 1).otherwise(0)).alias("both_null"),
            F.sum(F.when(F.col("LATITUDE").isNull() != F.col("LONGITUDE").isNull(), 1).otherwise(0)).alias("one_null"),
        ).first().asDict()
        if coords != {"both_null": 3, "one_null": 0}:
            raise ValueError(f"airports: coordinate invariant failed after write: {coords}")
    print(f"READBACK_LOOKUP_OK|{name}|{stats}", flush=True)


def recursive_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def part_file_count(path: Path) -> int:
    return sum(1 for item in path.glob("part-*.parquet") if item.is_file())


def validate_flights_readback(df: DataFrame, flights_path: Path) -> None:
    stats = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("ARRIVAL_DELAY").isNotNull(), 1).otherwise(0)).alias("arrival_nonnull"),
        F.sum(F.when(F.col("ARRIVAL_DELAY").isNull(), 1).otherwise(0)).alias("arrival_null"),
        F.sum(F.when(F.col("ARRIVAL_DELAY").isNull() & (F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0), 1).otherwise(0)).alias("arrival_null_outside_ineligible"),
        F.sum(F.when((F.col("MONTH") == 10) & (F.col("ORIGIN_AIRPORT").rlike("^[0-9]+$")), 1).otherwise(0)).alias("month10_numeric_origin"),
        F.sum(F.when((F.col("MONTH") == 10) & (F.col("DESTINATION_AIRPORT").rlike("^[0-9]+$")), 1).otherwise(0)).alias("month10_numeric_destination"),
    ).first().asDict()
    expected = {
        "rows": EXPECTED_COUNTS["flights"],
        "arrival_nonnull": EXPECTED_ARRIVAL_NONNULL,
        "arrival_null": EXPECTED_ARRIVAL_NULL,
        "arrival_null_outside_ineligible": 0,
        "month10_numeric_origin": EXPECTED_MONTH_10_DOT_ROWS,
        "month10_numeric_destination": EXPECTED_MONTH_10_DOT_ROWS,
    }
    if stats != expected:
        raise ValueError(f"flights: read-back invariant failed: actual={stats}, expected={expected}")
    print(f"READBACK_FLIGHTS_INVARIANTS_OK|{stats}", flush=True)
    print("READBACK_FULL_ROW_DUPLICATES|skipped|previous_raw_profile_result=0", flush=True)

    partition_dirs = sorted(path for path in flights_path.glob("MONTH=*") if path.is_dir())
    parsed = {}
    for partition_dir in partition_dirs:
        value = partition_dir.name.split("=", 1)[1]
        if value == "__HIVE_DEFAULT_PARTITION__":
            raise ValueError("flights: null MONTH partition found")
        try:
            month = int(value)
        except ValueError as error:
            raise ValueError(f"flights: invalid partition directory {partition_dir.name}") from error
        parsed[month] = part_file_count(partition_dir)
    if set(parsed) != set(range(1, 13)):
        raise ValueError(f"flights: partition values differ from 1..12: {sorted(parsed)}")

    rows_by_month = {row["MONTH"]: row["count"] for row in df.groupBy("MONTH").count().collect()}
    if set(rows_by_month) != set(range(1, 13)) or sum(rows_by_month.values()) != EXPECTED_COUNTS["flights"]:
        raise ValueError(f"flights: rows by month invalid: {rows_by_month}")
    for month in range(1, 13):
        print(f"PARTITION|MONTH={month}|part_files={parsed[month]}|rows={rows_by_month[month]}", flush=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Final output already exists; refusing to overwrite: {output_dir}")
    required = {name: input_dir / f"{name}.csv" for name in ("flights", "airports", "airlines")}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f"_staging_typed_parquet_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging path unexpectedly exists: {staging}")
    print(f"INPUT_DIR|{input_dir}", flush=True)
    print(f"FINAL_OUTPUT_DIR|{output_dir}", flush=True)
    print(f"STAGING_DIR|{staging}", flush=True)

    temp_root = Path(tempfile.gettempdir()).resolve()
    warehouse = temp_root / "prepare_data_warehouse"
    local_dir = Path(os.environ.get("SPARK_LOCAL_DIRS", str(temp_root / "prepare_data_spark_local"))).resolve()
    warehouse.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    spark: SparkSession | None = None
    raw_flights: DataFrame | None = None
    started = time.perf_counter()
    try:
        spark = (
            SparkSession.builder.master(args.master)
            .appName("PrepareFlightTypedParquet")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
            .config("spark.sql.warehouse.dir", str(warehouse))
            .config("spark.local.dir", str(local_dir))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        java_version = spark.sparkContext._jvm.java.lang.System.getProperty("java.version")
        print(
            f"VERSIONS|spark={spark.version}|pyspark={pyspark.__version__}|"
            f"java={java_version}|python={sys.version.split()[0]}|python_executable={sys.executable}",
            flush=True,
        )
        if spark.version != "3.5.9" or pyspark.__version__ != "3.5.9" or not java_version.startswith("17."):
            raise RuntimeError(
                f"Unexpected runtime: Spark {spark.version}, PySpark {pyspark.__version__}, Java {java_version}"
            )

        with timed("read_raw"):
            raw_flights = read_csv(spark, required["flights"]).persist(StorageLevel.DISK_ONLY)
            raw_airports = read_csv(spark, required["airports"])
            raw_airlines = read_csv(spark, required["airlines"])
            require_raw_schema(raw_flights, FLIGHTS_COLUMNS, "flights")
            require_raw_schema(raw_airports, AIRPORTS_COLUMNS, "airports")
            require_raw_schema(raw_airlines, AIRLINES_COLUMNS, "airlines")

        with timed("prewrite_validation"):
            flights, _ = checked_typed(raw_flights, "flights", EXPECTED_COUNTS["flights"])
            airports, _ = checked_typed(raw_airports, "airports", EXPECTED_COUNTS["airports"])
            airlines, _ = checked_typed(raw_airlines, "airlines", EXPECTED_COUNTS["airlines"])
            validate_prewrite_domains(flights, airports)

        with timed("write_staging"):
            (
                flights.repartition(args.shuffle_partitions, "MONTH")
                .write.mode("errorifexists").option("compression", "snappy")
                .partitionBy("MONTH").parquet(str(staging / "flights"))
            )
            airlines.coalesce(1).write.mode("errorifexists").option("compression", "snappy").parquet(str(staging / "airlines"))
            airports.coalesce(1).write.mode("errorifexists").option("compression", "snappy").parquet(str(staging / "airports"))

        with timed("readback_validation"):
            saved_flights = spark.read.parquet(str(staging / "flights"))
            saved_airports = spark.read.parquet(str(staging / "airports"))
            saved_airlines = spark.read.parquet(str(staging / "airlines"))
            validate_schema(saved_flights, "flights", FLIGHTS_COLUMNS)
            validate_schema(saved_airports, "airports", AIRPORTS_COLUMNS)
            validate_schema(saved_airlines, "airlines", AIRLINES_COLUMNS)
            validate_flights_readback(saved_flights, staging / "flights")
            validate_lookup(saved_airports, "airports", EXPECTED_COUNTS["airports"])
            validate_lookup(saved_airlines, "airlines", EXPECTED_COUNTS["airlines"])

        input_bytes = sum(path.stat().st_size for path in required.values())
        output_bytes = recursive_size(staging)
        ratio = input_bytes / output_bytes if output_bytes else float("inf")
        print(f"SIZE|input_csv_bytes={input_bytes}|staging_parquet_bytes={output_bytes}|csv_to_parquet_ratio={ratio:.4f}", flush=True)

        if output_dir.exists():
            raise FileExistsError(f"Final output appeared during run; staging retained: {output_dir}")
        with timed("atomic_promote"):
            staging.rename(output_dir)
        print(f"PROMOTED|{output_dir}", flush=True)
        print(f"TOTAL_SECONDS|{time.perf_counter() - started:.3f}", flush=True)
        print("RESULT|PASS", flush=True)
        return 0
    except Exception:
        print(f"RESULT|FAIL|staging_retained_if_present={staging}", flush=True)
        raise
    finally:
        if raw_flights is not None:
            raw_flights.unpersist(blocking=False)
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
