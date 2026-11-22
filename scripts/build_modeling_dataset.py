"""Materialize optional batch history features from typed raw Parquet inputs."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from feature_integration import FlightFeatureContract, build_flight_feature_frame
from mllib_pipeline import configure_local_filesystem, configure_pyspark_python, prepare_eligible_flights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Typed raw flights Parquet")
    parser.add_argument("--airports", type=Path, required=True, help="Typed airports Parquet")
    parser.add_argument("--output", type=Path, required=True, help="New batch feature Parquet directory")
    parser.add_argument("--exclude-departure-delay", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be new; existing datasets are never overwritten")
    configure_pyspark_python()
    spark = configure_local_filesystem(SparkSession.builder.appName("flight-batch-features")).getOrCreate()
    try:
        flights = prepare_eligible_flights(spark.read.parquet(str(args.input)))
        airports = spark.read.parquet(str(args.airports))
        contract = FlightFeatureContract(include_departure_delay_features=not args.exclude_departure_delay)
        features = build_flight_feature_frame(flights, airports, contract)
        features.write.parquet(str(args.output))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
