"""Score pre-engineered streaming JSON with baseline scikit-learn Random Forest in Spark.

Each JSON record must contain the 23 numeric model features below. This is a
fallback for serving the original sklearn artifact; it does not create lag features.
"""

import argparse
from functools import lru_cache
from pathlib import Path
import sys

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_contract import SELECTED_FEATURES


DEFAULT_MODEL = ROOT.parent / "random_forest_final.pkl"
INPUT_SCHEMA = StructType(
    [StructField("event_id", StringType(), True)]
    + [StructField(name, DoubleType(), True) for name in SELECTED_FEATURES]
)


@lru_cache(maxsize=1)
def load_worker_model(path):
    import joblib

    model = joblib.load(path)
    actual = list(getattr(model, "feature_names_in_", []))
    if actual != SELECTED_FEATURES:
        raise ValueError(f"Model feature order differs: {actual}")
    return model


def probability_udf(model_path):
    @pandas_udf(DoubleType())
    def score(*columns: pd.Series) -> pd.Series:
        # Spark sends Arrow batches; the full microbatch is never collected.
        frame = pd.DataFrame({name: column for name, column in zip(SELECTED_FEATURES, columns)})
        frame = frame.fillna(0).astype("float64")
        model = load_worker_model(model_path)
        return pd.Series(model.predict_proba(frame[SELECTED_FEATURES])[:, 1])

    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Missing sklearn artifact: {args.model}")
    if not args.input_dir.is_dir():
        parser.error(f"Missing JSON input directory: {args.input_dir}")
    if args.output_dir == args.input_dir or args.checkpoint_dir == args.input_dir:
        parser.error("Input, output, and checkpoint directories must differ")

    spark = SparkSession.builder.appName("SklearnRandomForestStreamingInference").getOrCreate()
    try:
        incoming = spark.readStream.schema(INPUT_SCHEMA).json(str(args.input_dir))
        probability = probability_udf(str(args.model.resolve()))
        scored = incoming.select(
            "event_id",
            probability(*[F.col(name) for name in SELECTED_FEATURES]).alias("probability_late_over_30m"),
        ).withColumn(
            "prediction", (F.col("probability_late_over_30m") >= 0.5).cast("int")
        )
        query = (scored.writeStream.format("json").outputMode("append")
                 .option("path", str(args.output_dir))
                 .option("checkpointLocation", str(args.checkpoint_dir))
                 .start())
        query.awaitTermination()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
