"""Spark Structured Streaming inference for both flight-delay models.

Input JSON contains an optional ``event_id`` plus the 23 numeric model-ready
features in ``FEATURES_BY_TASK``. Historical lag features must be produced
upstream. Missing feature values are filled with zero, matching training.
"""

import argparse
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from feature_contract import FEATURES_BY_TASK


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "classification": ROOT / "models/random_forest_final_spark",
    "regression": ROOT / "models/random_forest_regression_final_spark",
}


def input_schema(task):
    return StructType(
        [StructField("event_id", StringType(), True)]
        + [StructField(name, DoubleType(), True) for name in FEATURES_BY_TASK[task]]
    )


def load_verified_model(path, task):
    model = PipelineModel.load(str(path))
    stage_names = [type(stage).__name__ for stage in model.stages]
    if len(model.stages) != 2 or not isinstance(model.stages[0], VectorAssembler):
        raise ValueError(f"Unexpected pipeline stages: {stage_names}")
    expected_model_type = (
        RandomForestClassificationModel
        if task == "classification"
        else RandomForestRegressionModel
    )
    if not isinstance(model.stages[1], expected_model_type):
        raise ValueError(f"Saved model is not a Spark Random Forest {task} model")
    actual = model.stages[0].getInputCols()
    if actual != FEATURES_BY_TASK[task]:
        raise ValueError(f"Model feature contract differs: {actual}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=sorted(DEFAULT_MODELS), default="classification"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--available-now", action="store_true")
    args = parser.parse_args()
    model_path = args.model or DEFAULT_MODELS[args.task]
    if not model_path.is_dir():
        parser.error(f"Spark PipelineModel is missing: {model_path}")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory is missing: {args.input_dir}")
    stream_paths = {
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        args.checkpoint_dir.resolve(),
    }
    if len(stream_paths) != 3:
        parser.error("Input, output, and checkpoint directories must differ")

    spark = (
        SparkSession.builder.appName(
            f"FlightDelay{args.task.title()}Streaming"
        ).getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        model = load_verified_model(model_path, args.task)
        feature_cols = FEATURES_BY_TASK[args.task]
        incoming = (
            spark.readStream.schema(input_schema(args.task))
            .json(str(args.input_dir))
        )
        prepared = incoming.select("event_id", *feature_cols).fillna(
            0, subset=feature_cols
        )
        transformed = model.transform(prepared)
        if args.task == "classification":
            scored = transformed.select(
                "event_id",
                F.col("prediction").cast("int").alias("prediction"),
                vector_to_array("probability")[1].alias("probability_late_over_30m"),
            )
        else:
            scored = transformed.select(
                "event_id",
                F.col("prediction").alias("predicted_arrival_delay_minutes"),
            )
        writer = (
            scored.writeStream.format("json").outputMode("append")
            .option("path", str(args.output_dir))
            .option("checkpointLocation", str(args.checkpoint_dir))
        )
        query = (
            writer.trigger(availableNow=True).start()
            if args.available_now
            else writer.start()
        )
        query.awaitTermination()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
