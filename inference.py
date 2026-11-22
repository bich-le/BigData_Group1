"""Spark Structured Streaming inference through the saved full ML pipeline."""

import argparse
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import ImputerModel, StringIndexerModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "classification": ROOT / "models/random_forest_final_spark",
    "regression": ROOT / "models/random_forest_regression_final_spark",
}


def input_schema(model):
    """Infer a raw-event schema from saved preprocessing stages.

    This intentionally does not use the retired 23-column model-ready
    contract.  Columns created by a pipeline transformer are excluded.
    """
    numeric, categorical, generated = set(), set(), set()
    for stage in model.stages:
        if isinstance(stage, ImputerModel):
            numeric.update(stage.getInputCols())
        elif isinstance(stage, StringIndexerModel):
            categorical.add(stage.getInputCol())
        if type(stage).__name__ == "ScheduledHourTransformer":
            generated.add(stage.getOrDefault(stage.outputCol))
            numeric.add(stage.getOrDefault(stage.inputCol))
    numeric.difference_update(generated)
    fields = [StructField("event_id", StringType(), True)]
    fields += [StructField(name, StringType(), True) for name in sorted(categorical)]
    fields += [StructField(name, DoubleType(), True) for name in sorted(numeric)]
    fields += [StructField("CANCELLED", DoubleType(), True), StructField("DIVERTED", DoubleType(), True)]
    return StructType(fields)


def load_verified_model(path, task):
    model = PipelineModel.load(str(path))
    stage_names = [type(stage).__name__ for stage in model.stages]
    expected_model_type = (
        RandomForestClassificationModel
        if task == "classification"
        else RandomForestRegressionModel
    )
    if not isinstance(model.stages[-1], expected_model_type):
        raise ValueError(f"Saved model is not a Spark Random Forest {task} model")
    required = {"FlightEligibilityAndLabelTransformer", "ImputerModel", "PCAModel"}
    if not required.issubset(set(stage_names)):
        raise ValueError(
            "Model is a legacy feature-vector artifact, not a full raw-input pipeline: "
            f"{stage_names}"
        )
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
        incoming = (
            spark.readStream.schema(input_schema(model))
            .json(str(args.input_dir))
        )
        transformed = model.transform(incoming)
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
