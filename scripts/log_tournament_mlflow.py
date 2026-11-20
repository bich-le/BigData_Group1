"""Show the group's reported model tournament in MLflow without retraining.

The input files are checked-in evaluation summaries. Every run created here is
tagged as an import of reported results; no run is presented as a fresh fit.
The fitted champion models are registered separately by register_mlflow.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ROOT / "docs" / "evaluations"
ASSETS = ROOT / "docs" / "assets"
TASK_FILES = {
    "classification": EVALUATIONS / "classification_tournament.json",
    "regression": EVALUATIONS / "regression_tournament.json",
}
GRID_METRICS = {
    "rmse_cv": "cv_rmse",
    "mae_cv": "cv_mae",
    "r2_cv": "cv_r2",
    "rmse_valid": "validation_rmse",
    "mae_valid": "validation_mae",
    "r2_valid": "validation_r2",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation source is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def numeric_metrics(values: dict) -> dict[str, float]:
    """Log only metrics actually present in the source document."""
    result = {}
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Metric {name} must be numeric: {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Metric {name} is not finite: {value!r}")
        result[name] = number
    return result


def _plot_candidates(candidates: list[dict], task: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "oot_gini" if task == "classification" else "oot_rmse"
    names = [candidate["algorithm"] for candidate in candidates]
    values = [candidate["metrics"][metric] for candidate in candidates]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], values[::-1], color="#2b5c8f")
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(f"{task.title()} tournament — reported OOT results")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_regression_grid(results: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [item for item in results if item.get("rmse_valid") is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [str(item["config"]) for item in completed],
        [item["rmse_valid"] for item in completed],
        color="#2b5c8f",
    )
    ax.set(xlabel="RF grid configuration", ylabel="Validation RMSE")
    ax.set_title("Reported common-validation grid results")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def log_candidate(task: str, candidate: dict, source: Path) -> str:
    name = candidate["algorithm"]
    metrics = numeric_metrics(candidate["metrics"])
    if not metrics:
        raise ValueError(f"No metrics for {task} candidate {name}")
    with mlflow.start_run(run_name=f"{task}: {name}", nested=True) as run:
        mlflow.log_params({"task": task, "algorithm": name})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "evidence.kind": "reported_result_import",
                "training.executed": "false",
                "result.source": source.relative_to(ROOT).as_posix(),
                "tournament.verdict": candidate["verdict"],
            }
        )
        return run.info.run_id


def log_regression_grid(selection: dict) -> list[str]:
    winner = selection["winner"]["config"]
    run_ids = []
    for item in selection["results"]:
        with mlflow.start_run(
            run_name=f"regression RF grid config {item['config']:02d}", nested=True
        ) as run:
            params = {
                "task": "regression",
                "algorithm": "RandomForestRegressor",
                "config": item["config"],
                "numTrees": item["numTrees"],
                "maxDepth": item["maxDepth"],
                "minInstancesPerNode": item["minInstancesPerNode"],
            }
            if item.get("folds_used") is not None:
                params["folds_used"] = item["folds_used"]
            mlflow.log_params(params)
            mlflow.log_metrics(
                numeric_metrics(
                    {
                        target: item.get(source)
                        for source, target in GRID_METRICS.items()
                    }
                )
            )
            mlflow.set_tags(
                {
                    "evidence.kind": "reported_result_import",
                    "training.executed": "false",
                    "result.source": "docs/evaluations/regression_selection.json",
                    "result.status": item["status"],
                    "selection.eligible": str(
                        item["status"] == "complete"
                        and item.get("rmse_valid") is not None
                    ).lower(),
                    "selection.winner": str(item["config"] == winner).lower(),
                }
            )
            run_ids.append(run.info.run_id)
    return run_ids


def log_task(task: str, temporary: Path) -> dict:
    source = TASK_FILES[task]
    tournament = read_json(source)
    candidates = tournament["candidates"]
    champion = next(
        candidate for candidate in candidates if candidate["verdict"] == "champion"
    )
    plot = temporary / f"{task}_tournament.png"
    _plot_candidates(candidates, task, plot)

    with mlflow.start_run(run_name=f"{task} tournament (reported)") as parent:
        mlflow.log_params({"task": task, "candidate_count": len(candidates)})
        mlflow.log_metrics(
            {
                f"champion_{name}": value
                for name, value in numeric_metrics(champion["metrics"]).items()
            }
        )
        mlflow.set_tags(
            {
                "evidence.kind": "reported_result_import",
                "training.executed": "false",
                "model.artifact.logged": "false",
                "result.source": source.relative_to(ROOT).as_posix(),
                "selection.metric": tournament["selection_metric"],
            }
        )
        mlflow.log_artifact(str(source), artifact_path="sources")
        mlflow.log_artifact(str(ROOT / "docs" / "REPORT.md"), artifact_path="sources")
        mlflow.log_artifact(str(ASSETS / "pca_variance.png"), artifact_path="plots")
        mlflow.log_artifact(str(plot), artifact_path="plots")

        if task == "regression":
            selection_path = EVALUATIONS / "regression_selection.json"
            evaluation_path = EVALUATIONS / "regression_evaluation.json"
            selection = read_json(selection_path)
            mlflow.log_artifact(str(selection_path), artifact_path="sources")
            mlflow.log_artifact(str(evaluation_path), artifact_path="sources")
            grid_plot = temporary / "regression_rf_grid.png"
            _plot_regression_grid(selection["results"], grid_plot)
            mlflow.log_artifact(str(grid_plot), artifact_path="plots")
            # The checked-in feature-importance chart is generated from
            # representative hard-coded values in benchmark_results.py.
            # Label it honestly until a fitted model is available.
            mlflow.log_artifact(
                str(ASSETS / "feature_importance.png"),
                artifact_path="plots/representative_only",
            )
            mlflow.set_tag(
                "feature_importance.provenance", "representative_hardcoded_values"
            )
        else:
            selection = None

        candidate_ids = [log_candidate(task, item, source) for item in candidates]
        grid_ids = log_regression_grid(selection) if selection else []
        return {
            "task": task,
            "parent_run_id": parent.info.run_id,
            "candidate_run_ids": candidate_ids,
            "grid_run_ids": grid_ids,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI") or (ROOT / "mlruns").as_uri(),
        help="MLflow backend URI; defaults to the local gitignored mlruns directory",
    )
    parser.add_argument("--experiment", default="FlightDelay-Tournament-Reported")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with tempfile.TemporaryDirectory(prefix="tournament-mlflow-") as directory:
        temporary = Path(directory)
        results = [log_task(task, temporary) for task in TASK_FILES]
    print(json.dumps({"tracking_uri": args.tracking_uri, "runs": results}, indent=2))


if __name__ == "__main__":
    main()
