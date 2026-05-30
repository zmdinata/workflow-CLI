from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
import os
from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from social_media_addiction import (  # noqa: E402
    DEFAULT_PROCESSED_FILENAME,
    TARGET_COLUMN,
    build_default_regressor,
    build_train_test_split,
    evaluate_regression,
    feature_importance_frame,
    log_prediction_artifacts,
    save_model_artifact,
)
from social_media_addiction.data import load_preprocessing_schema  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training model untuk prediksi Addicted_Score.")
    parser.add_argument(
        "--data-path",
        default=str(Path(__file__).resolve().parent / DEFAULT_PROCESSED_FILENAME),
        help="Path ke dataset hasil preprocessing.",
    )
    parser.add_argument(
        "--schema-path",
        default=str(Path(__file__).resolve().parent / "preprocessing_schema.json"),
        help="Path ke schema preprocessing.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Tracking URI MLflow. Jika kosong, memakai penyimpanan lokal default.",
    )
    parser.add_argument(
        "--experiment-name",
        default="social-media-addiction-regression",
        help="Nama eksperimen MLflow.",
    )
    parser.add_argument(
        "--model-output",
        default=str(Path(__file__).resolve().parent / "artifacts" / "social_media_addiction_model.joblib"),
        help="Path output model lokal.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    return parser


def train(args: argparse.Namespace) -> dict:
    data_path = Path(args.data_path)
    schema_path = Path(args.schema_path)
    processed_df = pd.read_csv(data_path)
    schema = load_preprocessing_schema(schema_path)

    X_train, X_test, y_train, y_test = build_train_test_split(
        processed_df,
        target_column=TARGET_COLUMN,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    model = build_default_regressor(random_state=args.random_state)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    project_run_id = os.environ.get("MLFLOW_RUN_ID")
    project_client = MlflowClient() if project_run_id else None

    def log_params(params: dict) -> None:
        if project_client and project_run_id:
            for key, value in params.items():
                project_client.log_param(project_run_id, key, str(value))
        else:
            mlflow.log_params(params)

    def log_metrics(metrics: dict) -> None:
        if project_client and project_run_id:
            for key, value in metrics.items():
                project_client.log_metric(project_run_id, key, float(value))
        else:
            mlflow.log_metrics(metrics)

    def log_artifact(path: Path) -> None:
        if project_client and project_run_id:
            project_client.log_artifact(project_run_id, str(path))
        else:
            mlflow.log_artifact(str(path))

    if not project_run_id:
        mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    if mlflow.active_run() or project_run_id:
        run_context = nullcontext()
    else:
        run_context = mlflow.start_run()
    with run_context:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_regression(y_test, predictions)

        log_params(
            {
                "model_type": "RandomForestRegressor",
                "dataset_rows": int(processed_df.shape[0]),
                "dataset_features": int(processed_df.shape[1] - 1),
                "categorical_columns": len(schema.get("categorical_columns", [])),
                "numeric_columns": len(schema.get("numeric_columns", [])),
                "test_size": args.test_size,
                "random_state": args.random_state,
            }
        )
        log_metrics(metrics)

        artifact_dir = Path(args.model_output).parent / "reports"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        feature_frame = feature_importance_frame(model, X_train.columns)
        feature_path = artifact_dir / "feature_importance.csv"
        feature_frame.to_csv(feature_path, index=False)
        log_artifact(feature_path)

        evaluation_path = artifact_dir / "evaluation_summary.json"
        evaluation_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log_artifact(evaluation_path)

        prediction_artifacts = log_prediction_artifacts(
            y_true=y_test,
            y_pred=predictions,
            feature_importance=feature_frame,
            output_dir=artifact_dir,
        )
        for path in prediction_artifacts.values():
            log_artifact(path)

        save_model_artifact(model, args.model_output)
        signature = infer_signature(X_test, predictions)
        if project_client and project_run_id:
            log_artifact(Path(args.model_output))
            signature_path = artifact_dir / "model_signature.json"
            signature_payload = {
                "signature": signature.to_dict() if signature is not None else None,
                "input_example_columns": list(X_test.head(1).columns),
            }
            signature_path.write_text(json.dumps(signature_payload, indent=2), encoding="utf-8")
            log_artifact(signature_path)
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(1),
            )

    result = {
        "metrics": metrics,
        "model_output": str(args.model_output),
        "feature_columns": list(X_train.columns),
    }
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Model tersimpan di {result['model_output']}")


if __name__ == "__main__":
    main()
