from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
import os
from pathlib import Path
import sys

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from social_media_addiction import (  # noqa: E402
    DEFAULT_PROCESSED_FILENAME,
    TARGET_COLUMN,
    build_train_test_split,
    evaluate_regression,
    feature_importance_frame,
    log_prediction_artifacts,
    save_model_artifact,
)
from social_media_addiction.data import load_preprocessing_schema  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training model dengan tuning dan manual logging.")
    parser.add_argument(
        "--data-path",
        default=str(Path(__file__).resolve().parent / DEFAULT_PROCESSED_FILENAME),
    )
    parser.add_argument(
        "--schema-path",
        default=str(Path(__file__).resolve().parent / "preprocessing_schema.json"),
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
    )
    parser.add_argument(
        "--experiment-name",
        default="social-media-addiction-regression-tuning",
    )
    parser.add_argument(
        "--model-output",
        default=str(Path(__file__).resolve().parent / "artifacts" / "tuned_social_media_addiction_model.joblib"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=12)
    return parser


def train_with_tuning(args: argparse.Namespace) -> dict:
    processed_df = pd.read_csv(args.data_path)
    schema = load_preprocessing_schema(args.schema_path)

    X_train, X_test, y_train, y_test = build_train_test_split(
        processed_df,
        target_column=TARGET_COLUMN,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    search_space = {
        "n_estimators": [150, 200, 300, 400, 500],
        "max_depth": [None, 12, 18, 24, 30],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", 0.6, 0.8],
        "bootstrap": [True, False],
    }
    base_model = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=args.n_iter,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=args.random_state,
        n_jobs=-1,
        verbose=0,
    )

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

    def log_metric(key: str, value: float) -> None:
        if project_client and project_run_id:
            project_client.log_metric(project_run_id, key, float(value))
        else:
            mlflow.log_metric(key, value)

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

    if mlflow.active_run() or project_run_id:
        run_context = nullcontext()
    else:
        run_context = mlflow.start_run()
    with run_context:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        metrics = evaluate_regression(y_test, predictions)
        cv_rmse = float(abs(search.best_score_))

        log_params(
            {
                "model_type": "RandomForestRegressor",
                "dataset_rows": int(processed_df.shape[0]),
                "dataset_features": int(processed_df.shape[1] - 1),
                "categorical_columns": len(schema.get("categorical_columns", [])),
                "numeric_columns": len(schema.get("numeric_columns", [])),
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_iter": args.n_iter,
                "cv_folds": 5,
            }
        )
        log_params({f"best_{key}": value for key, value in search.best_params_.items()})
        log_metric("cv_rmse", cv_rmse)
        log_metrics(metrics)

        artifact_dir = Path(args.model_output).parent / "reports"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        feature_frame = feature_importance_frame(best_model, X_train.columns)
        feature_frame.to_csv(artifact_dir / "feature_importance.csv", index=False)
        log_artifact(artifact_dir / "feature_importance.csv")

        prediction_frame = pd.DataFrame(
            {
                "actual": y_test.reset_index(drop=True),
                "predicted": pd.Series(predictions),
                "residual": y_test.reset_index(drop=True) - pd.Series(predictions),
            }
        )
        prediction_frame.to_csv(artifact_dir / "prediction_samples.csv", index=False)
        log_artifact(artifact_dir / "prediction_samples.csv")

        artifact_paths = log_prediction_artifacts(
            y_true=y_test,
            y_pred=predictions,
            feature_importance=feature_frame,
            output_dir=artifact_dir,
        )
        for path in artifact_paths.values():
            log_artifact(path)

        summary_path = artifact_dir / "best_params.json"
        summary_path.write_text(json.dumps(search.best_params_, indent=2), encoding="utf-8")
        log_artifact(summary_path)

        save_model_artifact(best_model, args.model_output)
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
                best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(1),
            )

    return {
        "metrics": metrics,
        "cv_rmse": cv_rmse,
        "best_params": search.best_params_,
        "model_output": str(args.model_output),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = train_with_tuning(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
