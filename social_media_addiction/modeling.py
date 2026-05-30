from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data import TARGET_COLUMN


def build_train_test_split(
    processed_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
):
    feature_frame = processed_df.drop(columns=[target_column])
    target = processed_df[target_column]
    return train_test_split(feature_frame, target, test_size=test_size, random_state=random_state)


def build_default_regressor(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )


def evaluate_regression(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100),
    }


def save_model_artifact(model, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def _prepare_plot_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_prediction_artifacts(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    feature_importance: pd.DataFrame | None,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    residuals = y_true - y_pred

    paths: dict[str, Path] = {}

    scatter_path = _prepare_plot_path(output_dir / "actual_vs_predicted.png")
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color="#0f766e")
    min_value = float(min(y_true.min(), y_pred.min()))
    max_value = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#7c3aed")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Addicted Score")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    paths["actual_vs_predicted"] = scatter_path

    residual_path = _prepare_plot_path(output_dir / "residual_distribution.png")
    plt.figure(figsize=(7, 6))
    plt.hist(residuals, bins=20, color="#2563eb", alpha=0.85)
    plt.axvline(0, color="#111827", linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(residual_path, dpi=200)
    plt.close()
    paths["residual_distribution"] = residual_path

    if feature_importance is not None and not feature_importance.empty:
        feature_path = _prepare_plot_path(output_dir / "feature_importance.png")
        top_features = feature_importance.head(15).iloc[::-1]
        plt.figure(figsize=(8, max(4, 0.35 * len(top_features))))
        plt.barh(top_features["feature"], top_features["importance"], color="#16a34a")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Top Feature Importance")
        plt.tight_layout()
        plt.savefig(feature_path, dpi=200)
        plt.close()
        paths["feature_importance"] = feature_path

    summary = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "residual": residuals,
        }
    )
    summary_path = _prepare_plot_path(output_dir / "prediction_summary.csv")
    summary.to_csv(summary_path, index=False)
    paths["prediction_summary"] = summary_path

    return paths


def feature_importance_frame(model, feature_names: Iterable[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    importance = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)

