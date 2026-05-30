from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

TARGET_COLUMN = "Addicted_Score"
ID_COLUMNS = ["Student_ID"]
DEFAULT_RAW_FILENAME = "Students Social Media Addiction.csv"
DEFAULT_PROCESSED_FILENAME = "Students_Social_Media_Addiction_preprocessing.csv"
DEFAULT_SCHEMA_FILENAME = "preprocessing_schema.json"


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def _strip_string_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    return cleaned


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan.")

    cleaned = _strip_string_values(df)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.drop(columns=[col for col in ID_COLUMNS if col in cleaned.columns])

    target = cleaned[[TARGET_COLUMN]].copy()
    features = cleaned.drop(columns=[TARGET_COLUMN])

    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    encoded_features = pd.get_dummies(features, columns=categorical_columns, dtype=int)
    feature_columns = encoded_features.columns.tolist()
    processed = encoded_features.copy()
    processed[TARGET_COLUMN] = target[TARGET_COLUMN].astype(int)
    processed = processed[feature_columns + [TARGET_COLUMN]]

    schema = {
        "target_column": TARGET_COLUMN,
        "dropped_columns": [col for col in ID_COLUMNS if col in df.columns],
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "feature_columns": feature_columns,
        "row_count": int(processed.shape[0]),
        "feature_count": int(len(feature_columns)),
    }
    return processed, schema


def save_preprocessed_dataset(processed_df: pd.DataFrame, output_paths: Iterable[str | Path]) -> list[Path]:
    saved_paths: list[Path] = []
    for output_path in output_paths:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(path, index=False)
        saved_paths.append(path)
    return saved_paths


def save_preprocessing_schema(schema: dict, output_paths: Iterable[str | Path]) -> list[Path]:
    saved_paths: list[Path] = []
    for output_path in output_paths:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_paths.append(path)
    return saved_paths


def load_preprocessing_schema(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preprocess_raw_file(
    raw_path: str | Path,
    output_paths: Iterable[str | Path],
    schema_paths: Iterable[str | Path] | None = None,
) -> tuple[pd.DataFrame, dict, list[Path]]:
    raw_df = load_raw_dataset(raw_path)
    processed_df, schema = preprocess_dataframe(raw_df)
    saved_dataset_paths = save_preprocessed_dataset(processed_df, output_paths)
    if schema_paths is not None:
        save_preprocessing_schema(schema, schema_paths)
    return processed_df, schema, saved_dataset_paths


def transform_raw_inputs(raw_inputs: pd.DataFrame, schema: dict) -> pd.DataFrame:
    frame = _strip_string_values(raw_inputs)
    if TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[TARGET_COLUMN])
    frame = frame.drop(columns=[col for col in ID_COLUMNS if col in frame.columns], errors="ignore")

    missing_columns = [column for column in schema.get("numeric_columns", []) if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Kolom numerik wajib belum ada pada payload: {missing_columns}")

    categorical_columns = [column for column in schema.get("categorical_columns", []) if column in frame.columns]
    encoded = pd.get_dummies(frame, columns=categorical_columns, dtype=int)
    feature_columns = schema["feature_columns"]
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded

