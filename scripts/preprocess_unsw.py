from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_val_indices(labels: np.ndarray, val_size: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        count = max(1, int(round(len(class_indices) * val_size)))
        val_indices.extend(class_indices[:count].tolist())
    return np.array(sorted(val_indices), dtype=np.int64)


def encode_frame(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    categories: dict[str, list[str]],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    numeric = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    numeric = (numeric - mean) / std
    parts = [numeric.astype(np.float32)]
    feature_names = numeric_columns.copy()

    for column in categorical_columns:
        values = frame[column].astype(str).fillna("-")
        for category in categories[column]:
            parts.append((values == category).to_numpy(dtype=np.float32).reshape(-1, 1))
            feature_names.append(f"{column}={category}")

    return np.concatenate(parts, axis=1), feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/UNSW_NB15_training-set.csv")
    parser.add_argument("--test", default="data/raw/UNSW_NB15_testing-set.csv")
    parser.add_argument("--out", default="data/processed/unsw_nb15_multiclass.npz")
    parser.add_argument("--metadata-out", default="data/processed/unsw_nb15_multiclass_metadata.json")
    parser.add_argument("--label-column", default="attack_cat")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    categorical_columns = ["proto", "service", "state"]
    drop_columns = ["id", "label"]

    train_source = pd.read_csv(args.train)
    test_source = pd.read_csv(args.test)
    train_source = train_source.drop(columns=drop_columns)
    test_source = test_source.drop(columns=drop_columns)

    train_labels_raw = train_source.pop(args.label_column).astype(str).str.strip()
    test_labels_raw = test_source.pop(args.label_column).astype(str).str.strip()
    class_names = sorted(pd.concat([train_labels_raw, test_labels_raw], ignore_index=True).unique().tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    y_train_all = train_labels_raw.map(class_to_index).to_numpy(dtype=np.int64)
    y_test = test_labels_raw.map(class_to_index).to_numpy(dtype=np.int64)

    val_indices = stratified_val_indices(y_train_all, args.val_size, args.seed)
    train_mask = np.ones(len(y_train_all), dtype=bool)
    train_mask[val_indices] = False

    train_frame = train_source.loc[train_mask].reset_index(drop=True)
    val_frame = train_source.loc[val_indices].reset_index(drop=True)
    y_train = y_train_all[train_mask]
    y_val = y_train_all[val_indices]

    numeric_columns = [column for column in train_frame.columns if column not in categorical_columns]
    train_numeric = train_frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    mean = train_numeric.mean(axis=0)
    std = train_numeric.std(axis=0)
    std[std == 0] = 1.0

    categories = {
        column: sorted(train_frame[column].astype(str).fillna("-").unique().tolist())
        for column in categorical_columns
    }

    x_train, feature_names = encode_frame(train_frame, numeric_columns, categorical_columns, categories, mean, std)
    x_val, _ = encode_frame(val_frame, numeric_columns, categorical_columns, categories, mean, std)
    x_test, _ = encode_frame(test_source, numeric_columns, categorical_columns, categories, mean, std)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=np.array(class_names, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
    )

    metadata = {
        "class_names": class_names,
        "feature_count": len(feature_names),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

