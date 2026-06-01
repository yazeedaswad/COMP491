from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSplits:
    train: "TabularIdsDataset"
    val: "TabularIdsDataset"
    test: "TabularIdsDataset"
    class_names: list[str]
    input_dim: int
    num_classes: int


class TabularIdsDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def load_tabular_splits(config: dict) -> DatasetSplits:
    data_config = config["data"]
    if "processed_npz_path" in data_config:
        return load_processed_splits(data_config["processed_npz_path"])

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    seed = int(config["experiment"]["seed"])
    label_column = data_config["label_column"]

    if "train_csv_path" in data_config and "test_csv_path" in data_config:
        train_source = pd.read_csv(data_config["train_csv_path"])
        test_source = pd.read_csv(data_config["test_csv_path"])
    else:
        source = pd.read_csv(data_config["csv_path"])
        train_source, test_source = train_test_split(
            source,
            test_size=float(data_config["test_size"]),
            random_state=seed,
            stratify=source[label_column],
        )

    missing = [
        column
        for column in [label_column, *data_config.get("drop_columns", [])]
        if column not in train_source.columns or column not in test_source.columns
    ]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    train_source = train_source.drop(columns=data_config.get("drop_columns", []))
    test_source = test_source.drop(columns=data_config.get("drop_columns", []))
    train_labels_raw = train_source.pop(label_column).astype(str).str.strip()
    test_labels_raw = test_source.pop(label_column).astype(str).str.strip()
    labels_raw = pd.concat([train_labels_raw, test_labels_raw], ignore_index=True)
    class_names = sorted(labels_raw.unique().tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_labels_all = train_labels_raw.map(class_to_index).to_numpy()
    test_labels = test_labels_raw.map(class_to_index).to_numpy()

    categorical_columns = data_config.get("categorical_columns", [])
    numeric_columns = [column for column in train_source.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )

    train_frame, val_frame, train_labels, val_labels = train_test_split(
        train_source,
        train_labels_all,
        test_size=float(data_config["val_size"]),
        random_state=seed,
        stratify=train_labels_all,
    )

    train_features = preprocessor.fit_transform(train_frame)
    val_features = preprocessor.transform(val_frame)
    test_features = preprocessor.transform(test_source)

    return DatasetSplits(
        train=TabularIdsDataset(train_features, train_labels),
        val=TabularIdsDataset(val_features, val_labels),
        test=TabularIdsDataset(test_features, test_labels),
        class_names=class_names,
        input_dim=train_features.shape[1],
        num_classes=len(class_names),
    )


def load_processed_splits(path: str) -> DatasetSplits:
    archive = np.load(path, allow_pickle=True)
    class_names = archive["class_names"].tolist()
    return DatasetSplits(
        train=TabularIdsDataset(archive["x_train"], archive["y_train"]),
        val=TabularIdsDataset(archive["x_val"], archive["y_val"]),
        test=TabularIdsDataset(archive["x_test"], archive["y_test"]),
        class_names=class_names,
        input_dim=archive["x_train"].shape[1],
        num_classes=len(class_names),
    )
