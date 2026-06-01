from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_unsw_to_nsl_binary import (
    PROTO_BUCKETS,
    SERVICE_BUCKETS,
    STATE_BUCKETS,
    encode_common_features,
    read_nsl,
    stratified_val_indices,
)


def feature_names() -> np.ndarray:
    return np.array(
        [
            "log_duration",
            "log_src_bytes",
            "log_dst_bytes",
            *[f"proto={name}" for name in PROTO_BUCKETS],
            *[f"service={name}" for name in SERVICE_BUCKETS],
            *[f"state={name}" for name in STATE_BUCKETS],
        ],
        dtype=object,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsl-train", default="data/raw/KDDTrain+.txt")
    parser.add_argument("--unsw-test", default="data/raw/UNSW_NB15_testing-set.csv")
    parser.add_argument("--out", default="data/processed/nsl_to_unsw_binary_aligned.npz")
    parser.add_argument("--metadata-out", default="data/processed/nsl_to_unsw_binary_aligned_metadata.json")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nsl_path = Path(args.nsl_train)
    if not nsl_path.exists():
        raise FileNotFoundError(
            f"Missing NSL-KDD training file: {nsl_path}. "
            "Place KDDTrain+.txt under data/raw/ or pass --nsl-train."
        )

    unsw_path = Path(args.unsw_test)
    if not unsw_path.exists():
        raise FileNotFoundError(f"Missing UNSW-NB15 test file: {unsw_path}.")

    nsl = read_nsl(nsl_path)
    unsw = pd.read_csv(unsw_path)

    nsl_features_all, nsl_labels_all, mean, std = encode_common_features(nsl, "nsl")
    unsw_features, unsw_labels, _, _ = encode_common_features(unsw, "unsw", mean, std)

    val_indices = stratified_val_indices(nsl_labels_all, args.val_size, args.seed)
    train_mask = np.ones(len(nsl_labels_all), dtype=bool)
    train_mask[val_indices] = False

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=nsl_features_all[train_mask],
        y_train=nsl_labels_all[train_mask],
        x_val=nsl_features_all[val_indices],
        y_val=nsl_labels_all[val_indices],
        x_test=unsw_features,
        y_test=unsw_labels,
        class_names=np.array(["Normal", "Attack"], dtype=object),
        feature_names=feature_names(),
    )

    metadata = {
        "task": "binary reverse cross-dataset transfer",
        "source_train": str(nsl_path),
        "target_test": "UNSW-NB15 official testing set",
        "train_rows": int(train_mask.sum()),
        "val_rows": int(len(val_indices)),
        "test_rows": int(len(unsw_labels)),
        "feature_count": int(nsl_features_all.shape[1]),
        "class_names": ["Normal", "Attack"],
        "alignment_note": "Reverse transfer using the same coarse semantic alignment as UNSW-to-NSL.",
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

