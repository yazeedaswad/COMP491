from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

PROTO_BUCKETS = ["tcp", "udp", "icmp", "other"]
SERVICE_BUCKETS = ["http", "ftp", "smtp", "dns", "ssh", "none", "other"]
STATE_BUCKETS = ["established", "reset", "reject", "other"]


def stratified_val_indices(labels: np.ndarray, val_size: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * val_size)))
        val_indices.extend(indices[:count].tolist())
    return np.array(sorted(val_indices), dtype=np.int64)


def protocol_bucket(value: object) -> str:
    text = str(value).lower().strip()
    return text if text in {"tcp", "udp", "icmp"} else "other"


def service_bucket(value: object) -> str:
    text = str(value).lower().strip()
    if text in {"-", "", "nan"}:
        return "none"
    if "http" in text:
        return "http"
    if "ftp" in text:
        return "ftp"
    if "smtp" in text:
        return "smtp"
    if text in {"dns", "domain", "domain_u"}:
        return "dns"
    if text in {"ssh", "sshv1"}:
        return "ssh"
    return "other"


def state_bucket(value: object, source: str) -> str:
    text = str(value).upper().strip()
    if source == "unsw":
        if text in {"FIN", "CON"}:
            return "established"
        if text == "RST":
            return "reset"
        if text == "REQ":
            return "reject"
        return "other"
    if text == "SF":
        return "established"
    if text in {"RSTO", "RSTR", "RSTOS0"}:
        return "reset"
    if text == "REJ":
        return "reject"
    return "other"


def one_hot(values: pd.Series, buckets: list[str]) -> np.ndarray:
    matrix = np.zeros((len(values), len(buckets)), dtype=np.float32)
    bucket_to_index = {bucket: index for index, bucket in enumerate(buckets)}
    for row_index, value in enumerate(values):
        matrix[row_index, bucket_to_index[value]] = 1.0
    return matrix


def encode_common_features(frame: pd.DataFrame, source: str, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source == "unsw":
        duration = frame["dur"]
        src_bytes = frame["sbytes"]
        dst_bytes = frame["dbytes"]
        protocol = frame["proto"].map(protocol_bucket)
        service = frame["service"].map(service_bucket)
        state = frame["state"].map(lambda value: state_bucket(value, source))
        labels = (frame["label"].astype(int) > 0).astype(np.int64).to_numpy()
    elif source == "nsl":
        duration = frame["duration"]
        src_bytes = frame["src_bytes"]
        dst_bytes = frame["dst_bytes"]
        protocol = frame["protocol_type"].map(protocol_bucket)
        service = frame["service"].map(service_bucket)
        state = frame["flag"].map(lambda value: state_bucket(value, source))
        labels = (frame["label"].astype(str).str.strip().str.rstrip(".").str.lower() != "normal").astype(np.int64).to_numpy()
    else:
        raise ValueError(f"Unsupported source: {source}")

    numeric = pd.DataFrame(
        {
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    numeric = np.log1p(numeric.to_numpy(dtype=np.float32))
    if mean is None or std is None:
        mean = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        std[std == 0] = 1.0
    numeric = (numeric - mean) / std

    features = np.concatenate(
        [
            numeric.astype(np.float32),
            one_hot(protocol, PROTO_BUCKETS),
            one_hot(service, SERVICE_BUCKETS),
            one_hot(state, STATE_BUCKETS),
        ],
        axis=1,
    )
    return features, labels, mean, std


def read_nsl(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, names=NSL_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsw-train", default="data/raw/UNSW_NB15_training-set.csv")
    parser.add_argument("--nsl-test", default="data/raw/KDDTest+.txt")
    parser.add_argument("--out", default="data/processed/unsw_to_nsl_binary_aligned.npz")
    parser.add_argument("--metadata-out", default="data/processed/unsw_to_nsl_binary_aligned_metadata.json")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nsl_path = Path(args.nsl_test)
    if not nsl_path.exists():
        raise FileNotFoundError(
            f"Missing NSL-KDD test file: {nsl_path}. "
            "Place KDDTest+.txt under data/raw/ or pass --nsl-test."
        )

    unsw = pd.read_csv(args.unsw_train)
    nsl = read_nsl(nsl_path)

    unsw_features_all, unsw_labels_all, mean, std = encode_common_features(unsw, "unsw")
    nsl_features, nsl_labels, _, _ = encode_common_features(nsl, "nsl", mean, std)

    val_indices = stratified_val_indices(unsw_labels_all, args.val_size, args.seed)
    train_mask = np.ones(len(unsw_labels_all), dtype=bool)
    train_mask[val_indices] = False

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=unsw_features_all[train_mask],
        y_train=unsw_labels_all[train_mask],
        x_val=unsw_features_all[val_indices],
        y_val=unsw_labels_all[val_indices],
        x_test=nsl_features,
        y_test=nsl_labels,
        class_names=np.array(["Normal", "Attack"], dtype=object),
        feature_names=np.array(
            [
                "log_duration",
                "log_src_bytes",
                "log_dst_bytes",
                *[f"proto={name}" for name in PROTO_BUCKETS],
                *[f"service={name}" for name in SERVICE_BUCKETS],
                *[f"state={name}" for name in STATE_BUCKETS],
            ],
            dtype=object,
        ),
    )

    metadata = {
        "task": "binary cross-dataset transfer",
        "source_train": "UNSW-NB15 official training set",
        "target_test": str(nsl_path),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(len(val_indices)),
        "test_rows": int(len(nsl_labels)),
        "feature_count": int(unsw_features_all.shape[1]),
        "class_names": ["Normal", "Attack"],
        "alignment_note": "Coarse semantic alignment using duration, byte counts, protocol bucket, service bucket, and connection-state bucket.",
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

