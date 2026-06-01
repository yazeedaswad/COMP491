from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml

from ids_capstone.data import load_processed_splits
from ids_capstone.train import build_model, resolve_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_model(config_path: Path, warmup_batches: int, timed_batches: int) -> dict[str, float | int | str]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    splits = load_processed_splits(config["data"]["processed_npz_path"])
    device = resolve_device(config["training"]["device"])
    model = build_model(config, splits).to(device)
    checkpoint_path = Path(config["outputs"]["run_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    batch_size = int(config["training"]["batch_size"])
    sample_count = min(batch_size, len(splits.test))
    sample = splits.test.features[:sample_count].to(device)

    with torch.no_grad():
        for _ in range(warmup_batches):
            _ = model(sample)
        synchronize(device)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(timed_batches):
                _ = model(sample)
        synchronize(device)
        elapsed = time.perf_counter() - start

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    avg_batch_ms = elapsed / timed_batches * 1000
    avg_sample_ms = avg_batch_ms / sample_count

    return {
        "run": config["experiment"]["name"],
        "model": config["model"]["name"],
        "task": config["experiment"]["task"],
        "device": str(device),
        "input_dim": splits.input_dim,
        "num_classes": splits.num_classes,
        "batch_size": sample_count,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint_size_mb": checkpoint_size_mb,
        "avg_batch_ms": avg_batch_ms,
        "avg_sample_ms": avg_sample_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/model_profiles.csv")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--timed-batches", type=int, default=100)
    parser.add_argument(
        "configs",
        nargs="*",
        default=[
            "configs/unsw_nb15_mlp_unweighted.yaml",
            "configs/unsw_nb15_mlp_baseline.yaml",
            "configs/unsw_nb15_deep_autoencoder_baseline.yaml",
            "configs/unsw_nb15_cnn_bilstm_baseline.yaml",
            "configs/unsw_nb15_cnn_transformer_weighted_ce.yaml",
        ],
    )
    args = parser.parse_args()

    rows = [profile_model(Path(config_path), args.warmup_batches, args.timed_batches) for config_path in args.configs]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['run']}: "
            f"params={row['parameters']}, "
            f"checkpoint={row['checkpoint_size_mb']:.3f} MB, "
            f"batch={row['avg_batch_ms']:.3f} ms, "
            f"sample={row['avg_sample_ms']:.6f} ms"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

