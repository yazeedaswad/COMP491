from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_run(run_dir: Path) -> dict[str, float | str]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {"run": run_dir.name, "status": "missing"}

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = data["test"]
    return {
        "run": run_dir.name,
        "status": "complete",
        "accuracy": test["accuracy"],
        "macro_precision": test["macro_precision"],
        "macro_recall": test["macro_recall"],
        "macro_f1": test["macro_f1"],
        "weighted_f1": test["weighted_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="reports/runs")
    parser.add_argument("--out", default="reports/run_comparison.csv")
    parser.add_argument(
        "runs",
        nargs="*",
        default=[
            "unsw_nb15_mlp_unweighted",
            "unsw_nb15_mlp_baseline",
            "unsw_nb15_deep_autoencoder_baseline",
            "unsw_nb15_cnn_bilstm_baseline",
            "unsw_nb15_cnn_transformer",
            "unsw_nb15_cnn_transformer_unweighted",
            "unsw_nb15_cnn_transformer_weighted_ce",
            "unsw_nb15_cnn_transformer_weighted_ce_tuned",
        ],
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows = [load_run(runs_root / run) for run in args.runs]
    fieldnames = ["run", "status", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["status"] == "missing":
            print(f"{row['run']}: missing")
        else:
            print(
                f"{row['run']}: "
                f"accuracy={row['accuracy']:.4f}, "
                f"macro_f1={row['macro_f1']:.4f}, "
                f"weighted_f1={row['weighted_f1']:.4f}"
            )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
