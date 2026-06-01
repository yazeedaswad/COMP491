from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="reports/runs")
    parser.add_argument("--out", default="reports/per_class_comparison.csv")
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

    rows = []
    for run in args.runs:
        metrics_path = Path(args.runs_root) / run / "metrics.json"
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        report = data["test"]["report"]
        for class_name, values in report.items():
            if not isinstance(values, dict) or "f1-score" not in values:
                continue
            if class_name in {"macro avg", "weighted avg"}:
                continue
            rows.append(
                {
                    "run": run,
                    "class": class_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1-score"],
                    "support": values["support"],
                }
            )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run", "class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)

    for class_name in ["Normal", "Generic", "Exploits", "Worms", "Shellcode", "Analysis", "Backdoor"]:
        print(f"\n{class_name}")
        for row in rows:
            if row["class"] == class_name:
                print(
                    f"{row['run']}: "
                    f"P={row['precision']:.3f} "
                    f"R={row['recall']:.3f} "
                    f"F1={row['f1']:.3f}"
                )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
