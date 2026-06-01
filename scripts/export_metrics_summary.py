from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = data["test"]
    rows = [
        {"metric": "accuracy", "value": test["accuracy"]},
        {"metric": "macro_precision", "value": test["macro_precision"]},
        {"metric": "macro_recall", "value": test["macro_recall"]},
        {"metric": "macro_f1", "value": test["macro_f1"]},
        {"metric": "weighted_f1", "value": test["weighted_f1"]},
    ]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['metric']}: {row['value']:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

