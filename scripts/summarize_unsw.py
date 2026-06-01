from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(path: Path, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[label_column])
    labels = frame[label_column].astype(str).str.strip()
    counts = labels.value_counts().rename_axis("class").reset_index(name="count")
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    counts.insert(0, "split", path.stem)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/UNSW_NB15_training-set.csv")
    parser.add_argument("--test", default="data/raw/UNSW_NB15_testing-set.csv")
    parser.add_argument("--label-column", default="attack_cat")
    parser.add_argument("--out", default="reports/unsw_nb15_class_distribution.csv")
    args = parser.parse_args()

    summary = pd.concat(
        [
            summarize(Path(args.train), args.label_column),
            summarize(Path(args.test), args.label_column),
        ],
        ignore_index=True,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

