from __future__ import annotations

import argparse
import json

from ids_capstone.config import load_config
from ids_capstone.data import load_tabular_splits
from ids_capstone.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    splits = load_tabular_splits(config)
    result = train_model(config, splits)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()

