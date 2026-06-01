# Imbalance-Aware CNN-Transformer IDS

Capstone project for network intrusion detection using UNSW-NB15 and NSL-KDD. The project implements a proposed CNN-Transformer IDS model and compares it with prior-work-style baselines under same-dataset, imbalance-aware, ablation, cross-dataset, and deployment-oriented evaluations.

## Project Summary

The original research gap was that many IDS studies report high accuracy on one benchmark dataset, but do not sufficiently test class imbalance, cross-dataset generalization, or deployment practicality.

This repository addresses that gap with:

- Multiclass UNSW-NB15 experiments.
- Prior-work baselines: Deep Autoencoder IDS and CNN-BiLSTM IDS.
- Extra MLP reference baselines.
- Loss ablations: unweighted cross entropy, weighted cross entropy, focal loss.
- Architecture ablations: removing the CNN stage and reducing Transformer layers.
- Binary cross-dataset transfer in both directions:
  - UNSW-NB15 -> NSL-KDD
  - NSL-KDD -> UNSW-NB15
- Deployment profiling: parameter count, checkpoint size, and inference latency.

## Main Findings

- The best overall same-dataset model was the unweighted MLP baseline.
- The strongest prior-work-style baseline was the Deep Autoencoder.
- The proposed CNN-Transformer with weighted cross entropy was competitive with CNN-BiLSTM on UNSW-NB15.
- Removing the CNN stage improved the Transformer variant, suggesting that CNN locality is not always useful for engineered tabular flow features.
- In reverse transfer, NSL-KDD -> UNSW-NB15, the CNN-Transformer achieved the best macro-F1 among the tested transfer models.
- Cross-dataset performance was much weaker than same-dataset performance, confirming that dataset shift remains difficult.

## Repository Layout

```text
configs/              YAML experiment configurations
data/                 Local datasets and generated processed arrays
reports/              Generated metrics, summaries, and report artifacts
scripts/              Preprocessing, training, comparison, profiling
src/ids_capstone/     Python package with models, data loading, metrics, and training
tests/                Lightweight model smoke tests
```

Large datasets, processed arrays, model checkpoints, and generated report outputs are intentionally ignored by Git.

## Setup

Use Python 3.10+ with PyTorch.

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest
```

## Dataset Files

You can download the datasets from:
UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

NSL-KDD: https://www.kaggle.com/datasets/hassan06/nslkdd


```text
data/raw/
```

Required for UNSW-NB15 same-dataset experiments:

```text
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
data/raw/NUSW-NB15_features.csv
```

Required for UNSW-NB15 -> NSL-KDD transfer:

```text
data/raw/KDDTest+.txt
```

Required for NSL-KDD -> UNSW-NB15 reverse transfer:

```text
data/raw/KDDTrain+.txt
```

## Preprocessing

UNSW-NB15 multiclass preprocessing:

```bash
python scripts/preprocess_unsw.py
```

UNSW-NB15 -> NSL-KDD binary transfer preprocessing:

```bash
python scripts/preprocess_unsw_to_nsl_binary.py
```

NSL-KDD -> UNSW-NB15 binary reverse transfer preprocessing:

```bash
python scripts/preprocess_nsl_to_unsw_binary.py
```

## Same-Dataset Experiments

Run the main UNSW-NB15 experiments:

```bash
python scripts/train.py --config configs/unsw_nb15_mlp_unweighted.yaml
python scripts/train.py --config configs/unsw_nb15_mlp_baseline.yaml
python scripts/train.py --config configs/unsw_nb15_deep_autoencoder_baseline.yaml
python scripts/train.py --config configs/unsw_nb15_cnn_bilstm_baseline.yaml
python scripts/train.py --config configs/unsw_nb15_cnn_transformer_weighted_ce.yaml
python scripts/train.py --config configs/unsw_nb15_cnn_transformer_unweighted.yaml
python scripts/train.py --config configs/unsw_nb15_cnn_transformer.yaml
```

Compare runs:

```bash
python scripts/compare_runs.py
python scripts/export_per_class_comparison.py
```

## Ablation Experiments

Run architecture ablations:

```bash
python scripts/train.py --config configs/unsw_nb15_cnn_transformer_no_cnn_ablation.yaml
python scripts/train.py --config configs/unsw_nb15_cnn_transformer_one_layer_ablation.yaml
```

Export the ablation table:

```bash
python scripts/compare_runs.py unsw_nb15_cnn_transformer_weighted_ce unsw_nb15_cnn_transformer_no_cnn_ablation unsw_nb15_cnn_transformer_one_layer_ablation unsw_nb15_cnn_transformer --out reports/cnn_transformer_ablation.csv
```

## Cross-Dataset Transfer

UNSW-NB15 -> NSL-KDD:

```bash
python scripts/train.py --config configs/unsw_to_nsl_deep_autoencoder_baseline.yaml
python scripts/train.py --config configs/unsw_to_nsl_cnn_bilstm_baseline.yaml
python scripts/train.py --config configs/unsw_to_nsl_cnn_transformer_weighted_ce.yaml

python scripts/compare_runs.py unsw_to_nsl_deep_autoencoder_baseline unsw_to_nsl_cnn_bilstm_baseline unsw_to_nsl_cnn_transformer_weighted_ce --out reports/unsw_to_nsl_run_comparison.csv
```

NSL-KDD -> UNSW-NB15:

```bash
python scripts/train.py --config configs/nsl_to_unsw_deep_autoencoder_baseline.yaml
python scripts/train.py --config configs/nsl_to_unsw_cnn_bilstm_baseline.yaml
python scripts/train.py --config configs/nsl_to_unsw_cnn_transformer_weighted_ce.yaml

python scripts/compare_runs.py nsl_to_unsw_deep_autoencoder_baseline nsl_to_unsw_cnn_bilstm_baseline nsl_to_unsw_cnn_transformer_weighted_ce --out reports/nsl_to_unsw_run_comparison.csv
```

## Notes

- The cross-dataset experiments use binary Normal-vs-Attack labels because UNSW-NB15 and NSL-KDD do not share the same multiclass attack catogorization.
- Cross-dataset feature alignment is intentionally compact and documented in `docs/CROSS_DATASET_PLAN.md`.
- Results should be interpreted as empirical findings, not as a claim that the proposed model outperforms all baselines.

