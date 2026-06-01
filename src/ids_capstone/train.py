from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ids_capstone.data import DatasetSplits
from ids_capstone.losses import FocalLoss
from ids_capstone.metrics import compute_metrics
from ids_capstone.models import CNNBiLSTMIDS, CNNTransformerIDS, DeepAutoencoderIDS, MLPIDS


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_loss(config: dict, splits: DatasetSplits, device: torch.device) -> nn.Module:
    labels = splits.train.labels.numpy()
    training_config = config["training"]
    class_weighting = training_config.get("class_weighting")
    weight = None
    if class_weighting == "balanced":
        weights = compute_class_weight("balanced", classes=np.arange(splits.num_classes), y=labels)
        weight = torch.tensor(weights, dtype=torch.float32, device=device)

    if training_config["loss"] == "focal":
        return FocalLoss(gamma=float(training_config.get("focal_gamma", 2.0)), weight=weight)
    if training_config["loss"] == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    if training_config["loss"] == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {training_config['loss']}")


def build_model(config: dict, splits: DatasetSplits) -> nn.Module:
    model_config = config["model"]
    model_name = model_config.get("name", "cnn_transformer")
    if model_name == "mlp":
        return MLPIDS(
            input_dim=splits.input_dim,
            num_classes=splits.num_classes,
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            dropout=float(model_config["dropout"]),
        )
    if model_name == "deep_autoencoder":
        return DeepAutoencoderIDS(
            input_dim=splits.input_dim,
            num_classes=splits.num_classes,
            latent_dim=int(model_config.get("latent_dim", 64)),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            dropout=float(model_config["dropout"]),
        )
    if model_name == "cnn_bilstm":
        return CNNBiLSTMIDS(
            input_dim=splits.input_dim,
            num_classes=splits.num_classes,
            cnn_channels=int(model_config["cnn_channels"]),
            lstm_hidden_dim=int(model_config.get("lstm_hidden_dim", 64)),
            lstm_layers=int(model_config.get("lstm_layers", 1)),
            dropout=float(model_config["dropout"]),
        )
    if model_name != "cnn_transformer":
        raise ValueError(f"Unsupported model: {model_name}")
    return CNNTransformerIDS(
        input_dim=splits.input_dim,
        num_classes=splits.num_classes,
        cnn_channels=int(model_config["cnn_channels"]),
        transformer_dim=int(model_config["transformer_dim"]),
        transformer_layers=int(model_config["transformer_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        use_cnn=bool(model_config.get("use_cnn", True)),
    )


def evaluate(model: nn.Module, loader: DataLoader, class_names: list[str], device: torch.device) -> dict:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            output = model(features)
            logits = output[0] if isinstance(output, tuple) else output
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return compute_metrics(np.array(targets), np.array(predictions), class_names)


def train_model(config: dict, splits: DatasetSplits) -> dict:
    torch.manual_seed(int(config["experiment"]["seed"]))
    device = resolve_device(config["training"]["device"])
    run_dir = Path(config["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(splits.train, batch_size=int(config["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(splits.val, batch_size=int(config["training"]["batch_size"]))
    test_loader = DataLoader(splits.test, batch_size=int(config["training"]["batch_size"]))

    model = build_model(config, splits).to(device)
    criterion = build_loss(config, splits, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_macro_f1 = -1.0
    best_path = run_dir / "best_model.pt"
    history = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for features, labels in progress:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            if isinstance(output, tuple):
                logits, reconstruction = output
                classification_loss = criterion(logits, labels)
                reconstruction_loss = nn.functional.mse_loss(reconstruction, features)
                loss = classification_loss + float(config["training"].get("reconstruction_weight", 0.1)) * reconstruction_loss
            else:
                loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            progress.set_postfix(loss=loss.item())

        val_metrics = evaluate(model, val_loader, splits.class_names, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / len(splits.train),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, splits.class_names, device)
    result = {"history": history, "test": test_metrics, "class_names": splits.class_names}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
