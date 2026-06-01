from __future__ import annotations

import torch
from torch import nn


class CNNTransformerIDS(nn.Module):
    """Lightweight CNN-Transformer for tabular flow features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_channels: int = 64,
        transformer_dim: int = 64,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.2,
        use_cnn: bool = True,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(1, transformer_dim)
        self.use_cnn = use_cnn
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, input_dim, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=attention_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            local_features = self.cnn(features.unsqueeze(1)).transpose(1, 2)
        else:
            local_features = features.unsqueeze(-1)
        tokens = self.feature_projection(local_features) + self.position_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


class MLPIDS(nn.Module):
    """Simple feed-forward baseline for tabular intrusion detection."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class DeepAutoencoderIDS(nn.Module):
    """Autoencoder-based IDS baseline with a supervised classifier head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(features)
        reconstruction = self.decoder(latent)
        logits = self.classifier(latent)
        return logits, reconstruction


class CNNBiLSTMIDS(nn.Module):
    """CNN-BiLSTM hybrid baseline for tabular flow-feature sequences."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_channels: int = 64,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim * 2),
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local_features = self.cnn(features.unsqueeze(1)).transpose(1, 2)
        sequence, _ = self.bilstm(local_features)
        pooled = sequence.mean(dim=1)
        return self.classifier(pooled)
