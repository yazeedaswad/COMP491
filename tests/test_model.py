import torch

from ids_capstone.models import CNNBiLSTMIDS, CNNTransformerIDS, DeepAutoencoderIDS, MLPIDS


def test_cnn_transformer_forward_shape():
    model = CNNTransformerIDS(input_dim=12, num_classes=4)
    logits = model(torch.randn(8, 12))
    assert logits.shape == (8, 4)


def test_transformer_without_cnn_forward_shape():
    model = CNNTransformerIDS(input_dim=12, num_classes=4, use_cnn=False)
    logits = model(torch.randn(8, 12))
    assert logits.shape == (8, 4)


def test_mlp_forward_shape():
    model = MLPIDS(input_dim=12, num_classes=4)
    logits = model(torch.randn(8, 12))
    assert logits.shape == (8, 4)


def test_deep_autoencoder_forward_shape():
    model = DeepAutoencoderIDS(input_dim=12, num_classes=4)
    logits, reconstruction = model(torch.randn(8, 12))
    assert logits.shape == (8, 4)
    assert reconstruction.shape == (8, 12)


def test_cnn_bilstm_forward_shape():
    model = CNNBiLSTMIDS(input_dim=12, num_classes=4)
    logits = model(torch.randn(8, 12))
    assert logits.shape == (8, 4)
