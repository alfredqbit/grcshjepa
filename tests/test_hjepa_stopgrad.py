import torch

from grcshjepa.models.encoders import SortingEncoder
from grcshjepa.models.hjepa import HJEPA
from grcshjepa.models.predictors import MLPPredictor


def make_model():
    return HJEPA(
        encoder=SortingEncoder(length=8, latent_dim=16),
        latent_dim=16,
        projection_dim=8,
        max_horizon=2,
        predictor=MLPPredictor(latent_dim=16, horizon_dim=16),
        horizon_dim=16,
    )


def test_stopgrad_target_embedding_has_no_gradient():
    model = make_model()
    x = torch.rand(4, 8)
    x_future = torch.rand(4, 8)
    horizon = torch.ones(4, dtype=torch.long)
    out = model(x, x_future, horizon)
    assert out["z_tgt"].requires_grad is False


def test_online_encoder_receives_gradients():
    model = make_model()
    x = torch.rand(4, 8)
    x_future = torch.rand(4, 8)
    horizon = torch.ones(4, dtype=torch.long)
    out = model(x, x_future, horizon)
    loss = (out["z_hat"] - out["z_tgt"]).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
