import torch

from grcshjepa.losses import anti_collapse_loss, normalized_prediction_loss


def test_anti_collapse_penalizes_constant_codes():
    y_const = torch.zeros(32, 8)
    loss, stats = anti_collapse_loss(y_const)
    assert loss.item() >= 7.5
    assert stats["ac_cov_trace"] == 0.0


def test_anti_collapse_has_gradients():
    y = torch.randn(32, 8, requires_grad=True)
    loss, _ = anti_collapse_loss(y)
    loss.backward()
    assert y.grad is not None
    assert torch.isfinite(y.grad).all()


def test_normalized_prediction_loss_zero_on_same_vector():
    z = torch.randn(8, 6)
    assert normalized_prediction_loss(z, z).item() < 1e-10
