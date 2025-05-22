import unittest
import torch
from torch.optim.optimizer import Optimizer

###############################################
# Reference Implementation (as given above)
###############################################
class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        # TODO: Implement code here
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        # TODO: Implement code here
        return loss

###############################################
# Tests for the Lion optimizer constructor
###############################################
class TestLionInit(unittest.TestCase):
    def test_valid_init(self):
        """Test that valid hyperparameters do not raise errors."""
        # 3 groups of (lr, betas, weight_decay) to mimic realistic usage
        test_configs = [
            (1e-4, (0.9, 0.99), 0.0),
            (1e-3, (0.8, 0.9), 1e-4),
            (5e-5, (0.95, 0.999), 1e-5),
        ]
        for idx, (lr, betas, wd) in enumerate(test_configs, start=1):
            params = [torch.randn(10, 10, requires_grad=True)]
            try:
                opt = Lion(params, lr=lr, betas=betas, weight_decay=wd)
            except Exception as e:
                self.fail(f"Config #{idx} raised an unexpected exception: {e}")
            self.assertEqual(opt.defaults["lr"], lr)
            self.assertEqual(opt.defaults["betas"], betas)
            self.assertEqual(opt.defaults["weight_decay"], wd)

    def test_invalid_lr(self):
        """Test that invalid learning rates raise a ValueError."""
        params = [torch.randn(10, 10, requires_grad=True)]
        with self.assertRaises(ValueError):
            Lion(params, lr=-1e-4)

    def test_invalid_betas(self):
        """Test that invalid betas raise a ValueError."""
        params = [torch.randn(10, 10, requires_grad=True)]
        # Beta[0] out of range
        with self.assertRaises(ValueError):
            Lion(params, betas=(-0.1, 0.99))
        # Beta[1] out of range
        with self.assertRaises(ValueError):
            Lion(params, betas=(0.9, 1.0))


if __name__ == "__main__":
    unittest.main()