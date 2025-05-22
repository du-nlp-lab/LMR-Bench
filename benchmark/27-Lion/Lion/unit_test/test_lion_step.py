import unittest
import torch
import math
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
# Helper function to run a single step
# with the reference Lion and return the
# parameters and exp_avg after stepping.
###############################################
def run_reference_lion_step(
    param_init,
    grad_value,
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.0
):
    """
    param_init: torch.Tensor (initial parameter)
    grad_value: torch.Tensor (gradient)
    lr: learning rate
    betas: (beta1, beta2)
    weight_decay: weight decay
    Returns:
      updated_param: torch.Tensor (after one Lion step)
      exp_avg: torch.Tensor (the momentum buffer after the step)
    """
    # Clone to avoid modifying inputs
    p = param_init.clone().detach().requires_grad_(True)
    p.grad = grad_value.clone().detach()

    optimizer = Lion([p], lr=lr, betas=betas, weight_decay=weight_decay)
    optimizer.step()

    # Retrieve the exp_avg from optimizer state
    exp_avg = optimizer.state[p]["exp_avg"].clone().detach()
    return p.detach(), exp_avg


class TestLionStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Precompute the expected outputs using the reference Lion implementation
        for 3 representative groups of parameters and gradients.
        """
        torch.manual_seed(42)  # For reproducibility

        # Group 1
        param1 = torch.randn(2, 2)
        grad1 = torch.randn(2, 2)
        cls.ref_p1, cls.ref_exp_avg1 = run_reference_lion_step(
            param1, grad1, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0
        )
        cls.group1_data = (param1, grad1, 1e-4, (0.9, 0.99), 0.0)

        # Group 2
        param2 = torch.randn(2, 2)
        grad2 = torch.randn(2, 2)
        cls.ref_p2, cls.ref_exp_avg2 = run_reference_lion_step(
            param2, grad2, lr=1e-3, betas=(0.8, 0.9), weight_decay=1e-4
        )
        cls.group2_data = (param2, grad2, 1e-3, (0.8, 0.9), 1e-4)

        # Group 3
        param3 = torch.randn(3, 3)
        grad3 = torch.randn(3, 3)
        cls.ref_p3, cls.ref_exp_avg3 = run_reference_lion_step(
            param3, grad3, lr=5e-5, betas=(0.95, 0.999), weight_decay=1e-5
        )
        cls.group3_data = (param3, grad3, 5e-5, (0.95, 0.999), 1e-5)

    def test_step_group1(self):
        """Test if Group 1 parameters match the reference output."""
        param, grad, lr, betas, wd = self.group1_data
        test_p, test_exp_avg = run_reference_lion_step(param, grad, lr, betas, wd)
        # Compare with precomputed reference
        self.assertTrue(
            torch.allclose(test_p, self.ref_p1, rtol=1e-6, atol=1e-8),
            f"Group 1 final parameters mismatch.\nGot:\n{test_p}\nExpected:\n{self.ref_p1}"
        )
        self.assertTrue(
            torch.allclose(test_exp_avg, self.ref_exp_avg1, rtol=1e-6, atol=1e-8),
            f"Group 1 exp_avg mismatch.\nGot:\n{test_exp_avg}\nExpected:\n{self.ref_exp_avg1}"
        )

    def test_step_group2(self):
        """Test if Group 2 parameters match the reference output."""
        param, grad, lr, betas, wd = self.group2_data
        test_p, test_exp_avg = run_reference_lion_step(param, grad, lr, betas, wd)
        # Compare with precomputed reference
        self.assertTrue(
            torch.allclose(test_p, self.ref_p2, rtol=1e-6, atol=1e-8),
            f"Group 2 final parameters mismatch.\nGot:\n{test_p}\nExpected:\n{self.ref_p2}"
        )
        self.assertTrue(
            torch.allclose(test_exp_avg, self.ref_exp_avg2, rtol=1e-6, atol=1e-8),
            f"Group 2 exp_avg mismatch.\nGot:\n{test_exp_avg}\nExpected:\n{self.ref_exp_avg2}"
        )

    def test_step_group3(self):
        """Test if Group 3 parameters match the reference output."""
        param, grad, lr, betas, wd = self.group3_data
        test_p, test_exp_avg = run_reference_lion_step(param, grad, lr, betas, wd)
        # Compare with precomputed reference
        self.assertTrue(
            torch.allclose(test_p, self.ref_p3, rtol=1e-6, atol=1e-8),
            f"Group 3 final parameters mismatch.\nGot:\n{test_p}\nExpected:\n{self.ref_p3}"
        )
        self.assertTrue(
            torch.allclose(test_exp_avg, self.ref_exp_avg3, rtol=1e-6, atol=1e-8),
            f"Group 3 exp_avg mismatch.\nGot:\n{test_exp_avg}\nExpected:\n{self.ref_exp_avg3}"
        )


if __name__ == "__main__":
    unittest.main()