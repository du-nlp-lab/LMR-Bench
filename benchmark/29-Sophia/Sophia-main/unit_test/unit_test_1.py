import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List, Optional

from sophia import SophiaG, sophiag, _single_tensor_sophiag

import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")


class TinyModel(nn.Module):
    """
    A tiny feedforward network for testing.
    """
    def __init__(self, in_dim=4, hidden_dim=5, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class TestSophiaG(unittest.TestCase):
    """
    Unit tests for the SophiaG optimizer. We validate that
    for three sets of hyperparameters, the final parameters
    match known-good reference values.
    """

    def setUp(self):
        """
        Create a small synthetic dataset and a function to train for a few steps.
        """
        torch.manual_seed(42)

        # Synthetic dataset: 16 samples, 4 features
        self.X = torch.randn(16, 4)
        # Two-class classification
        self.y = torch.randint(0, 2, (16,))

        # We'll train a small model
        self.model = TinyModel()

    def run_sophia_test(
        self,
        lr,
        betas,
        rho,
        weight_decay,
        batch_size,
        steps,
        ref_params
    ):
        """
        Helper to run a short training loop using SophiaG and compare final parameters
        to a reference. 'ref_params' is a dictionary of param_name -> reference_tensor.
        """

        # Reinitialize model so that each test starts from the same parameters
        torch.manual_seed(42)
        model = TinyModel()

        optimizer = SophiaG(
            model.parameters(),
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay
        )

        # Very simple training loop with all samples at once (for demonstration).
        for step in range(steps):
            # Forward
            pred = model(self.X)
            loss = F.cross_entropy(pred, self.y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # (Optional) update the Hessian approximation
            optimizer.update_hessian()

            # Update parameters
            optimizer.step(bs=batch_size)

        # Compare final parameters to reference
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.assertIn(name, ref_params,
                              f"Missing reference for parameter '{name}'")

                # Check shape is the same
                self.assertEqual(
                    param.shape, ref_params[name].shape,
                    f"Shape mismatch in parameter '{name}'"
                )

                # Compare values: adjust rtol, atol as needed
                torch.testing.assert_close(
                    param, ref_params[name],
                    rtol=1e-2, atol=1e-3,
                    msg=f"Parameter '{name}' not close to reference"
                )

    def test_sophia_group1(self):
        """
        Test SophiaG with the first group of hyperparameters.
        """
        # Pre-recorded reference parameters after running the exact same steps.
        # You must fill in these reference weights from a known-good run.
        ## Reference Parameters for Group 1
        REF_PARAMS_GROUP1 = {
            "fc1.weight": torch.tensor([
                [0.3821, 0.4151, -0.1173, 0.4594],
                [-0.1100, 0.1007, -0.2429, 0.2931],
                [0.4412, -0.3670, 0.4341, 0.0940],
                [0.3689, 0.0682, 0.2406, -0.0701],
                [0.3859, 0.0743, -0.2339, 0.1278]
            ]),
            "fc1.bias": torch.tensor([-0.2303, -0.0589, -0.2029,  0.3322, -0.3942]),
            "fc2.weight": torch.tensor([
                [-0.2062, -0.1265, -0.2693, 0.0420, -0.4422],
                [0.4040, -0.3796, 0.3456, 0.0746, -0.1447]
            ]),
            "fc2.bias": torch.tensor([0.2763, 0.0697]),
        }

        lr = 1e-4
        betas = (0.965, 0.99)
        rho = 0.04
        weight_decay = 1e-1
        batch_size = 5120  # This was from the original prompt
        steps = 5

        log_details = (f"lr={lr}, betas={betas}, rho={rho}, wd={weight_decay}, "
                       f"batch_size={batch_size}, steps={steps}")
        try:
            self.run_sophia_test(
                lr=lr,
                betas=betas,
                rho=rho,
                weight_decay=weight_decay,
                batch_size=batch_size,
                steps=steps,
                ref_params=REF_PARAMS_GROUP1
            )
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")
        except AssertionError as e:
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise

    def test_sophia_group2(self):
        """
        Test SophiaG with the second group of hyperparameters (different betas).
        """
        # Pre-recorded reference parameters after the same steps.
        # Again, replace these with real reference values from a known-good run.
        REF_PARAMS_GROUP2 = {
            "fc1.weight": torch.tensor([
                [0.3818, 0.4153, -0.1176, 0.4595],
                [-0.1100, 0.1005, -0.2429, 0.2931],
                [0.4412, -0.3673, 0.4341, 0.0941],
                [0.3689, 0.0682, 0.2406, -0.0701],
                [0.3859, 0.0744, -0.2339, 0.1279]
            ]),
            "fc1.bias": torch.tensor([-0.2301, -0.0591, -0.2026,  0.3322, -0.3942]),
            "fc2.weight": torch.tensor([
                [-0.2064, -0.1268, -0.2694, 0.0417, -0.4422],
                [0.4041, -0.3794, 0.3457, 0.0749, -0.1447]
            ]),
            "fc2.bias": torch.tensor([0.2762, 0.0698]),
        }

        lr = 1e-4
        betas = (0.9, 0.99)  # Different betas
        rho = 0.04
        weight_decay = 1e-1
        batch_size = 5120
        steps = 5

        log_details = (f"lr={lr}, betas={betas}, rho={rho}, wd={weight_decay}, "
                       f"batch_size={batch_size}, steps={steps}")
        try:
            self.run_sophia_test(
                lr=lr,
                betas=betas,
                rho=rho,
                weight_decay=weight_decay,
                batch_size=batch_size,
                steps=steps,
                ref_params=REF_PARAMS_GROUP2
            )
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")
        except AssertionError as e:
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise

    def test_sophia_group3(self):
        """
        Test SophiaG with a third group of hyperparameters
        (larger lr, bigger rho, etc.).
        """
        # Pre-recorded reference parameters after the same steps.
        # Replace with real reference values from a known-good run.
        REF_PARAMS_GROUP3 = {
            "fc1.weight": torch.tensor([
                [0.3814, 0.4152, -0.1177, 0.4593],
                [-0.1136, 0.1002, -0.2383, 0.2908],
                [0.4423, -0.3673, 0.4317, 0.0954],
                [0.3642, 0.0727, 0.2360, -0.0656],
                [0.3871, 0.0755, -0.2371, 0.1290]
            ]),
            "fc1.bias": torch.tensor([-0.2298, -0.0598, -0.2022,  0.3359, -0.3928]),
            "fc2.weight": torch.tensor([
                [-0.2064, -0.1273, -0.2703, 0.0414, -0.4434],
                [0.4040, -0.3786, 0.3466, 0.0751, -0.1432]
            ]),
            "fc2.bias": torch.tensor([0.2760, 0.0698]),
        }

        lr = 1e-3  # Larger lr
        betas = (0.965, 0.99)
        rho = 0.1  # Bigger rho
        weight_decay = 1e-1
        batch_size = 5120
        steps = 5

        log_details = (f"lr={lr}, betas={betas}, rho={rho}, wd={weight_decay}, "
                       f"batch_size={batch_size}, steps={steps}")
        try:
            self.run_sophia_test(
                lr=lr,
                betas=betas,
                rho=rho,
                weight_decay=weight_decay,
                batch_size=batch_size,
                steps=steps,
                ref_params=REF_PARAMS_GROUP3
            )
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")
        except AssertionError as e:
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise


if __name__ == '__main__':
    unittest.main()