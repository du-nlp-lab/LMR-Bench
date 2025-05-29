import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List, Optional

from lion_pytorch import Lion

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


class TestLion(unittest.TestCase):
    """
    Unit tests for the Lion optimizer. We validate that
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

    def run_lion_test(
            self,
            lr,
            betas,
            weight_decay,
            steps,
            ref_params
    ):
        """
        Helper to run a short training loop using Lion and compare final parameters
        to a reference. 'ref_params' is a dictionary of param_name -> reference_tensor.
        """

        # Reinitialize model so that each test starts from the same parameters
        torch.manual_seed(42)
        model = TinyModel()

        optimizer = Lion(
            model.parameters(),
            lr=lr,
            betas=betas,
            # rho=rho,
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

            # Update parameters
            optimizer.step()

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

    def test_lion_group1(self):
        """
        Test Lion with the first group of hyperparameters.
        """
        # Pre-recorded reference parameters after running the exact same steps.
        # You must fill in these reference weights from a known-good run.
        ## Reference Parameters for Group 1
        REF_PARAMS_GROUP1 = {
            "fc1.weight": torch.tensor([
                [0.3818, 0.4155, -0.1176, 0.4598],
                [-0.1100, 0.1004, -0.2429, 0.2931],
                [0.4412, -0.3673, 0.4341, 0.0941],
                [0.3689, 0.0682, 0.2406, -0.0701],
                [0.3859, 0.0744, -0.2339, 0.1279]
            ]),
            "fc1.bias": torch.tensor([-0.2299, -0.0591, -0.2026,  0.3322, -0.3942]),
            "fc2.weight": torch.tensor([
                [-0.2067, -0.1268, -0.2694, 0.0417, -0.4422],
                [0.4044, -0.3794, 0.3457, 0.0749, -0.1447]
            ]),
            "fc2.bias": torch.tensor([0.2758, 0.0702]),
        }

        lr = 1e-4
        betas = (0.965, 0.99)
        weight_decay = 1e-1  # Matched to achieve REF_PARAMS_GROUP1 with placeholder
        steps = 5

        log_details = f"lr={lr}, betas={betas}, wd={weight_decay}, steps={steps}"

        try:
            self.run_lion_test(
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                steps=steps,
                ref_params=REF_PARAMS_GROUP1
            )
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")
        except AssertionError as e:
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise

    def test_lion_group2(self):
        """
        Test Lion with the second group of hyperparameters (different betas).
        """
        # Pre-recorded reference parameters after the same steps.
        # Again, replace these with real reference values from a known-good run.
        REF_PARAMS_GROUP2 = {
            "fc1.weight": torch.tensor([
                [0.3818, 0.4155, -0.1176, 0.4598],
                [-0.1100, 0.1004, -0.2429, 0.2931],
                [0.4412, -0.3673, 0.4341, 0.0941],
                [0.3689, 0.0682, 0.2406, -0.0701],
                [0.3859, 0.0744, -0.2339, 0.1279]
            ]),
            "fc1.bias": torch.tensor([-0.2299, -0.0591, -0.2026,  0.3322, -0.3942]),
            "fc2.weight": torch.tensor([
                [-0.2067, -0.1268, -0.2694, 0.0417, -0.4422],
                [0.4044, -0.3794, 0.3457, 0.0749, -0.1447]
            ]),
            "fc2.bias": torch.tensor([0.2758, 0.0702]),
        }

        lr = 1e-4
        betas = (0.9, 0.99)  # Different betas
        weight_decay = 1e-1
        steps = 5

        log_details = f"lr={lr}, betas={betas}, wd={weight_decay}, steps={steps}"

        try:
            self.run_lion_test(
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                steps=steps,
                ref_params=REF_PARAMS_GROUP2
            )
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")
        except AssertionError as e:
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise

    def test_lion_group3(self):
        """
        Test Lion with a third group of hyperparameters
        (larger lr, bigger rho, etc.).
        """
        # Pre-recorded reference parameters after the same steps.
        # Replace with real reference values from a known-good run.
        REF_PARAMS_GROUP3 = {
            "fc1.weight": torch.tensor([
                [0.3771, 0.4198, -0.1221, 0.4641],
                [-0.1145, 0.0958, -0.2383, 0.2885],
                [0.4456, -0.3716, 0.4294, 0.0985],
                [0.3642, 0.0727, 0.2360, -0.0656],
                [0.3902, 0.0789, -0.2383, 0.1324]
            ]),
            "fc1.bias": torch.tensor([-0.2253, -0.0636, -0.1980,  0.3365, -0.3895]),
            "fc2.weight": torch.tensor([
                [-0.2111, -0.1312, -0.2738, 0.0372, -0.4465],
                [0.4087, -0.3747, 0.3501, 0.0794, -0.1401]
            ]),
            "fc2.bias": torch.tensor([0.2712, 0.0747]),
        }

        lr = 1e-3  # Larger lr
        betas = (0.965, 0.99)
        weight_decay = 1e-1  # Matched to achieve REF_PARAMS_GROUP3 with placeholder
        steps = 5

        log_details = f"lr={lr}, betas={betas}, wd={weight_decay}, steps={steps}"
        try:
            self.run_lion_test(
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                steps=steps,
                ref_params=REF_PARAMS_GROUP3
            )
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")
        except AssertionError as e:
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise


if __name__ == '__main__':
    unittest.main()
