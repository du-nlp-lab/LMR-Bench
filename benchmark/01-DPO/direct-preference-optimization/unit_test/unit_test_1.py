import torch
import random
import unittest
from trainers import preference_loss
import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'unit_test_1.log'
logging.basicConfig(filename=log_file,
                    filemode="w",
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s")

class TestModelOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fix the seed so everyone gets the same "random" indices
        random.seed(42)

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load inputs and outputs from .pt files
        self.policy_chosen_logps   = torch.load("tensors/policy_chosen_logps.pt",  map_location=self.device)
        self.policy_rejected_logps = torch.load("tensors/policy_rejected_logps.pt", map_location=self.device)
        self.reference_chosen_logps= torch.load("tensors/reference_chosen_logps.pt", map_location=self.device)
        self.reference_rejected_logps= torch.load("tensors/reference_rejected_logps.pt", map_location=self.device)
        self.beta   = 0.1
        self.losses = torch.load("tensors/losses.pt", map_location=self.device)

        # choose 3 distinct random indices in [0, len-1]
        max_idx = len(self.policy_chosen_logps)
        assert max_idx >= 3, "Need at least 3 examples in your tensors"
        self.indices = random.sample(range(max_idx), 3)
        logging.info(f"Selected random indices for testing: {self.indices}")

    def test_random_pairs(self):
        """Compute losses for each random index and assert all pass."""
        all_passed = True
        failed_indices = []

        for idx in self.indices:
            pc = self.policy_chosen_logps[idx]
            pr = self.policy_rejected_logps[idx]
            rc = self.reference_chosen_logps[idx]
            rr = self.reference_rejected_logps[idx]
            expected_loss = self.losses[idx]

            loss, _, _ = preference_loss(pc, pr, rc, rr, beta=self.beta)
            if torch.allclose(loss, expected_loss, rtol=0.01):
                logging.info(f"Index {idx}: Test Passed: (expected {expected_loss}, got {loss})")
            else:
                logging.info(f"Index {idx}: Test Failed: (expected {expected_loss}, got {loss})")
                all_passed = False
                failed_indices.append(idx)

        if not all_passed:
            logging.error(f"Failed at indices: {failed_indices}")
        else:
            logging.info("All random‐pair tests passed.")

if __name__ == "__main__":
    unittest.main()
