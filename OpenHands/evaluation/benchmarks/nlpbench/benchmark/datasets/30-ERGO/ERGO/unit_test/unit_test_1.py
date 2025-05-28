import logging
import torch
import pickle
import random
import sys
import unittest
sys.path.append("../")

from model import focal_loss
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

class TestModelOutputs(unittest.TestCase):
    def setUp(self):
        self.focal_loss = focal_loss(alpha=0.75, gamma=2, num_classes=2, size_average=True)

        # Load inputs and outputs from pkl files
        self.preds = pickle.load(open("unit_test/pickles/preds.pkl", "rb"))
        self.labels = pickle.load(open("unit_test/pickles/labels.pkl", "rb"))
        self.losses = pickle.load(open("unit_test/pickles/losses.pkl", "rb"))

        # Ensure inputs and outputs have the same length
        assert len(self.preds) == len(self.labels) == len(self.losses)
        print("Len:", len(self.preds))
    
    def test_random_pairs(self):
        # Select three random indices
        indices = random.sample(range(len(self.preds)), 5)
        
        for idx in indices:
            pred = self.preds[idx]
            label = self.labels[idx]
            loss_expected = self.losses[idx]

            loss = self.focal_loss(pred, label)
            if torch.allclose(torch.tensor(loss), torch.tensor(loss_expected), rtol=0.01):
                logging.info(f"Test Passed: Loss Expected {loss_expected}, got {loss}")
            else:
                logging.info(f"Test Failed: Loss Expected {loss_expected}, got {loss}")


if __name__ == "__main__":
    unittest.main()
