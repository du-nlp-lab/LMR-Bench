import unittest
import torch
from transformers import AutoTokenizer
from hierarchy_transformers import HierarchyTransformer
from hierarchy_transformers.losses import HierarchyTransformerLoss
import logging
from pathlib import Path


# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")


class TestHierarchyTransformerLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure reproducibility
        torch.manual_seed(42)

        # Load model & tokenizer once for all tests
        cls.device = torch.device("cpu")
        cls.model = HierarchyTransformer.from_pretrained(
            "Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun"
        ).to(cls.device)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun"
        )
        cls.loss_fn = HierarchyTransformerLoss(cls.model)

    def test_loss_values(self):
        # Prepare inputs
        entity_texts = ["fruit", "berry", "vegetable"]
        entity_embeddings = [
            self.tokenizer(text, return_tensors="pt").to(self.device)
            for text in entity_texts
        ]
        labels = torch.tensor([0, 1, 2], device=self.device)

        # Compute loss
        loss_dict = self.loss_fn(entity_embeddings, labels)

        # Extract scalar losses
        total_loss   = loss_dict["loss"].item()
        cluster_loss = loss_dict["cluster_loss"].item()
        centri_loss  = loss_dict["centri_loss"].item()

        try:
            # Assert close to expected values
            self.assertAlmostEqual(total_loss,   5.8003, places=3)
            self.assertAlmostEqual(cluster_loss, 3.9675, places=3)
            self.assertAlmostEqual(centri_loss,  1.8328, places=3)

            # If test passes, log success
            logging.info(f"Test Passed: total_loss={total_loss}, cluster_loss={cluster_loss}, centri_loss={centri_loss}")
        
        except AssertionError as e:
            # If test fails, log failure
            logging.error(f"Test Failed: total_loss={total_loss}, cluster_loss={cluster_loss}, centri_loss={centri_loss}. Error: {str(e)}")
            raise  # Re-raise the exception to make the test fail

if __name__ == "__main__":
    unittest.main()
