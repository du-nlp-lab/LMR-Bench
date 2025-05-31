import torch
import unittest
from bert_score_goal import greedy_cos_idf


import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'unit_test_1.log'
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s")

class TestModelOutputs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load inputs and outputs from .pt files
        self.ref_embedding = torch.load("tensors/tensor_ref_embedding.pt").to(self.device)
        self.ref_masks = torch.load("tensors/tensor_ref_masks.pt").to(self.device)
        self.ref_idf = torch.load("tensors/tensor_ref_idf.pt").to(self.device)
        self.hyp_embedding = torch.load("tensors/tensor_hyp_embedding.pt").to(self.device)
        self.hyp_masks = torch.load("tensors/tensor_hyp_masks.pt").to(self.device)
        self. hyp_idf= torch.load("tensors/tensor_hyp_idf.pt").to(self.device)

        self.bertscore = torch.load("tensors/bert_scores.pt").to(self.device)

        # Ensure inputs and outputs have the same length
        assert len(self.ref_embedding) == len(self.bertscore), "Inputs and outputs must have the same length"
        logging.info(f"len: {len(self.bertscore)}")

    def test_random_pairs(self):
        # Select three random indices
        # indices = random.sample(range(len(self.policy_chosen_logps)), 3)
        indices = [0, 1]

        for idx in indices:
            ref_embedding = self.ref_embedding[idx].unsqueeze(0)
            ref_masks = self.ref_masks[idx].unsqueeze(0)
            ref_idf = self.ref_idf[idx].unsqueeze(0)
            hyp_embedding = self.hyp_embedding[idx].unsqueeze(0)
            hyp_masks = self.hyp_masks[idx].unsqueeze(0)
            hyp_idf = self.hyp_idf[idx].unsqueeze(0)

            bertscore_expected = self.bertscore[idx].unsqueeze(0)
            bertscore = greedy_cos_idf(ref_embedding, ref_masks, ref_idf, hyp_embedding, hyp_masks , hyp_idf)
            
            if torch.allclose(bertscore, bertscore_expected, rtol=0.01):
                logging.info(f"Test passed: Losses Expected {bertscore_expected}, got {bertscore}")
            else:
                logging.info(f"Test Failed: Losses Expected {bertscore_expected}, got {bertscore}")



if __name__ == "__main__":
    unittest.main()