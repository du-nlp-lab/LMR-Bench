import torch
import random
import unittest
from movescore import word_mover_score

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
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load inputs and outputs from .pt files
        self.ref_embedding = torch.load("unit_test/tensor/ref_embedding.pt", map_location=self.device)
        self.ref_tokens = torch.load("unit_test/tensor/ref_tokens.pt")
        self.ref_idf = torch.load("unit_test/tensor/ref_idf.pt",  map_location=self.device)
        self.hyp_embedding = torch.load("unit_test/tensor/hyp_embedding.pt", map_location=self.device)
        self.hyp_tokens = torch.load("unit_test/tensor/hyp_tokens.pt")
        self.hyp_idf= torch.load("unit_test/tensor/hyp_idf.pt",  map_location=self.device)

        self.moverscore = torch.load("unit_test/tensor/movescore.pt", map_location=self.device)


        # Ensure inputs and outputs have the same length
        assert self.ref_embedding.size()[1]  == len(self.moverscore), "Inputs and outputs must have the same length"
        logging.info(f"len: len({self.moverscore})")

    def test_random_pairs(self):
        # Select three random indices
        # indices = random.sample(range(len(self.policy_chosen_logps)), 3)
        indices = [0, 1]

        for idx in indices:
            ref_embedding =  self.ref_embedding[:,idx].unsqueeze(1)
            ref_tokens = [self.ref_tokens[idx]]
            ref_idf = self.ref_idf[idx].unsqueeze(0)
            hyp_embedding =self.hyp_embedding[:,idx].unsqueeze(1)
            hyp_tokens =[self.hyp_tokens[idx]]
            hyp_idf = self.hyp_idf[idx].unsqueeze(0)

            moverscore_expected = self.moverscore[idx].unsqueeze(0)

            moverscore = word_mover_score(ref_embedding, ref_idf, ref_tokens, hyp_embedding, hyp_idf, hyp_tokens)

            moverscore = torch.tensor(moverscore, device=self.device)

            if torch.allclose(torch.tensor(moverscore), moverscore_expected, atol=0.01):
                logging.info(f"Test Passed: Losses Expected {moverscore_expected}, got {moverscore}")
            else:
                logging.info(f"Test Failed: Losses Expected {moverscore_expected}, got {moverscore}")



if __name__ == "__main__":
    unittest.main()
