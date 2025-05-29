import random
import pickle
import random
import sys
import unittest
sys.path.append("../")

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

from conll_tree import spacy_result_to_conll

class TestModelOutputs(unittest.TestCase):
    def setUp(self):
        # Load inputs and outputs from pkl files
        self.sentences = pickle.load(open("unit_test/pickles/sentences.pkl", "rb"))
        self.structures = pickle.load(open("unit_test/pickles/structures.pkl", "rb"))

        # Ensure inputs and outputs have the same length
        assert len(self.sentences) == len(self.structures)
        print("Len:", len(self.sentences))
    
    def test_random_pairs(self):
        # Select three random indices
        indices = random.sample(range(len(self.sentences)), 3)
        
        for idx in indices:
            sentence = self.sentences[idx]
            structure_expected = self.structures[idx]

            structure =  spacy_result_to_conll(sentence)

            if structure == structure_expected:
                logging.info(f"Test Passed: Structure Expected\n{structure_expected}, got\n{structure}")
            else:
                logging.info(f"Test Failed: Structure Expected\n{structure_expected}, got\n{structure}")


if __name__ == "__main__":
    unittest.main()
