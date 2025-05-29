import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
from src.data import generate_counter_parametric_knowledge_dataset
import logging
from pathlib import Path

log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test.log'

logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format="%(asctime)s = %(message)s")

class TestGenerateCounterParametricKnowledgeDataset(unittest.TestCase):
    def setUp(self):
        with open("unit_test/generate_counter_parametric_knowledge_dataset_input.pkl", "rb") as f:
            self.parametric_knowledge_dataset, self.nb_counter_parametric_knowledge = pickle.load(f)

        with open("unit_test/generate_counter_parametric_knowledge_dataset_output.pkl", "rb") as f:
            self.expected_output = pickle.load(f)

    def test_output_matches_expected(self):
        result = generate_counter_parametric_knowledge_dataset(
            self.parametric_knowledge_dataset, self.nb_counter_parametric_knowledge
        )
        try:
            self.assertTrue(
                result.equals(self.expected_output),
                "The output of generate_counter_parametric_knowledge_dataset does not match the expected result."
            )
            logging.info("Test Passed")
        except AssertionError as e:
            logging.info(f"Test Failed: {e}")

if __name__ == "__main__":
    unittest.main()
