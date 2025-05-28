import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
from src.data import (
    remove_object_subject_overlap,
    generate_parametric_knowledge_dataset,
    generate_counter_parametric_knowledge_dataset,
    remove_entity_overlap_between_relation_groups,
)
import logging
from pathlib import Path

class TestRemoveObjectSubjectOverlap(unittest.TestCase):
    def setUp(self):
        with open("unit_test_2/remove_object_subject_overlap_input.pkl", "rb") as f:
            self.input_data = pickle.load(f)

        with open("unit_test_2/remove_object_subject_overlap_param.pkl", "rb") as f:
            self.similarity_threshold = pickle.load(f)

        with open("unit_test_2/remove_object_subject_overlap_output.pkl", "rb") as f:
            self.expected_output = pickle.load(f)

    def test_output_matches_expected(self):
        result = remove_object_subject_overlap(self.input_data, string_similarity_ratio=self.similarity_threshold)

        # 比较 DataFrame 是否完全一致
        PASS = True
        try:
            self.assertTrue(
                result.equals(self.expected_output),
                "The output of remove_object_subject_overlap does not match the expected result."
            )
        except:
            PASS = False

        if PASS:
            logging.info("Test Passed")
        else:
            logging.info("Test Failed")

if __name__ == "__main__":
    # Set up logging to capture test pass/fail results
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'unit_test_1.log'
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        level=logging.INFO,
                        format="%(asctime)s - %(message)s")
    unittest.main()
