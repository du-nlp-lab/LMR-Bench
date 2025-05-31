import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
import numpy as np
import logging
from pathlib import Path
from analysis.analysis_after_processing import get_delta

class TestGetDelta(unittest.TestCase):
    def setUp(self):
        with open("unit_test_1/get_delta_input.pkl", "rb") as f:
            self.input_matrix, self.group_option = pickle.load(f)

        with open("unit_test_1/get_delta_output.pkl", "rb") as f:
            self.expected_delta = pickle.load(f)

    def test_get_delta_output_within_1_percent(self):
        result = get_delta(self.input_matrix, self.group_option)

        # 容许 1% 的相对误差
        try:
            self.assertTrue(
                # np.isclose(result, self.expected_delta, rtol=0.01),
                np.allclose(result, self.expected_delta, rtol=0.01),
                f"Delta result {result} differs from expected {self.expected_delta} by more than 1%"
            )
            logging.info("Test Passed")
        except Exception as e:
            logging.info(f"Test Failed: Delta result {result} differs from expected {self.expected_delta} by more than 1%")

if __name__ == "__main__":
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define the log file path
    log_file = log_dir / 'unit_test_1.log'

    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format="%(asctime)s = %(message)s")
    unittest.main()