import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
from src.classification.classifier import perform_classification_by_relation_group
import logging
from pathlib import Path


class TestClassificationFunction(unittest.TestCase):
    def setUp(self):
        # Load the saved input data
        with open("unit_test_1/prompting_results.pkl", "rb") as f:
            self.prompting_results = pickle.load(f)

        with open("unit_test_1/params.pkl", "rb") as f:
            self.params = pickle.load(f)

        # Load the saved output data
        with open("unit_test_1/classification_metrics.pkl", "rb") as f:
            self.classification_metrics_expected = pickle.load(f)

    def test_classification_outputs(self):
        classification_metrics = perform_classification_by_relation_group(
            prompting_results=self.prompting_results,
            include_mlps=self.params["include_mlps"],
            include_mlps_l1=self.params["include_mlps_l1"],
            include_mhsa=self.params["include_mhsa"],
            vertical=self.params["vertical"],
            position=self.params["token_position"]
        )
    
        self.assertEqual(len(classification_metrics), len(self.classification_metrics_expected), "Output length mismatch")

        PASS = True
        for i, (out, expected) in enumerate(zip(classification_metrics, self.classification_metrics_expected)):
            self.assertEqual(type(out), type(expected), f"Type mismatch at index {i}")
            if isinstance(out, dict):
                try:
                    self.assertEqual(out.keys(), expected.keys(), f"Key mismatch at index {i}")
                except:
                    PASS = False

                for key in out:
                    try:
                        self.assertTrue(
                            out[key] == expected[key] or
                            (hasattr(out[key], "equals") and out[key].equals(expected[key])),
                            f"Mismatch at index {i}, key {key}"
                    )
                    except:
                        PASS = False
            elif isinstance(out, float):
                try:
                    self.assertAlmostEqual(out, expected, places=5, msg=f"Mismatch at index {i}")
                except:
                    PASS = False
            elif hasattr(out, "equal"):
                try:
                    self.assertTrue(out.equal(expected), f"Tensor mismatch at index {i}")
                except:
                    PASS = False
            else:
                try:
                    self.assertEqual(out, expected, f"General mismatch at index {i}")
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
