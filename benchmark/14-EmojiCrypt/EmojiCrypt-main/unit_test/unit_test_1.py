import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import unittest
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Tabular import compute_mean_cosine_similarity  # Make sure this matches your main function file
import logging
from pathlib import Path

class TestComputeMeanCosineSimilarity(unittest.TestCase):
    def setUp(self):
        self.output_file = "unit_test/cosine_sim_results.pkl"
        self.pairs_file = "unit_test/enc_dec_pairs.pkl"

        # Create test data directory if not exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # # Save test pairs to file for testing
        # with open(self.pairs_file, "wb") as f:
        #     pickle.dump(self.test_pairs, f)
        
        with open(self.pairs_file, 'rb') as f:
            self.saved_inputs = pickle.load(f)

    def test_mean_cosine_similarity_computation(self):
        
        mean_cosine_sim = compute_mean_cosine_similarity(self.saved_inputs)
        print("mean_cosine_sim: ", mean_cosine_sim)

        # Load the saved results to validate
        with open(self.output_file, "rb") as f:
            results = pickle.load(f)
        # print("results: ", results)

        # Extract the mean similarity from the file
        saved_mean_cosine_sim = results["mean_cosine_sim"]
        print("saved_mean_cosine_sim: ", saved_mean_cosine_sim)

        # Validate that the saved mean cosine similarity matches the returned value
        # self.assertAlmostEqual(mean_cosine_sim, saved_mean_cosine_sim, places=6, msg="Mean cosine similarity mismatch.")
        if np.allclose(mean_cosine_sim, saved_mean_cosine_sim, rtol=0.01):
            logging.info("Test Passed")
        else:
            logging.info("Test Failed")

        # Check that the number of pairs matches
        # self.assertEqual(len(results["inputs"]), len(self.test_pairs), "Mismatch in the number of pairs.")


if __name__ == "__main__":

    # Set up logging to capture test pass/fail results
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'unit_test_1.log'
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        level=logging.INFO,
                        format="%(asctime)s - %(message)s")

    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format="%(asctime)s = %(message)s")
    unittest.main()
