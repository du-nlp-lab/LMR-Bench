import unittest
import random

# ------------------------------------------------------------------------------
# Either paste the entire Hyena code here or import it if it's in another module:
#
# from your_hyena_module import HyenaOperator
#
# For clarity in this snippet, we'll assume you have the HyenaOperator code
# in a file "hyena.py" and do:
#
# from hyena import HyenaOperator
#
# ------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging
from pathlib import Path


from standalone_hyena import HyenaOperator


# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")


class TestHyenaOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We fix the random seeds for reproducibility (so we generate the exact same
        # test input in both your reference code and the tested code).
        random.seed(0)
        torch.manual_seed(0)

        # We'll prepare three sets of test inputs of varying shapes.
        # Feel free to adjust shapes to match your constraints or desired coverage.

        # --- Test case 1 ---
        cls.input_1 = torch.randn(1, 8, 8)  # (batch_size=1, seq_len=8, d_model=8)
        cls.layer_1 = torch.nn.Sequential(
            HyenaOperator(d_model=8, l_max=8, order=2, filter_order=4)
        )
        # Replace the following with the actual baseline output from your known-good code
        # after running: reference_output_1 = cls.layer_1(cls.input_1)
        # and storing reference_output_1 somewhere safe.
        cls.expected_output_1 = torch.tensor([[[-0.2499, 0.3312, 0.1442, 0.1878, -0.2253, 0.0747, -0.1653,
                                                0.2745],
                                               [-0.2404, 0.3150, 0.1500, 0.1757, -0.2260, 0.0932, -0.1479,
                                                0.2817],
                                               [-0.2883, 0.3044, 0.0776, 0.0847, -0.2294, 0.1464, -0.1972,
                                                0.3776],
                                               [-0.2913, 0.3046, 0.1412, 0.1694, -0.2236, 0.0646, -0.1867,
                                                0.3253],
                                               [-0.2885, 0.3103, 0.0917, 0.1154, -0.2709, 0.1069, -0.1779,
                                                0.3252],
                                               [-0.3276, 0.3210, 0.0772, 0.1194, -0.2661, 0.0630, -0.2181,
                                                0.3827],
                                               [-0.1796, 0.3299, 0.1579, 0.1776, -0.2431, 0.1432, -0.0890,
                                                0.2354],
                                               [-0.2529, 0.3083, 0.1791, 0.1802, -0.2290, 0.0756, -0.1331,
                                                0.2707]]])

        # --- Test case 2 ---
        cls.input_2 = torch.randn(2, 10, 8)  # (batch_size=2, seq_len=10, d_model=8)
        cls.layer_2 = torch.nn.Sequential(
            HyenaOperator(d_model=8, l_max=10, order=2, filter_order=8)
        )
        # Replace the following with baseline output from your reference code
        cls.expected_output_2 = torch.tensor(
            [[[0.0809, -0.0955, 0.2319, 0.0376, 0.0843, -0.1453, -0.0165, -0.0529],
              [0.3062, -0.1737, 0.0986, -0.1702, 0.0331, 0.0181, -0.1812,
               0.1001],
              [0.1951, 0.0072, 0.1360, 0.1180, 0.1956, -0.1114, -0.1592,
               -0.1342],
              [0.1005, -0.0995, 0.1823, 0.1348, 0.2055, -0.0987, -0.1333,
               -0.1475],
              [0.0059, -0.2544, 0.2801, 0.0918, 0.1486, -0.1635, -0.1793,
               0.0642],
              [0.0790, -0.0551, 0.2446, 0.0449, 0.0596, -0.1387, -0.0734,
               -0.0215],
              [0.0408, -0.0137, 0.2762, 0.1893, 0.1879, -0.2374, -0.1393,
               -0.0611],
              [0.0940, -0.0582, 0.2164, 0.0418, 0.0712, -0.1100, -0.0448,
               -0.0744],
              [0.0786, -0.1946, 0.2228, 0.0314, 0.1227, -0.0965, -0.0879,
               -0.0336],
              [0.1696, -0.1262, 0.1877, 0.0278, 0.1740, -0.1986, -0.0343,
               -0.0240]],

             [[0.0703, -0.1012, 0.2377, 0.0304, 0.0730, -0.1257, -0.0132,
               -0.0614],
              [0.0315, -0.0470, 0.2598, 0.0754, 0.0682, -0.1717, 0.0366,
               -0.0979],
              [0.1423, -0.0851, 0.1971, 0.0410, 0.1462, -0.1133, -0.0623,
               -0.0926],
              [-0.0222, -0.0226, 0.3105, 0.1018, 0.0234, -0.2433, 0.0235,
               -0.0165],
              [0.1255, -0.0490, 0.1885, 0.0577, 0.1061, -0.0856, -0.0949,
               -0.0957],
              [0.0740, -0.0957, 0.2322, 0.0398, 0.0867, -0.1294, -0.0038,
               -0.0806],
              [-0.1761, 0.1410, 0.3809, 0.2605, -0.0470, -0.3592, 0.0487,
               -0.0408],
              [0.0857, -0.0624, 0.2428, 0.0158, 0.0663, -0.0457, -0.0493,
               -0.1119],
              [0.0782, -0.1035, 0.2314, 0.0280, 0.0835, -0.1282, 0.0044,
               -0.0759],
              [0.0300, -0.0980, 0.2596, 0.0650, 0.0889, -0.1684, 0.0325,
               -0.0872]]])

        # --- Test case 3 ---
        cls.input_3 = torch.randn(1, 8, 16)  # (batch_size=1, seq_len=8, d_model=16)
        cls.layer_3 = torch.nn.Sequential(
            HyenaOperator(d_model=16, l_max=8, order=2, filter_order=8)
        )
        # Replace the following with baseline output
        cls.expected_output_3 = torch.tensor([[[-0.1549, 0.0332, 0.2246, 0.1926, -0.2730, -0.0134, -0.0147,
                                                -0.1307, -0.1494, -0.1739, 0.1844, -0.0989, -0.0353, 0.0458,
                                                -0.1252, 0.0141],
                                               [-0.2219, -0.0383, 0.2317, 0.1663, -0.2268, -0.0259, 0.0227,
                                                -0.0160, -0.2080, -0.2005, 0.1757, -0.1025, 0.0075, 0.1009,
                                                -0.1452, 0.0046],
                                               [-0.1203, 0.0892, 0.2451, 0.2053, -0.3023, -0.0226, -0.0671,
                                                -0.1603, -0.0900, -0.1672, 0.1443, -0.0611, -0.0875, 0.0460,
                                                -0.1362, -0.0067],
                                               [-0.1740, -0.0039, 0.2111, 0.2088, -0.3108, -0.0234, -0.0788,
                                                -0.1572, -0.2184, -0.1010, 0.2155, -0.0663, -0.0809, 0.0945,
                                                -0.0715, -0.0037],
                                               [-0.1160, 0.0586, 0.2707, 0.2273, -0.2680, -0.0286, -0.0643,
                                                -0.0984, -0.1716, -0.2345, 0.1939, -0.1808, 0.0034, -0.0295,
                                                -0.1039, 0.0421],
                                               [-0.2800, -0.0561, 0.0844, 0.1213, -0.3365, 0.0523, 0.1276,
                                                -0.1508, -0.1407, -0.1317, 0.3185, -0.1021, -0.1021, 0.1045,
                                                0.0012, -0.0519],
                                               [-0.2750, -0.0210, 0.1329, 0.1084, -0.2885, 0.0235, 0.1142,
                                                -0.0893, -0.1447, -0.1502, 0.2547, -0.0622, -0.0473, 0.1419,
                                                -0.0939, -0.0261],
                                               [-0.1095, 0.0889, 0.2462, 0.1969, -0.2852, -0.0067, -0.0744,
                                                -0.1521, -0.1317, -0.1579, 0.1347, -0.0997, -0.0239, 0.0192,
                                                -0.1586, 0.0027]]])

    def test_hyena_case_1(self):
        """
        Test the HyenaOperator output for a small (1,8,8) input.
        """
        computed_output = None
        log_details = "Output not computed"

        try:
            with torch.no_grad():
                computed_output = self.layer_1(self.input_1)

            if computed_output is not None:
                log_details = f"output_shape={str(computed_output.shape)}, output_norm={torch.norm(computed_output).item():.4f}"

            # Check shape
            self.assertEqual(computed_output.shape, self.expected_output_1.shape)
            # Check numerical closeness
            self.assertTrue(
                torch.allclose(computed_output, self.expected_output_1, atol=1e-4, rtol=1e-3),
                f"Case 1: output differs from expected by more than tol"
            )

            # If test passes, log success
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")

        except AssertionError as e:
            # If test fails, log failure
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise  # Re-raise the exception to make the test fail

    def test_hyena_case_2(self):
        """
        Test the HyenaOperator output for a small (2,10,8) input.
        """
        computed_output = None
        log_details = "Output not computed"
        try:
            with torch.no_grad():
                computed_output = self.layer_2(self.input_2)

            if computed_output is not None:
                log_details = f"output_shape={str(computed_output.shape)}, output_norm={torch.norm(computed_output).item():.4f}"

            # Check shape
            self.assertEqual(computed_output.shape, self.expected_output_2.shape)
            # Check numerical closeness
            self.assertTrue(
                torch.allclose(computed_output, self.expected_output_2, atol=1e-4, rtol=1e-3),
                f"Case 2: output differs from expected by more than tol"
            )
            # If test passes, log success
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")

        except AssertionError as e:
            # If test fails, log failure
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise  # Re-raise the exception to make the test fail

    def test_hyena_case_3(self):
        """
        Test the HyenaOperator output for a small (1,8,16) input.
        """
        computed_output = None
        log_details = "Output not computed"
        try:
            with torch.no_grad():
                computed_output = self.layer_3(self.input_3)

            if computed_output is not None:
                log_details = f"output_shape={str(computed_output.shape)}, output_norm={torch.norm(computed_output).item():.4f}"

            # Check shape
            self.assertEqual(computed_output.shape, self.expected_output_3.shape)
            # Check numerical closeness
            self.assertTrue(
                torch.allclose(computed_output, self.expected_output_3, atol=1e-4, rtol=1e-3),
                f"Case 3: output differs from expected by more than tol"
            )
            # If test passes, log success
            logging.info(f"Test Passed: {self._testMethodName}. Details: {log_details}")

        except AssertionError as e:
            # If test fails, log failure
            logging.error(f"Test Failed: {self._testMethodName}. Details: {log_details}. Error: {str(e)}")
            raise  # Re-raise the exception to make the test fail


if __name__ == "__main__":
    unittest.main()
