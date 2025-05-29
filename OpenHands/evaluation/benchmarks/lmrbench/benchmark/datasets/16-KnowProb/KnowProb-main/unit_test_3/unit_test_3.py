import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
from src.parametric_knowledge import is_parametric_object_not_in_the_prompt
import logging
from pathlib import Path

class TestIsParametricObjectNotInThePrompt(unittest.TestCase):
    def setUp(self):
        # 载入输入数据
        with open("unit_test_3/is_parametric_object_not_in_the_prompt_input.pkl", "rb") as f:
            self.parametric_objects, self.relations_with_gen_desc = pickle.load(f)

        # 载入期望输出数据
        with open("unit_test_3/is_parametric_object_not_in_the_prompt_output.pkl", "rb") as f:
            self.expected_output = pickle.load(f)

    def test_output_matches_expected(self):
        # 保持遍历式计算格式
        result = [
            is_parametric_object_not_in_the_prompt(parametric_object, self.relations_with_gen_desc)
            for parametric_object in self.parametric_objects
        ]


        PASS = True
        try:
            self.assertEqual(
                result, self.expected_output,
                "The output of is_parametric_object_not_in_the_prompt does not match the expected result."
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
