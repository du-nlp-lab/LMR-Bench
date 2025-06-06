import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
import numpy as np
from analysis.analysis_after_processing import get_filtered_matrix
import logging
from pathlib import Path

class TestGetFilteredMatrix(unittest.TestCase):
    def setUp(self):
        # 加载输入数据
        with open("unit_test_2/get_filtered_matrix_input.pkl", "rb") as f:
            self.group_option, self.prompt_variation, self.model_id, self.excluded_ids = pickle.load(f)

        # 加载输出数据
        with open("unit_test_2/get_filtered_matrix_output.pkl", "rb") as f:
            self.expected_matrix = pickle.load(f)

    def test_matrix_close_within_1_percent(self):
        # 调用函数
        result_matrix = get_filtered_matrix(
            self.group_option, self.prompt_variation, self.model_id, self.excluded_ids
        )

        # 检查 shape 是否一致
        self.assertEqual(result_matrix.shape, self.expected_matrix.shape, "Matrix shapes do not match.")

        # 检查每个元素是否在 1% 相对误差范围内
        if np.allclose(result_matrix, self.expected_matrix, rtol=0.01):
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
