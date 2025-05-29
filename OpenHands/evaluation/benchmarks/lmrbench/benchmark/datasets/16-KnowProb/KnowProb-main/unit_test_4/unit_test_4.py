import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
from src.data import remove_entity_overlap_between_relation_groups
import logging
from pathlib import Path

class TestRemoveEntityOverlapBetweenRelationGroups(unittest.TestCase):
    def setUp(self):
        # 载入输入数据
        with open("unit_test_4/remove_entity_overlap_between_relation_groups_input.pkl", "rb") as f:
            self.input_data = pickle.load(f)

        # 载入期望输出数据
        with open("unit_test_4/remove_entity_overlap_between_relation_groups_output.pkl", "rb") as f:
            self.expected_output = pickle.load(f)

    def test_output_matches_expected(self):
        # 调用被测试的函数
        result = remove_entity_overlap_between_relation_groups(self.input_data)

        # 检查输出是否和保存的一致
        PAS = True
        try:
            self.assertTrue(
                result.equals(self.expected_output),
                "The output of remove_entity_overlap_between_relation_groups does not match the expected result."
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
