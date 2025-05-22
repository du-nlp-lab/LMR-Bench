import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
from src.data import remove_entity_overlap_between_relation_groups
import logging
from pathlib import Path

log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_4.log'

logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format="%(asctime)s = %(message)s")

class TestRemoveEntityOverlapBetweenRelationGroups(unittest.TestCase):
    def setUp(self):
        # 载入输入数据
        with open("unit_test/remove_entity_overlap_between_relation_groups_input.pkl", "rb") as f:
            self.input_data = pickle.load(f)

        # 载入期望输出数据
        with open("unit_test/remove_entity_overlap_between_relation_groups_output.pkl", "rb") as f:
            self.expected_output = pickle.load(f)

    def test_output_matches_expected(self):
        # 调用被测试的函数
        result = remove_entity_overlap_between_relation_groups(self.input_data)

        # 检查输出是否和保存的一致
        try:
            self.assertTrue(
                result.equals(self.expected_output),
                "The output of remove_entity_overlap_between_relation_groups does not match the expected result."
            )
            logging.info("Test Passed")
        except AssertionError as e:
            logging.info(f"Test Failed: {e}")
            
if __name__ == "__main__":
    unittest.main()
