import sys
import os
import torch
import unittest
import numpy as np
import re
import logging
from pathlib import Path

# 添加模块路径，导入推理函数
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../golden_files')))
from run_inference_golden import get_sentence


def extract_answer(text):
    """
    从模型输出中提取第一个出现的数字（整数或小数），用于作为预测结果。
    示例： "A: The answer is 3.5" → 3.5
    """
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return float(matches[0]) if matches else None


class TestResults(unittest.TestCase):
    def setUp(self):
        # 加载所有测试用例数据
        self.demo1 = torch.load("unit_test/demo1_1.pt")
        self.x = torch.load("unit_test/x_1.pt")
        self.args_method = torch.load("unit_test/args_method.pt")
        self.args_cot_trigger = torch.load("unit_test/args_cot_trigger.pt")
        self.args_direct_answer_trigger_for_zeroshot = torch.load("unit_test/args_direct_answer_trigger_for_zeroshot.pt")
        self.k = torch.load("unit_test/k_1.pt")  # ground truth

    def test_accuracy(self):
        k = get_sentence(
            self.demo1,
            self.x,
            self.args_method,
            self.args_cot_trigger,
            self.args_direct_answer_trigger_for_zeroshot
        )

        try:
            # 提取预测值
            k_pred = extract_answer(k)
            k_gt = extract_answer(self.k)

            if k_pred is None or k_gt is None:
                self.fail(f"Answer extraction failed: pred={k}, gt={self.k}")
                logging.info(f"Test Failed")
            elif np.allclose(k_pred, k_gt, rtol=0.01):
                logging.info(f"Test Passed")
            else:
                logging.info(f"Test Failed")
                self.fail(f"Mismatch: expected {k_gt}, got {k_pred}")
        except Exception as e:
            self.fail(f"Type conversion failed: pred={k}, gt={self.k}, error={e}")
            logging.info(f"Test Failed")


if __name__ == "__main__":
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'unit_test_1.log'

    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format="%(asctime)s = %(message)s")

    unittest.main()
