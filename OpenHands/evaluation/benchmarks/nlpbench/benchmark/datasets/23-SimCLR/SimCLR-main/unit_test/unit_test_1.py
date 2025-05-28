import unittest
import torch
import torch.nn as nn
import os
import sys
import re
from argparse import Namespace, ArgumentParser

# ✅ 自定义参数解析（避免 unittest 冲突）
parser = ArgumentParser()
parser.add_argument("--overwrite_gt", action="store_true", help="Overwrite logits and labels in debug .pt files")
args, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args  # 保留 unittest 支持的参数

# 添加 simclr 所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simclr import SimCLR


import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'unit_test_1.log'
logging.basicConfig(filename=log_file,
                    filemode="w",
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s")


class TestInfoNCELoss(unittest.TestCase):
    def setUp(self):
        self.overwrite = args.overwrite_gt

        self.args = Namespace()
        self.args.batch_size = 256
        self.args.n_views = 2
        self.args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.args.temperature = 0.5

        dummy_model = nn.Identity()
        self.simclr = SimCLR(args=self.args, model=dummy_model, optimizer=None, scheduler=None)

        self.debug_dir = os.path.dirname(__file__)
        self.debug_files = [
            "../info_nce_debug_9.pt",
            "../info_nce_debug_42.pt",
            "../info_nce_debug_88.pt"
        ]

    def extract_seed(self, filename):
        match = re.search(r"debug_(\d+)\.pt", filename)
        return int(match.group(1)) if match else None

    def test_info_nce_loss_against_debug_files(self):
        for debug_file in self.debug_files:
            with self.subTest(file=debug_file):
                seed = self.extract_seed(debug_file)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                path = os.path.join(self.debug_dir, debug_file)
                assert os.path.exists(path), f"{path} not found!"
                data = torch.load(path, map_location=self.args.device)

                features = data["features"].to(self.args.device)
                expected_logits = data["logits"].to(self.args.device)
                expected_labels = data["labels"].to(self.args.device)

                logits, labels = self.simclr.info_nce_loss(features)

                if self.overwrite:
                    # ✅ 保存当前结果覆盖原来的 ground truth
                    data["logits"] = logits.detach().cpu()
                    data["labels"] = labels.detach().cpu()
                    torch.save(data, path)
                    logging.info(f"✅ Overwritten ground truth in: {debug_file}")
                    continue  # 不再断言，直接跳过当前样本

                # ✅ 输出 logits 差异
                if not torch.allclose(logits, expected_logits, atol=0.01):
                    logging.info("Test Failed")
                    logging.info(f"\n❌ {debug_file}: Logits mismatch")
                    diff = (logits - expected_logits).abs()
                    logging.info(f"Max diff: {diff.max().item()}")
                    logging.info(f"Mean diff: {diff.mean().item()}")
                    logging.info(f"Expected logits:\n{expected_logits[:5]}")
                    logging.info(f"Got logits:\n{logits[:5]}")
                else:
                    logging.info("Test Passed")

                # ✅ 输出 labels 差异
                if not torch.equal(labels, expected_labels):
                    logging.info("Test Failed")
                    logging.info(f"\n❌ {debug_file}: Labels mismatch")
                    logging.info(f"Expected labels:\n{expected_labels}")
                    logging.info(f"Got labels:\n{labels}")
                else:
                    logging.info("Test Passed")

                # ✅ 断言 shape 和数值一致性
                try:
                    self.assertEqual(logits.shape, expected_logits.shape,
                                    f"{debug_file}: Logits shape mismatch")
                    self.assertEqual(labels.shape, expected_labels.shape,
                                    f"{debug_file}: Labels shape mismatch")

                    self.assertTrue(torch.allclose(logits, expected_logits, atol=1e-4),
                                    f"{debug_file}: Logits values differ")
                    self.assertTrue(torch.equal(labels, expected_labels),
                                    f"{debug_file}: Labels not equal")
                    logging.info("Test Passed")
                except:
                    logging.info("Test Failed")


if __name__ == '__main__':
    unittest.main()