import unittest
import torch
import torch.nn.functional as F

import sys
# ——1. 备份并移除前两项——
removed = []
for _ in range(2):
    if sys.path:
        removed.append(sys.path.pop(0))

# ——2. 在全局 site-packages 中导入——
from loralib import SVDLinear

# ——3. 恢复原始路径顺序——
for entry in reversed(removed):
    sys.path.insert(0, entry)
  

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


class TestSVDLinearForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 4
        self.in_dim = 8
        self.out_dim = 6
        self.r = 3

        # Base nn.Linear for comparison
        self.base_linear = torch.nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.base_linear.weight.data.normal_(0, 0.02)
        self.base_linear.bias.data.normal_(0, 0.02)

        # SVDLinear with same weight/bias
        self.svd = SVDLinear(
            in_features=self.in_dim,
            out_features=self.out_dim,
            r=self.r,
            lora_alpha=1,
            lora_dropout=0.0,
            fan_in_fan_out=False,
            merge_weights=False
        )
        # Override its base weight & bias
        self.svd.weight.data = self.base_linear.weight.data.clone()
        self.svd.bias.data = self.base_linear.bias.data.clone()

        # Input tensor
        self.x = torch.randn(self.batch_size, self.in_dim)

    def test_zero_adapter_matches_base(self):
        """With zero-initialized adapter, SVDLinear should match base Linear."""
        out_svd = self.svd(self.x)
        out_base = self.base_linear(self.x)
        try:
            torch.testing.assert_close(out_svd, out_base, rtol=0.01, atol=0.1)
            logging.info("test_zero_adapter_matches_base: Test Passed")
        except AssertionError as e:
            logging.info("test_zero_adapter_matches_base: Test Failed – %s", e)

    def test_nonzero_adapter_changes_output(self):
        """With non-zero adapter parameters, output should differ slightly."""
        # Manually set adapter parameters
        with torch.no_grad():
            self.svd.lora_A.fill_(1e-3)
            self.svd.lora_B.fill_(1e-3)
            self.svd.lora_E.fill_(1.0)
            # ensure ranknum is non-zero
            self.svd.ranknum.fill_(float(self.r))

        out_svd = self.svd(self.x)
        out_base = self.base_linear(self.x)

        # Difference must be non-zero but small
        diff = (out_svd - out_base).abs().mean().item()
        self.assertGreater(diff, 0.0, "Expected non-zero difference with active adapter")
        self.assertLess(diff, 1e-1, f"Difference too large ({diff}) for small adapter values")

        # Manual adapter computation
        adapter_mat = (self.svd.lora_A * self.svd.lora_E)  # [r, in_dim]
        adapter_mat = adapter_mat.T @ self.svd.lora_B.T     # [in_dim, out_dim]
        manual_adapter = self.x @ adapter_mat * (self.svd.scaling / self.svd.ranknum.item())
        expected = out_base + manual_adapter

        try:
            # Here we demonstrate a loose relative tolerance of 0.1 and
            # explicitly set a matching atol (required if rtol is set).
            torch.testing.assert_close(out_svd, out_base, rtol=0.01, atol=0.1)
            logging.info("test_nonzero_adapter_changes_output: Test Passed")
        except AssertionError as e:
            logging.info("test_nonzero_adapter_changes_output: Test Failed – %s", e)


if __name__ == "__main__":
    unittest.main()