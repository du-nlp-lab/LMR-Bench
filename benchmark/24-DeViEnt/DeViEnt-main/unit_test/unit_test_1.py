import torch
import unittest
import argparse
import sys
import os

# ✅ 手动处理自定义参数，避免 unittest 报错
custom_parser = argparse.ArgumentParser()
custom_parser.add_argument('--seed', type=int, default=42, help="Seed for test data")
custom_parser.add_argument('--overwrite_gt', action='store_true', help="Overwrite ground truth with current model output")
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv  # 只保留 unittest 支持的参数

# ✅ 加载模型相关模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Evaluator')))
from visual_text_training import BERTModelModule, set_seed


class TestLossFunction(unittest.TestCase):
    def setUp(self):
        self.seed = custom_args.seed
        self.overwrite_gt = custom_args.overwrite_gt

        set_seed(self.seed)

        self.test_file = f"Evaluator/test_data_seed{self.seed}.pt"
        self.device = "cpu"

        # ✅ 使用本地缓存路径加载模型
        local_model_path = os.path.join(os.path.dirname(__file__), ".cache", "bert-base-uncased")
        print(f"[DEBUG] Loading model from: {local_model_path}")

        self.model = BERTModelModule(model_name=local_model_path, use_classification_head=True).to(self.device)

        torch.use_deterministic_algorithms(True)
        self.model.eval()

        if os.path.exists(self.test_file):
            self.test_data = torch.load(self.test_file)
        else:
            raise FileNotFoundError(f"{self.test_file} not found!")


    def print_diff(self, name, actual, expected):
        if actual is not None and expected is not None:
            diff = (actual - expected).abs().max().item()
            print(f"[DEBUG] Max diff for {name}: {diff}")

    def test_loss_consistency(self):
        """确保相同输入得到相同的 loss 和输出"""
        batch = (
            self.test_data["images"],
            {k: v for k, v in self.test_data["hypo_premise_inputs"].items()},
            {k: v for k, v in self.test_data["hypo_update_inputs"].items()},
            self.test_data["update_types"]
        )

        with torch.no_grad():
            loss, logits_hypo_premise, logits_hypo_update, score_hypo_premise, score_hypo_update = self.model.compute_loss_and_scores(batch)

        if self.overwrite_gt:
            torch.save({
                "loss": loss,
                "logits_hypo_premise": logits_hypo_premise,
                "logits_hypo_update": logits_hypo_update,
                "score_hypo_premise": score_hypo_premise,
                "score_hypo_update": score_hypo_update,
                "hypo_premise_inputs": self.test_data["hypo_premise_inputs"],
                "hypo_update_inputs": self.test_data["hypo_update_inputs"],
                "images": self.test_data["images"],
                "update_types": self.test_data["update_types"]
            }, self.test_file)
            print(f"✅ Overwritten ground truth saved to {self.test_file}")
            return

        # 比较 loss 和 logits 等
        self.print_diff("loss", loss, self.test_data["loss"])
        self.assertTrue(torch.allclose(loss, self.test_data["loss"], atol=0.1),
                        f"[LOSS MISMATCH]\nExpected: {self.test_data['loss']}\nGot: {loss}")

        if logits_hypo_premise is not None and self.test_data["logits_hypo_premise"] is not None:
            self.print_diff("logits_hypo_premise", logits_hypo_premise, self.test_data["logits_hypo_premise"])
            self.assertTrue(torch.allclose(logits_hypo_premise, self.test_data["logits_hypo_premise"], atol=0.1))

        if logits_hypo_update is not None and self.test_data["logits_hypo_update"] is not None:
            self.print_diff("logits_hypo_update", logits_hypo_update, self.test_data["logits_hypo_update"])
            self.assertTrue(torch.allclose(logits_hypo_update, self.test_data["logits_hypo_update"], atol=0.1))

        if score_hypo_premise is not None and self.test_data["score_hypo_premise"] is not None:
            self.print_diff("score_hypo_premise", score_hypo_premise, self.test_data["score_hypo_premise"])
            self.assertTrue(torch.allclose(score_hypo_premise, self.test_data["score_hypo_premise"], atol=0.1))

        if score_hypo_update is not None and self.test_data["score_hypo_update"] is not None:
            self.print_diff("score_hypo_update", score_hypo_update, self.test_data["score_hypo_update"])
            self.assertTrue(torch.allclose(score_hypo_update, self.test_data["score_hypo_update"], atol=0.1))

        print("✅ All tests passed!")


if __name__ == "__main__":
    unittest.main()