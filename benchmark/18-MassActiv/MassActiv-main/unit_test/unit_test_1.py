import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
import numpy as np
import torch
from main_vit import run_exp1  # Ensure this is the correct import path
import logging
from pathlib import Path
from torchvision import transforms
import argparse
import lib
import monkey_patch as mp 
# import huggingface_hub
# from huggingface_hub import hf_hub_download
# os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./unit_test/.cache")
# os.environ["HF_HOME"] = os.path.abspath("./unit_test/.cache")  # extra safe
# os.environ["HF_HUB_OFFLINE"] = "1"  # 强制脱机模式（不联网

# # Monkey patch to force local cache only
# def hf_hub_download_local_only(*args, **kwargs):
#     kwargs["local_files_only"] = True
#     kwargs["cache_dir"] = "./unit_test/.cache"
#     return huggingface_hub.real_hf_hub_download(*args, **kwargs)

# # 替换 huggingface_hub 的默认下载逻辑
# huggingface_hub.real_hf_hub_download = hf_hub_download
# huggingface_hub.hf_hub_download = hf_hub_download_local_only

log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format="%(asctime)s = %(message)s")
    



class TestExp1(unittest.TestCase):
    def setUp(self):
        self.layer_id = 40  # Adjust as needed
        self.savedir = "results/vit/3d_feat_vis"
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.model_family = "dinov2_reg"
        self.model_size = "giant"
        self.input_file = os.path.join(self.savedir, f"run_exp1_inputs_layer_{self.layer_id}.pkl")
        self.output_file = os.path.join(self.savedir, f"feat_abs_layer_{self.layer_id}.pkl")
        
        os.makedirs(self.savedir, exist_ok=True)
        
        # Load the saved inputs for the test
        with open(self.input_file, 'rb') as f:
            self.inputs = pickle.load(f)

    def test_exp1(self):
        os.environ["HF_HOME"] = "unit_test/.cache"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_family',default="dinov2_reg", type=str)
        parser.add_argument('--model_size', default="giant", type=str)
        parser.add_argument("--layer_id", type=int, default=1)
        parser.add_argument('--exp1', action="store_true", help="plot 3d feature")
        parser.add_argument('--imagenet_dir', type=str, default="/home/mingjies/imagenet-data/val")
        parser.add_argument('--linear_head_path', type=str, default="/data/locus/project_data/project_data2/mingjies/dinov2")
        parser.add_argument('--reg_feat_mean', type=str, default="assets/reg_feat_mean/")
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--num_imgs_mean', type=int, default=10)
        parser.add_argument('--savedir', type=str)
        args = parser.parse_args()
        # Load the ViT model and layers
        model, layers, val_transform = lib.load_vit(args)
        
        # Run the function
        run_exp1(
            self.inputs["layer_id"], 
            layers, 
            model, 
            val_transform, 
            self.inputs["savedir"], 
            self.inputs["model_family"], 
            self.inputs["model_size"]
        )
        
        # Load the expected outputs
        with open(self.output_file, "rb") as f:
            expected_feat_abs = pickle.load(f)

        # Extract the calculated feat_abs
        with open(os.path.join(self.inputs["savedir"], f"feat_abs_layer_{self.inputs['layer_id']}.pkl"), "rb") as f:
            calculated_feat_abs = pickle.load(f)
        
        # Validate that the feature activations match within 1% relative error
        try:
            self.assertTrue(
                np.allclose(calculated_feat_abs, expected_feat_abs, rtol=0.01),
                f"Feature activations differ by more than 1% (expected {expected_feat_abs}, got {calculated_feat_abs})"
            )
            logging.info("Test Passed")
        except AssertionError as e:
            logging.info(f"Test Failed: {e}")

    def mock_load_vit(self):
        # Mock function to simulate the model, layers, and transform
        layers = [torch.nn.Identity() for _ in range(50)]
        model = torch.nn.Sequential(*layers)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return model, layers, val_transform


if __name__ == "__main__":
    unittest.main()
