import torch
import random
import unittest
import pickle
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import from the models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.TimeMixer import MultiScaleSeasonMixing

class Args:
    """Mock arguments class to simulate the config parameters used in the model"""
    def __init__(self):
        self.seq_len = 96  # Default from run.py
        self.down_sampling_window = 2  # Default value, adjust if needed
        self.down_sampling_layers = 3  # Default value, adjust if needed

class TestSeasonTransformation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create a mock config similar to what the model would use
        self.config = Args()

        # Load inputs and outputs from pickle files
        with open('tensors/season_list.pkl', 'rb') as f:
            self.season_list = []
            while True:
                try:
                    obj = pickle.load(f)
                    self.season_list.append(obj)
                except EOFError:
                    break

        with open('tensors/out_season_list.pkl', 'rb') as f:
            self.out_season_list = []
            while True:
                try:
                    obj = pickle.load(f)
                    self.out_season_list.append(obj)
                except EOFError:
                    break

    def test_random_samples(self):
        # Select three random indices if there are at least 3 elements, otherwise use all available
        num_samples = min(3, len(self.season_list))
        indices = random.sample(range(len(self.season_list)), num_samples)
        
        print(f"Testing indices: {indices}")
        
        # Check if the model state exists
        model_state_path = 'tensors/season_model_state.pt'
        has_saved_model = os.path.exists(model_state_path)
        
        for idx in indices:
            season_input = self.season_list[idx]
            out_season_expected = self.out_season_list[idx]
            
            # For sanity check, print the shapes of inputs and outputs
            print(f"\nTesting sample {idx}:")
            print(f"Input shape: {[t.shape for t in season_input]}")
            print(f"Expected output shape: {[t.shape for t in out_season_expected]}")
            
            # Create model with parameters determined from the data
            model = MultiScaleSeasonMixing(
                seq_len=self.config.seq_len,
                down_sampling_window=self.config.down_sampling_window,
                down_sampling_layers=self.config.down_sampling_layers
            )
            
            # Load model state if available
            if has_saved_model:
                print(f"Loading model state from {model_state_path}")
                model.load_state_dict(torch.load(model_state_path, map_location=self.device))
            else:
                print("Warning: No saved model state found. Using randomly initialized model.")
                
            model.to(self.device)
            model.eval()
            
            # Move input tensors to the same device as the model
            season_input_device = [t.to(self.device) for t in season_input]
            
            # Run the model to get the actual output
            with torch.no_grad():
                out_season_actual = model(season_input_device)
            
            # Check that the output length matches the expected output length
            self.assertEqual(len(out_season_actual), len(out_season_expected), 
                            f"Output length mismatch: expected {len(out_season_expected)}, got {len(out_season_actual)}")
            
            for i, (actual, expected) in enumerate(zip(out_season_actual, out_season_expected)):
                expected = expected.to(self.device)  # Make sure expected is on the same device
                
                # Compare tensor values with tolerance
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5), 
                               f"Tensor values don't match at position {i}")
                
            print(f"Sample {idx} passed all tests!")

if __name__ == "__main__":
    unittest.main() 