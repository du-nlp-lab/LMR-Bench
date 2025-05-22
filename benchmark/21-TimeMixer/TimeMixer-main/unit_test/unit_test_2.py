import torch
import random
import unittest
import pickle
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import from the models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.TimeMixer import MultiScaleTrendMixing

class Args:
    """Mock arguments class to simulate the config parameters used in the model"""
    def __init__(self):
        self.seq_len = 96  # Default from run.py
        self.down_sampling_window = 2  # Default value, adjust if needed
        self.down_sampling_layers = 3  # Default value, adjust if needed

class TestTrendTransformation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create a mock config similar to what the model would use
        self.config = Args()

        # Load inputs and outputs from pickle files
        with open('tensors/trend_list.pkl', 'rb') as f:
            self.trend_list = []
            while True:
                try:
                    obj = pickle.load(f)
                    self.trend_list.append(obj)
                except EOFError:
                    break

        with open('tensors/out_trend_list.pkl', 'rb') as f:
            self.out_trend_list = []
            while True:
                try:
                    obj = pickle.load(f)
                    self.out_trend_list.append(obj)
                except EOFError:
                    break

    def test_random_samples(self):
        # Select three random indices if there are at least 3 elements, otherwise use all available
        num_samples = min(3, len(self.trend_list))
        indices = random.sample(range(len(self.trend_list)), num_samples)
        
        print(f"Testing indices: {indices}")
        
        # Check if the model state exists
        model_state_path = 'tensors/trend_model_state.pt'
        has_saved_model = os.path.exists(model_state_path)
        
        for idx in indices:
            trend_input = self.trend_list[idx]
            out_trend_expected = self.out_trend_list[idx]
            
            # For sanity check, print the shapes of inputs and outputs
            print(f"\nTesting sample {idx}:")
            print(f"Input shape: {[t.shape for t in trend_input]}")
            print(f"Expected output shape: {[t.shape for t in out_trend_expected]}")
            
            # Create model with parameters determined from the data
            model = MultiScaleTrendMixing(
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
            trend_input_device = [t.to(self.device) for t in trend_input]
            
            # Run the model to get the actual output
            with torch.no_grad():
                out_trend_actual = model(trend_input_device)
            
            # Check that the output length matches the expected output length
            self.assertEqual(len(out_trend_actual), len(out_trend_expected), 
                            f"Output length mismatch: expected {len(out_trend_expected)}, got {len(out_trend_actual)}")
            
            for i, (actual, expected) in enumerate(zip(out_trend_actual, out_trend_expected)):
                expected = expected.to(self.device)  # Make sure expected is on the same device
                
                # Compare tensor values with tolerance
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5), 
                               f"Tensor values don't match at position {i}")
                
            print(f"Sample {idx} passed all tests!")

if __name__ == "__main__":
    unittest.main() 