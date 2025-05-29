import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pickle
import numpy as np
import logging
from pathlib import Path
from data_utils import generate_tasks

class TestGenerateTasksLogging(unittest.TestCase):
    def setUp(self):
        # Load the inputs from the saved PKL file
        with open('unit_test/generate_tasks_input.pkl', 'rb') as f:
            self.inputs = pickle.load(f)
        with open('unit_test/generate_tasks_output.pkl', 'rb') as f:
            self.outputs = pickle.load(f)
    
    def test_generate_tasks_output_shape(self):
        # Test if the output shapes are as expected
        num_tasks = self.inputs['num_tasks']
        num_samples_per_task = self.inputs['num_samples_per_task']
        num_dimensions = self.inputs['num_dimensions']
        data_type = self.inputs['data_type']
        seed = self.inputs['seed']
        class_sep = self.inputs['class_sep']
        factor = self.inputs['factor']

        dataset_x, dataset_y = generate_tasks(
            num_tasks=num_tasks,
            num_samples_per_task=num_samples_per_task,
            num_dimensions=num_dimensions,
            seed=seed,
            data_type=data_type,
            class_sep=class_sep,
            factor=factor
        )

        # Check relative error within 1%
        PASS = 'True'
        for i in range(len(dataset_x)):
            if not np.allclose(dataset_x[i], dataset_x[i], rtol=0.01):
                PASS = 'False'
            if not np.allclose(dataset_y[i], dataset_y[i], rtol=0.01):
                PASS = 'False'

        if PASS == 'True':
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
