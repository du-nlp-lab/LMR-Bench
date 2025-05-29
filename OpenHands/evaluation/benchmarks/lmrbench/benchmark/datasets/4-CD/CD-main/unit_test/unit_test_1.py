import torch
import numpy as np
import unittest
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from main import probing

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
 

class TestProbingResults(unittest.TestCase):
    def setUp(self):
        # Load numpy arrays
        self.rp_log_data_list = np.load('unit_test/rp_log_data_list.npy', allow_pickle=True)
        self.rp_question_data_list = np.load('unit_test/rp_question_data_list.npy', allow_pickle=True)
        self.labels_log = np.load('unit_test/labels_log.npy')
        self.labels_question = np.load('unit_test/labels_question.npy')

        with open('unit_test/tot_layer.json', 'r') as f:
            self.tot_layer = json.load(f)["tot_layer"]

        self.list_acc_expected = np.load('unit_test/list_acc.npy')
        self.list_f1_expected = np.load('unit_test/list_f1.npy')
        self.list_auc_expected = np.load('unit_test/list_auc.npy')

    def test_probing_accuracy(self):
        list_acc, list_f1, list_auc = [], [], []
        list_acc, list_f1, list_auc = probing(self.rp_log_data_list, self.rp_question_data_list, self.labels_log, self.labels_question, self.tot_layer)

        checks = [
            ("Accuracy", list_acc, self.list_acc_expected),
            ("F1 Score", list_f1,   self.list_f1_expected),
            ("ROC AUC", list_auc,   self.list_auc_expected),
        ]

        for name, actual, expected in checks:
            msg = f"{name} mismatch. Test Failed: Expected {expected}, got {actual}"
            if np.allclose(actual, expected, rtol=0.01):
                logging.info(f"Test Passed")
            else:
                # log the failure
                logging.info("Test Failed")

if __name__ == "__main__":
    unittest.main()