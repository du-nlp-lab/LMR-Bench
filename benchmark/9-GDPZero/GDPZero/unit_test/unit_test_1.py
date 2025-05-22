import logging
import torch
import random
import pickle
import sys
import unittest
import numpy as np

sys.path.append("../")

from core.mcts import OpenLoopMCTS
from pathlib import Path
from utils.utils import dotdict

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

class TestModelOutputs(unittest.TestCase):
    def setUp(self):
        self.args = dotdict({
            "cpuct": 1.0,
            "num_MCTS_sims": 10,
            "max_realizations": 3,
            "Q_0": 0.25,
        })
        self.open_mcts = OpenLoopMCTS(None, None, self.args)

        # Load inputs and outputs from pkl files
        self.valid_moves = pickle.load(open("unit_test/pickles/valid_moves.pkl", "rb"))
        self.hashable_state = pickle.load(open("unit_test/pickles/hashable_state.pkl", "rb"))
        self.Ns = pickle.load(open("unit_test/pickles/Ns.pkl", "rb"))
        self.Nsa = pickle.load(open("unit_test/pickles/Nsa.pkl", "rb"))
        self.P = pickle.load(open("unit_test/pickles/P.pkl", "rb"))
        self.Q = pickle.load(open("unit_test/pickles/Q.pkl", "rb"))
        self.best_action = pickle.load(open("unit_test/pickles/best_action.pkl", "rb"))
        self.best_uct = pickle.load(open("unit_test/pickles/best_uct.pkl", "rb"))

        # Ensure inputs and outputs have the same length
        assert len(self.valid_moves) == len(self.hashable_state) == len(self.Ns) == len(self.Nsa) == \
               len(self.P) == len(self.best_action) == len(self.best_uct)
        print("Len:", len(self.valid_moves))
    
    def test_random_pairs(self):
        # Select three random indices
        indices = random.sample(range(len(self.valid_moves)), 3)
        
        for idx in indices:
            self.open_mcts.valid_moves = self.valid_moves[idx]
            hashable_state = self.hashable_state[idx]
            self.open_mcts.Ns = self.Ns[idx]
            self.open_mcts.Nsa = self.Nsa[idx]
            self.open_mcts.P = self.P[idx]
            self.open_mcts.Q = self.Q[idx]
            
            best_action_exptected = self.best_action[idx]
            best_uct_exptected = self.best_uct[idx]
            
            try:
                best_action, best_uct = self.open_mcts.find_best_action(hashable_state)
            except:
                logging.info("Test Failed")

            if torch.allclose(torch.tensor(best_action, dtype=torch.int64), torch.tensor(best_action_exptected, dtype=torch.int64), rtol=0.01):
                logging.info(f"Test Passed: Best Action Expected {best_action_exptected}, got {best_action}")
            else:
                logging.info(f"Test Failed: Best Action Expected {best_action_exptected}, got {best_action}")

            if torch.allclose(torch.tensor(best_uct), torch.tensor(best_uct_exptected), rtol=0.01):
                logging.info(f"Test Passed: Best UCT Expected {best_uct_exptected}, got {best_uct}")
            else:
                logging.info(f"Test Failed: Best UCT Expected {best_uct_exptected}, got {best_uct}")


if __name__ == "__main__":
    unittest.main()
