import sys

sys.path.append('../')

import torch
import random
import unittest
from trainers import tdpo_loss

import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")


class TestModelOutputs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load inputs and outputs from .pt files
        self.tdpo_output_chosen_data = [torch.load('unit_test/tensors/tdpo_output_chosen_1.pt', weights_only=True),
                                        torch.load('unit_test/tensors/tdpo_output_chosen_2.pt', weights_only=True)]
        self.tdpo_output_rejected_data = [torch.load('unit_test/tensors/tdpo_output_rejected_1.pt', weights_only=True),
                                          torch.load('unit_test/tensors/tdpo_output_rejected_2.pt', weights_only=True)]
        self.chosen_logps_margin_data = [data['logps_margin'].to(self.device) for data in self.tdpo_output_chosen_data]
        self.rejected_logps_margin_data = [data['logps_margin'].to(self.device) for data in
                                           self.tdpo_output_rejected_data]
        self.chosen_position_kl_data = [data['per_position_kl'].to(self.device) for data in
                                        self.tdpo_output_chosen_data]
        self.rejected_position_kl_data = [data['per_position_kl'].to(self.device) for data in
                                          self.tdpo_output_rejected_data]

        self.losses = [torch.load('unit_test/tensors/tdpo_loss_1.pt', weights_only=True).to(self.device),
                       torch.load('unit_test/tensors/tdpo_loss_2.pt', weights_only=True).to(self.device)]
        self.chosen_rewards = [torch.load('unit_test/tensors/tdpo_chosen_reward_1.pt', weights_only=True).to(self.device),
                               torch.load('unit_test/tensors/tdpo_chosen_reward_2.pt', weights_only=True).to(self.device)]
        self.rejected_rewards = [torch.load('unit_test/tensors/tdpo_rejected_reward_1.pt', weights_only=True).to(self.device),
                                 torch.load('unit_test/tensors/tdpo_rejected_reward_2.pt', weights_only=True).to(self.device)]

        self.beta = 0.1
        self.alpha = 0.5
        self.if_tdpo2 = True

        # Ensure inputs and outputs have the same length
        assert len(self.tdpo_output_chosen_data) == len(self.losses), "Inputs and outputs must have the same length"

    def test_random_pairs(self):
        # Select three random indices
        indices = [0, 1]

        for idx in indices:
            chosen_logps_margin = self.chosen_logps_margin_data[idx]
            rejected_logps_margin = self.rejected_logps_margin_data[idx]
            chosen_position_kl = self.chosen_position_kl_data[idx]
            rejected_position_kl = self.rejected_position_kl_data[idx]

            losses_expected = self.losses[idx]
            chosen_reward_expected = self.chosen_rewards[idx]
            rejected_reward_expectedd = self.rejected_rewards[idx]

            losses, chosen_reward, rejected_reward = tdpo_loss(chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl,
                               beta=self.beta, alpha=self.alpha, if_tdpo2=self.if_tdpo2)
            
            if torch.allclose(losses, losses_expected, rtol=0.01):
                logging.info(f"Test Passed: Losses Expected {losses_expected}, got {losses}")
            else:
                logging.info(f"Test Failed: Losses Expected {losses_expected}, got {losses}")

            # if torch.allclose(chosen_reward_expected, chosen_reward, rotl=0.01):
            #     logging.info(f"Test Passed: Chosen Reward Expected {chosen_reward_expected}, got {chosen_reward}")
            # else:
            #     logging.info(f"Test Failed: Chosen Reward Expected {chosen_reward_expected}, got {chosen_reward}")
            
            # if torch.allclose(rejected_reward_expectedd, rejected_reward, rtol=0.01):
            #     logging.info(f"Test Passed: Rejected Reward Expected {rejected_reward_expectedd}, got {rejected_reward}")
            # else:
            #     logging.info(f"Test Failed: Rejected Reward Expected {rejected_reward_expectedd}, got {rejected_reward}")




if __name__ == "__main__":
    unittest.main()
