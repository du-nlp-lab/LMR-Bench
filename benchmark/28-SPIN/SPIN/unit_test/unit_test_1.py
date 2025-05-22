import torch
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    GPT2LMHeadModel,
)
from datasets import Dataset
import unittest
from spin.alignment.trainer import SPINTrainer
from unittest.mock import Mock, MagicMock
import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)


# Define the log file path
log_file = log_dir / 'unit_test_1.log'
logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")


# Mock the create_reference_model function to handle our mock model
def create_reference_model(model=None):
    """Create a properly mocked reference model that has the necessary methods."""
    # Create a new mock with the methods we need
    

    ref_model = MagicMock(spec=GPT2LMHeadModel)
    # ref_model = MagicMock()

    # ref_model.hf_device_map.values()
    # ref_model.hf_device_map = MagicMock(return_value=model)  # Return the model itself to simulate moving to device
    # ref_model.modules = MagicMock(return_value=model)  # Return the model itself to simulate moving to device
    
    # Add any specific attributes needed for the function to work
    return ref_model


class TestModelOutputs(unittest.TestCase):

    def setUp(self):
        # Create mock objects
        model = MagicMock(spec=GPT2LMHeadModel)  # Use a specific model class
        model.to = MagicMock(return_value=model)  # Return the model itself to simulate moving to device
        model.model_parallel = MagicMock(return_value=model)  # Return the model itself to simulate moving to device
        model.config = Mock(is_encoder_decoder=False)
        tokenizer = Mock(spec=AutoTokenizer)
        # Mock datasets
        train_dataset = Mock(spec=Dataset)
        eval_dataset = Mock(spec=Dataset)
        # Create training arguments
        training_args = Mock(spec=TrainingArguments)
        training_args.remove_unused_columns = True
        training_args.gradient_checkpointing = False
        training_args.seed = 42
        training_args.deepspeed_plugin = None
        training_args.get_process_log_level.return_value = logging.INFO  # Use a valid log level
        training_args.fsdp_config = {"xla": False}
        training_args.fsdp = []
        training_args.report_to = []
        training_args.output_dir = 'outputs/iter0-ckpt'
        training_args.max_steps = 1

        # Initialize the SPINTrainer
        trainer = SPINTrainer(
            model=model,
            ref_model=create_reference_model(),  # Let SPINTrainer create a reference model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=512,
            max_prompt_length=128,
            disable_dropout=False,
            generate_during_eval=False,
        )

        self.spin_loss = trainer.spin_loss
        self.device = 'cuda'
        # Load inputs and outputs from .pt files
        self.policy_real_logps = torch.load("unit_test/tensors/policy_real_logps.pt", weights_only=True).to(self.device)  
        self.policy_generated_logps = torch.load("unit_test/tensors/policy_generated_logps.pt", weights_only=True).to(self.device) 
        self.opponent_real_logps = torch.load("unit_test/tensors/opponent_real_logps.pt", weights_only=True).to(self.device) 
        self.opponent_generated_logps = torch.load("unit_test/tensors/opponent_generated_logps.pt", weights_only=True).to(self.device) 

        self.losses = torch.load("unit_test/tensors/losses.pt", weights_only=True).to(self.device)  

        # Ensure inputs and outputs have the same length
        assert len(self.policy_real_logps) == len(self.losses), "Inputs and outputs must have the same length"

    def test_random_pairs(self):
        
        losses_expected = self.losses
        try:
            losses, real_rewards, generated_rewards = self.spin_loss(
                self.policy_real_logps,
                self.policy_generated_logps,
                self.opponent_real_logps,
                self.opponent_generated_logps,
            )
        except:
            logging.info("Test Failed")
        
        if torch.allclose(losses, losses_expected, rtol=0.01):
            logging.info(f"Test Passed: Losses Expected {losses_expected}, got {losses}")
        else:
            logging.info(f"Test Failed: Losses Expected {losses_expected}, got {losses}")



if __name__ == "__main__":
    unittest.main()
