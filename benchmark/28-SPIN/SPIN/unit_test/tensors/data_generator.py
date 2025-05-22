import torch
import os
from spin.alignment.trainer_golden import SPINTrainer

import torch
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    GPT2LMHeadModel,
)
from datasets import Dataset
from unittest.mock import Mock, MagicMock
import logging

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



def generate_spin_loss_test_data(batch_size: int = 256, seed: int = 42):
    """
    Generate synthetic log probability data for testing the spin_loss function.

    Args:
        batch_size (int): Number of samples in the batch.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple of four tensors:
            - policy_real_logps
            - policy_generated_logps
            - opponent_real_logps
            - opponent_generated_logps
    """
    # Create a directory to store the tensors if it doesn't exist
    os.makedirs("tensors", exist_ok=True)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Generate synthetic log-probabilities (random negative values simulating logps)
    policy_real_logps = torch.randn(batch_size) - 1.0  # e.g., log probs ~ N(-1, 1)
    policy_generated_logps = torch.randn(batch_size) - 1.2
    opponent_real_logps = torch.randn(batch_size) - 1.5
    opponent_generated_logps = torch.randn(batch_size) - 1.3

    # Save tensors
    torch.save(policy_real_logps, "tensors/policy_real_logps.pt")
    torch.save(policy_generated_logps, "tensors/policy_generated_logps.pt")
    torch.save(opponent_real_logps, "tensors/opponent_real_logps.pt")
    torch.save(opponent_generated_logps, "tensors/opponent_generated_logps.pt")




        

if __name__ == "__main__":

    generate_spin_loss_test_data()
    print("generate_spin_loss_test_data, done!")
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

    spin_loss = trainer.spin_loss

    # Load inputs and outputs from .pt files

    device = 'cuda'
    policy_real_logps = torch.load("tensors/policy_real_logps.pt", weights_only=True).to(device)  
    policy_generated_logps = torch.load("tensors/policy_generated_logps.pt", weights_only=True).to(device) 
    opponent_real_logps = torch.load("tensors/opponent_real_logps.pt", weights_only=True).to(device) 
    opponent_generated_logps = torch.load("tensors/opponent_generated_logps.pt", weights_only=True).to(device) 

    losses, chosen_rewards, rejected_rewards = spin_loss(
        policy_real_logps,
        policy_generated_logps,
        opponent_real_logps,
        opponent_generated_logps
    )
    torch.save(losses, os.path.join("tensors", "losses.pt"))
    print('Done!')
            
    
    
 
 