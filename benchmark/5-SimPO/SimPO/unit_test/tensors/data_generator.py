import torch
import os
from scripts.simpo_trainer import SimPOTrainer


def generate_dpo_test_data(
    num_samples=100,
    mean_chosen=-2.5,
    std_chosen=0.5,
    mean_rejected=-4.0,
    std_rejected=0.7,
    save_dir="tensors"
):
    """
    Generate test data for DPO (Direct Preference Optimization) algorithm.
    
    This generates two tensors:
    - policy_chosen_logps: Log probabilities for chosen responses
    - policy_rejected_logps: Log probabilities for rejected responses
    
    For a realistic DPO setup, chosen responses typically have higher log probabilities
    than rejected responses, but with some overlap in distributions.
    
    Args:
        num_samples: Number of test samples to generate
        mean_chosen: Mean of log probabilities for chosen responses
        std_chosen: Standard deviation for chosen responses
        mean_rejected: Mean of log probabilities for rejected responses
        std_rejected: Standard deviation for rejected responses
        save_dir: Directory to save the tensors
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate log probabilities for chosen responses
    # These should generally be higher (less negative) as they're preferred
    policy_chosen_logps = torch.normal(mean=mean_chosen, std=std_chosen, size=(num_samples,))
    
    # Generate log probabilities for rejected responses
    # These should generally be lower (more negative) as they're less preferred
    policy_rejected_logps = torch.normal(mean=mean_rejected, std=std_rejected, size=(num_samples,))
    
    # Save tensors
    torch.save(policy_chosen_logps, os.path.join(save_dir, "policy_chosen_logps.pt"))
    torch.save(policy_rejected_logps, os.path.join(save_dir, "policy_rejected_logps.pt"))
    
    print(f"Generated {num_samples} samples of DPO test data")
    print(f"Chosen log probs - Mean: {policy_chosen_logps.mean().item():.4f}, Std: {policy_chosen_logps.std().item():.4f}")
    print(f"Rejected log probs - Mean: {policy_rejected_logps.mean().item():.4f}, Std: {policy_rejected_logps.std().item():.4f}")
    
    # Verify that chosen responses generally have higher probabilities
    higher_chosen = (policy_chosen_logps > policy_rejected_logps).float().mean().item()
    print(f"Percentage where chosen > rejected: {higher_chosen * 100:.2f}%")
    
    return policy_chosen_logps, policy_rejected_logps










from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F

def simpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        beta = 2.0
        device = 'cuda'
        gamma_beta_ratios = 0.25
        label_smoothing = 0
        loss_type = "sigmoid"


        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(device)
        logits = pi_logratios - gamma_beta_ratios

        if loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        elif loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = beta * policy_chosen_logps.to(device).detach()
        rejected_rewards = beta * policy_rejected_logps.to(device).detach()

        return losses, chosen_rewards, rejected_rewards


def generate_golden_loss():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simpo = SimPOTrainer()
    simpo_loss = simpo.simpo_loss
    # Load inputs and outputs from .pt files
    policy_chosen_logps = torch.load("tensors/policy_chosen_logps.pt").to(device)  
    policy_rejected_logps = torch.load("tensors/policy_rejected_logps.pt").to(device) 
    
    losses, chosen_rewards, rejected_rewards = simpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
    )
    print(losses)
        

if __name__ == "__main__":
    # You can adjust these parameters to match your specific requirements
    # chosen_logps, rejected_logps = generate_dpo_test_data(
    #     num_samples=1000,  # Number of preference pairs
    #     mean_chosen=-2.5,  # Chosen responses have higher (less negative) log probs
    #     std_chosen=0.5,
    #     mean_rejected=-4.0,  # Rejected responses have lower (more negative) log probs
    #     std_rejected=0.7,
    #     save_dir="tensors"
    # )
    
    # # Optional: Show some example values
    # print("\nExample values (first 5):")
    # print("Chosen log probs:", chosen_logps[:5].tolist())
    # print("Rejected log probs:", rejected_logps[:5].tolist())

    # Load inputs and outputs from .pt files
    device = 'cuda'
    policy_chosen_logps = torch.load("tensors/policy_chosen_logps.pt").to(device)  
    policy_rejected_logps = torch.load("tensors/policy_rejected_logps.pt").to(device) 
    losses, chosen_rewards, rejected_rewards = simpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
    )
    torch.save(losses, os.path.join("tensors", "losses.pt"))
    print('Done!')
            
    
    
 
 