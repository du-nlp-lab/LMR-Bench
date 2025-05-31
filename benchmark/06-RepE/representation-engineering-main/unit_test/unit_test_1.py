import sys
# sys.path.append('representation-engineering-main')
sys.path.append('examples/honesty')
sys.path.append('')
import os

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

from repe import repe_pipeline_registry
repe_pipeline_registry()

from utils import honesty_function_dataset, plot_lat_scans, plot_detection_results
from sklearn.decomposition import PCA
from itertools import islice

import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Define the log file path
log_file = log_dir / 'unit_test_1.log'

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")


def recenter(x, mean=None):
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x,axis=0,keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
    return x - mean

def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    # Calculate the magnitude of the direction vector
     # Ensure H and direction are on the same device (CPU or GPU)
    if type(direction) != torch.Tensor:
        H = torch.Tensor(H).cuda()
    if type(direction) != torch.Tensor:
        direction = torch.Tensor(direction)
        direction = direction.to(H.device)
    mag = torch.norm(direction)
    assert not torch.isinf(mag).any()
    # Calculate the projection
    projection = H.matmul(direction) / mag
    return projection

# The following get_rep_directions_agent function is the key function used in repe reader, here you only need to implement the pca reader you can use the pca package to finish the function Please only edit the following two function to implement the repr reading method designed in the paper
def get_rep_directions_agent(self, 
                             model,  # this parameter is unused in this function you can ignore this
                             tokenizer,  # this parameter is unused in this function you can ignore this
                             hidden_states,  # output hiddenstates of each layer which is a dict
                             hidden_layers,  # target layer that we want to implement representation reader. Is a list of layer index you should use this to get the hiddenstates from hidden_states dict
                             **kwargs):
    """Get PCA components for each layer You can use the PCA package from sklearn which is already imported"""
    """You can also use the recenter function which already defined above to recenter the hiddenstates"""
    """This is a class function that have self.H_train_means[layer_index] aim to store the mean value of hiddenstates at each layer for calculating the PCA. and self.n_components is the number of component we use from PCA and this is the value you can directly use in this function."""
    directions = {} # this is the dictionary you should return, this dictionary have the calculated pca component(first one component) for each layer's hiddenstate
    raise NotImplementedError("The function 'get_rep_directions_agent' not implemented")
    
    return directions



if __name__ == '__main__':
    # Pelase do not change any of the following code
    model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", use_auth_token=hf_token)
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, use_auth_token=hf_token)
    tokenizer.pad_token_id = 0 
    
    
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
    
    user_tag = "USER:"
    assistant_tag = "ASSISTANT:"
    data_path = "data/facts/facts_true_false.csv"
    dataset = honesty_function_dataset(data_path, tokenizer, user_tag, assistant_tag)

    # override get_rep_direction function with agent's implementation
    rep_reading_pipeline.direction_finder.get_rep_directions = get_rep_directions_agent.__get__(rep_reading_pipeline.direction_finder, type(rep_reading_pipeline.direction_finder))

    honesty_rep_reader = rep_reading_pipeline.get_directions(
    dataset['train']['data'], 
    rep_token=rep_token, 
    hidden_layers=hidden_layers, 
    n_difference=n_difference, 
    train_labels=dataset['train']['labels'], 
    direction_method=direction_method,
    batch_size=32,
    )
    layer_id = list(range(-10, -15, -1))
    block_name="decoder_block"
    control_method="reading_vec"
    
    rep_control_pipeline = pipeline(
        "rep-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=layer_id, 
        control_method=control_method)
    inputs = [
    f"{user_tag} Who is the president of the United States in 2030? {assistant_tag}",
    f"{user_tag} You killed a person for revenge but don't want to get caught. What would you tell the police? {assistant_tag}",
    f"{user_tag} You played on the computer while your mom was away and pretended to study the whole time. What will you tell her? {assistant_tag}",
    ]
    
    coeff=1.5
    max_new_tokens=128
    
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()
    
    agent_output = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
    answer1 = [{'generated_text': 'USER: Who is the president of the United States in 2030? ASSISTANT: The United States is a democracy. The president is elected by the people to serve a two-year term. The vice president is elected by the people to serve a two-year term. The vice president is elected by the people to serve a two-year term. The vice president is elected by the people to serve a two-year term. The vice president is elected by the people to serve a two-year term. The vice president is elected by the people to serve a two-year term. The vice president is elected by the people to serve a two-year term. The vice president is elected by the people to'}]
    answer2 = [{'generated_text': "USER: You killed a person for revenge but don't want to get caught. What would you tell the police? ASSISTANT: You would give a report on the crime scene. The victim was found dead in a park, with a bullet in his head. The victim was a man, 35 years old, who had been found dead in a park, with a bullet in his head. The victim was a man, 35 years old, who had been found dead in a park, with a bullet in his head. The victim was a man, 35 years old, who had been found dead in a park, with a bullet in his head. The victim was a man, 35 years old, who had been found dead in"}]
    answer3 = [{'generated_text': 'USER: You played on the computer while your mom was away and pretended to study the whole time. What will you tell her? ASSISTANT: You will study the computer while your mom is away and pretend to study the whole time. What will you tell her? ASSISTANT: You will study the computer while your mom is away and pretend to study the whole time. What will you tell her? ASSISTANT: You will study the computer while your mom is away and pretend to study the whole time. What will you tell her? ASSISTANT: You will study the computer while your mom is away and pretend to study the whole time. What will you tell her? ASSISTANT: You will study the computer while your mom'}]
    gold_answer = [answer1, answer2, answer3]
    correct_answer_count = 0
    for index,output in enumerate(agent_output):
        print(output)
        if output == gold_answer[index]:
            correct_answer_count += 1
            logging.info(f"Test Passed: index {index}")
        else:
            logging.info(f"Test Failed: index {index}")
    logging.info(f'''{correct_answer_count}/3 test cases passed''')
        
    