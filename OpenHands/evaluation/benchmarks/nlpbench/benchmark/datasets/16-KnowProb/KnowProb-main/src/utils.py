import argparse
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
import torch


def query(query: str, subject: str, relation: str, instruct: bool) -> str:
    """creates a query to get the object given the subject and relation

    Args:
        query (str): the query that is provided in the ParaRel dataset
        subject (str): subject of the knowledge triplet (subject, relation, object)
        relation (str): relation of the knowledge triplet (subject, relation, object)
        instruct (bool): whether to return an instructed query or not

    Returns:
        str: a query to directly prompt the LLM on the subject
    """
    if instruct:
        return f"What is the capital city of {subject}? Capital name:"
    else:
        return query


def statement(query: str, obj: str) -> str:
    """returns a statement about a triplet knowledge

    Args:
        query (str): query containing a subject and relation
        obj (str): object of the knowledge triplet (subject, relation, object)

    Returns:
        str: a statement of the subject, relation, and object
    """
    return query + " " + obj + "."


def parse_args():
    parser = argparse.ArgumentParser(description="Processing")
    parser.add_argument("--model_name", type=str, help="Model name to build the knowledge datasets.")
    parser.add_argument("--device", type=str, help="Device on which to run.")
    parser.add_argument(
        "--nb_counter_parametric_knowledge",
        "-nb_cpk",
        type=int,
        help="Number of counter-parametric-knowledge triplets per triplet knowledge.",
    )
    parser.add_argument("--include_mlps", action="store_true", help="Whether to include MLPs.")
    parser.add_argument("--include_mhsa", action="store_true", help="Whether to include MHSAs.")
    parser.add_argument("--include_mlps_l1", action="store_true", help="Whether to include MLP first layer.")
    parser.add_argument(
        "--test_on_head",
        action="store_true",
        help="Whether to test on the header (n_head first rows of the ParaRel dataset).",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        nargs="?",
        const=0,
        help="The number of rows to test on (only valid if test_on_head is True.)",
    )
    parser.add_argument(
        "--token_position",
        type=str,
        choices=["first", "subject_query", "relation_query", "object"],
        help="Label of the token to consider.",
    )
    return parser.parse_args()


def setup_directories(permanent_path, model_name):
    directories = [
        f"{permanent_path}",
        f"{permanent_path}/paper_data",
        f"{permanent_path}/paper_figures",
        f"{permanent_path}/paper_data/prompting_results",
        f"{permanent_path}/classification_metrics",
        f"{permanent_path}/classification_metrics/{model_name}"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)


def generate_output(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, input_txt: str, device: str, max_new_tokens: int
) -> tuple[Any, str]:
    """
    Generates output from the model given an input text and a tokenizer.

    Args:
        model (AutoModelForCausalLM): The loaded LLM.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        input_txt (str): The input text for generation.
        device (str): The device to run inference on.

    Returns:
        tuple: A tuple containing the generated output token IDs and the output text.
    """
    inputs = tokenizer(input_txt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_txt = tokenizer.decode(output_ids.sequences[0][inputs.input_ids[0].shape[0] :], skip_special_tokens=True)
    return output_ids, output_txt
