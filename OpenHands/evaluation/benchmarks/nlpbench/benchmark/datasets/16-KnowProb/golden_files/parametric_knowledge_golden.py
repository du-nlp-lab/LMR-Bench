from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from math import exp
import numpy as np
import pandas as pd


def build_parametric_knowledge(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_gen_tokens: int,
) -> dict:
    """this function builds the parametric knowledge

    Args:
        prompt: a prompt in natural language format asking for the object given the subject and relation
        tokenizer: the loaded tokenizer
        model: the loaded LLM
        device: the device where to run the inference on the LLM
        max_gen_tokens: the maximum number of tokens to generate

    Returns:
        A dictionary containing the following information:
            - model_output: the model output sequence
            - confidence_score: the confidence score of the model for generating `model_output`
                                (
                                this confidence score is computed based on the first token such that:
                                confidence_score = p(top_1) - p(top_2)
                                where:
                                - top_1 is the token with the highest probability
                                - top_2 is the token with the second highest probability.
                                )
            - log_likelihood_sequence: log-likelihood of the generated sequence
            - perplexity_sequence: perplexity of the generated sequence
            - log_probas: the log-probabilities of the generated tokens
            - model_output_tokens: the list of generated tokens given the prompt
            - model_output_tokens_ids: the list of IDs of the generated tokens given the prompt
    """

    # print(f"prompt: {prompt}")
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids.input_ids,
            do_sample=False,  # greedy decoding strategy
            max_new_tokens=max_gen_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            attention_mask=prompt_ids.attention_mask,
        )
    model_output_tokens_ids = [token_id for token_id in output_ids.sequences[0][prompt_ids.input_ids.shape[1] :]]
    model_output = tokenizer.decode(model_output_tokens_ids, skip_special_tokens=True)
    model_output_tokens = [tokenizer.decode(token_id) for token_id in model_output_tokens_ids]

    # compute the confidence score
    probas = torch.softmax(output_ids.scores[0], dim=1)[0]
    arg_top2 = torch.topk(probas, 2)
    proba_top_1 = probas[arg_top2.indices[0]]
    proba_top_2 = probas[arg_top2.indices[1]]
    confidence_score = float(proba_top_1 - proba_top_2)

    # compute the log-likelihood
    all_log_probas = tuple(torch.log(torch.softmax(logits, dim=-1))[0].tolist() for logits in output_ids.scores)
    log_likelihood = 0
    nb_gen_tokens = len(all_log_probas)
    for i in range(nb_gen_tokens):
        k = np.argmax(all_log_probas[i])
        log_likelihood += all_log_probas[i][k]
    log_likelihood = float(log_likelihood)

    # compute the perplexity
    perplexity = exp(-(1 / nb_gen_tokens) * log_likelihood)

    return {
        "model_output": model_output,
        "confidence_score": confidence_score,
        "log_likelihood_sequence": log_likelihood,
        "perplexity_sequence": perplexity,
        "log_probas": all_log_probas,
        "model_output_tokens": model_output_tokens,
        "model_output_tokens_ids": model_output_tokens_ids,
    }


def is_parametric_object_not_in_the_prompt(
    parametric_object: str, relations_with_generated_description: pd.DataFrame
) -> bool:
    """this function checks whether the parametric_object is included in the
    one-shot examples that were used to guide the LLM during parametric knowledge
    building as it would mean that it's biased by the prompt.

    Args:
        parametric_object (str): the generated parametric object
        relations_with_generated_description (pd.DataFrame): the dataframe containing all the relations
        along with the objects and subjects that were used in the one-shot example in the prompt
        to guide the LLMs towards generating coherent objects (the used columns are `generation_object_1`
        and `generation_subject_1` which respectively represent the object and the subject used in the
        one-shot examples.)

    Returns:
        bool
    """

    return (
        parametric_object.lower() not in relations_with_generated_description.generation_object_1.str.lower().tolist()
        or parametric_object.lower()
        not in relations_with_generated_description.generation_subject_1.str.lower().tolist()
    )
