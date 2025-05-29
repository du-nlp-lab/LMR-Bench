import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
from tqdm import tqdm

from src.utils import generate_output
from src.store_activations import forward_pass_with_hooks


def get_tokens_positions(tokenizer: AutoTokenizer, input_context: str, input_query: str, subject_query: str) -> tuple:
    """returns the indexes of the last token for the context object, query subject, and query relation

    Args:
        tokenizer (AutoTokenizer): the tokenizer of the LLM
        input_context (str): the context just before the query which represents a statement about a subject, relation, and a counter-knowledge object
        input_query (str): the query that comes after the context
        subject_query (str): the subject in the query (which is also the same as the one in the context)

    Returns:
        Tuple: a triplet with the positions of the context object, query subject, and query relation respectively
    """
    object_context_idx = len(tokenizer.encode(f"{input_context}")) - 2  # -2 to not consider the point after the context
    subject_query_idx = len(tokenizer.encode(f"{input_context} {subject_query}")) - 1
    relation_query_idx = len(tokenizer.encode(f"{input_context} {input_query}")) - 1

    return object_context_idx, subject_query_idx, relation_query_idx


def build_prompt_for_knowledge_probing(input_context: str, input_query: str):
    return f"{input_context} {input_query}"


def identify_knowledge_source(
    parametric_knowledge_object: str, counter_knowledge_object: str, output_object: str
) -> str:
    parametric_knowledge_object = parametric_knowledge_object.lower().strip()
    output_object = output_object.lower().strip()
    counter_knowledge_object = counter_knowledge_object.lower().strip()

    # counter_knowledge_object_words = (
    #     counter_param_knowledge_record
    #     .counter_knowledge_object
    #     .lower()
    #     .strip()
    #     .split(" ")
    # )

    return (
        "CK"
        if " ".join(output_object.split(" ")[: len(counter_knowledge_object.split(" "))]) == counter_knowledge_object
        else (
            "PK"
            if " ".join(output_object.split(" ")[: len(parametric_knowledge_object.split(" "))])
            == parametric_knowledge_object
            else "ND"
        )
    )


def prompt_model_and_store_activations(
    counter_parametric_knowledge_dataset: pd.DataFrame,
    device: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    model: AutoModelForCausalLM,
    include_first_token: bool,
    include_object_context_token: bool,
    include_subject_query_token: bool,
    include_relation_query_token: bool,
    include_mlps: bool = True,
    include_mlps_l1: bool = False,
    include_mhsa: bool = False,
    vertical: bool = True,
    max_gen_tokens: str = 10,
    split_regexp: str = r'"|\.|\n|\\n|,|\(|\<s\>',
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """
    Prompts the LLM to determine the knowledge source and saves the activations of specific modules (MLP, MLP-L1, MHSA).

    Args:
        parametric_knowledge (pd.DataFrame): Dataframe containing the parametric knowledge dataset.
        counter_parametric_knowledge_dataset (pd.DataFrame): Dataframe containing the counter-parametric-knowledge dataset.
        device (str): Device where inference on the LLM will be run.
        tokenizer (AutoTokenizer): Tokenizer for the LLM.
        model_name (str): Name of the model (mainly for printing and path reference purposes).
        model (AutoModelForCausalLM): Loaded LLM.
        include_first_token (bool): whether to probing on the first token or not (see control experiment of the paper)
        include_object_context_token (bool): whether to probing on the object token in the context or not
        include_subject_query_token (bool): whether to probing on the object token in the query or not
        include_relation_query_token (bool): whether to probing on the relation token in the query or not
        include_mlps (bool): Whether to include the second layer of the MLP (output) in the activations.
        include_mlps_l1 (bool): Whether to include the first layer of the MLP.
        include_mhsa (bool): Whether to include the activations of the output of the MHSA.
        vertical (bool): Whether the token positions are vertical.
        split_regexp (str): the regexp pattern to parse the output of the LLM after prompting

    Returns:
        prompting_results: A DataFrame containing various results and metrics of the probe as well as the activations
    """

    prompting_results = []

    activations = {"mlp": [], "mlp-l1": [], "mhsa": []}

    # We copy to avoid modifying inplace
    counter_parametric_knowledge_dataset = counter_parametric_knowledge_dataset.copy()

    for _, counter_param_knowledge_record in tqdm(
        counter_parametric_knowledge_dataset.iterrows(),
        total=counter_parametric_knowledge_dataset.shape[0],
        desc=f"Probing {model_name} with prompts...",
    ):
        # Example values:
        # - statement: "Paris is the capital of France"
        # - statement_object: "France"
        # - counter_knowledge_object: "Italy"
        # - statement_query: "Paris is the capital of"
        knowledge_probing_prompt_context = counter_param_knowledge_record.statement.replace(
            counter_param_knowledge_record.statement_object, counter_param_knowledge_record.counter_knowledge_object
        )
        knowledge_probing_prompt_query = counter_param_knowledge_record.statement_query

        counter_param_knowledge_record.knowledge_probing_prompt = (
            f"{knowledge_probing_prompt_context} {knowledge_probing_prompt_query}"
        )

        # get the last token position (index) for the context object, query subject, and query relation
        object_context_idx, subject_query_idx, relation_query_idx = get_tokens_positions(
            tokenizer=tokenizer,
            input_context=knowledge_probing_prompt_context,
            input_query=knowledge_probing_prompt_query,
            subject_query=counter_param_knowledge_record.statement_subject,
        )

        # register the hooks at the right module
        forward_pass_with_hooks(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            knowledge_probing_prompt=counter_param_knowledge_record.knowledge_probing_prompt,
            device=device,
            first_token_idx=0,
            object_context_idx=object_context_idx,
            subject_query_idx=subject_query_idx,
            relation_query_idx=relation_query_idx,
            include_first_token=include_first_token,
            include_object_context_token=include_object_context_token,
            include_subject_query_token=include_subject_query_token,
            include_relation_query_token=include_relation_query_token,
            include_mlps=include_mlps,
            include_mlps_l1=include_mlps_l1,
            include_mhsa=include_mhsa,
            vertical=vertical,
            activations=activations,
        )
        output_ids, output_txt = generate_output(
            model=model,
            tokenizer=tokenizer,
            input_txt=counter_param_knowledge_record.knowledge_probing_prompt,
            device=device,
            max_new_tokens=max_gen_tokens,
        )
        output_object = re.split(split_regexp, output_txt)[0]

        knowledge_source = identify_knowledge_source(
            parametric_knowledge_object=counter_param_knowledge_record["parametric_knowledge_object"],
            counter_knowledge_object=counter_param_knowledge_record["counter_knowledge_object"],
            output_object=output_object,
        )
        prompting_results.append(
            {
                "knowledge_source": knowledge_source,
                "output_txt": output_txt,
                "output_object": output_object.lower().strip(),
                "counter_knowledge_object": counter_param_knowledge_record["counter_knowledge_object"].lower().strip(),
                "parametric_knowledge_object": counter_param_knowledge_record["parametric_knowledge_object"]
                .lower()
                .strip(),
                "statement_query": counter_param_knowledge_record["statement_query"],
                "statement_subject": counter_param_knowledge_record["statement_subject"],
                "rel_lemma": counter_param_knowledge_record.rel_lemma,
                "relation_group_id": counter_param_knowledge_record.relation_group_id,
            }
        )

    prompting_results_df = pd.DataFrame(prompting_results)

    if include_mlps:
        prompting_results_df["mlp"] = activations["mlp"]
    if include_mlps_l1:
        prompting_results_df["mlp-l1"] = activations["mlp-l1"]
    if include_mhsa:
        prompting_results_df["mhsa"] = activations["mhsa"]

    return prompting_results_df
