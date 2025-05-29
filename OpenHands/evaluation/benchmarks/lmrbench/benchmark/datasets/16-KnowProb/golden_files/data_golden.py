import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rapidfuzz.distance import JaroWinkler

from src.context_knowledge import build_counter_parametric_knowledge, get_objects_to_first_token_id_dict
from src.parametric_knowledge import build_parametric_knowledge


PRE_PROMPT_FOR_PARAMETRIC_KNOWLEDGE_GENERATION = """<s>[INST] <<SYS>>Complete the following statement directly and concisely with <OBJECT_GENERATION_DESCRIPTION>. Do not try to answer with a detailed explanation, just give the answer in few words without being specific. Do not use any specific formatting and end the output with a point.<</SYS>>
Here is an example:<EXAMPLE_STATEMENTS></s><s>[INST]<STATEMENT_QUERY> [/INST]"""


def read_data(
    permanent_path: str, test_on_head: bool, n_head: int = 100, filename: str = "pararel.csv"
) -> pd.DataFrame:
    """this function reads the ParaRel dataset

    Args:
        permanent_path (str): the path where ParaRel is stored
        test_on_head: weither to consider only the first elements of the dataframe
        n_head: the number of elements in the head of the dataframe to consider (only useful if test_on_head is true)

    Returns:
        pd.DataFrame: the ParaRel dataset as a pandas DataFrame
    """
    pararel = pd.read_csv(f"{permanent_path}/paper_data/{filename}")
    data = pararel
    if test_on_head:
        data = data.iloc[:n_head]
    return data


def build_prompt_for_parametric_knowledge_generation(
    statement: str,
    statement_subject: str,
    statement_object: str,
    example_statements: list,
    example_statement_objects: list,
    object_generation_description: str,
    pre_prompt_for_parametric_knowledge_generation: str = PRE_PROMPT_FOR_PARAMETRIC_KNOWLEDGE_GENERATION,
) -> str:
    """this function build a parametric knowledge prompt to generate the parametric object, it uses a template defined in
    PRE_PROMPT_FOR_PARAMETRIC_KNOWLEDGE_GENERATION.

    Args:
        statement (str): the generated statement by Mistral-large
        statement_subject (str): the subject as it appears in the statement
        statement_object (str): the object as it appears in the statement
        example_statements (list): a list of statements to use as one-shot examples in the parametric knowledge prompt
        example_statement_objects (list): a list of objects corresponding to the statements in the example_statements to use
            as one-shot examples in the parametric knowledge prompt
        object_generation_description (str): a description of the type of expected objects (it is specific to the relation in the prompt)
        pre_prompt_for_parametric_knowledge_generation (str, optional): the parametric knowledge prompt template. Defaults to PRE_PROMPT_FOR_PARAMETRIC_KNOWLEDGE_GENERATION.

    Returns:
        str: a prompt that is specific to `statement`
    """

    generation_statements_examples = ""
    # for example_statement, example_statement_object in zip(example_statements, example_statement_objects):
    generation_statements_examples += f" {example_statements[0]} [/INST]{example_statement_objects[0]}."

    prompt = pre_prompt_for_parametric_knowledge_generation.replace(
        "<OBJECT_GENERATION_DESCRIPTION>", object_generation_description
    )
    prompt = prompt.replace("<GENERATION_STATEMENT_EXAMPLES>", generation_statements_examples)
    prompt = prompt.replace("<STATEMENT_QUERY>", statement.replace(statement_object + ".", "").strip())
    prompt = prompt.replace("the subject", statement_subject)
    prompt = prompt.replace("<EXAMPLE_STATEMENTS>", generation_statements_examples)

    return prompt


def generate_parametric_knowledge_dataset(
    data: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    relations_with_gen_desc: pd.DataFrame,
    device: str,
    max_gen_tokens: int,
    split_regexp: str,
    pre_prompt_for_parametric_knowledge_generation: str = PRE_PROMPT_FOR_PARAMETRIC_KNOWLEDGE_GENERATION,
) -> pd.DataFrame:
    """this function builds parametric knowledge for the model and returns parametric-based object in the parametric_object column

    Args:
        data (pd.DataFrame): the ParaRel dataframe
        model (transformers.AutoModelForCausalLM): the LLM
        tokenizer (transformers.AutoTokenizer): the tokenizer
        relations_with_gen_desc (pd.DataFrame): a dataframe containing all the relation with their corresponding generation description
        device (str): where to process the inference (cuda, mbp, or cpu)
        max_gen_tokens (int): the maximum number of tokens to generate
        split_regexp (str): the regular expression that is used to split the ouput of the LLMs
        pre_prompt_for_parametric_knowledge_generation (str): the prompt template that is used to generate the parametric knowledge (default to
            PRE_PROMPT_FOR_PARAMETRIC_KNOWLEDGE_GENERATION)

    Returns:
        pd.DataFrame: the dataframe containing the parametric knowledge:
            - parametric_object: the parametric-knowledge-based object
            - model_output: the model output sequence
            - confidence_score: the confidence score of the model for generating `model_output`
            - log_probability_parametric_object_first_token: the log-probability of the parametric object's first token
            - log_likelihood_sequence: log-likelihood of the generated sequence
            - perplexity_sequence: perplexity of the generated sequence
    """

    tqdm.pandas(desc="Generating parametric knowledge...")

    parametric_knowledge_data = data.progress_apply(
        lambda e: build_parametric_knowledge(
            build_prompt_for_parametric_knowledge_generation(
                e.statement,
                e.statement_subject,
                e.statement_object,
                [
                    relations_with_gen_desc[
                        relations_with_gen_desc.relation == e.rel_lemma
                    ].generation_statement_1.iloc[0],
                    relations_with_gen_desc[
                        relations_with_gen_desc.relation == e.rel_lemma
                    ].generation_statement_2.iloc[0],
                ],
                [
                    relations_with_gen_desc[relations_with_gen_desc.relation == e.rel_lemma].generation_object_1.iloc[
                        0
                    ],
                    relations_with_gen_desc[relations_with_gen_desc.relation == e.rel_lemma].generation_object_2.iloc[
                        0
                    ],
                ],
                relations_with_gen_desc[
                    relations_with_gen_desc.relation == e.rel_lemma
                ].object_generation_description.iloc[0],
            ),
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_gen_tokens=max_gen_tokens,
        ),
        axis=1,
    )
    parametric_knowledge_data = pd.DataFrame.from_dict(parametric_knowledge_data.tolist(), orient="columns")

    # the log-probability of the first generated token
    parametric_knowledge_data["log_probability_parametric_object_first_token"] = [
        log_prob[0] for log_prob in parametric_knowledge_data["log_probas"]
    ]
    data = pd.concat([data.reset_index(drop=True), parametric_knowledge_data.reset_index(drop=True)], axis=1)

    data = pd.merge(
        data,
        relations_with_gen_desc[["relation_group_id", "relation"]],
        how="left",
        left_on="rel_lemma",
        right_on="relation",
    )

    cleaned_model_outputs = [
        seq[0].strip().lower() if isinstance(seq, list) else None
        for seq in data.model_output.str.split(split_regexp).tolist()
    ]
    data = data.rename(columns={"object": "knowledge_base_object"})
    data["parametric_object"] = cleaned_model_outputs

    return data


def generate_counter_parametric_knowledge_dataset(
    parametric_knowledge: pd.DataFrame, nb_counter_parametric_knowledge: int
) -> pd.DataFrame:
    """this function generates the counter-parametric knowledge using the
    parametric knowledge. It generates `nb_counter_parametric_knowledge`
    counter-knowledge units for each parametric knowledge element.

    Args:
        parametric_knowledge (pd.DataFrame): the parametric knowledge dataframe
        nb_counter_parametric_knowledge (int): the number of counter-knowledge
            units to generate for each parametric knowledge element.

    Returns:
        pd.DataFrame: a dataframe that contains the counter-knowledge dataset with
        the following columns:
            - statement_subject
            - rel_lemma
            - statement_object
            - relation_group_id
            - parametric_object
            - statement
            - model_output_tokens
            - parametric_object_first_token_log_probas_distribution
            - nb_counter_parametric_knowledge
            - all_dataset_objects_to_their_first_token_id
            - parametric_knowledge_rel
    """

    object_to_first_token_id = get_objects_to_first_token_id_dict(parametric_knowledge)

    counter_parametric_knowledge_list = []
    # tqdm.pandas(desc="Generating counter parametric knowledge...")
    for _, param_knowledge_record in tqdm(parametric_knowledge.iterrows()):
        # build nb_counter_parametric_knowledge counter parametric knowledge triplets for each parametric knowledge element

        counter_parametric_knowledge_list += build_counter_parametric_knowledge(
            statement_subject=param_knowledge_record.statement_subject,
            rel_lemma=param_knowledge_record.rel_lemma,
            statement_object=param_knowledge_record.statement_object,
            relation_group_id=param_knowledge_record.relation_group_id,
            parametric_object=param_knowledge_record.parametric_object,
            statement=param_knowledge_record.statement,
            parametric_object_first_token_log_probas_distribution=param_knowledge_record.log_probability_parametric_object_first_token,
            nb_counter_parametric_knowledge=nb_counter_parametric_knowledge,
            all_dataset_objects_to_their_first_token_id=object_to_first_token_id,
            parametric_knowledge_rel=parametric_knowledge[
                parametric_knowledge.rel_lemma == param_knowledge_record.rel_lemma
            ],
        )

    counter_parametric_knowledge_df = pd.DataFrame(counter_parametric_knowledge_list)

    return counter_parametric_knowledge_df



def detect_subject_object_overlap(subject_str: str, object_str: str, string_similarity_threshold: float) -> bool:
    """Determine whether there is overlap between a subject and an object in a statement.

    If either of them is contained in the other when cast to lowercase, return True.

    Otherwise, return True iff the **cased** normalized jaro-winkler similarity between the two
    if lower than the specified threshold.
    """
    assert isinstance(subject_str, str)
    assert isinstance(object_str, str)

    if (subject_str.lower() in object_str.lower()) or (object_str.lower() in subject_str.lower()):
        return True

    return JaroWinkler.normalized_similarity(subject_str, object_str) >= string_similarity_threshold


def remove_object_subject_overlap(
    parametric_knowledge_dataset: pd.DataFrame, string_similarity_ratio=0.8
) -> pd.DataFrame:
    """this function deletes the examples where the subject and the parametric object are similar using
    a Jaro-Winkler string similarity.

    Args:
        parametric_knowledge_dataset (pd.DataFrame): the parametric knowledge dataset
        string_similarity_ratio (float, optional): the threshold to use in the Jaro-Winkler string distance. Defaults to 0.8.

    Returns:
        pd.DataFrame: the new parametric knowledge dataset with no similar objects and subjects.
    """

    have_overlap = parametric_knowledge_dataset.apply(
        lambda row: detect_subject_object_overlap(
            row["statement_subject"], row["parametric_object"], string_similarity_threshold=string_similarity_ratio
        ),
        axis=1,
    )

    return parametric_knowledge_dataset[~have_overlap]


def return_entity_overlap_between_relation_groups(parametric_knowledge_dataset: pd.DataFrame) -> dict:
    """this function returns a dict of dicts representing a kind of matrix
    of shape (nb_relation_groups, nb_relation_groups), where each value
    is the number of examples in a relation group that have the same subject
    or object in another relation group.

    Args:
        parametric_knowledge_dataset (pd.DataFrame): the parametric knowledge dataset
    """

    relation_groups_entity_overlap_count = {
        relation_group_id: {
            sub_relation_group_id: None
            for sub_relation_group_id in parametric_knowledge_dataset.relation_group_id.unique()
        }
        for relation_group_id in parametric_knowledge_dataset.relation_group_id.unique()
    }

    for relation_group_id in relation_groups_entity_overlap_count:
        for sub_relation_group_id in relation_groups_entity_overlap_count[relation_group_id]:
            relation_group_subjects = parametric_knowledge_dataset[
                parametric_knowledge_dataset.relation_group_id == relation_group_id
            ].statement_subject
            sub_relation_group_subjects = parametric_knowledge_dataset[
                parametric_knowledge_dataset.relation_group_id == sub_relation_group_id
            ].statement_subject

            relation_group_objects = parametric_knowledge_dataset[
                parametric_knowledge_dataset.relation_group_id == relation_group_id
            ].parametric_object
            sub_relation_group_objects = parametric_knowledge_dataset[
                parametric_knowledge_dataset.relation_group_id == sub_relation_group_id
            ].parametric_object

            # For each pair of relations, we count the number of examples of the first relation where EITHER
            # - the subject of the example is contained in any of the subjects of the examples in the dataset that are of the second relation
            # - OR the object of the example is contained in any of the objects of the examples in the dataset that are of the second relation
            relation_groups_entity_overlap_count[relation_group_id][sub_relation_group_id] = np.sum(
                relation_group_subjects.isin(sub_relation_group_subjects)
                | relation_group_objects.isin(sub_relation_group_objects)
            )

    return relation_groups_entity_overlap_count


def remove_entity_overlap_between_relation_groups(parametric_knowledge_dataset: pd.DataFrame) -> pd.DataFrame:
    """this function return the parametric knowledge dataset
    with only examples that don't have entity (subject or object)
    overlap between relation groups

    Args:
        parametric_knowledge_dataset (pd.DataFrame): the parametric knowledge
            dataset

    Returns:
        pd.DataFrame
    """

    parametric_knowledge_dataset = parametric_knowledge_dataset.copy()
    
    for idx in tqdm(parametric_knowledge_dataset.index):
        if (
            parametric_knowledge_dataset.loc[idx].statement_subject
            in parametric_knowledge_dataset[
                parametric_knowledge_dataset.relation_group_id
                != parametric_knowledge_dataset.loc[idx].relation_group_id
            ].statement_subject.tolist()
            or parametric_knowledge_dataset.loc[idx].parametric_object
            in parametric_knowledge_dataset[
                parametric_knowledge_dataset.relation_group_id
                != parametric_knowledge_dataset.loc[idx].relation_group_id
            ].parametric_object.tolist()
        ):
            parametric_knowledge_dataset = parametric_knowledge_dataset.drop(index=[idx])

    return parametric_knowledge_dataset
