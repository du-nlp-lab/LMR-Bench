import pandas as pd
from tqdm import tqdm
import re
import asyncio

import httpx
import more_itertools
from openai import OpenAI

from src.utils import generate_answer_with_retries

PRE_PROMPT_FOR_STATEMENT_GENERATION = """Below you are given a subject, a relation, and an object. Output a statement in the following form 'subject relation object'. It is very important to respect the given order, the subject should always come before the relation, you should never start with the relation even though it does not sound natural.
Also output the used forms of subject, relation, and object exactly as they appear in the statement, even if you use "the" or any other token such as the possessive "'s" in the statement then do not forget to include it.

Here are some examples:

<FEW_SHOT_LEARNING_EXAMPLES>
"""

OPENAI_API_KEY = "YOUR_API_KEY_HERE"
OPENAI_API_URL = "YOUR_API_URL_HERE"

def is_statement_form_correct(
    prompt_subject: str,
    prompt_object: str,
    statement: str,
    statement_subject: str,
    statement_relation: str,
    statement_object: str,
) -> tuple:
    """this function checks if the generated statement is in the correct form

    Args:
        prompt_subject (str): the given subject in the prompt
        prompt_object (str): the given object in the prompt
        statement (str): the returned statement by the LLM
        statement_subject (str): the returned subject statement
        statement_relation (str): the returned relation statement
        statement_object (str): the returned object statement

    Returns:
        tuple: the first element is a boolean indicating if the form of
        the statement is correct and the second element is a string
        giving details about the form.
    """
    # 1. check si y a bien les bons statement_subject, statement_relation et statement_object
    if (
        (statement != f"{statement_subject} {statement_relation} {statement_object}.")
        and (statement != f"The {statement_subject} {statement_relation} {statement_object}.")
        and (statement != f"{statement_subject}'s {statement_relation} {statement_object}.")
        and (statement != f"{statement_subject}, {statement_relation} {statement_object}.")
    ):
        return (
            False,
            "The returned attributes (subject, relation, and object) do not allow to reconstruct the statement.",
        )

    # 2. check si le vrai objet et sujet donnés dans le prompt sont dans le statement
    if prompt_subject not in statement or prompt_object not in statement:
        return False, "The returned subject and/or object is not in the statement."

    # la relation ne peut pas etre vérifiée car ce n'ai pas la meme forme

    return True, "The statement's form seems correct."


def adjust_statement_subject_form(
    statement: str, statement_subject: str, statement_relation: str, statement_object: str
) -> str:
    """this function corrects the statement' subject form that is
    returned by Mistral-large by making
    sure that it matches the statement either by adding "The" in
    the beginning or "'s" "," in the end

    Args:
        statement (str): the returned statement by the LLM
        statement_subject (str): the returned subject statement
        statement_relation (str): the returned relation statement
        statement_object (str): the returned object statement

    Returns:
        str: the correct form of statement_subject
    """

    if statement == f"{statement_subject} {statement_relation} {statement_object}.":
        return statement_subject
    elif statement == f"The {statement_subject} {statement_relation} {statement_object}.":
        statement_subject = f"The {statement_subject}"
    elif statement == f"{statement_subject}'s {statement_relation} {statement_object}.":
        statement_subject = f"{statement_subject}'s"
    elif statement == f"{statement_subject}, {statement_relation} {statement_object}.":
        statement_subject = f"{statement_subject},"

    return statement_subject


def parse_statement_output(output_txt: str) -> dict:
    """this function parses the output of the LLM that generates statements

    Args:
        output_txt (str): the output of the LLM (Mistral-large is used a by default)

    Returns:
        dict: a dict that contains the following elements:
            - statement: the full statement as it was returned by the LLM
            - statement_subject: the subject that is used in the statement (it can be slightly different
                                 from the original form, e.g. United States of America -> The United States of
                                 America, see `PRE_PROMPT_FOR_STATEMENT_GENERATION`)
            - statement_relation: the relation that is used in the statement (it is different from the original form
                                  which is a lemma, e.g. `is-headquarter` -> `is headquartered in`)
            - statement_object: the object that is used in the statement (it can be slightly different
                                 from the original form)
    """
    pattern = r"{{\s*(.*?)\s*}}"

    matches = re.findall(pattern, output_txt)

    if len(matches) > 0:
        parametric_knowledge_triplet = {
            "statement": output_txt.split("\n")[0],
            "statement_subject": matches[0],
            "statement_relation": matches[1],
            "statement_object": matches[2],
        }
    else:
        parametric_knowledge_triplet = None

    return parametric_knowledge_triplet


async def generate_statements(
    pararel_dataset: pd.DataFrame,
    relations_with_gen_desc: pd.DataFrame,
    requests_batch_size: int = 100,
    pre_prompt_for_statement_generation: str = PRE_PROMPT_FOR_STATEMENT_GENERATION,
    model_name="Mistral-large",
) -> pd.DataFrame:
    """this function generate statements in a form that is specified by the instructions in `pre_prompt_for_statement_generation`.
       More specifically, it was created to transform statements where the subject comes after the relation (e.g. The
       official language of France is French) to statements where the subject comes before the relation (e.g. France's
       official language is French), but this can be applied to any desired form of statement by changing the instructions
       in `pre_prompt_for_statement_generation`. The default LLM for generating statements is Mistral-Large because it is open-weight,
       which allows reproducibility, and has capabilities that are very close to GPT-4.

    Args:
        pararel_dataset (pd.DataFrame): the raw pararel dataset
        pre_prompt_for_statement_generation (str, optional): the pre-prompt with the instructions about the form of the statement to
                                                  generate. Defaults to PRE_PROMPT_FOR_STATEMENT_GENERATION.
        model_name (str, optional): the name of the LLM to use to generate the statements. It is recommended to use a very
                                    good instructed LLM to have high quality statements. Defaults to "Mistral-large".
        relations_with_gen_desc (pd.DataFrame): a dataframe containing all the relation with their corresponding generation description

    Returns:
        pd.DataFrame: the pararel dataset with four additional columns:
            - statement: the generated statement.
            - statement_object: the object as it appears exactly in the statement.
            - statement_subject: the subject as it appears exactly in the statement.
            - statement_relation: the relation as it appears exactly in the statement.
    """

    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_URL + "/v1",
        http_client=httpx.AsyncClient(timeout=60, verify=False),
    )

    tasks = []
    for _, row in tqdm(pararel_dataset.iterrows(), desc="Preparing all the queries."):
        system_prompt = pre_prompt_for_statement_generation.replace(
            "<FEW_SHOT_LEARNING_EXAMPLES>",
            f"""subject: {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].raw_subject}
relation: {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].relation}
object: {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].raw_object}
statement: {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].generation_statement_1} {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].generation_object_1}.

statement-subject: {{{{ {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].generation_subject_1} }}}}
statement-relation: {{{{ {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].generation_relation_1} }}}}
statement-object: {{{{ {relations_with_gen_desc[relations_with_gen_desc.relation == row.rel_lemma].iloc[0].generation_object_1} }}}}

""",
        )

        prompt_txt = f"subject: {row.subject}\nrelation: {row.rel_lemma}\nobject: {row.object}\nstatement: "
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_txt}]

        tasks.append(client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=500,
        ))

    answers = []
    if len(tasks) <= requests_batch_size:
        answers = [await asyncio.gather(*tasks, return_exceptions=True)]
    else:
        for batch in tqdm(
            more_itertools.chunked(tasks, n=requests_batch_size),
            desc=f"Requesting batches, each with a size of {requests_batch_size} requests",
        ):
            answers += await asyncio.gather(*batch, return_exceptions=True)

    statements = []
    statement_subjects = []
    statement_objects = []
    statement_relations = []
    for answer in tqdm(answers, desc=f"Processing the batches answers, each with {requests_batch_size} answers"):
        parametric_knowledge_triplet = None
        if answer is not None:
            # parse
            parametric_knowledge_triplet = parse_statement_output(answer)

            if parametric_knowledge_triplet is not None:
                statements.append(parametric_knowledge_triplet["statement"])
                statement_subjects.append(parametric_knowledge_triplet["statement_subject"])
                statement_objects.append(parametric_knowledge_triplet["statement_object"])
                statement_relations.append(parametric_knowledge_triplet["statement_relation"])
                answer = None
            else:
                statements.append("wrong format")
                statement_subjects.append("wrong format")
                statement_objects.append("wrong format")
                statement_relations.append("wrong format")
        else:
            statements.append(None)
            statement_subjects.append(None)
            statement_objects.append(None)
            statement_relations.append(None)

    pararel_dataset["statement"] = statements
    pararel_dataset["statement_subject"] = statement_subjects
    pararel_dataset["statement_object"] = statement_objects
    pararel_dataset["statement_relation"] = statement_relations

    return "succeeded", pararel_dataset
