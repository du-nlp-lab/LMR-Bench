import os
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

def truncate(prompt: str, max_num_tokens: int, side: str, tokenizer) -> str:
    """Truncate prompt from side given the token budget"""

    tokens = tokenizer.tokenize(prompt)
    num_tokens = len(tokens)

    if num_tokens > max_num_tokens:
        if side == 'left':
            prompt_tokens = tokens[num_tokens - max_num_tokens:]
        elif side == 'right':
            prompt_tokens = tokens[:max_num_tokens]
        prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
        new_len = len(tokenizer.tokenize(prompt))
        if new_len > max_num_tokens:
            logger.warning(
                f'Number of tokens after truncation is greater than max tokens allowed: {new_len=} {num_tokens=}')
    return prompt


def construct_prompt(data, tokenizer=None, prompt_budget=64000, retrieval_budget=32000):
    implementations = data['implementations']
    paper_name = data['paper_name']
    repo_path = data['repo_path']
    for implementation in implementations:
        instruction = implementation['instruction']
        file_to_be_implemented = implementation['file_to_be_implemented']
        file_content = read_file(file_to_be_implemented)
        file_content = "#" + path + "\n" + file_content
        retrieval_context = implementation['retrieval_context']
        if len(retrieval_context) > 0:
            retrieval_code = ""
            for retrieval_elements in retrieval_context:
                identifier = retrieval_elements['identifier']
                path = retrieval_elements['path']
                snippet = "#" + path + "\n" + retrieval_elements['snippet'] +"\n"
                retrieval_code += snippet
            retrieval_code = truncate(retrieval_code, retrieval_budget, 'right', tokenizer)

            prompt = f"""Assume you are a Natural Language Processing(NLP) research scientist.
            You are now provided a NLP paper and trying to implement an idea mentioned in the paper. You are also given an unfinished repository and corresponding instruction to the implementation you need to do.
            Follow the instruction and finish the implementation based on the provided paper and code. You need to make sure your implementation can be correctly integrated into the repository and consider possible situations such as device management and edge cases in the NLP problem coding.
            If you think the given information is not enough and need to know more information to make sure you can implement the idea correctly, tell me where the problem is and what you need.
            The output should be the complete file you have finished.
            Instruction:
            {instruction}
            The file you need to implement:
            {file_content}
            Other related files you may need during the implementation:
            {retrieval_code}
            """
        else:

            prompt = f"""Assume you are a Natural Language Processing(NLP) research scientist.
            You are now provided a NLP paper and trying to implement an idea mentioned in the paper. You are also given an unfinished repository and corresponding instruction to the implementation you need to do.
            Follow the instruction and finish the implementation based on the provided paper and code. You need to make sure your implementation can be correctly integrated into the repository and consider possible situations such as device management and edge cases in the NLP problem coding.
            If you think the given information is not enough and need to know more information to make sure you can implement the idea correctly, tell me where the problem is and what you need.
            The output should be the complete file you have finished.
            Instruction:
            {instruction}
            The file you need to implement:
            {file_content}
            """

        prompt = truncate(prompt, prompt_budget, 'right', tokenizer)

        return prompt




        

