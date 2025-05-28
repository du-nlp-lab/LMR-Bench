import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from torch.nn import functional as F
import sys
import os

sys.path.append(os.path.abspath('..'))


import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'unit_test_1.log'
logging.basicConfig(filename=log_file,
                    filemode="w",
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s")

def context_aware_sampling(model,       # LLM used for generating
                           tokenizer,   # tokenizer for this LLM
                           input_ids,   # tokenized input ids
                           context_ids, # ids for context
                           alpha=0.9,   
                           max_length=128, 
                           temperature=0.00000000000001):
    '''
    Implement the function context_aware_sampling function and pass the test by this .py file, you may ignore the following parameters: alpha, max_length, temperature.

    The input of this function is two hiddenstates, context_ids is the representation of the context and input_ids is the representation of all the question and context.
    '''
    # Following you need to finish the context aware decoding method introduced in the paper
    # This is the only function that you should edit in this file
    raise NotImplementedError("The function 'context_aware_sampling' not implemented")
    return generated_tokens


# test function for agent
# You should not edit the following function
def test_function(context, 
                  question,  # string 
                  answer,    # ground_truth_answer
                  model,     # LLM used for generating
                  tokenizer):
    context_input = tokenizer(context, return_tensors="pt").input_ids.to(device)
    question_input = tokenizer(question, return_tensors="pt").input_ids.to(device)

    input_ids = torch.cat([context_input, question_input], dim=-1)
    model.eval()
    # this function generate the output with context aware decoding
    output_tokens = context_aware_sampling(  
                                            model,
                                            tokenizer,
                                            input_ids,
                                            context_ids=context_input,
                                            alpha=0.5,
                                            max_length=50,
                                            temperature=0.0000000000000000000000000001, 
                                        )

    context_aware_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print('---')
    print(context_aware_output)
    print('---')
    return context_aware_output == answer


if __name__ == '__main__':
    # load model, here we use a tiny model for efficient so the result may not be accurate
    # to pass the test cases you will need to use exactly the same model below
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    test_case_1 = {
        'context' : "The current year is 2027. Argentina won World Cups in 1978,1986,2022 and 2026.",
        'question' : "How many world cups has Argentina won?",        
        'answer' : '''The current year is 2027. Argentina won World Cups in 1978,1986,2022 and 2026. How many world cups has Argentina won?
How many world cups has Argentina won?
Argentina has won 19 World Cups, the most of any country.
Argentina won its first World Cup in 1978, beating West Germany ''' 
    }

    test_case_2 = {
    'context' : "Prison Link Cymru had 1099 referrals in 2015-16 and said some ex-offenders were living rough for up to a year before finding suitable accommodation ......",
    'question' : "Summarize the article in one sentence. ",
    'answer' : '''Prison Link Cymru had 1099 referrals in 2015-16 and said some ex-offenders were living rough for up to a year before finding suitable accommodation ...... Summarize the article in one sentence. ›
The article discusses the challenges faced by ex-offenders in finding suitable accommodation, and the efforts of Prison Link Cymru to address this issue.''' 
}

    test_case_3 = {
    'context' : '''Write a quote that ends in the word "early":''',
    'question' : "Better late than",
    'answer' : '''Write a quote that ends in the word "early": Better late than never.

Vocabulary: late, end, end, end, end, end, end, end, end, end, end, end, end, end, end, end, end, end, end, end,'''
}
    test_cases = [test_case_1, test_case_2, test_case_3]
    result = 0
    for case in test_cases:
        context = case['context']
        qestion = case['question']
        answer = case['answer']
        result += test_function(context, qestion, answer, model, tokenizer)
        if result:
            logging.info("Test Passed")
        else:
            logging.info("Test Failed")
    print(f'''{result}/3 test cases passed''')
