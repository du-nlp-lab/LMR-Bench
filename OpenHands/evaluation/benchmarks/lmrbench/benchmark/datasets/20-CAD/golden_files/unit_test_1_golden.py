import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from torch.nn import functional as F
import sys
import os

sys.path.append(os.path.abspath('..'))

def context_aware_sampling(model, tokenizer, input_ids, context_ids, alpha=0.9, max_length=128, temperature=0.00000000000001):
    generated_tokens = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            full_context_outputs = model(generated_tokens)
            full_context_logits = full_context_outputs.logits[:, -1, :] 

            question_only_input = generated_tokens[:, len(context_ids):]
            question_only_outputs = model(question_only_input)
            question_only_logits = question_only_outputs.logits[:, -1, :] 

        adjusted_logits = (1 + alpha) * full_context_logits - alpha * question_only_logits
        adjusted_probs = F.softmax(adjusted_logits / temperature, dim=-1)

        next_token = torch.multinomial(adjusted_probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

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
    print(f'''{result}/3 test cases passed''')
