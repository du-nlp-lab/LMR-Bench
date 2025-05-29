import os
import json
from tqdm import tqdm
import openai
from conll_tree import spacy_result_to_conll

def write_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f)


def read_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

openai.api_key = '...'


def get_completion_from_messages(messages, model="gpt-35-turbo-1106"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=800,
            top_p=0.00001,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        response_content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        generated_tokens = response.usage.completion_tokens
        return response_content, input_tokens, generated_tokens
    except BaseException as e:
        print("Error:", e)
        return "", 0, 0
    except RuntimeError as e:
        print(f"RuntimeError encountered: {e}")
        return "", 0, 0
    except Exception as e:
        print(f"Unexpected error encountered: {e}")
        return "", 0, 0


def safe_str_concat(*args):
    return "".join([str(arg) if arg is not None else "" for arg in args])


def prompt_for_aspect_information(sentence, aspect, structure):
    system_content = "You are an AI assistant that helps people find information.Please refine your reply and ensure its accuracy. The reply length is limited to 200 words or less."
    new_context = f"Given the sentence '{sentence}', "
    prompt = new_context + f""""{structure}"is the CoNLL-U format for the syntactic dependency relationship of this
    sentence. Each row in the table represents a word in the sentence, and each column represents some specific
    properties of the word, including ID (Word position in the sentence), Starting from 1), text (word itself),
    Lemma (the base form of the Word), Pos (the simple UPOS part of speech tag), Tag (the detailed part of speech
    tag), FEATS (other grammatical features, blank here), HEAD (the header word of the dependency relationship of the
    current word, i.e. the ID of the word that the word depends on), DEPREL (dependency label, describing the
    relationship between the current word and the header word), DEPS (word dependency, empty here), MISC (other
    additional information, left blank here).""" + f"Based on the syntactic dependency information of the sentence,analyze information related to ‘{aspect}’ word in the sentence."
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return message


def label_prompt_for_aspect_information(sentence, aspect, structure, polarity):
    system_content = "You are an AI assistant that helps people find information.Please refine your reply and ensure its accuracy. The reply length is limited to 200 words or less."
    new_context = f"Given the sentence '{sentence}', "
    prompt = new_context + f""""{structure}"is the CoNLL-U format for the syntactic dependency relationship of this
    sentence. Each row in the table represents a word in the sentence, and each column represents some specific
    properties of the word, including ID (Word position in the sentence), Starting from 1), text (word itself),
    Lemma (the base form of the Word), Pos (the simple UPOS part of speech tag), Tag (the detailed part of speech
    tag), FEATS (other grammatical features, blank here), HEAD (the header word of the dependency relationship of the
    current word, i.e. the ID of the word that the word depends on), DEPREL (dependency label, describing the
    relationship between the current word and the header word), DEPS (word dependency, empty here), MISC (other
    additional information, left blank here).""" + f"Based on the syntactic dependency information of the sentence,analyze information related to ‘{aspect}’ word in the sentence. the sentiment polarity towards ‘{aspect}’ is {polarity}."
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return message


def prompt_for_polarity_inferring(sentence, aspect, aspect_info):
    system_content = "You are an AI assistant that helps people find information.Please refine your reply and ensure its accuracy. The reply length is limited to 120 words or less."
    prompt = f"Given the sentence '{sentence}', " + aspect_info + f"Considering the context and information related to ‘{aspect}’, what is the speaker's opinion towards {aspect}?"
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return message


def label_prompt_for_polarity_inferring(sentence, aspect, aspect_info, polarity):
    system_content = "You are an AI assistant that helps people find information.Please refine your reply and ensure its accuracy. The reply length is limited to 120 words or less."
    prompt = safe_str_concat(f"Given the sentence '{sentence}'," ,aspect_info ,f"Considering the context and information related to ‘{aspect}’,what is the speaker's opinion towards {aspect}? the sentiment polarity towards ‘{aspect}’ is {polarity}.")
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return message


def lone_prompt_for_polarity_inferring(sentence, aspect, polarity):
    system_content = "You are an AI assistant that helps people find information.Please refine your reply and ensure its accuracy. The reply length is limited to 120 words or less."
    prompt = f"Given the sentence '{sentence}'," + f"Considering the context and information related to ‘{aspect}’,what is the speaker's opinion towards {aspect}? the sentiment polarity towards ‘{aspect}’ is {polarity}."
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return message


def prompt_for_polarity_label(sentence, aspect, polarity_expr):
    system_content = (
        "You are an sentiment analysis expert.I will provide you with a sentence and a certain aspect mentioned in the sentence. "
        "Please analyze the sentiment polarity of that aspect in a given sentence,"
        "Output:\n'''\nThe sentiment towards {{Aspect}} in the given sentence is {{positive, negative or neutral}}.Because\n")
    prompt = f"Given the sentence '{sentence}', " + polarity_expr + f"Based on the common sense and such speaker's opinion, what is the sentiment polarity towards ‘{aspect}’?"
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return message


def label_prompt_for_polarity_label(sentence, aspect, polarity_expr, polarity):
    system_content = (
        "You are an sentiment analysis expert.I will provide you with a sentence and a certain aspect mentioned in the sentence. "
        "Please analyze the sentiment polarity of that aspect in a given sentence,"
        "Output:\n'''\nThe sentiment towards {{Aspect}} in the given sentence is {{positive, negative or neutral}}.Because\n")
    prompt = safe_str_concat(f"Given the sentence '{sentence}', " ,polarity_expr ,f"Based on the common sense and such speaker's opinion, what is the sentiment polarity towards ‘{aspect}’?the sentiment polarity towards ‘{aspect}’ is {polarity}.")
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return message


def get_completion(text):
    sentence = text['sentence']
    aspect = text['aspect']
    label = text['label']
    # aspect_info = text['aspect_info']
    # structure = text['structure']

    structure = spacy_result_to_conll(sentence)

    step_1_message = prompt_for_aspect_information(sentence, aspect, structure)
    aspect_info = get_completion_from_messages(step_1_message)[0]

    # step_2_message = label_prompt_for_polarity_inferring(sentence, aspect, aspect_info,label)
    # step_2_message = lone_prompt_for_polarity_inferring(sentence, aspect,label)
    step_2_message = prompt_for_polarity_inferring(sentence, aspect, aspect_info)
    polarity_expr = get_completion_from_messages(step_2_message)[0]

    step_3_message = prompt_for_polarity_label(sentence, aspect, polarity_expr)
    # step_3_message = label_prompt_for_polarity_label(sentence, aspect, polarity_expr, label)
    output_lb = get_completion_from_messages(step_3_message)[0]

    output_entry = {
        'sentence': sentence,
        'aspect': aspect,
        'label': label,
        'structure': structure,
        'aspect_info': aspect_info,
        'opinion': polarity_expr,
        'output_lb':output_lb,
        'output': safe_str_concat("1---", aspect_info, "2---", polarity_expr,"3---", output_lb),
        'match': "null"
    }
    # print(output_entry)
    return output_entry


def process_text_data(data, save_dir):
    batch_size = 100  
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        output_data = [] 
        for text in tqdm(batch_data, total=len(batch_data), desc=f"Processing data {i + 1}-{i + 1 + batch_size}:"):
            output_entry = get_completion(text)
            output_data.append(output_entry)

        temp_filename = f'temp_data_file_{i // batch_size}.json'
        temp_filepath = os.path.join(save_dir, temp_filename)
        write_json(temp_filepath, output_data)

    final_data = []
    for i in tqdm(range(0, len(data), batch_size), desc="Merging data"):
        temp_filename = f'temp_data_file_{i // batch_size}.json'
        temp_filepath = os.path.join(save_dir, temp_filename)
        batch_data = read_json(temp_filepath)

        final_data.extend(batch_data)

    final_filepath = os.path.join(save_dir, "gpt35_res.json")
    write_json(final_filepath, final_data)

    for i in tqdm(range(0, len(data), batch_size), desc="Cleaning up"):
        temp_filename = f'temp_data_file_{i // batch_size}.json'
        temp_filepath = os.path.join(save_dir, temp_filename)
        os.remove(temp_filepath)


if __name__ == '__main__':
    data_path = '...'
    save_dir = '...'
    data = read_json(data_path)
    process_text_data(data, save_dir)
