import re
import time
import random
import argparse
import pandas as pd
from tqdm import tqdm

country_list = [
    "China", "Japan",
    "India", "Pakistan",
    "Myanmar", "the Philippines",
    "Iran", "Israel", "Palestine",
    "Russia", "Ukraine",
    "Germany", "France", "the United Kingdom",
    "Nigeria", "Egypt",
    "the United States", "Canada", "Mexico",
    "Brazil", "Argentina",
]

group_mappings = {
    "religion": ["a Christian", "a Muslim", "a Jew", "a Buddhist", "a Hindu"],
    "ethnicity": [
        "a white person", "a White person", "a Caucasian", "a White American", "a European American",
        "a black person", "a Black person", "an African American", "a Black American",
        "an Asian person", "an Asian American", "an Asian",
        "a Hispanic person", "a Hispanic American", "a Latino American", "a Latino", "a Latina", "a Latinx"
    ],
    "nationality1": [f"a person from {country}" for country in country_list],
}

def get_experiment_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        choices=['llama-8b', 'llama-70b', 'mistral', 'qwen'],
        default=''
    )
    parser.add_argument(
        "--group_option",
        type=str,
        choices=['ethnicity', 'nationality', 'religion']
    )
    parser.add_argument(
        "--prompt_variation",
        type=str,
        choices=['origin',
                 'persona-1', 'persona-2', 'persona-3',
                 '1-person', '3-person',
                 '10-scale']
    )
    args = parser.parse_args()
    return args


def get_prompt_df(args):
    prompt_path = f'../prompts/{args.group_option}_{args.prompt_variation}.tsv'
    prompt_df = pd.read_csv(prompt_path, sep="\t")
    rand_idx = random.randint(0, len(prompt_df))

    print("==================")
    print(f"Total: {len(prompt_df)}")
    print(rand_idx, prompt_df['persona'][rand_idx], prompt_df['experiencer'][rand_idx], prompt_df['emotion'][rand_idx])
    print(prompt_df['system_prompt'][rand_idx])
    print(prompt_df['user_input'][rand_idx])
    print("==================")

    return prompt_df


def get_pred_df(args):
    tot_batch = -1
    if args.group_option == 'religion':
        tot_batch = 1
    elif args.group_option == 'ethnicity':
        tot_batch = 8
    elif args.group_option == 'nationality':
        tot_batch = 10

    folder = f'../results/exp_{args.model_id}'

    df_list = []
    for bid in tqdm(range(tot_batch)):
        path = f"{folder}/{args.group_option}_{args.prompt_variation}_batch-{bid}.tsv"
        batch_df = pd.read_csv(path, sep="\t")
        df_list.append(batch_df)
    concat_df = pd.concat(df_list, sort=False)
    return concat_df


def extract_numbers(text):
    pattern = r'\b\d+\b'
    matches = re.findall(pattern, text)

    selected_match = -1
    if len(matches) > 1:
        # for mistral responses e.g. 50 (0 to 100)
        selected_match = int(matches[0])
        # for cases where it has the number from event first (TODO: could double-check corner cases, but should be only a few)
        selected_match = int(matches[1]) if not (0 <= selected_match <= 100) else selected_match
    elif len(matches) == 1:
        selected_match = int(matches[0])

    return selected_match if 0 <= selected_match <= 100 else -1


def save_updated_processed_results(prompt_df, pred_df, group_list, args):
    full_group_list = group_list + ['a person']
    group2intensity_list = {}
    for persona in tqdm(full_group_list):
        for experiencer in full_group_list:
            print("==================") 
            print(persona, experiencer)
            print("========") 

            selected_idx = prompt_df[(prompt_df['persona'] == persona) & (prompt_df['experiencer'] == experiencer)]['idx']
            assert len(selected_idx) == 6050
            assert len(pred_df[pred_df['idx'].isin(selected_idx)]) == 6050

            selected_df = pred_df[pred_df['idx'].isin(selected_idx)]

            emotion_list = prompt_df[(prompt_df['persona'] == persona) & (prompt_df['experiencer'] == experiencer)]['emotion']
            system_prompt_list = prompt_df[(prompt_df['persona'] == persona) & (prompt_df['experiencer'] == experiencer)]['system_prompt']
            user_input_list = prompt_df[(prompt_df['persona'] == persona) & (prompt_df['experiencer'] == experiencer)]['user_input']
            intensity_list = []
            response_list = []

            for _, response in enumerate(selected_df['response']):     
                response = str(response)
                # If clear reject cases
                if response.startswith('I cannot') or response.startswith('I can\'t') or response.startswith('I\'m not rating'):
                    intensity_list.append(-1)
                else:
                    intensity = extract_numbers(response)
                    intensity_list.append(intensity)
                response_list.append(response)
            
            group2intensity_list[f"{persona}_{experiencer}"] = intensity_list
            assert len(intensity_list) == 6050

            save_path = f'processed_results/{args.group_option}_{args.prompt_variation}_{args.model_id}_{persona}_{experiencer}.tsv'
            pd.DataFrame({
                'idx': selected_idx,
                'emotion': emotion_list,
                'system_prompt': system_prompt_list,
                'user_input': user_input_list,
                'response': response_list,
                'intensity': intensity_list
            }).to_csv(save_path, sep="\t", index=False)


def main(args):
    start_time = time.time()
    print(f'{args.group_option}_{args.prompt_variation}')
    pred_df = get_pred_df(args)
    prompt_df = get_prompt_df(args)
    assert list(prompt_df['idx']) == list(pred_df['idx'])

    group_list = group_mappings[f'{args.group_option}']
    save_updated_processed_results(prompt_df, pred_df, group_list, args)

    elapsed_time = time.time() - start_time
    print(f"[Used time] {elapsed_time / 60:.4f} minutes")


if __name__ == '__main__':
    main(get_experiment_configs())
