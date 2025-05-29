import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
from scipy import stats

model_id_list = ['llama-8b', 'llama-70b', 'mistral', 'qwen']
group_option_list = ['ethnicity', 'nationality', 'religion']
prompt_variation_list = ['origin', 'persona-1', 'persona-2', 'persona-3', '10-scale', '1-person', '3-person']

country_list = [
    # English-Speaking
    "the United States", "Canada", "the United Kingdom",
    # Protestant Europe
    "Germany",
    # Catholic Europe
    "France",
    # Confucian
    "China", "Japan",
    # West & South Asia
    "India", "Myanmar", "Israel",
    # Orthodox Europe
    "Russia", "Ukraine",
    # Latin America
    "the Philippines", "Argentina", "Brazil", "Mexico",
    # African-Islamic
    "Iran", "Palestine", "Nigeria", "Egypt", "Pakistan",
]

group_mappings = {
    "religion": ["a Christian", "a Muslim", "a Jew", "a Buddhist", "a Hindu"],
    "ethnicity": [
        "a white person", "a White person", "a Caucasian", "a White American", "a European American",
        "a black person", "a Black person", "an African American", "a Black American",
        "an Asian person", "an Asian American", "an Asian",
        "a Hispanic person", "a Hispanic American", "a Latino American", "a Latino", "a Latina", "a Latinx"
    ],
    "nationality": [f"a person from {country}" for country in country_list],
}

crowd_enVent_emotions = ['anger', 'disgust', 'fear', 'guilt', 'sadness', 'shame', 'boredom', 'joy', 'pride', 'trust', 'relief', 'surprise']


def get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer):
    path_folder = 'processed_results'
    file_path = f'{path_folder}/{group_option}_{prompt_variation}_{model_id}_{persona}_{experiencer}.tsv'
    intensity_list = pd.read_csv(file_path, sep='\t')['intensity']
    assert len(intensity_list) == 6050
    return intensity_list, path_folder


def get_excluded_id_set(group_option, prompt_variation, model_id):
    print(group_option, prompt_variation, model_id)
    group_list = group_mappings[group_option]
    full_group_list = group_list + ['a person']
    
    excluded_ids = []
    for _, persona in tqdm(enumerate(full_group_list)):
        for _, experiencer in enumerate(full_group_list):
            intensity_list, _ = get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer)
            excluded_ids += [idx for idx, intensity in enumerate(intensity_list) if intensity == -1]
    
    excluded_id_set = set(excluded_ids)
    
    print(f'{len(excluded_id_set)}: {round(len(excluded_id_set) / len(intensity_list) * 100, 2)}%')
    return excluded_id_set


def get_mask_cells(group_option, prompt_variation, model_id, excluded_ids):
    print(group_option, prompt_variation, model_id)
    group_list = group_mappings[group_option]
    full_group_list = ['a person'] + group_list
    
    matrix_persona = np.zeros((len(full_group_list), len(full_group_list)))
    matrix_experiencer = np.zeros((len(full_group_list), len(full_group_list)))
    
    for i, persona in tqdm(enumerate(full_group_list)):
        for j, experiencer in enumerate(full_group_list):
            if persona == experiencer or persona == 'a person' or experiencer == 'a person':
                continue
            
            intensity_list, path_folder = get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer)
    
            selected_intensity_list = [intensity for idx, intensity in enumerate(intensity_list) if idx not in excluded_ids]
    
            # load in-group intensity (persona)
            file_path = f'{path_folder}/{group_option}_{prompt_variation}_{model_id}_{persona}_{persona}.tsv'
            intensity_list = pd.read_csv(file_path, sep='\t')['intensity']
            in_group_intensity_list = [intensity for idx, intensity in enumerate(intensity_list) if idx not in excluded_ids]
            
            t_test_result = stats.ttest_rel(in_group_intensity_list, selected_intensity_list)
            matrix_persona[i][j] = t_test_result.pvalue

            # load in-group intensity (experiencer)
            file_path = f'{path_folder}/{group_option}_{prompt_variation}_{model_id}_{experiencer}_{experiencer}.tsv'
            intensity_list = pd.read_csv(file_path, sep='\t')['intensity']
            in_group_intensity_list = [intensity for idx, intensity in enumerate(intensity_list) if idx not in excluded_ids]
            
            t_test_result = stats.ttest_rel(in_group_intensity_list, selected_intensity_list)
            matrix_experiencer[i][j] = t_test_result.pvalue
    
    pv = 0.05 / ((len(full_group_list))**2)
    mask = np.invert((matrix_persona < pv) & (matrix_experiencer < pv))

    return mask


def draw_single_heatmap(group_option, prompt_variation, model_id, excluded_ids, mask_matrix, fontsize=32):
    print(group_option, prompt_variation, model_id)
    group_list = group_mappings[group_option]
    full_group_list = ['a person'] + group_list

    matrix = np.zeros((len(full_group_list), len(full_group_list)))
    
    for i, persona in tqdm(enumerate(full_group_list)):
        for j, experiencer in enumerate(full_group_list):
            intensity_list, _ = get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer)
    
            selected_intensity_list = [intensity for idx, intensity in enumerate(intensity_list) if idx not in excluded_ids]
            matrix[i][j] = round(sum(selected_intensity_list) / len(selected_intensity_list), 2)

    mean = np.mean(matrix)
    std = np.std(matrix)
    print(f"{round(mean, 2)}\u00B1{round(std, 2)}")

    matrix = (matrix - mean) / std

    min_limit = np.amin(matrix)
    max_limit = np.amax(matrix)
    # print(min_limit, max_limit)
    print(f'({round(min_limit, 2)}, {round(max_limit, 2)})')
    fmt = ".2f"

    figure(figsize=(22, 20), dpi=80)
    sns.set_style("white")
    ax = sns.heatmap(matrix, 
                     vmin=min_limit,
                     vmax=max_limit,
                     fmt=fmt,
                     mask=mask_matrix,
                     cmap="YlGnBu",
                     cbar=True,
                     linewidths=0.5,
                     yticklabels=["unspecified"] + [g.replace('a person from', '').strip() for g in group_list],
                     xticklabels=["unspecified"] + [g.replace('a person from', '').strip() for g in group_list])

    plt.xticks(rotation=90, fontsize=fontsize)
    if group_option == 'religion':
        plt.yticks(rotation=360, fontsize=fontsize)
    else:
        plt.yticks(fontsize=fontsize)
    plt.xlabel("Experiencer", fontsize=fontsize)
    plt.ylabel("Perceiver", fontsize=fontsize)

    color1 = "white"
    if group_option == 'religion':
        start_p = 1
    else:
        start_p = 1
    ax.hlines(start_p, 0, len(group_list) + 1, color=color1, lw=5, clip_on=False)
    # x, ymin, ymax
    ax.vlines(start_p, 0, len(group_list) + 1, color=color1, lw=5, clip_on=False)
    
    plt.tight_layout()
    plt.savefig(f'figures/{group_option}_{prompt_variation}_{model_id}_main.pdf')
    # plt.show()
    plt.clf()
    plt.close()


model_id = model_id_list[0]
prompt_variation = 'origin'
# group_option = 'religion'
# group_option = 'ethnicity'
group_option = 'nationality'

# for group_option in group_option_list:
excluded_ids = get_excluded_id_set(group_option, prompt_variation, model_id)
mask_matrix = get_mask_cells(group_option, prompt_variation, model_id, excluded_ids)

draw_single_heatmap(group_option, prompt_variation, model_id, excluded_ids, mask_matrix)
