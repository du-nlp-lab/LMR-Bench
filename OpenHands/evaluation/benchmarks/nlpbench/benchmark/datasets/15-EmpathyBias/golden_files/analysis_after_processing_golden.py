#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

model_id_list = ['llama-8b', 'llama-70b', 'mistral', 'qwen']
group_option_list = ['ethnicity', 'nationality', 'religion']
prompt_variation_list = ['origin', 'persona-1', 'persona-2', 'persona-3', '10-scale', '1-person', '3-person']

model_id_list = ['llama-8b']
group_option_list = ['religion']
prompt_variation_list = ['persona-1']

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


# In[2]:


def get_excluded_id_set(group_option, prompt_variation, model_id):
    print(group_option, prompt_variation, model_id)
    group_list = group_mappings[group_option]
    full_group_list = group_list + ['a person']
    
    excluded_ids = []
    for i, persona in tqdm(enumerate(full_group_list)):
        for j, experiencer in enumerate(full_group_list):
            intensity_list, _ = get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer)
            excluded_ids += [idx for idx, intensity in enumerate(intensity_list) if intensity == -1]

            # if len([idx for idx, intensity in enumerate(intensity_list) if intensity == -1]) > 0:
            #     print(file_path)
    
    excluded_id_set = set(excluded_ids)
    
    print(f'{len(excluded_id_set)}: {round(len(excluded_id_set) / len(intensity_list) * 100, 2)}%')
    return excluded_id_set


# ### Refusal Statistics

# In[3]:


for model_id in model_id_list:
    for group_option in group_option_list:
        for prompt_variation in prompt_variation_list:
            get_excluded_id_set(group_option, prompt_variation, model_id)


# In[5]:


# Sanity check
# group_option = 'nationality'
# prompt_variation = 'persona-1'
# model_id = 'qwen'

# get_excluded_id_set(group_option, prompt_variation, model_id)


# ### Summarized Heatmap

# In[6]:


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


# In[7]:


def draw_heatmap(group_option, prompt_variation, model_id, excluded_ids, mask_matrix):
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

    sns.set_style("white")
    ax = sns.heatmap(matrix, 
                     vmin=min_limit,
                     vmax=max_limit,
                     fmt=fmt,
                     mask=mask_matrix,
                     cmap="YlGnBu",
                     cbar=False,
                     linewidths=0.05,
                     yticklabels=[], xticklabels=[])
    ax.set_aspect(1)
    

    color1 = "white"
    # y, xmin, xmax
    if group_option == 'religion':
        start_p = 0.95
    else:
        start_p = 0.8
    ax.hlines(start_p, 0, len(group_list) + 1, color=color1, lw=5, clip_on=False)
    # x, ymin, ymax
    ax.vlines(start_p, 0, len(group_list) + 1, color=color1, lw=5, clip_on=False)
    
    plt.tight_layout()
    plt.savefig(f'figures/{group_option}_{prompt_variation}_{model_id}.png')
    plt.clf()
    plt.close()


# In[13]:


# Generate all heatmaps for a specific model
model_id = model_id_list[0]
for group_option in group_option_list:
    for prompt_variation in prompt_variation_list:
        excluded_ids = get_excluded_id_set(group_option, prompt_variation, model_id)
        mask_matrix = get_mask_cells(group_option, prompt_variation, model_id, excluded_ids)
        draw_heatmap(group_option, prompt_variation, model_id, excluded_ids, mask_matrix)


# ### Gap Delta Statistics with Permutation

# In[9]:


def get_excluded_id_set_no_default(group_option, prompt_variation, model_id):
    print(group_option, prompt_variation, model_id)
    group_list = group_mappings[group_option]
    full_group_list = group_list  # we don't consider "a person" case here
    
    excluded_ids = []
    for i, persona in tqdm(enumerate(full_group_list)):
        for j, experiencer in enumerate(full_group_list):
            intensity_list, _ = get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer)
            
            excluded_ids += [idx for idx, intensity in enumerate(intensity_list) if intensity == -1]
    
    excluded_id_set = set(excluded_ids)
    
    print(f'{len(excluded_id_set)}: {round(len(excluded_id_set) / len(intensity_list) * 100, 2)}%')
    return excluded_id_set


# In[10]:


def get_filtered_matrix(group_option, prompt_variation, model_id, excluded_ids):  # not the full matrix
    print(group_option, prompt_variation, model_id)
    group_list = group_mappings[group_option]
    matrix = np.zeros((len(group_list), len(group_list)))
    
    for i, persona in tqdm(enumerate(group_list)):
        for j, experiencer in enumerate(group_list):
            intensity_list, _ = get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer)
    
            selected_intensity_list = [intensity for idx, intensity in enumerate(intensity_list) if idx not in excluded_ids]
            matrix[i][j] = round(sum(selected_intensity_list) / len(selected_intensity_list), 2)

    mean = np.mean(matrix)
    std = np.std(matrix)
    print(f"{round(mean, 2)}\u00B1{round(std, 2)}")

    matrix = (matrix - mean) / std

    min_limit = np.amin(matrix)
    max_limit = np.amax(matrix)
    print(f'({round(min_limit, 2)}, {round(max_limit, 2)})')

    return matrix


# #### Permutation

# In[11]:


def permute_matrix(matrix):
    row_len = matrix.shape[0]

    row_idx = [n for n in range(row_len)]
    random.shuffle(row_idx)

    col_idx = [n for n in range(row_len)]
    random.shuffle(col_idx)

    new_matrix = np.zeros(matrix.shape)
    for ridx, i in enumerate(row_idx):
        for cidx, j in enumerate(col_idx):
            new_matrix[ridx, cidx] = matrix[i, j]

    return new_matrix

ethnicity_group_list = group_mappings["ethnicity"]

def check_in_group(group_option, i, j):
    if group_option == 'ethnicity':
        ethnicity_map = {
            'a white person': 1,
            'a White person': 1,
            'a Caucasian': 1,
            'a White American': 1,
            'a European American': 1,
            'a black person': 2,
            'a Black person': 2,
            'an African American': 2,
            'a Black American': 2,
            'an Asian person': 3,
            'an Asian American': 3,
            'an Asian': 3,
            'a Hispanic person': 4,
            'a Hispanic American': 4,
            'a Latino American': 4,
            'a Latino': 4,
            'a Latina': 4,
            'a Latinx': 4
        }
        return ethnicity_map[ethnicity_group_list[i]] == ethnicity_map[ethnicity_group_list[j]]
    else:
        return i == j

def get_delta(matrix, group_option):
    tot_len = matrix.shape[0]
    in_group_list = []
    out_group_list = []
    for i in range(tot_len):
        for j in range(tot_len):
            if check_in_group(group_option, i, j):
                in_group_list.append(matrix[i, j])
            else:
                out_group_list.append(matrix[i, j])

    delta = sum(in_group_list)/len(in_group_list) - sum(out_group_list)/len(out_group_list)
    return delta

def get_interval(test_matrix, p_times=10000):
    permuted_delta = []
    for _ in tqdm(range(p_times)):
        new_matrix = permute_matrix(test_matrix)
        new_delta = get_delta(new_matrix, group_option)
        # print(new_delta)
        permuted_delta.append(new_delta)

    sorted_deltas = np.sort(permuted_delta)

    # Determine the 95% confidence interval (2.5th percentile and 97.5th percentile)
    lower_bound = np.percentile(sorted_deltas, 2.5)
    upper_bound = np.percentile(sorted_deltas, 97.5)

    return lower_bound, upper_bound


# In[17]:


# Sanity check
# model_id = model_id_list[-1]
# group_option = group_option_list[-1]
# print(prompt_vatiation_list)
# prompt_variation = prompt_variation_list[-3]

excluded_ids = get_excluded_id_set_no_default(group_option, prompt_variation, model_id)
actual_matrix = get_filtered_matrix(group_option, prompt_variation, model_id, excluded_ids)

print(f'{get_delta(actual_matrix, group_option)}')
print(f'{get_interval(actual_matrix)}')


# In[18]:


# model_id = model_id_list[0]
# group_option = group_option_list[0]
# prompt_variation = prompt_variation_list[0]

for model_id in model_id_list:
    for group_option in group_option_list:
        for prompt_variation in prompt_variation_list:
            excluded_ids = get_excluded_id_set_no_default(group_option, prompt_variation, model_id)
            actual_matrix = get_filtered_matrix(group_option, prompt_variation, model_id, excluded_ids)
            
            print(f'{get_delta(actual_matrix, group_option)}')
            print(f'{get_interval(actual_matrix)}')


# In[ ]:




