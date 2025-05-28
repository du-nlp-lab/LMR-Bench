from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

model_id_list = ['llama-8b', 'llama-70b', 'mistral', 'qwen']
group_option_list = ['ethnicity', 'nationality', 'religion']
prompt_variation_list = ['origin', 'persona-1', 'persona-2',
                         'persona-3', '10-scale', '1-person', '3-person']

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

crowd_enVent_emotions = ['anger', 'disgust', 'fear', 'guilt', 'sadness',
                         'shame', 'boredom', 'joy', 'pride', 'trust', 'relief', 'surprise']


def get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer):
    path_folder = 'processed_results'
    file_path = f'{path_folder}/{group_option}_{prompt_variation}_{model_id}_{persona}_{experiencer}.tsv'
    intensity_list = pd.read_csv(file_path, sep='\t')['intensity']
    assert len(intensity_list) == 6050
    return intensity_list, path_folder


# https://scikit-learn.org/dev/modules/generated/sklearn.manifold.TSNE.html

# ENGLISH-SPEAKING -- the United States, Canada, the United Kingdom
# PROTESTANT EUROPE -- Germany
# CATHOLIC EUROPE -- France
# CONFUCIAN -- China, Japan
# WEST & SOUTH ASIA -- India, Myanmar, Israel
# ORTHODOX EUROPE -- Russia, Ukraine
# LATIN AMERICA -- the Philippines, Argentina, Brazil, Mexico
# AFRICAN-ISLAMIC -- Iran, Palestine, Nigeria, Egypt, Pakistan

group_mapping = {
    'Germany': 'Protestant Europe',
    'the United Kingdom': 'English-Speaking',
    'the United States': 'English-Speaking',
    'Canada': 'English-Speaking',
    'China': 'Confucian',
    'Japan': 'Confucian',
    'France': 'Catholic Europe',
    'Russia': 'Orthodox Europe',
    'Ukraine': 'Orthodox Europe',
    'Israel': 'West & South Asia',
    'Argentina': 'Latin America',
    'Brazil': 'Latin America',
    'Mexico': 'Latin America',
    'the Philippines': 'Latin America',
    'Egypt': 'African-Islamic',
    'Palestine': 'African-Islamic',
    'Nigeria': 'African-Islamic',
    'India': 'West & South Asia',
    'Myanmar': 'West & South Asia',
    'Pakistan': 'African-Islamic',
    'Iran': 'African-Islamic'
}

group_colors = {
    'English-Speaking': '#7d4f81',
    'Protestant Europe': '#be75be',
    'Catholic Europe': '#c7a7d4',
    'Confucian': '#f9b5ac',
    'West & South Asia': '#d0d6b5',
    'Orthodox Europe': '#ee7674',
    'Latin America': '#7eb8bd',
    'African-Islamic': '#408089'
}


def get_excluded_id_set_no_default(group_option, prompt_variation, model_id):
    print(group_option, prompt_variation, model_id)
    group_list = group_mappings[group_option]
    full_group_list = group_list  # we don't consider "a person" case here

    excluded_ids = []
    for _, persona in tqdm(enumerate(full_group_list)):
        for _, experiencer in enumerate(full_group_list):
            intensity_list, _ = get_intensity_list(group_option, prompt_variation, model_id, persona, experiencer)

            excluded_ids += [idx for idx, intensity in enumerate(intensity_list) if intensity == -1]

    excluded_id_set = set(excluded_ids)

    print(f'{len(excluded_id_set)}: {round(len(excluded_id_set) / len(intensity_list) * 100, 2)}%')
    return excluded_id_set


# not the full matrix
def get_filtered_matrix(group_option, prompt_variation, model_id, excluded_ids):
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


# perceiver embeddings
group_option = group_option_list[1]

model_id = model_id_list[0]
prompt_variation = prompt_variation_list[0]

excluded_ids = get_excluded_id_set_no_default(group_option, prompt_variation, model_id)
actual_matrix = get_filtered_matrix(group_option, prompt_variation, model_id, excluded_ids)

data = pd.DataFrame(actual_matrix, index=country_list, columns=country_list)

colors = [group_colors[group_mapping[label]] for label in data.index]

# https://scikit-learn.org/dev/modules/generated/sklearn.manifold.TSNE.html
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(data)

plt.figure(figsize=(8, 5))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=150)
size = 12
for i, label in enumerate(data.index):
    if label in ["the United States", "the United Kingdom", "Argentina", "the Philippines", "France"]:
        plt.annotate(f"{label}  ", (tsne_result[i, 0], tsne_result[i, 1]), ha='right', fontsize=size, color='#434343')
    elif label in ["Ukraine"]:
        plt.annotate(f"{label}", (tsne_result[i, 0], tsne_result[i, 1]), ha='center', fontsize=size, color='#434343')
    else:
        plt.annotate(f"  {label}", (tsne_result[i, 0], tsne_result[i, 1]), ha='left', fontsize=size, color='#434343')

plt.tight_layout()
plt.savefig(f'figures/tsne_{group_option}_{prompt_variation}_{model_id}.pdf')
plt.show()
