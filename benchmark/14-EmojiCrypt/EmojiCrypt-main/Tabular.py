# ## Data Processing:

# In[2]:


import pandas as pd
import random
from openai import OpenAI
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# # census_df = pd.read_csv('census_small.csv', header = None)
# census_df = pd.read_csv('census_10.csv', header = None)

# # Define the column names
# column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
#                 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
#                 'capital-loss', 'hours-per-week', 'native-country', 'income']

# # Assign the column names to the DataFrame
# census_df.columns = column_names
# census_df.fillna('Unknown')

# print( census_df['income'].unique() )
# census_df['income'] = census_df['income'].replace(' <=50K', 'no')
# census_df['income'] = census_df['income'].replace(' >50K', 'yes')
# census_df.drop(columns=['fnlwgt', 'education-num'], inplace=True)
# print( census_df['income'].unique() )


# # ## Reusable Encryption (on feature levels)

# # In[3]:


# API_KEY = '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'
# client = OpenAI(api_key = API_KEY)
# model_id = 'gpt-4.1-mini-2025-04-14'

# feature_level_dict = {}
# for column in census_df.columns[:-1]:  # Exclude the discretized column for this part
#     unique_values = census_df[column].unique()
#     for value in unique_values:
#         feature_level_to_convert = (f"Feature: {column}, Value: {value}")
#         print(feature_level_to_convert)

#         compression_prompt = (
#             f"Analyze the feature and value: {feature_level_to_convert}. "
#             "Create a sequence using NON-natural language elements to depict the feature-value pair provided above. "
#             "Key guidelines: "
#             "- Avoid directly mentioning the feature or its value. Instead, use symbolic representation to represent the feature value precisely! "
#             "- Utilize a mix of abbreviated characters, emojis, emoticons, and logical/math operators (e.g., '->', '+', '<='). "
#             "- Aim for clarity and simplicity, ensuring the sequence is interpretable by advanced LLMs. Example: if feature and value = 'Feature: age, Value: 39', then the LLM need to be able to recognize it after sequence conversion."
#             "- Present your sequence as a single line, optimizing for diversity in symbols while minimizing token usage. "
#             "Example (for illustration ONLY): For 'Feature: gender, Value: female', you might write: 👩. "
#             "Important: Avoid directly mentioning the feature or its value in the output sequence!"
#         )

#         completion = client.chat.completions.create(
#             model = model_id, temperature = 0, max_tokens = 10,

#             messages=[{"role": "system", "content": 'Please come up with a NON-natural language sequence for the narrative provided'},
#                         {"role": "user", "content": compression_prompt}],
#             timeout = 1200)

#         compressed_description = completion.choices[0].message.content
#         print(compressed_description)
#         print()

#         feature_level_dict[feature_level_to_convert] = compressed_description


# # ## Performance Evaluation

# # In[4]:


# system_msg = "Please serve as a binary classifier on user income level."

# right_count_pos = 0
# right_count_neg = 0
# compressed_right_count_pos = 0
# compressed_right_count_neg = 0
# total_pos = 0
# total_neg = 0
# total = 0

# # you may also set range smaller for a subset
# for i in range( len(census_df) ):
#     row_series = census_df.iloc[i][:-1]
#     target = census_df.iloc[i]['income']

#     overall_feature_levels = ''
#     converted_feature_levels = ''
#     for feature, value in row_series.items():
#         feature_level = (f"Feature: {feature}, Value: {value}")

#         overall_feature_levels += feature_level + '; '
#         converted_feature_levels += (f"Feature: {feature}, Value: {feature_level_dict[feature_level]}") + '; '

#     print(overall_feature_levels)
#     print(converted_feature_levels)
#     print()

#     original_prompt = (
#         f"Given the following user description: \n\n {overall_feature_levels}; \n\n"
#         f"Please determine whether the user had an income higher than $50k in the year of 1993. Do NOT explain anything, just output 'yes' or 'no', in lower case:"
#     )

#     completion = client.chat.completions.create(
#             model = model_id, temperature = 0,
#             messages=[{"role": "system", "content": system_msg},
#                         {"role": "user", "content": original_prompt }],
#             timeout = 1200)

#     original_pred = completion.choices[0].message.content

#     total += 1
#     if target == 'yes':
#         total_pos += 1
#     else:
#         total_neg += 1

#     if target == original_pred:
#         if (target == 'yes'):
#             right_count_pos += 1
#         else:
#             right_count_neg += 1

#     census_prompt = (
#         "You are given an user description with each feature value presented as a compressed NON-natural language sequence, "
#         "utilizing a mix of abstract & abbreviated characters, emojis, emoticons, as well as math & logical operators. "
#         "This sequence encapsulates the essential information of the user's original feature value. "
#         "Please determine whether the user had an income higher than $50k in the year of 1993. Do NOT explain anything, just output 'yes' or 'no', in lower case, "
#         "Below is the compressed user description: \n\n"
#         f"{converted_feature_levels}"
#     )

#     compressed_completion = client.chat.completions.create(
#             model = model_id, temperature = 0,
#             messages=[{"role": "system", "content": system_msg},
#                         {"role": "user", "content": census_prompt}],
#             timeout = 1200)

#     compressed_pred = compressed_completion.choices[0].message.content

#     if target == compressed_pred:
#         if (target == 'yes'):
#             compressed_right_count_pos += 1
#         else:
#             compressed_right_count_neg += 1

#     if total % 10 == 0 or total == census_df.shape[0]:
#         print(f"Accuracy: { (right_count_pos/total_pos) * 0.5 + (right_count_neg/total_neg) * 0.5 }")
#         print(f"Compressed Accuracy: { (compressed_right_count_pos/total_pos) * 0.5 + (compressed_right_count_neg/total_neg) * 0.5 }")
#         print()


# # ## Decryption Robustness Test

# # In[5]:


# # perform feature value decryptions
# enc_dec_pairs = {}
# for k, v in feature_level_dict.items():
#     feature = k.split(', V')[0]
#     print(feature, v)

#     decryption_prompt = f"Given the {feature}, and the non-natural language sequence: '{v}' that represents a value of the {feature} feature given, please try to decode the value. Return the decoded value only, in lower case and on a new line."
#     print(decryption_prompt)
#     completion = client.chat.completions.create(
#             model = model_id, temperature = 0,

#             messages=[{"role": "system", "content": 'Please serve as a feature value decrypter for the encrypted value given'},
#                         {"role": "user", "content": decryption_prompt}],
#             timeout = 1200)

#     decryption = completion.choices[0].message.content

#     enc_dec_pairs[k] = feature + ", Value: " + decryption
#     print(k, ' ', enc_dec_pairs[k])
#     print()


# # # perform cos sim computations
# # overall_similarity_score = 0
# # for k, v in enc_dec_pairs.items():
# #     original_level = k.split(', Value: ')[1]
# #     decryption_level = v.split(', Value: ')[1]

# #     response = client.embeddings.create(
# #         input=original_level,
# #         model="text-embedding-3-small",
# #         dimensions = 100,
# #     )
# #     original_level_embedding = np.array(response.data[0].embedding)
# #     original_level_embedding = original_level_embedding.reshape(1, -1)

# #     # You can reduce the dimensions of the embedding by passing in the dimensions parameter without
# #     # the embedding losing its concept-representing properties: set to 100 to mitigate curse of dimensionality
# #     response = client.embeddings.create(
# #         input=decryption_level,
# #         model="text-embedding-3-small",
# #         dimensions = 100,
# #     )
# #     decryption_level_embedding = np.array(response.data[0].embedding)
# #     decryption_level_embedding = decryption_level_embedding.reshape(1, -1)

# #     similarity_score = cosine_similarity(original_level_embedding, decryption_level_embedding)
# #     overall_similarity_score += similarity_score

# # print()
# # print('Mean cosine sim: ', overall_similarity_score / len(enc_dec_pairs))


# def compute_mean_cosine_similarity(enc_dec_pairs, output_file="cosine_sim_results.pkl"):
#     overall_similarity_score = 0
#     similarity_inputs = []
    
#     for k, v in enc_dec_pairs.items():
#         original_level = k.split(', Value: ')[1]
#         decryption_level = v.split(', Value: ')[1]
        
#         # Prepare embeddings for original and decrypted levels
#         original_level_embedding = np.array([[np.random.rand(100)]]).reshape(1, -1)  # Mock embedding
#         decryption_level_embedding = np.array([[np.random.rand(100)]]).reshape(1, -1)  # Mock embedding
        
#         # Calculate cosine similarity
#         similarity_score = cosine_similarity(original_level_embedding, decryption_level_embedding)
#         overall_similarity_score += similarity_score
#         similarity_inputs.append((original_level, decryption_level, similarity_score))

#     # Calculate mean cosine similarity
#     mean_cosine_sim = overall_similarity_score / len(enc_dec_pairs)
    
#     # Save inputs and final score to PKL file
#     with open(output_file, "wb") as f:
#         pickle.dump({"inputs": similarity_inputs, "mean_cosine_sim": mean_cosine_sim}, f)
    
#     print(f"Mean cosine similarity: {mean_cosine_sim}")
#     return mean_cosine_sim

import pickle
def compute_mean_cosine_similarity(enc_dec_pairs, output_file="cosine_sim_results.pkl", pairs_file="enc_dec_pairs.pkl"):

    return mean_cosine_sim


# mean_cosine_sim = compute_mean_cosine_similarity(enc_dec_pairs)

