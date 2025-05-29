import os
import json
import pickle
import time
from openai import OpenAI
import pandas as pd


def generate_encrypted_features(df, api_key, model_id, log_dir='logs'):
    """
    Generate encrypted representations for each feature level in the given DataFrame using the specified OpenAI model.

    Args:
        df (pd.DataFrame): The DataFrame containing the feature levels to be encrypted.
        api_key (str): OpenAI API key for accessing the model.
        model_id (str): The ID of the OpenAI model to use for encryption.
        log_dir (str, optional): Directory to save input and output logs. Defaults to 'logs'.

    Returns:
        dict: A dictionary mapping each feature level to its encrypted representation.
    """
    client = OpenAI(api_key=api_key)
    feature_level_dict = {}
    os.makedirs(log_dir, exist_ok=True)

    # Save inputs as a pickle file
    input_filename = os.path.join(log_dir, 'generate_encrypted_features_input.pkl')
    with open(input_filename, 'wb') as f:
        pickle.dump(df, f)

    for column in df.columns[:-1]:  # Exclude the target column (e.g., income)
        unique_values = df[column].unique()
        for value in unique_values:
            feature_level_to_convert = f"Feature: {column}, Value: {value}"
            print(feature_level_to_convert)

            # Generate the compression prompt
            compression_prompt = (
                f"Analyze the feature and value: {feature_level_to_convert}. "
                "Create a sequence using NON-natural language elements to depict the feature-value pair provided above. "
                "Key guidelines: "
                "- Avoid directly mentioning the feature or its value. Instead, use symbolic representation to represent the feature value precisely! "
                "- Utilize a mix of abbreviated characters, emojis, emoticons, and logical/math operators (e.g., '->', '+', '<='). "
                "- Aim for clarity and simplicity, ensuring the sequence is interpretable by advanced LLMs. Example: if feature and value = 'Feature: age, Value: 39', then the LLM need to be able to recognize it after sequence conversion. "
                "- Present your sequence as a single line, optimizing for diversity in symbols while minimizing token usage. "
                "Example (for illustration ONLY): For 'Feature: gender, Value: female', you might write: 👩. "
                "Important: Avoid directly mentioning the feature or its value in the output sequence!"
            )

            # Generate encrypted representation
            try:
                completion = client.chat.completions.create(
                    model=model_id, temperature=0, max_tokens=10,
                    messages=[{"role": "system", "content": 'Please come up with a NON-natural language sequence for the narrative provided'},
                              {"role": "user", "content": compression_prompt}],
                    timeout=1200
                )

                compressed_description = completion.choices[0].message.content.strip()
                print(compressed_description)

                # Log input and output
                feature_level_dict[feature_level_to_convert] = compressed_description

            except Exception as e:
                print(f"Error generating compressed description for {feature_level_to_convert}: {e}")
                feature_level_dict[feature_level_to_convert] = "ERROR"

    # Save outputs as a pickle file
    output_filename = os.path.join(log_dir, 'generate_encrypted_features_output.pkl')
    with open(output_filename, 'wb') as f:
        pickle.dump(feature_level_dict, f)

    return feature_level_dict


# Example usage
if __name__ == "__main__":
    # Load the census data
    census_df = pd.read_csv('census_2.csv', header=None)
    census_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                         'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                         'capital-loss', 'hours-per-week', 'native-country', 'income']
    census_df.drop(columns=['fnlwgt', 'education-num'], inplace=True)
    census_df['income'] = census_df['income'].replace(' <=50K', 'no')
    census_df['income'] = census_df['income'].replace(' >50K', 'yes')

    # Generate encrypted features
    API_KEY = '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'
    MODEL_ID = 'gpt-4.1-mini-2025-04-14'

    encrypted_features = generate_encrypted_features(census_df, API_KEY, MODEL_ID)
    print(encrypted_features)