import os
import json
import pandas as pd
from datasets import Dataset


def build_dataset(root_directory = "benchmark/datasets"):


    data_rows = []

    # Loop through each entry in the root directory.
    for subfolder in os.listdir(root_directory):
        subfolder_path = os.path.join(root_directory, subfolder)
        # print("subfolder_path:", subfolder_path)
        if os.path.isdir(subfolder_path):
            print("subfolder_path:", subfolder_path)
        # if os.path.isdir(subfolder_path) and ("SimPO" in subfolder_path):
            # Build the full path to the info.json file in the subfolder
            info_json_path = os.path.join(subfolder_path, "info.json")
            if os.path.exists(info_json_path):
                # Open and load the JSON content
                with open(info_json_path, "r", encoding="utf-8") as json_file:
                    info_dict = json.load(json_file)
                data_rows.append(info_dict)


    df = pd.DataFrame(data_rows)
    print("df:", df.head())
    print("df columns:", df.columns)


    from collections import Counter


    hf_dataset = Dataset.from_pandas(df)


    # Print the resulting Hugging Face dataset
    print(hf_dataset)

    hf_dataset.push_to_hub("Shinyy/LMR-Bench")


if __name__ == "__main__":
    build_dataset()