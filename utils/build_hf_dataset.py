import os
import json
import pandas as pd
from datasets import Dataset


def build_dataset(root_directory = "benchmark/datasets"):


    data_rows = []

    KEYWORDS = ["AdaLora", "4-CD", "StepLength", "DecBound", "EmojiCrypt", "EmpathyBias", "KnowProb", "MassActiv", "Native-Sparse-Attention", "TokenDPO", "SimCLR", "bertscore", "CAD", "DoLa", "DPO", "ERGO", "GDPZero", "HITs", "moverscore", "RepE", "SimPO", "SPIN", "Syn-Chain"]

    # Loop through each entry in the root directory.
    for subfolder in os.listdir(root_directory):
        subfolder_path = os.path.join(root_directory, subfolder)
        # print("subfolder_path:", subfolder_path)
        if os.path.isdir(subfolder_path) and any(kw in subfolder_path for kw in KEYWORDS):
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

    #     # 找出 implementations 不是 list 的行
    # mask = ~df['implementations'].apply(lambda x: isinstance(x, list))

    # # 查看这些行的所有内容
    # bad_rows = df[mask]
    # print("以下行的 implementations 不是 list：")
    # print(bad_rows)

    from collections import Counter

    # # 1. 列出 implementations 中所有值的 Python 类型分布
    # type_counts = Counter(df['implementations'].map(lambda x: type(x).__name__))
    # print("implementations 列的类型分布：", type_counts)

    #     # 2. 针对所有“非 list”类型的行做一次筛查并打印
    # bad_types = {typ for typ, cnt in type_counts.items() if typ != 'list'}
    # mask = df['implementations'].map(lambda x: type(x).__name__ in bad_types)
    # print("以下行的 implementations 不是 list：")
    # print(df[mask][['implementations']])

    hf_dataset = Dataset.from_pandas(df)


    # Print the resulting Hugging Face dataset
    print(hf_dataset)

    hf_dataset.push_to_hub("Shinyy/NLPBench")


if __name__ == "__main__":
    build_dataset()