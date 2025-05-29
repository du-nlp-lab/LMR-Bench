import pandas as pd

# 读取原始 CSV 文件
df = pd.read_csv("pararel_ori.csv")
dp = pd.read_csv("raw_pararel_with_correctly_reformatted_statements_ori.csv")

# 保留前 1000 行
df_first_100 = df.head(30)
dp_first_100 = dp.head(30)

# 保存到新的 CSV 文件中
df_first_100.to_csv("pararel.csv", index=False)
dp_first_100.to_csv("raw_pararel_with_correctly_reformatted_statements.csv")