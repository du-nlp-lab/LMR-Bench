import os
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
target_dir = os.path.join(
    os.path.dirname(__file__),
    ".cache",
    "bert-base-uncased"
)

# 如果目录不存在就创建
os.makedirs(target_dir, exist_ok=True)

# 下载并保存模型和 tokenizer 到目标目录
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(target_dir)
tokenizer.save_pretrained(target_dir)

print(f"✅ 模型和 tokenizer 已保存到: {target_dir}")
