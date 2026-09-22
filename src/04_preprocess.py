from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

sample = dataset["train"][0]

mr = sample["meaning_representation"]
target = sample["target"]

# 把原来“两个字段”的监督学习样本，改造成 GPT-2 熟悉的“一整段文本”
text = f"MR: {mr}\nText: {target}"

print("MR:")
print(mr)

print("\nTarget:")
print(target)

print("\nCombined text:")
print(text)

prompt = f"MR: {mr}\nText:"
target_text = f" {target}"

prompt_ids = tokenizer(prompt)["input_ids"]  # ids=Identifiers
target_ids = tokenizer(target_text)["input_ids"]

input_ids = prompt_ids + target_ids

labels = [-100] * len(prompt_ids) + target_ids  # PyTorch 的 CrossEntropyLoss 约定ignore_index = -100