from datasets import load_dataset
from transformers import AutoTokenizer

# 加载模型
model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)

# GPT-2 默认没有 pad_token,使用 eos_token 来充当 pad_token
tokenizer.pad_token = tokenizer.eos_token  # eos_token = end of sequence token


def preprocess(example):
    mr = example["meaning_representation"]
    target = example["target"]

    prompt = f"MR: {mr}\nText:"
    target_text = f" {target}"

    prompt_ids = tokenizer(prompt)["input_ids"]
    target_ids = tokenizer(target_text)["input_ids"]

    input_ids = prompt_ids + target_ids
    # Labels 标签决定是否参与 loss , mask 标签决定是否与其他 token 相关
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


"""
map：对各 split 的每条样本执行 preprocess
preprocess 返回 dict，dict 的 key 会成为新列名
remove_columns 用 train 的 column_names 取得原始列名，并在转换后删除旧列
最终只保留 input_ids / attention_mask / labels
"""
tokenized_dataset = dataset.map(  # dataset.map() 默认会返回一个新的 Dataset
    preprocess,
    remove_columns=dataset["train"].column_names,  # remove_columns参数决定丢掉哪些字段
)

print(tokenized_dataset)

print("\n--- tokenized sample ---")
sample = tokenized_dataset["train"][0]

print("input_ids:")
print(sample["input_ids"])

print("\nlabels:")
print(sample["labels"])

print("\nattention_mask:")
print(sample["attention_mask"])
