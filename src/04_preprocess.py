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

'''
提示词：
MR:......
Text:

目标：
target文本
'''
prompt = f"MR: {mr}\nText:"
target_text = f" {target}"

'''
tokenizer(prompt):
{'input_ids': [13599, 25, 1438, 58, 464, 18456, 4357, 4483, 6030, 58, 1073, 5853, 6128, 4357, 2057, 58, 25324, 4357, 2756, 17257, 58, 1203, 621, 4248, 1238, 4357, 6491, 7955, 58, 9319, 4357, 1989, 58, 380, 690, 485, 4357, 1641, 23331, 306, 58, 8505, 4357, 1474, 58, 22991, 1362, 2677, 60, 198, 8206, 25], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

attention_mask =1有效token =0Mask掉
被 mask 的 token 在前向传播时不会作为有效上下文参与与其他 token 的注意力关系计算
'''

prompt_ids = tokenizer(prompt)["input_ids"]  # ids=Identifiers
target_ids = tokenizer(target_text)["input_ids"]

input_ids = prompt_ids + target_ids

labels = [-100] * len(prompt_ids) + target_ids  # PyTorch 的 CrossEntropyLoss 约定ignore_index = -100 对应位置不参与loss的计算