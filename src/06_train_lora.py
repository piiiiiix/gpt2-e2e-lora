import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

model_name = "openai-community/gpt2"

# 1. tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. dataset
dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)


def preprocess(example):
    mr = example["meaning_representation"]
    target = example["target"]

    prompt = f"MR: {mr}\nText:"
    target_text = f" {target}"

    prompt_ids = tokenizer(prompt)["input_ids"]
    target_ids = tokenizer(target_text)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names,
)

# 3. smoke test
# train_dataset = tokenized_dataset["train"].select(range(500))
# eval_dataset = tokenized_dataset["validation"].select(range(100))

# 3.1 formal training
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# 4. GPT-2
model = AutoModelForCausalLM.from_pretrained(model_name)

# 5. LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["c_attn"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. 一个 batch 里面句子长度不同，所以需要动态 padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,  # 将所有数据 padding 到最长数据长度
    # padding="max_length",  # padding 到指定长度的写法
    # max_length=128,
    # Padding元素对应在mask中自动变为0，且mask延长
    label_pad_token_id=-100,
)

# 7. 训练参数
training_args = TrainingArguments(
    # checkpoint、日志相关输出存放位置
    # output_dir="outputs/lora-smoke-test",
    output_dir="outputs/gpt2-e2e-lora/checkpoints",

    # 训练、验证的batch size
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    # 初始/最大学习率
    learning_rate=2e-4,
    # 整个训练数据集跑几遍
    num_train_epochs=3,

    # 日志频率 练10个步长，打印一次训练信息
    logging_steps=10,

    # smoke test 验证触发频率 触发频率，单位为步长，每50个步长触发一次
    # eval_strategy="steps",
    # eval_steps=50,

    eval_strategy= "epoch",

    # smoke test 不设置 checkpoint ,即存档点
    # save_strategy="no",
    # 存档点设置方法 
    # save_strategy="steps",
    # save_steps=500,
    # save_total_limit=2,

    save_strategy= "epoch",
    save_total_limit= 2,
    
    # 不发送训练日志到外部日志工具
    report_to="none",

    # 16位浮点混合精度训练
    fp16=True,
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 9. 开练
trainer.train()
'''
{'loss': '3.517', 'grad_norm': '0.3274', 'learning_rate': '0.0001856', 'epoch': '0.08'} 
loss = -1/N sum(logP(yt|MR,y<t))
grad_norm = loss 对所有参数的梯度到原点的欧式距离对，又有，用于表示梯度的强度
learning_rate动态步长使用 schedule 策略，按照步数把 learning rate 从初始给定值均匀降到 0
步数 = num_train_epochs * train_dataset / per_device_train_batch_size
'''

# 10. 保存模型和tokenizer配置
trainer.save_model("outputs/gpt2-e2e-lora/final")
tokenizer.save_pretrained("outputs/gpt2-e2e-lora/final")

'''
# 10. smoke test 测试
test_sample = dataset["validation"][0]

prompt = f"MR: {test_sample['meaning_representation']}\nText:"

inputs = tokenizer(
    prompt,
    return_tensors="pt",  # pt = PyTorch tensor
).to(model.device)


# 切换评估模式，关闭dropout， Dropout会随机丢弃一部分激活值
model.eval()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
    )

print("\nMR:")
print(test_sample["meaning_representation"])

print("\nReference:")
print(test_sample["target"])

print("\nModel output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
'''