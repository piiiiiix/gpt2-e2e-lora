import json
import os

from datasets import load_dataset
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

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# 直接加载完整 GPT-2
# 不挂 LoRA，所以所有原始参数都会参与训练
model = AutoModelForCausalLM.from_pretrained(model_name)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)


os.environ["TENSORBOARD_LOGGING_DIR"] = "outputs/gpt2-e2e-full-ft/tensorboard"


training_args = TrainingArguments(
    output_dir="outputs/gpt2-e2e-full-ft/checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

save_dir = "outputs/gpt2-e2e-full-ft/final"

trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

# 11. 保存训练过程到日志
log_path = os.path.join(save_dir, "train_log.json")
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(
        trainer.state.log_history,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Training log saved to: {log_path}")
