"""train_lora(R): train three independent LoRA models with seeds 42, 43, 44."""
import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import TensorBoardCallback

MODEL_NAME = "openai-community/gpt2"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs/gpt2-e2e-lora"
# SEEDS = (42, )
SEEDS = (42, 43, 44)


def train_lora(R):
    """只接收正整数 R，返回 seed 42、43、44 三个模型的保存目录。"""
    if isinstance(R, bool) or not isinstance(R, int) or R <= 0:
        raise ValueError("R 必须是正整数")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)

    def preprocess(example):
        prompt_ids = tokenizer(f"MR: {example['meaning_representation']}\nText:")["input_ids"]
        target_ids = tokenizer(f" {example['target']}")["input_ids"]
        input_ids = prompt_ids + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + target_ids,
        }

    tokenized_dataset = dataset.map(
        preprocess, remove_columns=dataset["train"].column_names,
    )
    saved_paths = []
    for seed in SEEDS:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seed{seed}"
        run_name = f"r{R}_{run_id}"
        save_dir = OUTPUT_ROOT / "final" / f"r{R}" / run_id
        checkpoint_dir = OUTPUT_ROOT / "checkpoints" / f"r{R}" / run_id
        tensorboard_dir = OUTPUT_ROOT / "tensorboard" / f"r{R}" / run_id
        # 同参数重训也使用独立目录；目录重名时停止，避免覆盖已有结果。
        for run_dir in (save_dir, checkpoint_dir, tensorboard_dir):
            run_dir.mkdir(parents=True, exist_ok=False)
        print(f"\n===== Training {run_name} =====", flush=True)
        # 必须先设置 seed，再创建模型和随机初始化的 LoRA 权重。
        set_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model = get_peft_model(model, LoraConfig(
            r=R, lora_alpha=4, target_modules=["c_attn"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100,
        )
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            run_name=run_name,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=torch.cuda.is_available(),
            seed=seed,
            data_seed=seed,
        )
        # 显式绑定本次训练目录，避免自动 callback 使用默认或共享日志路径。
        tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks=[TensorBoardCallback(tb_writer=tb_writer)],
        )
        try:
            trainer.train()
            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            with (save_dir / "train_log.json").open("w", encoding="utf-8") as f:
                json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
            saved_paths.append(save_dir)
            print(f"Saved {run_name}: {save_dir}", flush=True)
        finally:
            tb_writer.close()
            # Trainer 和 collator 也持有模型引用。
            del trainer, data_collator, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("R", type=int, help="LoRA rank; trains seeds 42, 43, 44")
    train_lora(parser.parse_args().R)
