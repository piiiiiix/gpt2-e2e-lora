import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai-community/gpt2"
lora_path = "outputs/gpt2-e2e-lora/final"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 加载 tokenizer
# ============================================================

# 直接加载我们训练结束时保存的 tokenizer 配置
tokenizer = AutoTokenizer.from_pretrained(lora_path)


# ============================================================
# 2. 加载 baseline：原始 GPT-2
# ============================================================

baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
baseline_model = baseline_model.to(device)
baseline_model.eval()


# ============================================================
# 3. 加载 LoRA 模型
# ============================================================

# LoRA 不是一个完整 GPT-2，所以先加载原始 GPT-2
lora_base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 再把训练好的 LoRA adapter 挂上去
lora_model = PeftModel.from_pretrained(
    lora_base_model,
    lora_path,
)

lora_model = lora_model.to(device)
lora_model.eval()


# ============================================================
# 4. 加载 E2E 数据
# ============================================================

dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)

sample = dataset["validation"][0]

mr = sample["meaning_representation"]
reference = sample["target"]

prompt = f"MR: {mr}\nText:"


# ============================================================
# 5. tokenize
# ============================================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(device)

prompt_length = inputs["input_ids"].shape[1]


# ============================================================
# 6. 两个模型使用完全相同的生成设置
# ============================================================

def generate(model):
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取模型新生成的部分，不把 prompt 再打印一遍
    generated_ids = output[0][prompt_length:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


baseline_output = generate(baseline_model)
lora_output = generate(lora_model)


# ============================================================
# 7. 对比
# ============================================================

print("\n================ MR ================\n")
print(mr)

print("\n============= Reference =============\n")
print(reference)

print("\n=========== Baseline GPT-2 ==========\n")
print(baseline_output)

print("\n============ GPT-2 + LoRA ===========\n")
print(lora_output)