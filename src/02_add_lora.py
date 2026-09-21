from peft import (  # PEFT = Parameter-Efficient Fine-Tuning（参数高效微调）
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,  # LM是language model, Coral language model 因果语言模型
)

#缓存模型
model_name = "openai-community/gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)

config = LoraConfig(
    r=4,  # 中间低秩维度
    lora_alpha=4,  # alpha/r = LoRA 音量旋钮, 没有额外改 LoRA 矩阵输出的数值大小
    target_modules=["c_attn"],  # 目标模块,即QKV
    lora_dropout=0.0,  # 随机丢弃,随机给一定比例的激活值"打码",防止在特化过程的训练中过拟合,激活值/激活矩阵指上一层的运算结果
    bias="none",  # 不训练偏置
    task_type="CAUSAL_LM",  # 模型任务
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

print(model)