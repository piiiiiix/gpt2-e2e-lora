import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai-community/gpt2"

# 加载模型对应的tokenizer
print("1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型
print("2. Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained(model_name)
# 把模型搬到显卡
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("3. Device:",device)

prompt = "The meaning of life is"

# 切分prompt，把切分好的token转成token ID，再包装成PyTorch tensor这个PT
inputs = tokenizer(prompt, return_tensors = "pt")
# 把输入搬进显卡
inputs = inputs.to(device)

# 输出token ID
print("4. Input IDs:")
print(inputs["input_ids"])

print("5. Tokens:")
# 每人看一眼token ID所对应的token
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

with torch.no_grad():  # 不进行反向传播
    outputs = model.generate(  # 模型生成，词向量转换的过程被包含在内
        **inputs,
        max_new_tokens = 30,  # 最多30个token
        do_sample = False,  # 贪心,不进行随机采样
    )

# 解码模型输出,跳过特殊token
print("6. Generated text:")
print(tokenizer.decode(outputs[0], skip_special_tokens = True))

# 模型架构
print("\n7. GPT-2 structure:")
print(model)

'''
(c_attn): Conv1D(nf=2304, nx=768) Wq+Wv+Wk
(c_proj): Conv1D(nf=768, nx=768) Wo
nf 输出维度
nx 输入维度

此处维度指的是特征数量 不同于Axis


'''