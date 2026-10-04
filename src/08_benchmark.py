"""
benchmark : BLEU、NIST、METEOR、ROUGE-L、CIDEr
BLEU：主要看你的生成文本和参考答案之间，n-gram（连续词片段）重合得有多像。比如 reference 里有 near Raja Indian Cuisine，你的输出里也出现了类似连续词组，BLEU 就会涨。它还会综合 1-gram、2-gram、3-gram、4-gram，并对太短的输出做 brevity penalty（长度惩罚）。所以它本质上偏“字面重合度”。

NIST：可以理解成 BLEU 的一个变体，但它会给信息量更大的 n-gram 更高权重。像 the restaurant is 这种到处都有的短语，价值不大；像 Raja Indian Cuisine 这种更稀有、更有区分度的词组，贡献更大。所以它比 BLEU 更强调“你有没有对上有信息量的表达”。

METEOR：比 BLEU 更宽松一点，不只是死看完全一样的词，还会考虑词形变化、近义匹配、对齐情况。例如某些情况下 rated 和 rating，或者意义接近的词，比 BLEU 更容易得到部分匹配。它还会同时考虑 precision 和 recall，所以对“漏了很多 reference 内容”的情况也比较敏感。

ROUGE-L：这里的 L 是 Longest Common Subsequence（最长公共子序列）。它看的是你生成文本和 reference 之间，能不能找到一条比较长、顺序一致的共同词序列。它不要求这些词必须连续，所以比 BLEU 的 n-gram 更宽松一些。你可以理解成：整体句子骨架和词序有多像。

CIDEr：先把 prediction 和 reference 都变成 TF-IDF 加权的 n-gram 向量，然后算余弦相似度。这个最有点“面向生成任务”的味道。它也是看 n-gram，但会给稀有、具有描述区分度的词组更高权重，而高频套话权重更低。最早大量用于 image captioning（图像描述），核心思想是：生成的描述是不是抓住了参考答案里真正有辨识度的信息，而不是只会说一些万能句。对 E2E 这种生成任务也挺合适。
"""

from pathlib import Path

import nltk
import torch
from datasets import load_dataset
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import corpus_nist
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 0. 配置
# ============================================================

def ensure_wordnet():
    # Use the same directory for downloading and loading, regardless of IDE cwd.
    data_dir = Path(__file__).resolve().parents[1] / "nltk_data"
    nltk.data.path.insert(0, str(data_dir))
    try:
        wordnet.ensure_loaded()
        return
    except LookupError:
        pass

    data_dir.mkdir(parents=True, exist_ok=True)
    if not nltk.download("wordnet", download_dir=str(data_dir), raise_on_error=True):
        raise RuntimeError(f"WordNet download failed. Data directory: {data_dir}")
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:
        raise RuntimeError(
            f"WordNet is still unavailable after downloading to {data_dir}. "
            "Check the downloader output and your network connection."
        ) from exc


# English METEOR needs WordNet; omw-1.4 is for multilingual lookups.
ensure_wordnet()

MODEL_NAME = "openai-community/gpt2"

# 先在 validation 上确认评测链路
SPLIT = "validation"

MAX_NEW_TOKENS = 60

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. tokenizer + baseline GPT-2
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()  # 切换推理模式


# ============================================================
# 2. 加载 E2E
# ============================================================

dataset = load_dataset(
    "GEM/e2e_nlg",
    trust_remote_code=True,
)

eval_dataset = dataset[SPLIT].select(range(100))


# ============================================================
# 3. 逐条生成
# ============================================================

predictions = []
references = []

for i, sample in enumerate(eval_dataset):

    mr = sample["meaning_representation"]
    reference = sample["target"]

    prompt = f"MR: {mr}\nText:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",  # 结果直接返回成 PyTorch Tensor
    ).to(device)

    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只保留新生成的部分
    generated_ids = output[0][prompt_length:]

    prediction = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()  # .strip()删开头结尾空白字符

    predictions.append(prediction)
    references.append(reference)

    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1}/{len(eval_dataset)}")


# ============================================================
# 4. 转换成 pycocoevalcap 所需格式
# ============================================================

# 格式：
# {
#   0: ["reference sentence"],
#   1: ["reference sentence"],
# }
#
# prediction 同理，只不过每条只有一个生成结果

gts = {i: [references[i]] for i in range(len(references))}

res = {i: [predictions[i]] for i in range(len(predictions))}


# ============================================================
# 5. BLEU
# ============================================================

bleu_scorer = Bleu(4)
# Bleu(n) = 计算从 BLEU-1 一直到 BLEU-n
# BLEU-4 本身也不是“只看 4-gram”。标准 BLEU-4 通常是把 1、2、3、4-gram 的 precision 一起综合，再乘 brevity penalty（长度惩罚）。

bleu_score, _ = bleu_scorer.compute_score(
    gts,
    res,
)

# bleu_score 会返回 BLEU-1 ~ BLEU-4
bleu1, bleu2, bleu3, bleu4 = bleu_score


# ============================================================
# 6. ROUGE-L
# ============================================================

rouge_scorer = Rouge()

rouge_l, _ = rouge_scorer.compute_score(
    gts,
    res,
)


# ============================================================
# 7. CIDEr
# ============================================================

cider_scorer = Cider()

cider, _ = cider_scorer.compute_score(
    gts,
    res,
)


# ============================================================
# 8. NIST
# ============================================================

# NIST 需要：
#
# references:
# [
#   [["reference", "tokens"]],
#   [["another", "reference"]],
# ]
#
# hypotheses:
# [
#   ["generated", "tokens"],
#   ["another", "generated"]
# ]

'''
[
    "The restaurant is good",
    "It is near Raja"
]
⬇️
[
    [
        ["The", "restaurant", "is", "good"]
    ],
    [
        ["It", "is", "near", "Raja"]
    ]
]
'''


nist_references = [[reference.split()] for reference in references]

nist_hypotheses = [prediction.split() for prediction in predictions]

nist = corpus_nist(
    nist_references,
    nist_hypotheses,
    n=5,
)


# ============================================================
# 9. METEOR
# ============================================================

meteor_scores = []

# for ... in zip() 多个可迭代对象一起遍历
for reference, prediction in zip(
    references,
    predictions,
):
    score = meteor_score(
        [reference.split()],
        prediction.split(),
    )

    meteor_scores.append(score)

meteor = sum(meteor_scores) / len(meteor_scores)


# ============================================================
# 10. 输出结果
# ============================================================

print("\n===================================")
print(f"Baseline GPT-2 Benchmark ({SPLIT})")
print("===================================")

print(f"BLEU-1   : {bleu1:.4f}")
print(f"BLEU-2   : {bleu2:.4f}")
print(f"BLEU-3   : {bleu3:.4f}")
print(f"BLEU-4   : {bleu4:.4f}")

print(f"NIST     : {nist:.4f}")
print(f"METEOR   : {meteor:.4f}")
print(f"ROUGE-L  : {rouge_l:.4f}")
print(f"CIDEr    : {cider:.4f}")
