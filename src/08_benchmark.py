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
from peft import PeftModel
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai-community/gpt2"
LORA_PATH = PROJECT_ROOT / "outputs/gpt2-e2e-lora/final/r4_attn_lr2e-4_best-eval_dirtest"
FT_PATH = PROJECT_ROOT / "outputs/gpt2-e2e-full-ft/final"
SPLIT = "validation"
NUM_SAMPLES = 100
MAX_NEW_TOKENS = 60


def benchmark(model, tokenizer, eval_dataset, *, model_name="Model", max_new_tokens=60):
    """生成文本并计算指标，返回指标字典。"""
    if len(eval_dataset) == 0:
        raise ValueError("评测数据不能为空")
    ensure_wordnet()
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    references = []

    for i, sample in enumerate(eval_dataset):
        prompt = f"MR: {sample['meaning_representation']}\nText:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # 只保留新生成的文本，不包含 prompt。
        prediction = tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True,
        ).strip()
        predictions.append(prediction)
        references.append(sample["target"])
        if (i + 1) % 100 == 0 or i + 1 == len(eval_dataset):
            print(f"[{model_name}] Generated {i + 1}/{len(eval_dataset)}", flush=True)

    # pycocoevalcap 要求每条 reference/prediction 放在列表中。
    gts = {i: [reference] for i, reference in enumerate(references)}
    res = {i: [prediction] for i, prediction in enumerate(predictions)}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    rouge_l, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)

    nist_references = [[reference.split()] for reference in references]
    nist_hypotheses = [prediction.split() for prediction in predictions]
    nist = corpus_nist(nist_references, nist_hypotheses, n=5)
    meteor = sum(
        meteor_score([reference.split()], prediction.split())
        for reference, prediction in zip(references, predictions)
    ) / len(predictions)

    return {
        **{f"BLEU-{i + 1}": float(score) for i, score in enumerate(bleu_scores)},
        "NIST": float(nist),
        "METEOR": float(meteor),
        "ROUGE-L": float(rouge_l),
        "CIDEr": float(cider),
    }


def main():
    if not (LORA_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(f"找不到 LoRA adapter：{LORA_PATH}")
    if not (FT_PATH / "config.json").is_file():
        raise FileNotFoundError(f"找不到 full fine-tuning 模型：{FT_PATH}")
    ensure_wordnet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GPT-2 和 LoRA 共用训练时保存的 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(str(LORA_PATH))
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)
    # 冒烟测试
    # eval_dataset = dataset[SPLIT].select(range(min(NUM_SAMPLES, len(dataset[SPLIT]))))

    eval_dataset = dataset[SPLIT]

    baseline_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    baseline_scores = benchmark(
        baseline_model, tokenizer, eval_dataset,
        model_name="GPT-2", max_new_tokens=MAX_NEW_TOKENS,
    )
    # 先释放 baseline，避免两个完整模型同时占用显存。
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    lora_base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(lora_base_model, str(LORA_PATH)).to(device)
    lora_scores = benchmark(
        lora_model, tokenizer, eval_dataset,
        model_name="GPT-2 + LoRA", max_new_tokens=MAX_NEW_TOKENS,
    )
    # PeftModel 持有 base model；两个引用都释放后再加载全量微调模型。
    del lora_model, lora_base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    ft_tokenizer = AutoTokenizer.from_pretrained(str(FT_PATH))
    ft_tokenizer.pad_token = ft_tokenizer.eos_token
    ft_model = AutoModelForCausalLM.from_pretrained(str(FT_PATH)).to(device)
    ft_scores = benchmark(
        ft_model, ft_tokenizer, eval_dataset,
        model_name="GPT-2 E2E Full FT", max_new_tokens=MAX_NEW_TOKENS,
    )

    print(f"\nBenchmark ({SPLIT}, {len(eval_dataset)} samples)")
    print(f"{'Metric':<10} {'GPT-2':>12} {'GPT-2 + LoRA':>14} {'GPT-2 Full FT':>16}")
    print("-" * 55)
    for metric, baseline_score in baseline_scores.items():
        print(
            f"{metric:<10} {baseline_score:>12.4f} "
            f"{lora_scores[metric]:>14.4f} {ft_scores[metric]:>16.4f}"
        )


if __name__ == "__main__":
    main()
