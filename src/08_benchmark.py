"""
benchmark : BLEU、NIST、METEOR、ROUGE-L、CIDEr
BLEU：主要看你的生成文本和参考答案之间，n-gram（连续词片段）重合得有多像。比如 reference 里有 near Raja Indian Cuisine，你的输出里也出现了类似连续词组，BLEU 就会涨。它还会综合 1-gram、2-gram、3-gram、4-gram，并对太短的输出做 brevity penalty（长度惩罚）。所以它本质上偏“字面重合度”。

NIST：可以理解成 BLEU 的一个变体，但它会给信息量更大的 n-gram 更高权重。像 the restaurant is 这种到处都有的短语，价值不大；像 Raja Indian Cuisine 这种更稀有、更有区分度的词组，贡献更大。所以它比 BLEU 更强调“你有没有对上有信息量的表达”。

METEOR：比 BLEU 更宽松一点，不只是死看完全一样的词，还会考虑词形变化、近义匹配、对齐情况。例如某些情况下 rated 和 rating，或者意义接近的词，比 BLEU 更容易得到部分匹配。它还会同时考虑 precision 和 recall，所以对“漏了很多 reference 内容”的情况也比较敏感。

ROUGE-L：这里的 L 是 Longest Common Subsequence（最长公共子序列）。它看的是你生成文本和 reference 之间，能不能找到一条比较长、顺序一致的共同词序列。它不要求这些词必须连续，所以比 BLEU 的 n-gram 更宽松一些。你可以理解成：整体句子骨架和词序有多像。

CIDEr：先把 prediction 和 reference 都变成 TF-IDF 加权的 n-gram 向量，然后算余弦相似度。这个最有点“面向生成任务”的味道。它也是看 n-gram，但会给稀有、具有描述区分度的词组更高权重，而高频套话权重更低。最早大量用于 image captioning（图像描述），核心思想是：生成的描述是不是抓住了参考答案里真正有辨识度的信息，而不是只会说一些万能句。对 E2E 这种生成任务也挺合适。
"""

import csv
import gc
import json
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
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
SPLIT = "validation"
MAX_NEW_TOKENS = 60


def benchmark(model, tokenizer, eval_dataset, *, model_name="Model", max_new_tokens=60, predictions_path=None):
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

    if predictions_path is not None:
        with Path(predictions_path).open("w", encoding="utf-8") as stream:
            for sample, prediction, reference in zip(eval_dataset, predictions, references):
                stream.write(json.dumps({"meaning_representation": sample["meaning_representation"],
                                         "prediction": prediction, "reference": reference},
                                        ensure_ascii=False) + "\n")

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


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_benchmark(model_config, eval_dataset, output_dir, *, split=SPLIT,
                  max_new_tokens=MAX_NEW_TOKENS):
    """评测一个模型；配置含 name、type（base/lora/full）、path、可选 base_model。"""
    import sys
    config = dict(model_config)
    kind = config.get("type")
    if kind not in {"base", "lora", "full"}:
        raise ValueError("模型 type 必须为 base、lora 或 full")
    path = str(config["path"])
    if kind in {"lora", "full"}:
        local_path = Path(path)
        if not local_path.is_absolute():
            local_path = PROJECT_ROOT / local_path
        required = "adapter_config.json" if kind == "lora" else "config.json"
        if not (local_path / required).is_file():
            raise FileNotFoundError(f"找不到模型文件：{local_path / required}")
        path = str(local_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    started = time.perf_counter()
    with (output_dir / "benchmark.log").open("w", encoding="utf-8") as log:  # noqa: SIM117
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                print(f"Model: {config['name']}\nConfig: {config}\nSplit: {split}\nSamples: {len(eval_dataset)}\nDevice: {device}", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(path)
                tokenizer.pad_token = tokenizer.eos_token
                if kind == "lora":
                    from peft import PeftConfig
                    base_path = config.get("base_model") or PeftConfig.from_pretrained(path).base_model_name_or_path
                    model = AutoModelForCausalLM.from_pretrained(base_path)
                    model = PeftModel.from_pretrained(model, path).to(device)
                else:
                    model = AutoModelForCausalLM.from_pretrained(path).to(device)
                scores = benchmark(model, tokenizer, eval_dataset, model_name=config["name"],
                                   max_new_tokens=max_new_tokens,
                                   predictions_path=output_dir / "predictions.jsonl")
                result = {"name": config["name"], "config": {**config, "path": path},
                          "split": split, "num_samples": len(eval_dataset), "device": device,
                          "max_new_tokens": max_new_tokens, "do_sample": False,
                          "elapsed_seconds": time.perf_counter() - started, "metrics": scores}
                (output_dir / "metrics.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
                return result
            except Exception:
                traceback.print_exc()
                raise
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def save_model_chart(result, output_dir):
    """单个模型的五项原始得分，共用同一组 XY 轴；BLEU 使用 BLEU-4。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["BLEU", "NIST", "METEOR", "ROUGE-L", "CIDEr"]
    keys = ["BLEU-4", "NIST", "METEOR", "ROUGE-L", "CIDEr"]
    values = [result["metrics"][key] for key in keys]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=plt.get_cmap("tab10").colors[:5])
    ax.bar_label(bars, labels=[f"{value:.4g}" for value in values], padding=4)
    ax.set_xlabel("Metric (BLEU = BLEU-4)")
    ax.set_ylabel("Score (raw values)")
    ax.set_ylim(0, max(max(values) * 1.2, 0.01))
    ax.set_title(f"{result['name']} | {result['split']} | {result['num_samples']} samples")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    try:
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"benchmark_scores.{extension}", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def benchmark_model(model_config, *, split=SPLIT, num_samples=None,
                    max_new_tokens=MAX_NEW_TOKENS, output_root=None):
    """接收一个模型，评测并保存独立日志、指标和五项得分图。"""
    if num_samples is not None and (isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0):
        raise ValueError("num_samples 必须为正整数或 None（完整 split）")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须为正整数")
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError("生成图表需要 matplotlib，请在训练环境运行 pip install matplotlib")
    dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)[split]
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs/benchmarks"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    import re
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", model_config["name"]).strip("_") or "model"
    run_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_config.json").write_text(json.dumps(
        {"model": model_config, "split": split, "num_samples": num_samples,
         "max_new_tokens": max_new_tokens}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    result = run_benchmark(model_config, dataset, run_dir,
                           split=split, max_new_tokens=max_new_tokens)
    # metrics.json 保留模型路径、评测设置及全部原始指标，供以后生成模型对比图。
    with (run_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        row = {"model": result["name"], "split": split, "samples": result["num_samples"],
               "elapsed_seconds": result["elapsed_seconds"], **result["metrics"]}
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    save_model_chart(result, run_dir)
    print(f"{result['name']} 日志、指标和五项得分图已保存：{run_dir}", flush=True)
    return run_dir

if __name__ == "__main__":
    raise SystemExit("请运行 src/00_benchmark_automation.py，在其中配置模型列表。")
