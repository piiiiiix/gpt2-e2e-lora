"""FFT 逐样本 CIDEr 分布页面。

启动：python -m streamlit run src/10_CIDEr_analysis.py
依赖：streamlit torch transformers datasets pycocoevalcap matplotlib pandas
评分口径与 08_benchmark.py 相同：每条数据的 target 为单个参考答案，
不额外分词；在整个选定 split 上计算 CIDEr 的文档频率，再取逐样本分数。
"""

import csv
import gc
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "outputs/gpt2-e2e-full-ft/final"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/CIDEr_analysis"


def discover_models():
    """只展示已保存完整权重的 FFT 模型，名称包含训练时间和 seed。"""
    if not MODEL_ROOT.exists():
        return {}
    return {
        path.name: path
        for path in sorted(MODEL_ROOT.iterdir())
        if path.is_dir() and (path / "config.json").is_file()
        and any(next(path.glob(pattern), None) is not None for pattern in
                ("*.safetensors", "pytorch_model*.bin"))
    }


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def score_samples(samples, predictions):
    from pycocoevalcap.cider.cider import Cider

    if len(samples) < 2:
        raise ValueError("CIDEr 的文档频率需要至少两条样本；单条样本会退化为零分。")
    if len(samples) != len(predictions):
        raise ValueError("生成结果数量与样本数量不一致。")
    references = {i: [sample["target"]] for i, sample in enumerate(samples)}
    hypotheses = {i: [prediction] for i, prediction in enumerate(predictions)}
    mean, scores = Cider().compute_score(references, hypotheses)
    return float(mean), [float(score) for score in scores]


def generate_predictions(model_path, samples, max_new_tokens, device, progress):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True,
        ).to(device)
        model.eval()
        predictions = []
        for index, sample in enumerate(samples):
            prompt = f"MR: {sample['meaning_representation']}\nText:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_length = inputs["input_ids"].shape[1]
            context_limit = getattr(model.config, "max_position_embeddings", 1024)
            if prompt_length + max_new_tokens > context_limit:
                raise ValueError(f"样本 {index} 的 prompt 与生成长度超过模型上下文上限 {context_limit}。")
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            predictions.append(tokenizer.decode(
                output[0][prompt_length:], skip_special_tokens=True,
            ).strip())
            progress(index + 1, len(samples))
        return predictions
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_scatter(records, output_dir, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(row["model"] for row in records))
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3.6 * len(names)),
                             squeeze=False, sharex=True, sharey=True)
    try:
        for index, name in enumerate(names):
            rows = [row for row in records if row["model"] == name]
            ax = axes[index, 0]
            ax.scatter([row["sample_index"] for row in rows],
                       [row["cider"] for row in rows], s=9, alpha=.55,
                       color=plt.get_cmap("tab10")(index % 10), rasterized=True)
            mean = sum(row["cider"] for row in rows) / len(rows)
            ax.axhline(mean, color="#c0392b", linestyle="--", linewidth=1,
                       label=f"Mean = {mean:.6f}")
            ax.set_title(name, fontsize=10)
            ax.set_ylabel("CIDEr (raw score)")
            ax.grid(alpha=.2)
            ax.legend(loc="upper right")
        axes[-1, 0].set_xlabel("Sample index (original dataset row, zero-based)")
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, .97))
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"cider_scatter.{extension}", dpi=180)
    finally:
        plt.close(fig)


def main():
    import pandas as pd
    import streamlit as st
    import torch
    from datasets import load_dataset

    st.set_page_config(page_title="FFT CIDEr analysis", layout="wide")
    st.title("FFT · 逐样本 CIDEr 分布")
    models = discover_models()
    if not models:
        st.error(f"未找到完整 FFT 模型：{MODEL_ROOT}")
        st.stop()

    # 所有运行控制都放在页面顶部；提交后才加载数据和模型。
    with st.form("run_controls"):
        selected = st.multiselect("选择 FFT 模型（可多选）", list(models), default=list(models))
        left, middle, right = st.columns(3)
        dataset_name = left.text_input("Hugging Face 数据集", "GEM/e2e_nlg")
        dataset_config = middle.text_input("数据集配置名（可留空）", "")
        split = right.text_input("数据 split", "validation")
        left, middle, right = st.columns(3)
        sample_limit = left.number_input("样本数（0 = 完整 split）", min_value=0, value=0, step=100)
        max_tokens = middle.number_input("max_new_tokens", min_value=1, max_value=1024, value=60)
        device_choice = right.selectbox("计算设备", ["自动", "cpu", "cuda"])
        submitted = st.form_submit_button("开始评估并保存图片", type="primary")

    st.caption("数据集须含 meaning_representation 和 target 文本列。默认与现有 benchmark 一致："
               "greedy 生成、60 个新 token、每行使用一个 target 参考答案。"
               "更换 split 或样本数会改变 CIDEr 的 IDF，分数应在同一评估集内比较。")

    if submitted:
        if not selected or not dataset_name.strip() or not split.strip():
            st.error("请选择至少一个模型，并填写数据集和 split。")
            st.stop()
        if device_choice == "cuda" and not torch.cuda.is_available():
            st.error("当前环境没有可用的 CUDA。请选择 cpu 或自动。")
            st.stop()
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device_choice == "自动" else device_choice
        run_dir = None
        try:
            with st.spinner("正在加载评估数据…"):
                dataset = load_dataset(dataset_name.strip(),
                                       name=dataset_config.strip() or None,
                                       split=split.strip())
                if sample_limit:
                    dataset = dataset.select(range(min(int(sample_limit), len(dataset))))
                samples = list(dataset)
                if len(samples) < 2:
                    raise ValueError("评估集至少需要两条样本。")
                for i, sample in enumerate(samples):
                    for column in ("meaning_representation", "target"):
                        if not isinstance(sample.get(column), str) or not sample[column].strip():
                            raise ValueError(f"样本 {i} 的 {column} 缺失、为空或不是文本。")

            run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir.mkdir(parents=True, exist_ok=False)
            config = {"dataset": dataset_name.strip(), "dataset_config": dataset_config.strip() or None,
                      "split": split.strip(), "num_samples": len(samples),
                      "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
                      "models": {name: str(models[name]) for name in selected},
                      "device": device, "max_new_tokens": int(max_tokens), "do_sample": False,
                      "reference_policy": "single target per dataset row",
                      "metric": "pycocoevalcap.cider.cider.Cider (default n=4, sigma=6.0)"}
            save_json(run_dir / "run_config.json", config)
            records, summaries = [], []
            bar = st.progress(0.0)
            status = st.empty()
            for model_index, name in enumerate(selected):
                def update(done, total):
                    bar.progress((model_index + done / total) / len(selected))
                    status.text(f"{name}：已生成 {done}/{total}")

                predictions = generate_predictions(models[name], samples, int(max_tokens), device, update)
                mean, scores = score_samples(samples, predictions)
                model_records = [
                    {"model": name, "dataset": dataset_name.strip(), "split": split.strip(),
                     "sample_index": i, "sample_id": sample.get("gem_id", sample.get("id", i)),
                     "meaning_representation": sample["meaning_representation"],
                     "reference": sample["target"], "prediction": prediction, "cider": score}
                    for i, (sample, prediction, score) in enumerate(zip(samples, predictions, scores))
                ]
                # 每完成一个模型立即保存，后续模型失败也能保留已完成结果。
                with (run_dir / f"{name}_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(model_records[0]))
                    writer.writeheader()
                    writer.writerows(model_records)
                save_scatter(model_records, run_dir, f"{dataset_name} / {split} | {len(samples)} samples")
                # 单模型图片保留独立文件名，最后再写总对比图。
                for extension in ("png", "svg"):
                    (run_dir / f"cider_scatter.{extension}").rename(run_dir / f"{name}_scatter.{extension}")
                records.extend(model_records)
                values = pd.Series(scores)
                summaries.append({"model": name, "samples": len(scores), "mean_CIDEr": mean,
                                  "std": float(values.std()), "min": float(values.min()),
                                  "median": float(values.median()), "p95": float(values.quantile(.95)),
                                  "max": float(values.max()), "zero_count": int((values == 0).sum())})
                save_json(run_dir / "summary.json", summaries)
            pd.DataFrame(records).to_csv(run_dir / "all_sample_scores.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
            save_scatter(records, run_dir, f"{dataset_name} / {split} | {len(samples)} samples")
            st.session_state["cider_analysis_result"] = str(run_dir)
            status.text("全部模型评估完成。")
        except Exception as exc:
            st.error(f"评估失败：{exc}")
            if run_dir is not None:
                st.info(f"已完成模型的结果保存在：{run_dir}")
            st.stop()

    if "cider_analysis_result" in st.session_state:
        run_dir = Path(st.session_state["cider_analysis_result"])
        st.success(f"结果已保存：{run_dir}")
        st.dataframe(pd.read_csv(run_dir / "summary.csv"), hide_index=True, use_container_width=True)
        st.image(str(run_dir / "cider_scatter.png"))
        frame = pd.read_csv(run_dir / "all_sample_scores.csv")
        st.subheader("逐样本分数与生成文本")
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button("下载全部逐样本分数 CSV", (run_dir / "all_sample_scores.csv").read_bytes(),
                           file_name="all_sample_scores.csv", mime="text/csv")


if __name__ == "__main__":
    main()
