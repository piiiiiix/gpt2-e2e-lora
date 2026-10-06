"""从 benchmark.log 汇总三个训练 seed，并为每项指标绘制 LoRA / FFT 对比图。

运行：python src/09_comparison.py
可选：--benchmark-root PATH --output-dir PATH --start-date 20261005
      --end-date 20261006。日期筛选依据 benchmark 文件夹名，跨午夜需包含两天。
"""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKS = (1, 2, 4, 8, 16, 32, 64)
SEEDS = (42, 43, 44)
METRICS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "NIST", "METEOR", "ROUGE-L", "CIDEr")
GROUPS = tuple(f"R{rank}" for rank in RANKS) + ("FFT",)


def read_log(path):
    """读取日志尾部已完成的 JSON 结果；不依赖 metrics.json。"""
    content = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    result = None
    for match in re.finditer(r"^\s*\{", content, re.MULTILINE):
        try:
            candidate, _ = decoder.raw_decode(content[match.end() - 1:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "metrics" in candidate and "config" in candidate:
            result = candidate
    return result


def collect_runs(root, start_date=None, end_date=None):
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 benchmark 目录：{root}")
    selected = {}
    settings = None
    # 按时间目录排序，同一模型和 seed 重跑时只取最新的已完成结果。
    for path in sorted(root.rglob("benchmark.log")):
        if "smoke" in path.relative_to(root).parts:
            continue
        date_match = re.match(r"(\d{8})_", path.parent.name)
        if start_date or end_date:
            if not date_match:
                continue
            date = date_match[1]
            if (start_date and date < start_date) or (end_date and date > end_date):
                continue
        result = read_log(path)
        if result is None:
            print(f"跳过未完成日志：{path}")
            continue
        config = result["config"]
        kind = config.get("type")
        if kind not in ("lora", "full", "fft"):
            continue
        identity = f"{result.get('name', '')}/{config.get('path', '')}"
        seed_match = re.search(r"seed[_-]?(\d+)(?!\d)", identity, re.IGNORECASE)
        rank_match = re.search(r"(?:^|[/\\_])r(\d+)(?=[/\\_]|$)", identity, re.IGNORECASE)
        if not seed_match or (kind == "lora" and not rank_match):
            raise ValueError(f"无法从日志识别 rank / seed：{path}")
        seed = int(seed_match[1])
        group = f"R{int(rank_match[1])}" if kind == "lora" else "FFT"
        if seed not in SEEDS or group not in GROUPS:
            continue
        scores = result["metrics"]
        if any(key not in scores or not isinstance(scores[key], (int, float))
               or not math.isfinite(scores[key]) for key in METRICS):
            raise ValueError(f"指标缺失或无效：{path}")
        key = (group, seed)
        if key in selected:
            print(f"重复结果，使用较新的日志：{path}")
        selected[key] = {"group": group, "seed": seed, "log": str(path),
                         "settings": tuple(result.get(k) for k in
                                           ("split", "num_samples", "max_new_tokens", "do_sample")),
                         "metrics": {key: float(scores[key]) for key in METRICS}}
    if not selected:
        raise ValueError("没有找到已完成的 LoRA / FFT benchmark 日志。")
    for run in selected.values():
        if settings is None:
            settings = run["settings"]
        elif run["settings"] != settings:
            raise ValueError(f"benchmark 设置不一致，不能直接比较：{run['log']}")
    return selected


def summarize(runs):
    summary = []
    for group in GROUPS:
        members = [runs[(group, seed)] for seed in SEEDS if (group, seed) in runs]
        if not members:
            print(f"缺少 {group}：不绘制该组，不计算均值。")
            continue
        if len(members) != len(SEEDS):
            missing = [seed for seed in SEEDS if (group, seed) not in runs]
            raise ValueError(f"{group} 缺少 seed {missing}，不能计算三个 seed 的平均值。")
        for metric in METRICS:
            values = [run["metrics"][metric] for run in members]
            summary.append({"group": group, "metric": metric, "n_seeds": len(values),
                            "mean": statistics.mean(values), "std": statistics.stdev(values),
                            **{f"seed{seed}": value for seed, value in zip(SEEDS, values)}})
    fft_means = {row["metric"]: row["mean"] for row in summary if row["group"] == "FFT"}
    for row in summary:
        reference = fft_means.get(row["metric"])
        row["percent_of_fft"] = 100 * row["mean"] / reference if reference else None
        row["difference_from_fft"] = row["mean"] - reference if reference is not None else None
    return summary


def save_results(runs, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "comparison.json").write_text(
        json.dumps({"seeds": SEEDS, "summary": summary, "source_runs": list(runs.values())},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    for metric in METRICS:
        rows = {row["group"]: row for row in summary if row["metric"] == metric}
        # 只绘制有完整数据的组；缺失的 rank 不占用横轴位置。
        groups = [group for group in GROUPS if group in rows]
        positions = list(range(len(groups)))
        means = [rows[group]["mean"] for group in groups]
        errors = [rows[group]["std"] for group in groups]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.23, top=0.88)
        bars = ax.bar(positions, means, yerr=errors, capsize=5, width=0.65,
                      color=["#e8913a" if group == "FFT" else "#487eb0" for group in groups])
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=8, fontsize=9)
        if "FFT" in rows:
            ax.axhline(rows["FFT"]["mean"], color="#bc691d", linestyle="--",
                       linewidth=1.3, label="FFT mean", zorder=1)
            ax.legend(loc="upper left", frameon=False)
        ax.set_xticks(positions, groups)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0, max(value + error for value, error in zip(means, errors)) * 1.22 or 1)
        ax.set_xlabel("LoRA rank / full fine-tuning")
        ax.set_ylabel(f"{metric} score")
        ax.set_title(f"{metric}: LoRA vs. FFT", fontsize=15)
        fig.text(0.10, 0.09, "Mean of 3 seeds (42, 43, 44); error bars: sample standard deviation.",
                 fontsize=9, color="#555555")
        relative = [rows[group]["percent_of_fft"] for group in groups
                    if group != "FFT" and rows[group]["percent_of_fft"] is not None]
        if relative:
            fig.text(0.10, 0.045,
                     f"LoRA scores are {min(relative):.1f}%–{max(relative):.1f}% of the FFT mean. "
                     "Higher scores are better.", fontsize=9, color="#555555")
        ax.yaxis.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        for extension in ("png", "svg"):
            fig.savefig(output / f"{metric}.{extension}", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=PROJECT_ROOT / "outputs/benchmarks")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/comparison")
    parser.add_argument("--start-date", help="起始 benchmark 日期，格式 YYYYMMDD")
    parser.add_argument("--end-date", help="结束 benchmark 日期，格式 YYYYMMDD（包含当天）")
    args = parser.parse_args()
    for value in (args.start_date, args.end_date):
        if value and not re.fullmatch(r"\d{8}", value):
            parser.error("日期格式必须为 YYYYMMDD")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("起始日期不能晚于结束日期")
    runs = collect_runs(args.benchmark_root, args.start_date, args.end_date)
    summary = summarize(runs)
    save_results(runs, summary, args.output_dir)
    print(f"已汇总 {len(runs)} 个模型，生成 {len(METRICS)} 项指标对比图：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
