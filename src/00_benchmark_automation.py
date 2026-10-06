"""Select LoRA ranks, FFT and GPT-2; run the 08 benchmark interface."""
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ============================================================
# 运行选择：True = 打开，False = 关闭。可同时打开多个类别。
# 仅原生 GPT-2：保持下面默认设置。
# 仅 R8：RUN_LORA=True，LORA_RANKS=(8,)，其余类别关闭。
# R4 + R8 + FFT：RUN_LORA=True，LORA_RANKS=(4, 8)，RUN_FFT=True。
# 全部：RUN_ALL=True，自动扫描已有的所有 LoRA 秩和 FFT seed。
# ============================================================
RUN_ALL = False          # 打开后覆盖下面三个类别开关及 LORA_RANKS
RUN_LORA = False
LORA_RANKS = (8,)        # 单个秩写 (8,)，多个写 (1, 2, 4, 8, 16, 64)
                        # None = 所有已保存的 LoRA 秩（跳过不存在的秩）
RUN_FFT = True          # 所有已保存的 FFT seed
RUN_GPT2 = False          # 原生 GPT-2 只跑一次，不需要三个训练 seed

LORA_MODEL_ROOT = PROJECT_ROOT / "outputs/gpt2-e2e-lora/final"
FFT_MODEL_ROOT = PROJECT_ROOT / "outputs/gpt2-e2e-full-ft/final"
GPT2_MODEL_PATH = "openai-community/gpt2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/benchmarks"
SPLIT = "validation"
NUM_SAMPLES = None      # None = 完整 split；设为 100 可先做快速评测
MAX_NEW_TOKENS = 60


def select_models(*, run_all=None, run_lora=None, lora_ranks=None,
                  run_fft=None, run_gpt2=None):
    """根据开关生成模型列表，运行前检查本地模型；保留原有 benchmark API。"""
    import re
    run_all = RUN_ALL if run_all is None else run_all
    run_lora = RUN_LORA if run_lora is None else run_lora
    run_fft = RUN_FFT if run_fft is None else run_fft
    run_gpt2 = RUN_GPT2 if run_gpt2 is None else run_gpt2
    ranks = LORA_RANKS if lora_ranks is None else lora_ranks
    models = []

    def resolve_root(root):
        root = Path(root)
        return root if root.is_absolute() else PROJECT_ROOT / root

    def discover(root, filename, kind):
        if not root.is_dir():
            raise FileNotFoundError(f"找不到模型目录：{root}")
        configs = sorted(root.rglob(filename), key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.relative_to(root).as_posix())])
        if not configs:
            raise FileNotFoundError(f"目录中没有 {filename}：{root}")
        return [{"name": "_".join(path.parent.relative_to(root).parts) or root.name,
                 "type": kind, "path": str(path.parent)} for path in configs]

    if run_all or run_lora:
        root = resolve_root(LORA_MODEL_ROOT)
        if run_all or ranks is None:
            models.extend(discover(root, "adapter_config.json", "lora"))
        else:
            if isinstance(ranks, int) and not isinstance(ranks, bool):
                ranks = (ranks,)
            ranks = tuple(ranks)
            if not ranks or any(isinstance(rank, bool) or not isinstance(rank, int)
                                or rank <= 0 for rank in ranks):
                raise ValueError("LORA_RANKS 必须是正整数或非空正整数列表，例如 (8,)")
            for rank in sorted(set(ranks)):
                rank_models = discover(root / f"r{rank}", "adapter_config.json", "lora")
                for config in rank_models:
                    config["name"] = "_".join(Path(config["path"]).relative_to(root).parts)
                models.extend(rank_models)
    if run_all or run_fft:
        models.extend(discover(resolve_root(FFT_MODEL_ROOT), "config.json", "full"))
    if run_all or run_gpt2:
        models.append({"name": "GPT-2", "type": "base", "path": str(GPT2_MODEL_PATH)})
    if not models:
        raise ValueError("没有打开任何模型开关，请启用 RUN_LORA、RUN_FFT、RUN_GPT2 或 RUN_ALL")
    return models


def main():
    import argparse
    parser = argparse.ArgumentParser(description="按顶部开关运行 LoRA / FFT / 原生 GPT-2 benchmark")
    parser.add_argument("--dry-run", action="store_true", help="仅显示选中的模型，不加载模型或执行评测")
    args = parser.parse_args()
    models = select_models()
    print(f"本次选择 {len(models)} 个模型 | split={SPLIT} | samples={NUM_SAMPLES or 'all'}", flush=True)
    for config in models:
        print(f"  [{config['type']}] {config['name']}: {config['path']}", flush=True)
    if args.dry_run:
        return []
    module_path = Path(__file__).with_name("08_benchmark.py")
    spec = importlib.util.spec_from_file_location("benchmark_interface", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [module.benchmark_model(config, split=SPLIT, num_samples=NUM_SAMPLES,
                            max_new_tokens=MAX_NEW_TOKENS, output_root=OUTPUT_ROOT)
            for config in models]


if __name__ == "__main__":
    main()
