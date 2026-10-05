"""扫描训练输出的全部 LoRA 模型，每次传一个模型给 08 保存日志和五项得分图。"""
import importlib.util
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 对应 06_00_auto_train_lora.py 的 OUTPUT_ROOT / "final"。
MODEL_ROOT = PROJECT_ROOT / "outputs/gpt2-e2e-lora/final"
SPLIT = "validation"
NUM_SAMPLES = None  # None = 完整 split；设为 100 可先做快速评测。
MAX_NEW_TOKENS = 60
OUTPUT_ROOT = PROJECT_ROOT / "outputs/benchmarks"


def discover_models(model_root=MODEL_ROOT):
    """adapter_config.json 标识已保存的模型；支持 r{R}/seed{seed} 和旧目录。"""
    root = Path(model_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.is_dir():
        raise FileNotFoundError(f"找不到训练模型目录：{root}")

    def natural_key(path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", path.relative_to(root).as_posix())]

    models = []
    for config_path in sorted(root.rglob("adapter_config.json"), key=natural_key):
        model_path = config_path.parent
        relative = model_path.relative_to(root)
        name = "_".join(relative.parts) if relative.parts else root.name
        models.append({"name": name, "type": "lora", "path": str(model_path)})
    if not models:
        raise FileNotFoundError(f"没有找到已保存的 LoRA 模型：{root}")
    return models


def main():
    models = discover_models()
    print(f"扫描到 {len(models)} 个模型：", flush=True)
    for model_config in models:
        print(f"  {model_config['name']}: {model_config['path']}", flush=True)
    module_path = Path(__file__).with_name("08_benchmark.py")
    spec = importlib.util.spec_from_file_location("benchmark_interface", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved_paths = []
    for model_config in models:
        saved_paths.append(module.benchmark_model(
            model_config, split=SPLIT, num_samples=NUM_SAMPLES,
            max_new_tokens=MAX_NEW_TOKENS, output_root=OUTPUT_ROOT))
    return saved_paths


if __name__ == "__main__":
    main()
