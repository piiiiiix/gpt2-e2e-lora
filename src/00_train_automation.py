"""修改 R_VALUES 后运行，每个 R 自动训练三个 seed 的模型。"""

import importlib.util
from pathlib import Path

R_VALUES = [1, 2, 4, 8, 16, 64]


def main():
    # 文件名以数字开头，使用 importlib 加载接口模块。
    module_path = Path(__file__).with_name("06_00_auto_train_lora.py")
    spec = importlib.util.spec_from_file_location("auto_train_lora", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for R in R_VALUES:
        saved_paths = module.train_lora(R)
        print(f"R={R} 完成：{saved_paths}")


if __name__ == "__main__":
    main()
