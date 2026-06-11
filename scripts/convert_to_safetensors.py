"""
=== 模型格式转换工具 ===

将 pytorch_model.bin 转换为 safetensors 格式，
以兼容新版 transformers（需要 torch >= 2.6 才能加载 .bin 文件）。

此脚本只需运行一次：
    python convert_to_safetensors.py
"""

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models" / "hubert-base-ls960"


def convert():
    """将 pytorch_model.bin 转换为 model.safetensors"""
    bin_path = MODEL_DIR / "pytorch_model.bin"
    safetensors_path = MODEL_DIR / "model.safetensors"

    if not bin_path.exists():
        print(f"[错误] 未找到 {bin_path}")
        print("请先运行 python download_model.py 下载模型")
        return False

    if safetensors_path.exists():
        print(f"[提示] safetensors 已存在: {safetensors_path}")
        return True

    print(f"[转换] {bin_path}")
    print(f"    → {safetensors_path}")
    print(f"  文件大小约 360 MB，请稍候...")

    try:
        import torch
        from safetensors.torch import save_file

        # 加载 pytorch_model.bin
        print("  加载 pytorch_model.bin ...")
        # 使用 weights_only=False 因为模型来自可信源
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=False)

        print(f"  保存为 safetensors 格式 ...")
        save_file(state_dict, str(safetensors_path))

        print(f"\n[完成] 转换成功！")
        print(f"  原始文件: {bin_path} ({bin_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  新文件:   {safetensors_path} ({safetensors_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"\n  现在可以运行训练:")
        print(f"    python train.py")
        return True

    except ImportError as e:
        print(f"\n[错误] 缺少依赖: {e}")
        print(f"  请安装: pip install safetensors")
        return False
    except Exception as e:
        print(f"\n[错误] 转换失败: {e}")
        return False


def install_safetensors_check():
    """检查并提示安装 safetensors"""
    try:
        import safetensors
        print(f"[OK] safetensors 已安装 (version {safetensors.__version__})")
        return True
    except ImportError:
        print(f"[提示] safetensors 未安装")
        print(f"  正在安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "-q"])
        print(f"  [OK] safetensors 安装完成")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("PCL-MS-HuBERT-SER | 模型格式转换 (bin → safetensors)")
    print("=" * 60)

    if not MODEL_DIR.exists():
        print(f"\n[错误] 模型目录不存在: {MODEL_DIR}")
        print("请先运行: python download_model.py")
        sys.exit(1)

    install_safetensors_check()
    success = convert()

    if not success:
        print(f"\n手动转换方案:")
        print(f"  方法 1: pip install safetensors && python convert_to_safetensors.py")
        print(f"  方法 2: 升级 PyTorch: pip install torch>=2.6.0")
        print(f"  方法 3: 从镜像重新下载 safetensors 版本")
        print(f"    https://hf-mirror.com/facebook/hubert-base-ls960/resolve/main/model.safetensors")
        print(f"    放入: {MODEL_DIR}")
        sys.exit(1)
