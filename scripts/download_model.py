"""
=== HuBERT 模型下载工具 ===

用于手动下载 facebook/hubert-base-ls960 预训练权重到本地 models/ 目录，
适用于网络受限或离线环境。

使用方法：
    python download_model.py

下载内容（约 360 MB）：
    - config.json        (模型配置)
    - preprocessor_config.json
    - pytorch_model.bin  (预训练权重，~360 MB)
    - tokenizer.json, tokenizer_config.json, special_tokens_map.json

环境变量：
    HF_ENDPOINT: 设置镜像站点，如 https://hf-mirror.com
"""

import os
import sys
from pathlib import Path

# ⚠️ 必须在导入 transformers/huggingface_hub 之前设置镜像端点
# 否则 HF_ENDPOINT 环境变量不会生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")  # 5 分钟超时
print(f"[镜像] HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "hubert-base-ls960"
MODEL_NAME = "facebook/hubert-base-ls960"


def download_from_huggingface():
    """从 HuggingFace Hub 下载模型。"""
    try:
        from transformers import HubertModel, HubertConfig, Wav2Vec2FeatureExtractor
    except ImportError:
        print("[错误] 请先安装 transformers: pip install transformers")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    if (MODEL_DIR / "pytorch_model.bin").exists():
        print(f"[提示] 模型已存在于: {MODEL_DIR}")
        print(f"  如需重新下载，请先删除此目录")
        return True

    print(f"[下载] 正在下载 {MODEL_NAME} ...")
    print(f"  目标目录: {MODEL_DIR}")
    print(f"  大小约 360 MB，请耐心等待...")
    print(f"  如遇网络问题，可设置镜像: set HF_ENDPOINT=https://hf-mirror.com")
    print()

    try:
        # 下载模型权重和配置
        model = HubertModel.from_pretrained(MODEL_NAME)
        model.save_pretrained(str(MODEL_DIR))
        print(f"  [OK] pytorch_model.bin 已保存")

        # 下载预处理配置
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        feature_extractor.save_pretrained(str(MODEL_DIR))
        print(f"  [OK] preprocessor_config.json 已保存")

        print(f"\n{'='*60}")
        print(f"[完成] 模型已下载到: {MODEL_DIR}")
        print(f"")
        print(f"  请在 configs/model/hubert_multiscale.yaml 中确认:")
        print(f"    local_model_path: \"models/hubert-base-ls960\"")
        print(f"")
        print(f"  现在可以离线运行训练:")
        print(f"    python train.py")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        print(f"\n可能的原因和解决方案:")
        print(f"  1. 网络不通 → 设置镜像: set HF_ENDPOINT=https://hf-mirror.com")
        print(f"  2. 防火墙阻止 → 关闭代理或 VPN 后重试")
        print(f"  3. 超时 → 增加超时: set HF_HUB_DOWNLOAD_TIMEOUT=300")
        print(f"  4. 手动下载 → 浏览器访问:")
        print(f"     https://huggingface.co/facebook/hubert-base-ls960/tree/main")
        print(f"     下载所有文件到: {MODEL_DIR}")
        return False


def download_with_snapshot():
    """使用 snapshot_download 下载（备选方案）。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[错误] 请先安装 huggingface_hub: pip install huggingface_hub")
        return False

    print(f"[下载] 使用 snapshot_download 下载 {MODEL_NAME} ...")
    print(f"  目标目录: {MODEL_DIR}")

    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"\n[完成] 模型已下载到: {MODEL_DIR}")
        print(f"  现在可以运行: python train.py")
        return True
    except Exception as e:
        print(f"[错误] snapshot_download 失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PCL-MS-HuBERT-SER | HuBERT 模型下载工具")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"目标: {MODEL_DIR}")
    print()

    # 检查镜像站点
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"[镜像] 使用: {endpoint}")
    else:
        print("[提示] 若下载缓慢或失败，可设置镜像站点:")
        print("  Windows: set HF_ENDPOINT=https://hf-mirror.com")
        print("  Linux:   export HF_ENDPOINT=https://hf-mirror.com")

    print()

    # 尝试下载
    success = download_from_huggingface()
    if not success:
        print("\n尝试备选方案 (snapshot_download)...")
        success = download_with_snapshot()

    if not success:
        print(f"\n[手动方案] 请用浏览器下载以下文件并放入 {MODEL_DIR}:")
        print(f"  https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin")
        print(f"  https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json")
        print(f"  https://huggingface.co/facebook/hubert-base-ls960/resolve/main/preprocessor_config.json")
        sys.exit(1)
