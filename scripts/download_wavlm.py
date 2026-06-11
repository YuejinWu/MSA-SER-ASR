"""下载 WavLM .bin 并转 safetensors（镜像 + 本地转换）"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download
from pathlib import Path
import torch

save_dir = Path("models/wavlm-base-plus")
save_dir.mkdir(parents=True, exist_ok=True)

# 下载 pytorch_model.bin
print("下载 pytorch_model.bin (378 MB)...")
bin_path = hf_hub_download(
    "microsoft/wavlm-base-plus",
    "pytorch_model.bin",
    local_dir=str(save_dir),
)
print(f"OK: {bin_path}")

# 转 safetensors
print("转换 model.safetensors...")
state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
from safetensors.torch import save_file
save_file(state_dict, str(save_dir / "model.safetensors"))
os.remove(bin_path)
print("Done!")
