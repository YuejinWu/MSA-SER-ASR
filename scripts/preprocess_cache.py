"""=== 音频预处理缓存脚本 ===

一次性将所有音频文件处理为固定格式 tensor 并缓存到磁盘，
后续训练直接从缓存加载，无需重复重采样+归一化。

用法: python preprocess_cache.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

from src.data.mixed_datamodule import (
    _parse_ravdess, _parse_cremad, _parse_tess,
    NUM_CLASSES, COMMON_EMOTIONS,
)

# 配置
SAMPLE_RATE = 16000
MAX_LENGTH = 80000
CACHE_DIR = Path("data/cache")


def process_and_save(sample, cache_dir: Path) -> bool:
    """处理单个音频样本并保存为 .pt 文件。返回是否成功。"""
    cache_path = cache_dir / f"{sample.speaker_id}_{sample.file_path.stem}.pt"
    if cache_path.exists():
        return True  # 已缓存

    try:
        try:
            waveform, orig_sr = torchaudio.load(str(sample.file_path), backend="soundfile")
        except Exception:
            waveform, orig_sr = torchaudio.load(str(sample.file_path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != SAMPLE_RATE:
            waveform = AF.resample(waveform, orig_sr, SAMPLE_RATE)
        waveform = waveform.squeeze(0)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        if waveform.shape[0] < MAX_LENGTH:
            waveform = torch.cat([waveform, torch.zeros(MAX_LENGTH - waveform.shape[0])])
        else:
            waveform = waveform[:MAX_LENGTH]

        torch.save({"waveform": waveform, "label": sample.label}, cache_path)
        return True
    except Exception as e:
        print(f"  [错误] {sample.file_path}: {e}")
        return False


def main():
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有样本
    all_samples = []
    data_roots = {
        "RAVDESS": Path("data/speech-emotion-recognition-ravdess-data"),
        "CREMA-D": Path("data/CREMA-D"),
        "TESS": Path("data/TESS Toronto emotional speech set data"),
    }

    for name, root in data_roots.items():
        if not root.exists():
            print(f"[跳过] {name}: 目录不存在 ({root})")
            continue
        if name == "RAVDESS":
            samples = _parse_ravdess(root)
        elif name == "CREMA-D":
            samples = _parse_cremad(root)
        else:
            samples = _parse_tess(root)
        all_samples.extend(samples)
        print(f"[{name}] {len(samples)} 样本")

    print(f"\n总计 {len(all_samples)} 个样本，开始预处理...")
    print(f"缓存目录: {cache_dir}\n")

    success = 0
    fail = 0
    for sample in tqdm(all_samples, desc="预处理"):
        if process_and_save(sample, cache_dir):
            success += 1
        else:
            fail += 1

    print(f"\n完成！成功: {success}, 失败: {fail}")
    print(f"缓存大小: {sum(f.stat().st_size for f in cache_dir.glob('*.pt')) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
