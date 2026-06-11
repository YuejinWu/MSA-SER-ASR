"""=== 多数据集混合 DataModule ===

支持 RAVDESS + CREMA-D + TESS 联合训练。
自动检测预处理缓存，有缓存时直接从 .pt 加载（零 CPU 开销）。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# ---- 统一的情感标签映射 (6 类) ----
COMMON_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
NUM_CLASSES = 6

_RAVDESS_MAP = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
_CREMA_MAP = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}
_TESS_FOLDER_MAP = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "fear": 4, "disgust": 5}


class AudioSample:
    def __init__(self, file_path: Path, label: int, speaker_id: str):
        self.file_path = file_path
        self.label = label
        self.speaker_id = speaker_id


# ================================================================
# 数据集解析
# ================================================================

def _parse_ravdess(data_dir: Path) -> List[AudioSample]:
    samples = []
    for wav_file in sorted(data_dir.rglob("*.wav")):
        if "Actor_" not in str(wav_file):
            continue
        parts = wav_file.stem.split("-")
        if len(parts) < 7:
            continue
        emotion_code = int(parts[2])
        if emotion_code not in _RAVDESS_MAP:
            continue
        samples.append(AudioSample(wav_file, _RAVDESS_MAP[emotion_code], f"RAV_{parts[6]}"))
    return samples


def _parse_cremad(data_dir: Path) -> List[AudioSample]:
    samples = []
    for wav_file in sorted(data_dir.rglob("*.wav")):
        parts = wav_file.stem.split("_")
        if len(parts) < 3:
            continue
        emotion = parts[2].upper()
        if emotion not in _CREMA_MAP:
            continue
        samples.append(AudioSample(wav_file, _CREMA_MAP[emotion], f"CRE_{parts[0]}"))
    return samples


def _parse_tess(data_dir: Path) -> List[AudioSample]:
    samples = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_lower = folder.name.lower()
        matched_label = None
        for keyword, label in _TESS_FOLDER_MAP.items():
            if keyword in folder_lower and "surprise" not in folder_lower:
                matched_label = label
                break
        if matched_label is None:
            continue
        speaker_id = folder.name.split("_")[0]
        for wav_file in sorted(folder.glob("*.wav")):
            samples.append(AudioSample(wav_file, matched_label, f"TESS_{speaker_id}"))
    return samples


# ================================================================
# PyTorch Datasets
# ================================================================

class MixedAudioDataset(Dataset):
    """实时加载并预处理 WAV（无缓存时使用）。"""

    def __init__(self, samples: List[AudioSample], sample_rate=16000, max_length=80000):
        self.samples = samples
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            waveform, orig_sr = torchaudio.load(str(s.file_path), backend="soundfile")
        except Exception:
            waveform, orig_sr = torchaudio.load(str(s.file_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_sr, self.sample_rate)
        waveform = waveform.squeeze(0)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        if waveform.shape[0] < self.max_length:
            waveform = torch.cat([waveform, torch.zeros(self.max_length - waveform.shape[0])])
        else:
            waveform = waveform[:self.max_length]
        return {"input_values": waveform, "labels": torch.tensor(s.label, dtype=torch.long),
                "file_path": str(s.file_path)}


class CachedAudioDataset(Dataset):
    """从预处理缓存 .pt 文件直接加载（极快）。"""

    def __init__(self, cache_dir: Path, samples: List[AudioSample]):
        self.cache_dir = cache_dir
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cache_path = self.cache_dir / f"{s.speaker_id}_{s.file_path.stem}.pt"
        data = torch.load(cache_path, weights_only=True)
        return {"input_values": data["waveform"], "labels": torch.tensor(s.label, dtype=torch.long),
                "file_path": str(s.file_path)}


# ================================================================
# Lightning DataModule
# ================================================================

class MixedDataModule(pl.LightningDataModule):
    """RAVDESS + CREMA-D + TESS 混合数据模块，自动检测缓存。"""

    def __init__(self, ravdess_dir: str, cremad_dir: str, tess_dir: str,
                 sample_rate=16000, max_length=80000, batch_size=16, num_workers=4,
                 train_ratio=0.7, val_ratio=0.15, cache_dir: str = "data/cache"):
        super().__init__()
        self.ravdess_dir = Path(ravdess_dir)
        self.cremad_dir = Path(cremad_dir)
        self.tess_dir = Path(tess_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.cache_dir = Path(cache_dir)

        self.train_samples: List[AudioSample] = []
        self.val_samples: List[AudioSample] = []
        self.test_samples: List[AudioSample] = []

    def _split_speakers(self, samples: List[AudioSample]):
        speaker_set = sorted(set(s.speaker_id for s in samples))
        n = len(speaker_set)
        n_train = max(1, int(n * self.train_ratio))
        n_val = max(1, int(n * self.val_ratio))
        train_spks = set(speaker_set[:n_train])
        val_spks = set(speaker_set[n_train:n_train + n_val])
        test_spks = set(speaker_set[n_train + n_val:])
        train = [s for s in samples if s.speaker_id in train_spks]
        val = [s for s in samples if s.speaker_id in val_spks]
        test = [s for s in samples if s.speaker_id in test_spks]
        return train, val, test

    def prepare_data(self):
        self.train_samples.clear()
        self.val_samples.clear()
        self.test_samples.clear()

        print(f"\n[MixedData] 加载数据集...")
        for name, root, parser in [
            ("RAVDESS", self.ravdess_dir, _parse_ravdess),
            ("CREMA-D", self.cremad_dir, _parse_cremad),
            ("TESS", self.tess_dir, _parse_tess),
        ]:
            if not root.exists():
                print(f"  [跳过] {name}: 目录不存在")
                continue
            all_s = parser(root)
            tr, va, te = self._split_speakers(all_s)
            self.train_samples.extend(tr)
            self.val_samples.extend(va)
            self.test_samples.extend(te)
            print(f"  {name}: {len(all_s)} 样本 ({len(tr)}/{len(va)}/{len(te)} train/val/test)")

        total = len(self.train_samples) + len(self.val_samples) + len(self.test_samples)
        print(f"  合计: {total} 样本 "
              f"({len(self.train_samples)}/{len(self.val_samples)}/{len(self.test_samples)} train/val/test)")

        # 检测缓存
        cache_exists = self.cache_dir.exists() and any(self.cache_dir.glob("*.pt"))
        if cache_exists:
            cached = sum(1 for _ in self.cache_dir.glob("*.pt"))
            print(f"  [缓存] 检测到 {cached} 个 .pt 文件，将使用缓存加载")
        else:
            print(f"  [缓存] 未检测到，将实时预处理音频")
            print(f"  提示: 运行 python preprocess_cache.py 可大幅加速训练")

    def setup(self, stage: Optional[str] = None):
        use_cache = self.cache_dir.exists() and any(self.cache_dir.glob("*.pt"))
        ds_cls = CachedAudioDataset if use_cache else MixedAudioDataset

        if stage == "fit" or stage is None:
            ds_args = (self.cache_dir, self.train_samples) if use_cache else (self.train_samples, self.sample_rate, self.max_length)
            self.train_dataset = ds_cls(*ds_args)
            ds_args = (self.cache_dir, self.val_samples) if use_cache else (self.val_samples, self.sample_rate, self.max_length)
            self.val_dataset = ds_cls(*ds_args)

        if stage == "test" or stage is None:
            ds_args = (self.cache_dir, self.test_samples) if use_cache else (self.test_samples, self.sample_rate, self.max_length)
            self.test_dataset = ds_cls(*ds_args)

    def _dl(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=shuffle, persistent_workers=(self.num_workers > 0))

    def train_dataloader(self):
        return self._dl(self.train_dataset, True)

    def val_dataloader(self):
        return self._dl(self.val_dataset, False)

    def test_dataloader(self):
        return self._dl(self.test_dataset, False)
