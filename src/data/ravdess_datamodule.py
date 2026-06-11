"""
=== RAVDESS 数据集模块 (PyTorch Lightning DataModule) ===

RAVDESS 数据集：
    - 24 位演员（12 男 12 女），每人 60 条语音，共 1440 条
    - 48kHz / 16bit WAV，每条 3-5 秒
    - 8 种情感：neutral(1), calm(2), happy(3), sad(4),
                 angry(5), fearful(6), disgust(7), surprised(8)

文件名格式：
    03-01-01-01-01-01-01.wav
    前 3 个数字含义：
      第 1 位: 模态 (03 = 语音)
      第 2 位: 声道 (01 = 语音-only)
      第 3 位: 情感标签 (01-08)

数据划分策略：
    严格按说话人 ID（第 7 位数字）划分训练/验证/测试集，
    确保同一说话人不会同时出现在不同集合中。
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class RAVDESSSample:
    """RAVDESS 单条样本的数据结构。"""

    def __init__(
        self,
        file_path: Path,
        emotion_label: int,     # 0-7（内部标签，从 1-8 映射而来）
        speaker_id: str,
    ):
        self.file_path = file_path
        self.emotion_label = emotion_label
        self.speaker_id = speaker_id


class RAVDESSDataset(Dataset):
    """
    RAVDESS PyTorch Dataset。

    加载音频并预处理为固定长度 tensor。
    """

    def __init__(
        self,
        samples: List[RAVDESSSample],
        sample_rate: int = 16000,
        max_length: int = 80000,
    ):
        """
        Args:
            samples:     样本列表
            sample_rate: 目标采样率（Hz）
            max_length:  最大采样点数（不足补零，超长截断）
        """
        self.samples = samples
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        sample = self.samples[idx]

        try:
            # soundfile 后端加载音频，确保跨平台兼容
            waveform, orig_sr = torchaudio.load(
                str(sample.file_path),
                backend="soundfile",
            )
        except Exception as e:
            # 回退到默认后端
            waveform, orig_sr = torchaudio.load(
                str(sample.file_path),
            )

        # 转单声道（取平均）
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # shape: (1, num_samples)

        # 重采样到目标采样率
        if orig_sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_sr, self.sample_rate)
        # shape: (1, num_samples)

        # 去除通道维度
        waveform = waveform.squeeze(0)  # (num_samples,)

        # 归一化到 [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # Padding 或截断到固定长度
        if waveform.shape[0] < self.max_length:
            # 补零
            padding = torch.zeros(self.max_length - waveform.shape[0])
            waveform = torch.cat([waveform, padding])
        else:
            # 截断
            waveform = waveform[: self.max_length]

        return {
            "input_values": waveform,
            "labels": torch.tensor(sample.emotion_label, dtype=torch.long),
            "speaker_id": sample.speaker_id,
            "file_path": str(sample.file_path),
        }


class RAVDESSDataModule(pl.LightningDataModule):
    """
    RAVDESS 数据集 Lightning DataModule。

    职责：
        1. prepare_data(): 扫描数据目录，按说话人划分，保存为 CSV
        2. setup(): 加载 CSV，构建 Dataset
        3. train/val/test_dataloader(): 返回对应 DataLoader
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        max_length: int = 80000,
        batch_size: int = 16,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        """
        Args:
            data_dir:     RAVDESS 根目录（包含 Actor_01 ~ Actor_24 文件夹）
            sample_rate:  重采样目标采样率
            max_length:   最大采样点数
            batch_size:   批次大小
            num_workers:  DataLoader 工作进程数
            train_ratio:  训练集比例
            val_ratio:    验证集比例
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train_samples: List[RAVDESSSample] = []
        self.val_samples: List[RAVDESSSample] = []
        self.test_samples: List[RAVDESSSample] = []

        # 情感标签映射：文件名中的数字 1-8 → 0-7
        self.emotion_map = {
            1: 0, 2: 1, 3: 2, 4: 3,
            5: 4, 6: 5, 7: 6, 8: 7,
        }

        # 情感名称（中文，用于可视化）
        self.emotion_names = [
            "neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised",
        ]

    def prepare_data(self):
        """
        扫描 RAVDESS 目录，按说话人 ID 划分数据集。
        此方法在单个进程中调用，不分布式执行。
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"数据目录不存在: {self.data_dir}\n"
                f"请从 https://zenodo.org/record/1188976 下载 RAVDESS 数据集"
            )

        # 收集所有音频文件
        all_samples: List[RAVDESSSample] = []
        speaker_set: set[str] = set()

        for wav_file in sorted(self.data_dir.rglob("*.wav")):
            # 文件名格式: 03-01-01-01-01-01-01.wav
            parts = wav_file.stem.split("-")
            if len(parts) < 7:
                continue

            emotion_code = int(parts[2])
            speaker_id = parts[6]  # 第 7 位：说话人 ID (01-24)

            if emotion_code not in self.emotion_map:
                continue

            emotion_label = self.emotion_map[emotion_code]
            all_samples.append(
                RAVDESSSample(
                    file_path=wav_file,
                    emotion_label=emotion_label,
                    speaker_id=speaker_id,
                )
            )
            speaker_set.add(speaker_id)

        # 按说话人 ID 排序，确保划分可复现
        speakers = sorted(speaker_set, key=lambda x: int(x))
        num_speakers = len(speakers)
        n_train = max(1, int(num_speakers * self.train_ratio))
        n_val = max(1, int(num_speakers * self.val_ratio))

        # 严格划分：前 n_train 个说话人 → 训练，中间 n_val 个 → 验证，其余 → 测试
        train_speakers = set(speakers[:n_train])
        val_speakers = set(speakers[n_train : n_train + n_val])
        test_speakers = set(speakers[n_train + n_val :])

        self.train_samples = [
            s for s in all_samples if s.speaker_id in train_speakers
        ]
        self.val_samples = [
            s for s in all_samples if s.speaker_id in val_speakers
        ]
        self.test_samples = [
            s for s in all_samples if s.speaker_id in test_speakers
        ]

        print(f"[RAVDESS] 总样本: {len(all_samples)}")
        print(f"  说话人数: {num_speakers}")
        print(f"  训练: {len(self.train_samples)} 条 "
              f"({len(train_speakers)} 说话人)")
        print(f"  验证: {len(self.val_samples)} 条 "
              f"({len(val_speakers)} 说话人)")
        print(f"  测试: {len(self.test_samples)} 条 "
              f"({len(test_speakers)} 说话人)")

    def setup(self, stage: Optional[str] = None):
        """构建各阶段的 Dataset 对象。"""
        # Dataset 在 setup 中创建，支持多 GPU 分布式
        if stage == "fit" or stage is None:
            self.train_dataset = RAVDESSDataset(
                self.train_samples,
                sample_rate=self.sample_rate,
                max_length=self.max_length,
            )
            self.val_dataset = RAVDESSDataset(
                self.val_samples,
                sample_rate=self.sample_rate,
                max_length=self.max_length,
            )

        if stage == "test" or stage is None:
            self.test_dataset = RAVDESSDataset(
                self.test_samples,
                sample_rate=self.sample_rate,
                max_length=self.max_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # 丢弃不完整批次，避免 batch norm 问题
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
