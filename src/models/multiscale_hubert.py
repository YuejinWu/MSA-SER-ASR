"""=== 多尺度语音情感识别模型（HuBERT / WavLM 双主干 + Adapter 微调）==="""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, WavLMModel
from typing import Tuple, Optional

from src.models.components.temporal_attention import TemporalAttention
from src.models.components.prototype_layer import PrototypeLayer


class Adapter(nn.Module):
    """瓶颈 Adapter：残差连接 + 近零初始化。"""
    def __init__(self, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self._near_zero_init()

    def _near_zero_init(self):
        for m in [self.down, self.up]:
            nn.init.normal_(m.weight, std=1e-4)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


class MultiScaleHuBERT(nn.Module):
    """多尺度模型：CNN 局部 + Attention 全局 -> 融合 -> 分类"""

    def __init__(
        self,
        backbone: str = "facebook/hubert-base-ls960",
        hidden_size: int = 768,
        num_classes: int = 6,
        local_conv_channels: list[int] | None = None,
        attention_dim: int = 64,
        dropout: float = 0.4,
        adapter_bottleneck: int = 64,
        temperature: float = 0.3,
        prototype_momentum: float = 0.9,
        local_model_path: str = "",
        use_mirror: bool = True,
        mirror_endpoint: str = "https://hf-mirror.com",
    ):
        super().__init__()
        if local_conv_channels is None:
            local_conv_channels = [384, 384]
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # ---- 主干网络（HuBERT 或 WavLM）----
        self.backbone_name = backbone
        self.encoder = self._load_backbone(backbone, local_model_path, use_mirror, mirror_endpoint)

        # ---- 注入 Adapter ----
        self._adapter_handles = []
        self.adapters = nn.ModuleList()
        encoder_layers = self.encoder.encoder.layers
        for layer in encoder_layers:
            adapter = Adapter(hidden_size, adapter_bottleneck)
            self.adapters.append(adapter)
            handle = layer.register_forward_hook(
                self._make_adapter_hook(len(self.adapters) - 1)
            )
            self._adapter_handles.append(handle)

        # ---- 局部分支 ----
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_size, local_conv_channels[0], kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(local_conv_channels[0], local_conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ---- 全局分支 ----
        self.temporal_attention = TemporalAttention(hidden_dim=hidden_size, attention_dim=attention_dim)

        # ---- 融合层 ----
        fusion_input_dim = local_conv_channels[-1] + hidden_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # ---- 原型层 ----
        self.prototype_layer = PrototypeLayer(
            num_classes=num_classes, hidden_dim=hidden_size,
            temperature=temperature, momentum=prototype_momentum,
        )

    def _make_adapter_hook(self, idx: int):
        def hook(module, input, output):
            hidden = output[0]
            adapted = hidden + self.adapters[idx](hidden)
            return (adapted,) + output[1:]
        return hook

    @staticmethod
    def _load_backbone(backbone, local_model_path="", use_mirror=True, mirror_endpoint="https://hf-mirror.com"):
        is_wavlm = "wavlm" in backbone.lower()
        model_cls = WavLMModel if is_wavlm else HubertModel

        # ---- 尝试本地加载 ----
        candidate_paths = []
        if local_model_path and local_model_path.strip():
            p = Path(local_model_path.strip())
            if not p.is_absolute():
                project_root = Path(__file__).resolve().parent.parent.parent
                p = project_root / p
            candidate_paths.append(p)

        env_path = os.environ.get("HUBERT_LOCAL_PATH", "")
        if env_path:
            candidate_paths.append(Path(env_path))

        default_name = "wavlm-base-plus" if is_wavlm else "hubert-base-ls960"
        project_model = Path(__file__).resolve().parent.parent.parent / "models" / default_name
        candidate_paths.append(project_model)

        for local_path in candidate_paths:
            if not local_path.exists() or not (local_path / "config.json").exists():
                continue
            has_safetensors = (local_path / "model.safetensors").exists()
            has_bin_only = (not has_safetensors) and (local_path / "pytorch_model.bin").exists()
            if has_bin_only:
                print(f"\n[Backbone] 本地模型仅有 .bin 格式: {local_path}")
                print(f"  请先运行: python convert_to_safetensors.py")
                raise OSError("模型需要转换为 safetensors 格式。")
            print(f"[Backbone] 从本地路径加载: {local_path}")
            return model_cls.from_pretrained(str(local_path))

        # ---- 在线下载 ----
        if use_mirror and mirror_endpoint:
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = mirror_endpoint
            print(f"[Backbone] 使用镜像站点: {mirror_endpoint}")

        print(f"[Backbone] 尝试从 HuggingFace Hub 加载: {backbone}")
        try:
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
            model = model_cls.from_pretrained(backbone)
            print(f"[Backbone] 模型加载成功")
            return model
        except (OSError, RuntimeError, ValueError) as e:
            error_msg = str(e)
            if "10061" in error_msg or "Connection refused" in error_msg:
                print(f"\n[网络错误] 无法连接 HuggingFace，请先下载模型")
                raise OSError("无法连接 HuggingFace。") from e
            else:
                raise OSError(f"无法加载模型: {error_msg}") from e

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.encoder(input_values=input_values)
        features = output.last_hidden_state
        local_input = features.transpose(1, 2)
        local_feat = self.local_conv(local_input).squeeze(-1)
        global_feat, attn_weights = self.temporal_attention(features)
        fused = torch.cat([local_feat, global_feat], dim=-1)
        fused_features = self.fusion_layer(fused)
        logits = self.classifier(fused_features)
        return logits, fused_features, attn_weights

    def freeze_hubert(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True
