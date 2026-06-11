"""=== 语音情感识别系统 (PyTorch Lightning Module) ===

核心职责：
    1. 封装 MultiScaleHuBERT 模型的训练/验证/测试逻辑
    2. HuBERT 冻结 + Adapter 微调
    3. Cosine Annealing 学习率调度 + CrossEntropy + Label Smoothing
"""

import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from typing import Dict, Tuple, Any, Optional

from src.models.multiscale_hubert import MultiScaleHuBERT


# 三数据集混合后各类大致均衡
_CLASS_COUNTS = [1, 1, 1, 1, 1, 1]

def _compute_class_weights(counts: list, num_classes: int) -> torch.Tensor:
    total = sum(counts)
    weights = [total / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


class SERSystem(pl.LightningModule):
    """语音情感识别 Lightning 系统。"""

    def __init__(self, model_cfg: Dict[str, Any], train_cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.model = MultiScaleHuBERT(
            backbone=model_cfg.get("backbone", "facebook/hubert-base-ls960"),
            hidden_size=model_cfg.get("hidden_size", 768),
            num_classes=model_cfg.get("num_classes", 6),
            local_conv_channels=model_cfg.get("local_conv_channels", [384, 384]),
            attention_dim=model_cfg.get("attention_dim", 64),
            dropout=model_cfg.get("dropout", 0.4),
            adapter_bottleneck=model_cfg.get("adapter_bottleneck", 64),
            temperature=model_cfg.get("temperature", 0.3),
            prototype_momentum=model_cfg.get("prototype_momentum", 0.9),
            local_model_path=model_cfg.get("local_model_path", ""),
            use_mirror=model_cfg.get("use_mirror", True),
            mirror_endpoint=model_cfg.get("mirror_endpoint", "https://hf-mirror.com"),
        )

        self.model.freeze_hubert()
        self.num_classes = model_cfg.get("num_classes", 6)

        label_smoothing = train_cfg.get("label_smoothing", 0.1)
        class_weights = _compute_class_weights(_CLASS_COUNTS, self.num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing)

        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="weighted")
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_f1_weighted = F1Score(task="multiclass", num_classes=self.num_classes, average="weighted")
        self.val_f1_macro = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_f1_weighted = F1Score(task="multiclass", num_classes=self.num_classes, average="weighted")
        self.test_f1_macro = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

    def forward(self, input_values):
        return self.model(input_values)

    def training_step(self, batch, batch_idx):
        input_values, labels = batch["input_values"], batch["labels"]
        logits, _, _ = self.model(input_values)
        loss = self.ce_loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc.update(preds, labels)
        self.train_f1.update(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train/weighted_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_values, labels = batch["input_values"], batch["labels"]
        logits, _, _ = self.model(input_values)
        loss = self.ce_loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc.update(preds, labels)
        self.val_f1_weighted.update(preds, labels)
        self.val_f1_macro.update(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/weighted_f1", self.val_f1_weighted, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/macro_f1", self.val_f1_macro, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        input_values, labels = batch["input_values"], batch["labels"]
        logits, _, _ = self.model(input_values)
        preds = torch.argmax(logits, dim=-1)
        self.test_acc.update(preds, labels)
        self.test_f1_weighted.update(preds, labels)
        self.test_f1_macro.update(preds, labels)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/weighted_f1", self.test_f1_weighted, on_step=False, on_epoch=True)
        self.log("test/macro_f1", self.test_f1_macro, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.train_cfg.get("lr", 5e-4)
        weight_decay = self.train_cfg.get("weight_decay", 0.02)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.train_cfg.get("warmup_steps", 800)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
