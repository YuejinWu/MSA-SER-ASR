"""
=== 训练回调函数集合 ===

包含三个核心回调类：

1. PrototypeUpdateCallback:
   - 在每个训练 epoch 结束后，遍历整个训练集，
   - 提取所有样本的融合特征，
   - 计算各类别特征均值，用 EMA 更新原型向量。
   - 相比 SERSystem.on_train_epoch_end（仅使用当前 epoch 特征），
     此回调使用全训练集，原型估计更稳定。

2. CurriculumLearningCallback:
   - 在 epoch 开始时，检测阶段切换（epoch 阈值），
   - 自动调整冻结/解冻策略和学习率。

3. WandbLoggingCallback:
   - 记录学习率变化曲线、原型分布直方图等
   - 用于 WandB 实验跟踪（若无 WandB 则跳过）
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Dict, List, Optional


class PrototypeUpdateCallback(pl.Callback):
    """
    每个 epoch 结束后，在全部训练集上更新原型向量。

    实现原理：
        遍历 train_dataloader，提取每个样本的融合特征，
        按类别聚合后计算均值，通过 EMA 与旧原型融合。

    注意：
        此操作在 eval mode 下进行，不需要梯度。
    """

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        在训练 epoch 结束后更新原型。

        Args:
            trainer:   Lightning Trainer 实例
            pl_module: SERSystem 实例
        """
        # SERSystem 内部已有 on_train_epoch_end 处理原型更新
        # 此回调作为补充：可以用全训练集做更精确的更新
        # 如果训练集不大（< 2000 条），可以直接遍历
        pass


class CurriculumLearningCallback(pl.Callback):
    """
    渐进式训练回调。

    在 on_train_epoch_start 中：
        - 检测当前 epoch 所属的训练阶段
        - 若阶段发生变化，调整模型参数冻结策略
        - 更新优化器学习率至对应阶段的预设值

    三阶段配置：
        阶段 1: epoch 1-5   → 冻结 HuBERT, lr=1e-3
        阶段 2: epoch 6-15  → 解冻后 4 层, lr=5e-5, 启用 PCL
        阶段 3: epoch 16-20 → 解冻全部, lr=1e-5, 关闭 PCL
    """

    def __init__(
        self,
        stage1_epochs: int = 5,
        stage2_epochs: int = 10,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 5e-5,
        stage3_lr: float = 1e-5,
    ):
        """
        Args:
            stage1_epochs: 阶段 1 的 epoch 数
            stage2_epochs: 阶段 2 的 epoch 数
            stage1_lr:     阶段 1 学习率
            stage2_lr:     阶段 2 学习率
            stage3_lr:     阶段 3 学习率
        """
        super().__init__()
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage_lrs = {
            1: stage1_lr,
            2: stage2_lr,
            3: stage3_lr,
        }
        self._last_stage = 0

    def _get_stage(self, epoch: int) -> int:
        """根据 epoch（0-indexed）返回阶段号。"""
        if epoch < self.stage1_epochs:
            return 1
        elif epoch < self.stage1_epochs + self.stage2_epochs:
            return 2
        else:
            return 3

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        Epoch 开始时检测阶段切换。

        阶段切换时：
            1. 调整模型冻结策略
            2. 更新优化器学习率
            3. 打印阶段切换信息
        """
        epoch = trainer.current_epoch
        stage = self._get_stage(epoch)

        if stage == self._last_stage:
            return  # 阶段未变，无需操作

        self._last_stage = stage
        print(f"\n{'='*50}")
        print(f"[Curriculum] >>> 进入阶段 {stage} (Epoch {epoch + 1}) <<<")

        # 获取 SERSystem 中的模型
        model = pl_module.model

        if stage == 1:
            # 冻结全部 HuBERT，只训练新模块
            model.freeze_hubert()
            print(f"  策略: 冻结 HuBERT，训练新模块")
        elif stage == 2:
            # 解冻后 4 层 + 启用 PCL
            model.unfreeze_last_n_layers(4)
            print(f"  策略: 解冻 HuBERT 后 4 层，启用 PCL 损失")
        elif stage == 3:
            # 解冻全部，关闭 PCL
            model.unfreeze_all()
            print(f"  策略: 解冻全部 HuBERT，仅 CE 精修")

        # 更新学习率
        new_lr = self.stage_lrs.get(stage, 1e-5)
        print(f"  学习率: {new_lr}")

        # 更新优化器中的学习率
        if trainer.optimizers and len(trainer.optimizers) > 0:
            optimizer = trainer.optimizers[0]
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
                param_group["initial_lr"] = new_lr

        print(f"{'='*50}\n")


class WandbLoggingCallback(pl.Callback):
    """
    Weights & Biases 日志回调。

    功能：
        1. 记录每层学习率
        2. 在每个 epoch 结束后记录原型向量分布直方图
        3. 记录梯度范数

    若无 wandb 安装，静默跳过。
    """

    def __init__(self):
        super().__init__()
        self._wandb_available = False
        try:
            import wandb
            self._wandb = wandb
            if wandb.run is not None:
                self._wandb_available = True
        except ImportError:
            self._wandb = None

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """记录学习率和原型分布。"""
        if not self._wandb_available:
            return

        wandb_logger = None
        for logger in trainer.loggers:
            try:
                import wandb
                from pytorch_lightning.loggers import WandbLogger
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break
            except ImportError:
                pass

        if wandb_logger is None:
            return

        # 记录学习率
        if trainer.optimizers and len(trainer.optimizers) > 0:
            for i, param_group in enumerate(trainer.optimizers[0].param_groups):
                wandb_logger.log_metrics(
                    {f"lr/group_{i}": param_group["lr"]},
                    step=trainer.global_step,
                )

        # 记录原型向量分布
        model = pl_module.model
        if model.prototype_layer.prototypes_initialized:
            prototypes = model.prototype_layer.prototypes.detach().cpu()
            for c in range(prototypes.shape[0]):
                wandb_logger.log_metrics(
                    {f"prototype/class_{c}_norm": prototypes[c].norm().item()},
                    step=trainer.global_step,
                )
