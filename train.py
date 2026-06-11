"""=== 训练入口脚本 ===
使用 Hydra 管理配置，一键启动完整训练流程：
    python train.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.mixed_datamodule import MixedDataModule
from src.systems.ser_system import SERSystem
from src.utils.callbacks import WandbLoggingCallback


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ---- 镜像设置 ----
    use_mirror = cfg.model.get("use_mirror", True)
    mirror_endpoint = cfg.model.get("mirror_endpoint", "https://hf-mirror.com")
    if use_mirror and mirror_endpoint:
        os.environ["HF_ENDPOINT"] = mirror_endpoint
        print(f"[镜像] HF_ENDPOINT = {mirror_endpoint}")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    # ---- 随机种子 ----
    pl.seed_everything(cfg.seed, workers=True)

    print("=" * 60)
    print("PCL-MS-HuBERT-SER | 多数据集训练")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # ---- 设备检测 ----
    if torch.cuda.is_available():
        accelerator = "cuda"
        print(f"\n[设备] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        print("\n[设备] Apple Metal (MPS)")
    else:
        accelerator = "cpu"
        print("\n[设备] CPU")

    # ---- DataModule ----
    datamodule = MixedDataModule(
        ravdess_dir=cfg.data.ravdess_dir,
        cremad_dir=cfg.data.cremad_dir,
        tess_dir=cfg.data.tess_dir,
        sample_rate=cfg.data.sample_rate,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        cache_dir="data/cache",
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # ---- 系统 ----
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    system = SERSystem(model_cfg=model_cfg, train_cfg=train_cfg)

    # ---- 回调 ----
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-{epoch:02d}-{val/weighted_f1:.4f}",
            monitor=cfg.train.monitor_metric,
            mode="max", save_top_k=3, save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.train.monitor_metric,
            patience=cfg.train.early_stop_patience,
            mode="max", verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        WandbLoggingCallback(),
    ]

    # ---- 日志 ----
    loggers = [TensorBoardLogger(save_dir="outputs", name="tensorboard")]
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(project=cfg.project_name, name=f"run-{cfg.seed}",
                                    save_dir="outputs", offline=True))
    except (ImportError, Exception):
        pass

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator, devices=1,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks, logger=loggers,
        log_every_n_steps=10, deterministic=True,
        precision="16-mixed" if accelerator == "cuda" else "32-true",
    )

    # ---- 训练 ----
    print(f"\n{'='*60}")
    print(f"开始训练 ({cfg.train.max_epochs} epochs)")
    print(f"{'='*60}\n")
    trainer.fit(model=system, datamodule=datamodule)

    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}")
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            if cb.best_model_path:
                print(f"\n最佳模型: {cb.best_model_path}")
                if cb.best_model_score is not None:
                    print(f"最佳 {cfg.train.monitor_metric}: {cb.best_model_score:.4f}")
            break

    print(f"\n使用以下命令评估模型:")
    print(f"  python evaluate.py")
    print(f"\n启动 Gradio 演示:")
    print(f"  python app.py")


if __name__ == "__main__":
    main()

