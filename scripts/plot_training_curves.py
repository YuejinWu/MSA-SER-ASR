# -*- coding: utf-8 -*-
"""从 TensorBoard 事件文件提取训练曲线并生成 matplotlib 图表"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUTS = {
    "training_curves_loss.png": "Loss 曲线",
    "training_curves_acc_f1.png": "Accuracy & F1 曲线",
    "training_curves_lr.png": "Learning Rate 调度",
}

ea = EventAccumulator(
    str(Path(__file__).resolve().parent.parent / "outputs" / "tensorboard" / "version_0")
)
ea.Reload()

def get_scalars(tag):
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]

# ====== 读取数据 ======
epoch_steps, epoch_vals = get_scalars("epoch")
_, train_loss_step = get_scalars("train/loss_step")
_, train_loss_epoch = get_scalars("train/loss_epoch")
_, val_loss = get_scalars("val/loss")
_, train_acc = get_scalars("train/acc")
_, val_acc = get_scalars("val/acc")
_, train_f1 = get_scalars("train/weighted_f1")
_, val_wf1 = get_scalars("val/weighted_f1")
_, val_mf1 = get_scalars("val/macro_f1")
lr_steps, lr_vals = get_scalars("lr-AdamW")

epochs = np.arange(1, len(train_loss_epoch) + 1)

# ====== Chart 1: Loss 曲线 ======
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(epochs, train_loss_epoch, "o-", color="#3B82F6", linewidth=2, markersize=6, label="Train Loss (epoch avg)")
ax1.plot(epochs, val_loss, "s-", color="#EF4444", linewidth=2, markersize=6, label="Val Loss")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("CrossEntropy Loss", fontsize=12, color="#1E293B")
ax1.tick_params(axis="y", labelcolor="#1E293B")
ax1.grid(True, alpha=0.3)

# 标注最优 epoch
best_epoch = np.argmin(val_loss) + 1
best_val_loss = val_loss[best_epoch - 1]
ax1.annotate(
    f"Best: Epoch {best_epoch}\nVal Loss={best_val_loss:.4f}",
    xy=(best_epoch, best_val_loss), xytext=(best_epoch + 2, best_val_loss + 0.05),
    arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5),
    fontsize=10, color="#EF4444", fontweight="bold",
)

ax1.set_title("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right", fontsize=10)

# 插入 step-level loss 作为半透明背景参考
ax2 = ax1.twinx()
# 不画 twin axis 数据，只用 ax1

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] training_curves_loss.png")

# ====== Chart 2: Accuracy & F1 曲线 ======
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, train_acc, "o-", color="#3B82F6", linewidth=2, markersize=6, label="Train Accuracy")
ax.plot(epochs, val_acc, "s-", color="#10B981", linewidth=2, markersize=6, label="Val Accuracy")
ax.plot(epochs, val_wf1, "D-", color="#F59E0B", linewidth=2, markersize=6, label="Val Weighted F1")
ax.plot(epochs, val_mf1, "^-", color="#8B5CF6", linewidth=2, markersize=6, label="Val Macro F1")

best_epoch_f1 = np.argmax(val_wf1) + 1
best_f1 = val_wf1[best_epoch_f1 - 1]
ax.annotate(
    f"Best W-F1: {best_f1:.4f} @ Epoch {best_epoch_f1}",
    xy=(best_epoch_f1, best_f1),
    xytext=(best_epoch_f1 + 2, best_f1 - 0.03),
    arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.5),
    fontsize=10, color="#F59E0B", fontweight="bold",
)

# 画 70% 参考线
ax.axhline(y=0.70, color="gray", linestyle="--", alpha=0.5, label="70% Baseline")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0.35, 1.02)
ax.grid(True, alpha=0.3)
ax.set_title("Training & Validation Accuracy / F1 Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves_acc_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] training_curves_acc_f1.png")

# ====== Chart 3: Learning Rate 调度 ======
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(lr_steps, lr_vals, "-", color="#8B5CF6", linewidth=2)
ax.fill_between(lr_steps, 0, lr_vals, color="#8B5CF6", alpha=0.15)
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_title("Cosine Annealing Learning Rate Schedule (with 800-step Warmup)", fontsize=13, fontweight="bold")

# 标注 warmup 区域
ax.axvline(x=800, color="#EF4444", linestyle="--", linewidth=1, alpha=0.7)
ax.annotate("Warmup End\n(step 800)", xy=(800, lr_vals[-1] * 0.5),
            xytext=(1200, lr_vals[-1] * 0.55),
            arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1),
            fontsize=9, color="#EF4444")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves_lr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] training_curves_lr.png")

print("\nDone! All charts saved to results/")
