"""=== 模型评估与分析脚本 ===

功能：
    1. 加载最佳 checkpoint，在测试集上评估
    2. 计算 Accuracy、Weighted/Macro F1、每类 Precision/Recall
    3. 生成可视化：混淆矩阵、注意力图、t-SNE
    4. 错误分析：保存预测错误的样本到 CSV

用法：
    python evaluate.py
    python evaluate.py checkpoint_path="checkpoints/best.ckpt"
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体（静默回退）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.mixed_datamodule import MixedDataModule
from src.systems.ser_system import SERSystem


EMOTION_NAMES_CN = ["中性", "开心", "悲伤", "愤怒", "恐惧", "厌恶"]

EMOTION_NAMES_EN = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]


def _remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """向后兼容：将旧版 Linear classifier 键名映射到新版 Sequential 格式。

    旧: model.classifier.weight/bias  →  新: model.classifier.1.weight/bias
    """
    remap = {}
    for key, value in state_dict.items():
        if key == "model.classifier.weight":
            remap["model.classifier.1.weight"] = value
        elif key == "model.classifier.bias":
            remap["model.classifier.1.bias"] = value
        else:
            remap[key] = value
    return remap


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=EMOTION_NAMES_CN, yticklabels=EMOTION_NAMES_CN,
        cbar_kws={"label": "样本数"},
    )
    plt.xlabel("预测标签", fontsize=14)
    plt.ylabel("真实标签", fontsize=14)
    plt.title("PCL-MS-HuBERT-SER 混淆矩阵", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 混淆矩阵已保存: {save_path}")


def plot_attention_maps(
    attention_weights_list: List[Tuple[np.ndarray, int, int]],
    save_path: Path,
):
    selected = []
    seen_emotions = set()
    for attn, true_lbl, pred_lbl in attention_weights_list:
        if true_lbl not in seen_emotions:
            selected.append((attn, true_lbl, pred_lbl))
            seen_emotions.add(true_lbl)
        if len(selected) >= 3:
            break

    if len(selected) == 0:
        print("[可视化] 无注意力数据可展示")
        return

    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for i, (attn, true_lbl, pred_lbl) in enumerate(selected):
        ax = axes[i]
        x = np.arange(len(attn))
        ax.plot(x, attn, color="steelblue", linewidth=1.5, alpha=0.8)
        ax.fill_between(x, 0, attn, color="steelblue", alpha=0.2)
        ax.set_xlabel("时间帧", fontsize=11)
        ax.set_ylabel("注意力权重", fontsize=11)
        ax.set_title(
            f"样本 {i+1}: {EMOTION_NAMES_CN[true_lbl]} → 预测 {EMOTION_NAMES_CN[pred_lbl]}",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

    plt.suptitle("多尺度时间注意力可视化", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 注意力图已保存: {save_path}")


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray | None,
    save_path: Path,
):
    n_samples = features.shape[0]
    if n_samples > 1000:
        indices = np.random.choice(n_samples, 1000, replace=False)
        features = features[indices]
        labels = labels[indices]

    if prototypes is not None:
        all_data = np.concatenate([features, prototypes], axis=0)
    else:
        all_data = features

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    embeddings = tsne.fit_transform(all_data)

    if prototypes is not None:
        sample_embeddings = embeddings[:len(features)]
        prototype_embeddings = embeddings[len(features):]
    else:
        sample_embeddings = embeddings
        prototype_embeddings = None

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for c in range(6):
        mask = labels == c
        if mask.sum() == 0:
            continue
        plt.scatter(
            sample_embeddings[mask, 0], sample_embeddings[mask, 1],
            c=[colors[c]], label=EMOTION_NAMES_CN[c], alpha=0.5, s=15,
        )

    if prototype_embeddings is not None:
        plt.scatter(
            prototype_embeddings[:, 0], prototype_embeddings[:, 1],
            c=colors, marker="*", s=300,
            edgecolors="black", linewidths=1.5,
        )

    plt.legend(loc="upper left", fontsize=9, markerscale=2, ncol=2)
    plt.title("原型空间 t-SNE 可视化", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] t-SNE 图已保存: {save_path}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=" * 60)
    print("PCL-MS-HuBERT-SER | 模型评估")
    print("=" * 60)

    # 设置 HuggingFace 镜像
    use_mirror = cfg.model.get("use_mirror", True)
    mirror_endpoint = cfg.model.get("mirror_endpoint", "https://hf-mirror.com")
    if use_mirror and mirror_endpoint:
        os.environ.setdefault("HF_ENDPOINT", mirror_endpoint)

    # 查找 checkpoint
    checkpoint_path_str = cfg.get("checkpoint_path", None)
    if checkpoint_path_str is None:
        ckpt_dir = Path("checkpoints")
        ckpt_files = sorted(ckpt_dir.glob("best-*.ckpt"))
        if ckpt_files:
            checkpoint_path_str = str(ckpt_files[-1])
        elif (ckpt_dir / "last.ckpt").exists():
            checkpoint_path_str = str(ckpt_dir / "last.ckpt")
        else:
            print("[错误] 未找到 checkpoint，请先运行 python train.py")
            sys.exit(1)

    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        print(f"[错误] 模型文件不存在: {checkpoint_path}")
        sys.exit(1)

    print(f"\n加载模型: {checkpoint_path}")

    # 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # DataModule
    datamodule = MixedDataModule(
        ravdess_dir=cfg.data.ravdess_dir,
        cremad_dir=cfg.data.cremad_dir,
        tess_dir=cfg.data.tess_dir,
        sample_rate=cfg.data.sample_rate,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    # ---- 加载 checkpoint 并做键名兼容 ----
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)

    raw_ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    raw_state = raw_ckpt.get("state_dict", raw_ckpt)
    raw_state = _remap_state_dict(raw_state)

    # 先用旧 state_dict 初始化模型，再加载兼容后的权重
    system = SERSystem(model_cfg=model_cfg, train_cfg=train_cfg)
    missing, unexpected = system.load_state_dict(raw_state, strict=False)
    # "ce_loss.weight" 是类别权重，不在 state_dict 中，不算错误
    missing_filtered = [k for k in missing if "ce_loss.weight" not in k]
    if missing_filtered:
        print(f"[注意] 缺失键: {missing_filtered}")
    if unexpected:
        print(f"[注意] 多余键: {unexpected}")

    system.to(device)
    system.eval()

    # 评估
    print("\n开始在测试集上评估...\n")
    all_preds = []
    all_labels = []
    all_features = []
    all_logits = []
    all_attn_weights = []
    all_file_paths = []

    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            file_paths = batch["file_path"]

            logits, fused_features, attn_weights = system.model(input_values)
            preds = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_features.append(fused_features.cpu().numpy())
            all_logits.append(probs.cpu().numpy())
            all_attn_weights.append(attn_weights.cpu().numpy())
            all_file_paths.extend(file_paths)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_features = np.concatenate(all_features, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_attn_weights = np.concatenate(all_attn_weights, axis=0)

    # 指标
    acc = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average="weighted")
    f1_m = f1_score(all_labels, all_preds, average="macro")
    precision, recall, f1_per, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(6)), zero_division=0
    )

    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Weighted F1: {f1_w:.4f}")
    print(f"  Macro F1:    {f1_m:.4f}")
    print(f"\n  {'情感':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'样本数':<8}")
    print(f"  {'-'*46}")
    for i in range(6):
        print(f"  {EMOTION_NAMES_CN[i]:<8} {precision[i]:<10.4f} "
              f"{recall[i]:<10.4f} {f1_per[i]:<10.4f} {support[i]:<8}")

    # 保存可视化
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(6)))
    plot_confusion_matrix(cm, results_dir / "confusion_matrix.png")

    attn_data = [
        (all_attn_weights[i], all_labels[i], all_preds[i])
        for i in range(len(all_labels))
    ]
    plot_attention_maps(attn_data, results_dir / "attention_maps.png")

    try:
        prototypes = system.model.prototype_layer.prototypes.detach().cpu().numpy()
    except Exception:
        prototypes = None
    plot_tsne(all_features, all_labels, prototypes, results_dir / "prototype_tsne.png")

    # 错误分析
    errors = []
    for i in range(len(all_labels)):
        if all_preds[i] != all_labels[i]:
            conf = all_logits[i][all_preds[i]]
            errors.append({
                "file_path": all_file_paths[i],
                "true_label": EMOTION_NAMES_EN[all_labels[i]],
                "true_label_cn": EMOTION_NAMES_CN[all_labels[i]],
                "pred_label": EMOTION_NAMES_EN[all_preds[i]],
                "pred_label_cn": EMOTION_NAMES_CN[all_preds[i]],
                "confidence": round(float(conf), 4),
            })

    if errors:
        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(results_dir / "errors.csv", index=False, encoding="utf-8-sig")
        print(f"\n[错误分析] {len(errors)} 个预测错误，已保存: results/errors.csv")
    else:
        print(f"\n[错误分析] 全部正确！")

    print(f"\n评估完成！结果保存在 results/ 目录。")


if __name__ == "__main__":
    main()


