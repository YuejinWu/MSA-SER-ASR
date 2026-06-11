# MSA-SER-ASR：多尺度语音情感识别与转录分析系统

## 基于预训练模型 Adapter 迁移学习、多尺度时间建模与集成语音识别

> **Multi-Scale Speech Emotion Recognition with Adapter-based Transfer Learning and Integrated ASR**

本项目构建了一个端到端的**多尺度语音情感识别与转录分析系统**。系统以预训练 WavLM 为声学特征骨干，采用 Adapter 参数高效微调（仅 1.2M 可训练参数）实现跨数据集的迁移学习，通过 CNN 局部分支与 Self-Attention 全局分支构成"微观纹理 + 宏观韵律"的多尺度时间建模。联合 RAVDESS、CREMA-D、TESS 三个英文情感语音数据集（共 10,898 条），按严格说话人独立原则划分训练/验证/测试集，在 1,450 条测试样本上达到 **73.5% 准确率和 0.732 Macro F1**。系统同时集成 Whisper ASR 引擎，构建了基于 Gradio 的双模式交互界面（短音频即时分析 + 长音频滑动窗口情感时序标注），提供一站式语音情感识别与内容转录服务。

---

## 环境安装

```bash
# 创建虚拟环境（推荐 Python 3.10）
conda create -n pcl-ms-hubert python=3.10 -y
conda activate pcl-ms-hubert

# 安装依赖
pip install -r requirements.txt

# (可选) 登录 WandB 进行实验跟踪
wandb login
```

---

## 数据准备

本项目联合使用三个英文情感语音数据集，统一映射为 6 类情感标签：

| 数据集 | 样本量 | 说话人数 | 使用情感 | 下载链接 |
|--------|--------|:-------:|---------|------|
| **RAVDESS** | 1,056 | 24 | 6/8 类 | [Zenodo](https://zenodo.org/record/1188976) |
| **CREMA-D** | 7,442 | 91 | 6/6 类 | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |
| **TESS** | 2,400 | 2 | 6/7 类 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

### 目录结构

```
data/
├── speech-emotion-recognition-ravdess-data/
│   └── Actor_*/  *.wav
├── CREMA-D/
│   └── AudioWAV/  *.wav
└── TESS Toronto emotional speech set data/
    └── */  *.wav
```

### 配置数据路径

编辑 `configs/data/mixed.yaml` 设置各数据集路径：

```yaml
ravdess_dir: "data/speech-emotion-recognition-ravdess-data"
cremad_dir: "data/CREMA-D"
tess_dir: "data/TESS Toronto emotional speech set data"
```

---

## 运行命令

### 0. 预处理缓存（可选，推荐）

```bash
python scripts/preprocess_cache.py
```

一次性将音频预处理为 `.pt` tensor 缓存，训练 I/O 效率提升约 6 倍。

### 1. 训练模型

```bash
# 默认使用 WavLM + Adapter + 三数据集
python train.py

# 切换骨干或参数（命令行覆盖）
python train.py model.backbone=facebook/hubert-base-ls960 model.adapter_bottleneck=128
```

首次运行将自动下载预训练权重。最佳模型保存在 `checkpoints/` 目录。

### 2. 评估模型

```bash
# 自动加载最佳 checkpoint
python evaluate.py

# 指定 checkpoint
python evaluate.py checkpoint_path="checkpoints/best-epoch=04-val/weighted_f1=0.8399.ckpt"
```

输出：
- 测试集指标（Accuracy, Weighted F1, Macro F1, 每类 Precision/Recall）
- `results/confusion_matrix.png` — 混淆矩阵
- `results/attention_maps.png` — 多尺度注意力可视化
- `results/prototype_tsne.png` — 特征空间 t-SNE 散点图
- `results/errors.csv` — 错误分析（385 条）

### 3. Web 交互演示

```bash
python app.py
```

打开浏览器访问 `http://127.0.0.1:7860`，提供两种模式：
- **Short Audio**：上传短音频（≤5 秒），即时返回情感识别 + Whisper 转录 + 波形图 + 注意力可视化
- **Long Audio**：上传任意长度音频，滑动窗口分析输出情感时间线 + 逐句荧光笔转录 + 分布饼图

---

## 项目结构

```
PCL-MS-HuBERT-SER/
├── configs/
│   ├── config.yaml                     # Hydra 主配置
│   ├── model/
│   │   └── hubert_multiscale.yaml      # 模型参数（双骨干 + Adapter + 多尺度）
│   ├── data/
│   │   ├── mixed.yaml                  # 三数据集混合配置
│   │   └── ravdess.yaml                # RAVDESS 单数据集配置
│   └── train/
│       └── curriculum.yaml             # 训练策略
├── src/
│   ├── models/
│   │   ├── components/
│   │   │   ├── temporal_attention.py   # 全局时间注意力
│   │   │   └── prototype_layer.py      # 原型对比学习层（预留扩展）
│   │   └── multiscale_hubert.py        # 多尺度模型（HuBERT/WavLM + Adapter）
│   ├── losses/
│   │   └── prototype_contrastive.py    # PCL 损失函数
│   ├── data/
│   │   ├── mixed_datamodule.py         # 三数据集统一加载 + 缓存检测
│   │   └── ravdess_datamodule.py       # RAVDESS 数据模块
│   ├── systems/
│   │   └── ser_system.py               # Lightning 训练系统
│   └── utils/
│       └── callbacks.py                # 训练回调
├── scripts/
│   ├── preprocess_cache.py             # 一次性预处理缓存
│   ├── download_model.py               # 下载 HuBERT 模型
│   ├── download_wavlm.py               # 下载 WavLM 模型
│   └── plot_training_curves.py         # 训练曲线绘制
├── train.py                            # 训练入口
├── evaluate.py                         # 评估分析（混淆矩阵 + t-SNE + 注意力 + 错误分析）
├── app.py                              # Gradio 双模式界面（Short + Long Audio）
├── requirements.txt                    # 依赖列表
├── README.md                           # 本文件
├── REPORT.md                           # 详细实验报告
├── checkpoints/                        # 模型保存目录
└── results/                            # 评估结果输出
```

---

## 成员贡献

| 姓名 | 学号 | 承担工作 | 贡献比例 |
|------|------|----------|:------:|
| 熊烨 | — | 多尺度时间建模（CNN 局部 + Attention 全局）设计与实现 | — |
| 吴颂钊 | — | Adapter 迁移学习、训练策略与 Lightning 系统集成 | — |
| 吴彦霖 | — | 数据工程、评估分析、Whisper ASR 集成与 Gradio 演示 | — |

---

## 实验结果

> 最优模型：WavLM + Adapter (瓶颈 64) | Checkpoint: `best-epoch=04-val/weighted_f1=0.8399.ckpt`

### 测试集整体性能

| 指标 | 数值 |
|------|------|
| **Accuracy** | **73.5%** |
| Weighted F1 | 0.730 |
| **Macro F1** | **0.732** |
| 测试样本 | 1,450（3 数据集，~17 未见说话人） |
| 预测错误 | 385 |

### 每类性能

| 情感 | Precision | Recall | F1 | 样本数 |
|:----:|:---------:|:------:|:---:|:-----:|
| 中性 | 76.4% | 86.0% | **0.809** | 200 |
| 开心 | 80.3% | 78.4% | **0.794** | 250 |
| 悲伤 | 72.1% | 51.6% | 0.601 | 250 |
| 愤怒 | 76.5% | 87.2% | **0.815** | 250 |
| 恐惧 | 64.9% | 62.8% | 0.638 | 250 |
| 厌恶 | 70.2% | 77.2% | **0.735** | 250 |

### 对比实验

| 方案 | 骨干 | 微调 | Acc | Macro F1 |
|------|:---:|------|:---:|:--------:|
| E1 | HuBERT | 冻结 | 67.6% | 0.669 |
| E2 | HuBERT | Adapter(64) | 73.1% | 0.729 |
| E3 | HuBERT | Adapter(128) + Focal | 72.9% | 0.725 |
| **E4** | **WavLM** | **Adapter(64)** | **73.5%** | **0.732** |

> 详细实验过程与分析见 [REPORT.md](REPORT.md)

---

## 技术栈

- **Python 3.10**, **PyTorch 2.x**, **PyTorch Lightning 2.x**
- **HuggingFace Transformers** (WavLM / HuBERT 双骨干支持)
- **OpenAI Whisper** (ASR 语音转文字)
- **Hydra + OmegaConf** (配置管理)
- **Weights & Biases + TensorBoard** (实验跟踪)
- **torchaudio, soundfile, librosa** (音频处理)
- **matplotlib, seaborn** (可视化)
- **scikit-learn** (t-SNE 降维)
- **Gradio 4.x** (Web 交互界面)
