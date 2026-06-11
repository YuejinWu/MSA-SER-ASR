"""=== Speech Analyzer (Short + Long Audio) ===

Tab 1 - Short:  情感识别 + Whisper 转录
Tab 2 - Long:   长音频 → Timeline + 荧光笔标注转录（句子带情感底色） + Distribution

用法: python app.py
"""

import os, sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch, torch.nn.functional as F, torchaudio, torchaudio.functional as AF
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import whisper

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.systems.ser_system import SERSystem

EMOTION_NAMES = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]
EMOTION_COLORS = ["#5a6c6d", "#c8960c", "#7d3c98", "#c0392b", "#d35400", "#148f77"]
# 荧光笔半透明色
HL_COLORS = [
    "rgba(90,108,109,0.25)",   # Neutral - 淡灰
    "rgba(200,150,12,0.30)",   # Happy - 淡金
    "rgba(125,60,152,0.25)",   # Sad - 淡紫
    "rgba(192,57,43,0.25)",    # Angry - 淡红
    "rgba(211,84,0,0.28)",     # Fearful - 淡橙
    "rgba(20,143,119,0.28)",   # Disgust - 淡青
]

_WHISPER = None
def get_whisper():
    global _WHISPER
    if _WHISPER is None:
        print("[Whisper] Loading base model...")
        _WHISPER = whisper.load_model("base")
        print("[Whisper] Ready.")
    return _WHISPER

# ================================================================
def _remap_sd(sd):
    r = {}
    for k, v in sd.items():
        if k == "model.classifier.weight": r["model.classifier.1.weight"] = v
        elif k == "model.classifier.bias": r["model.classifier.1.bias"] = v
        else: r[k] = v
    return r

def _load_system(ckpt, mcfg, tcfg, dev):
    raw = torch.load(str(ckpt), map_location=dev, weights_only=False)
    sd = _remap_sd(raw.get("state_dict", raw))
    sys = SERSystem(model_cfg=mcfg, train_cfg=tcfg)
    sys.load_state_dict(sd, strict=False)
    sys.to(dev).eval()
    return sys

class EmotionPredictor:
    def __init__(self, ckpt, mcfg, tcfg, sr=16000, ml=80000):
        self.sr = sr; self.ml = ml
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sys = _load_system(ckpt, mcfg, tcfg, self.dev)
    def _load_audio(self, ap):
        if isinstance(ap, tuple):
            s, w = ap; wf = torch.from_numpy(w if w.ndim == 1 else w).float()
            if wf.ndim == 1: wf = wf.unsqueeze(0); return wf, s
        try: return torchaudio.load(str(ap), backend="soundfile")
        except: return torchaudio.load(str(ap))
    def _preprocess(self, wf_chunk, orig_sr):
        if wf_chunk.shape[0] > 1: wf_chunk = wf_chunk.mean(dim=0, keepdim=True)
        if orig_sr != self.sr: wf_chunk = AF.resample(wf_chunk, orig_sr, self.sr)
        pk = wf_chunk.abs().max()
        if pk > 0: wf_chunk = wf_chunk / pk
        wf_chunk = wf_chunk.squeeze(0)
        if wf_chunk.shape[0] < self.ml:
            wf_chunk = torch.cat([wf_chunk, torch.zeros(self.ml - wf_chunk.shape[0])])
        else: wf_chunk = wf_chunk[:self.ml]
        return wf_chunk.unsqueeze(0)
    def predict_chunk(self, wf_chunk, orig_sr):
        wf = self._preprocess(wf_chunk, orig_sr).to(self.dev)
        with torch.no_grad(): logits, _, _ = self.sys.model(wf)
        probs = F.softmax(logits, -1).squeeze(0).cpu().numpy()
        return int(np.argmax(probs)), probs.tolist()
    def predict(self, ap):
        wf_full, osr = self._load_audio(ap)
        idx, probs = self.predict_chunk(wf_full, osr)
        if isinstance(ap, tuple): raw = np.atleast_1d(np.asarray(ap[1]).squeeze())
        else:
            raw = wf_full.mean(dim=0).numpy().squeeze() if wf_full.shape[0] > 1 else wf_full.numpy().squeeze()
        wf_t = self._preprocess(wf_full, osr).to(self.dev)
        with torch.no_grad(): _, _, attn = self.sys.model(wf_t)
        return {"emotion": EMOTION_NAMES[idx], "idx": idx, "probs": probs, "wave": raw, "attn": attn.squeeze(0).cpu().numpy()}
    def analyze_long(self, ap, chunk_sec=4.0, overlap_sec=1.0):
        wf_full, osr = self._load_audio(ap)
        if wf_full.shape[0] > 1: wf_full = wf_full.mean(dim=0, keepdim=True)
        total_samples, total_sec = wf_full.shape[1], wf_full.shape[1] / osr
        cs = int(chunk_sec * osr)
        ss = max(1, int((chunk_sec - overlap_sec) * osr))
        segs = []; p = 0
        while p + cs <= total_samples:
            e = p + cs; idx, pr = self.predict_chunk(wf_full[:, p:e], osr)
            segs.append((p/osr, e/osr, EMOTION_NAMES[idx], idx, pr)); p += ss
        if p < total_samples and total_samples - p > osr:
            idx, pr = self.predict_chunk(wf_full[:, p:], osr)
            segs.append((p/osr, total_sec, EMOTION_NAMES[idx], idx, pr))
        wm = get_whisper()
        if isinstance(ap, tuple):
            tmp = Path("temp_asr.wav"); import soundfile as sf; wn = ap[1]
            sf.write(str(tmp), (wn if wn.ndim==1 else wn.T), 16000)
            wr = wm.transcribe(str(tmp)); tmp.unlink(missing_ok=True)
        else: wr = wm.transcribe(str(ap))
        def femo(tm):
            b, bd = segs[0], float("inf")
            for st2, et2, n, i, p in segs:
                d = abs(tm-(st2+et2)/2)
                if d < bd: bd = d; b = (st2, et2, n, i, p)
            return b
        lines = []
        for seg in wr.get("segments", []):
            _, _, en, ei, _ = femo((seg["start"]+seg["end"])/2)
            lines.append({"text": seg["text"].strip(), "ei": ei})
        return segs, lines, total_sec


# ================================================================
# Plots
# ================================================================
def plot_wave(wf, sr=16000):
    wf = np.atleast_1d(np.asarray(wf).squeeze())
    t = np.linspace(0, len(wf)/sr, len(wf))
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.plot(t, wf, color="steelblue", linewidth=0.6)
    ax.set_xlim(0, t[-1]); ax.grid(True, alpha=0.3); fig.tight_layout(); return fig

def plot_attn(at):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.plot(np.arange(len(at)), at, color="coral", linewidth=1.2)
    ax.grid(True, alpha=0.3); fig.tight_layout(); return fig

def plot_conf(cf):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.barh(range(len(cf)), cf, color=EMOTION_COLORS, height=0.6)
    for i, c in enumerate(cf): ax.text(c+0.01, i, f"{c:.0%}", va="center", fontsize=8)
    ax.set_yticks(range(len(EMOTION_NAMES))); ax.set_yticklabels(EMOTION_NAMES, fontsize=8)
    ax.set_xlim(0, 1.05); ax.invert_yaxis(); fig.tight_layout(); return fig

def plot_timeline(segments, total_sec):
    fig, ax = plt.subplots(figsize=(12, 2))
    for st, et, name, idx, _ in segments:
        ax.barh(0, et-st, left=st, height=0.8, color=EMOTION_COLORS[idx], edgecolor="white", linewidth=0.5)
        if et-st > total_sec*0.03: ax.text((st+et)/2, 0, name, ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_xlim(0, total_sec); ax.set_yticks([]); ax.set_xlabel("Time (s)"); ax.set_title("Emotion Timeline")
    fig.tight_layout(); return fig

def plot_distribution(segments):
    cnt = {}
    for _, _, n, _, _ in segments: cnt[n] = cnt.get(n, 0) + 1
    lbs, vls = list(cnt.keys()), list(cnt.values())
    cs = [EMOTION_COLORS[EMOTION_NAMES.index(n)] for n in lbs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(vls, labels=lbs, colors=cs, autopct="%1.1f%%", startangle=90)
    ax.set_title("Distribution"); fig.tight_layout(); return fig


# ================================================================
# UI
# ================================================================
def create_demo(predictor):
    with gr.Blocks(title="Speech Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Speech Analyzer | Emotion + Transcription")

        with gr.Tabs():
            with gr.Tab("Short Audio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        a1 = gr.Audio(type="filepath", label="Upload / Record")
                        b1 = gr.Button("Analyze", variant="primary", size="lg")
                        gr.Markdown("### Emotion")
                        eo = gr.Textbox(label="Predicted", interactive=False)
                        cp = gr.Plot(label="Confidence")
                    with gr.Column(scale=2):
                        gr.Markdown("### Transcription")
                        ao = gr.Textbox(label="Text", lines=3, interactive=False)
                        ai = gr.Textbox(label="Info", interactive=False)
                        wp = gr.Plot(label="Waveform")
                        ap = gr.Plot(label="Attention")
                def short_fn(aud):
                    if aud is None: return ("", None, "", "", None, None)
                    ser = predictor.predict(aud)
                    wr = get_whisper().transcribe(aud)
                    return (f"## {ser['emotion']}", plot_conf(ser["probs"]), wr["text"].strip(),
                            f"Lang: {wr.get('language','?')} | Whisper base", plot_wave(ser["wave"]), plot_attn(ser["attn"]))
                b1.click(fn=short_fn, inputs=[a1], outputs=[eo, cp, ao, ai, wp, ap])

            with gr.Tab("Long Audio"):
                gr.Markdown("Long speech: timeline + highlighted transcript + distribution.")
                with gr.Row():
                    with gr.Column(scale=1):
                        a2 = gr.Audio(type="filepath", label="Upload Long Audio")
                        cs2 = gr.Slider(2, 10, value=4, step=1, label="Chunk (s)")
                        os2 = gr.Slider(0, 3, value=1, step=0.5, label="Overlap (s)")
                        b2 = gr.Button("Analyze", variant="primary", size="lg")
                        gr.Markdown("### Summary")
                        st = gr.Textbox(label="Emotion Stats", lines=5, interactive=False)

                    with gr.Column(scale=2):
                        tl = gr.Plot(label="Emotion Timeline")
                        td = gr.HTML(label="Transcript (Highlighter Style)")
                        dp = gr.Plot(label="Distribution")

                def long_fn(aud, cs, ov):
                    if aud is None: return (None, "", None, "")
                    segs, lines, tsec = predictor.analyze_long(aud, cs, ov)
                    if not segs: return (None, "", None, "Audio too short")
                    cnt = {}
                    for _, _, n, _, _ in segs: cnt[n] = cnt.get(n, 0) + 1
                    total = sum(cnt.values())
                    sm = "\n".join(f"{k}: {v}/{total} ({v/total:.0%})" for k, v in sorted(cnt.items(), key=lambda x: -x[1]))
                    # 图例
                    legend_items = "".join(
                        f'<span style="display:inline-block;background:{HL_COLORS[i]};border-left:3px solid {EMOTION_COLORS[i]};padding:2px 10px;margin-right:8px;border-radius:3px;font-size:13px;font-weight:bold;color:#333">{EMOTION_NAMES[i]}</span>'
                        for i in range(6))
                    legend = f'<div style="margin-bottom:12px;padding:10px;background:#f5f5f5;border-radius:6px">{legend_items}</div>'
                    # 荧光笔文本
                    body = '<div style="max-height:420px;overflow-y:auto;background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;line-height:2.2">'
                    for ln in lines:
                        bg = HL_COLORS[ln["ei"]]
                        body += f'<span style="background:{bg};padding:3px 6px;font-size:17px;font-family:Microsoft YaHei,sans-serif;color:#111">{ln["text"]}</span> '
                    body += '</div>'
                    return (plot_timeline(segs, tsec), legend + body, plot_distribution(segs), sm)

                b2.click(fn=long_fn, inputs=[a2, cs2, os2], outputs=[tl, td, dp, st])

    return demo


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.model.get("use_mirror", True):
        os.environ.setdefault("HF_ENDPOINT", cfg.model.get("mirror_endpoint", "https://hf-mirror.com"))
    ckpt = cfg.get("checkpoint_path", None)
    if ckpt is None:
        cd = Path("checkpoints")
        ckpt = str(sorted(cd.glob("best-*.ckpt"))[-1]) if list(cd.glob("best-*.ckpt")) else str(cd / "last.ckpt")
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    tcfg = OmegaConf.to_container(cfg.train, resolve=True)
    predictor = EmotionPredictor(ckpt, mcfg, tcfg, cfg.data.sample_rate, cfg.data.max_length)
    get_whisper()
    demo = create_demo(predictor)
    print("\nhttp://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    main()
