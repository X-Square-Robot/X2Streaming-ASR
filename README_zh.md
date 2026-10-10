<div align="center">
  <h1>
    <img src="assets/beaver_logo.png" alt="X2Streaming-ASR 河狸吉祥物" width="72" align="center">
    X2Streaming-ASR
  </h1>
  <p>
    <strong>不确定时等待，确定时发射：结果不可回退的流式语音识别</strong><br>
    字/词说完几十毫秒内即可识别出来，且一经输出永不改写。
  </p>
  <p>
    <a href="https://arxiv.org/abs/2609.08672"><img src="https://img.shields.io/badge/arXiv-2609.08672-b31b1b" alt="arXiv 论文"></a>
    <a href="#-demo"><img src="https://img.shields.io/badge/Demo-Video-ff7a18" alt="Demo 视频"></a>
    <a href="https://huggingface.co/x-square-robot/X2Streaming-ASR-4B-1009"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face 模型"></a>
    <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python 3.11">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="Apache 2.0"></a>
  </p>
</div>

[English](README.md) | **简体中文**

<div align="center">
  <img src="assets/readme_use_cases.png" width="1000" alt="X2Streaming-ASR 应用场景：边说边出字幕、歧义时继续聆听、上下文消歧后立即发射、已提交文本可放心执行、基于 KV Cache 复用的原生因果流式">
  <p><em>边说边转写：字幕随语音实时出现，遇到歧义词先听清上下文再输出，出字即定稿，可直接交给下游 Agent 使用。</em></p>
</div>

<p align="center">
  <a href="#-最新动态">最新动态</a> ·
  <a href="#-项目简介">项目简介</a> ·
  <a href="#-demo">Demo</a> ·
  <a href="#-为什么选择-x2streaming-asr">为什么选择 X2</a> ·
  <a href="#-实验结果">实验结果</a> ·
  <a href="#-方法">方法</a> ·
  <a href="#-quick-start">快速开始</a> ·
  <a href="#-引用">引用</a>
</p>

---

## 🔥 最新动态

- **[2026-09]** 论文已发布于 arXiv：[2609.08672](https://arxiv.org/abs/2609.08672)。
- **[2026-10]** Hugging Face 模型权重已发布：
  [x-square-robot/X2Streaming-ASR-4B-1009](https://huggingface.co/x-square-robot/X2Streaming-ASR-4B-1009)。

## ✨ 项目简介

**X2Streaming-ASR 是一个面向中文、英文及中英混合语音的极低延迟流式语音识别模型，
输出结果不可回退。** 在所评测的数据集上，**中文平均发射延迟仅为 32–109 ms，英文仅为
12–85 ms，同时在多个测试集取得最优的识别效果**。

模型基于 [Voxtral Mini Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)，
将流式识别拆解为“何时提交”与“提交什么”两个问题。模型每 80 ms 在 **LISTEN** 与
**DECODE** 之间做一次决策：

- **不确定时继续聆听。** 最新音频存在歧义时，继续等待更多上下文。
- **确定时立即发射。** 信息充分后，立即识别新增音频。
- **结果不可回退。** 采用 append-only 输出，已经展示的文本不会被改写。
- **原生因果流式。** 只增量编码新到达的音频，历史状态通过 KV Cache 复用，
  无需重算，也不会被短滑动窗口截断。

本仓库仅提供**推理能力**：自适应流式推理与全上下文离线推理、Transformers / vLLM
后端、JSONL 批量转写以及网页 Demo。

## 🎬 Demo

*请取消静音以收听音频。*

https://github.com/user-attachments/assets/640f5a6d-c88b-4a59-a8c9-67ef3c833464

- 参考字幕中的彩色字块表示哪个模型最先给出该字/词的可靠提交结果：
  **X2** 青色、**Qwen** 蓝色、**R2T2** 橙色、**X-ASR** 紫色；颜色确定后不再改变。
- 右侧错误率只统计播放时间前 1.5 秒已经结束的字幕，并与各模型当前已提交文本比较。

## 💡 为什么选择 X2Streaming-ASR？

<div align="center">
  <img src="assets/readme_why.png" width="1000" alt="流式语音识别的四个痛点及 X2Streaming-ASR 的解决方式：自适应提交时机、不确定时等待、结果不可回退、原生因果流式">
</div>

固定分块或固定前瞻让每个字都等同样久；在声学边界处立刻提交，同音词容易出错；
反复改写中间结果会导致文字闪烁，也让语音 Agent 无法放心执行；每次重编码完整历史
或使用短滑动窗口，要么长音频越来越慢，要么丢失前文。X2Streaming-ASR 通过学习逐字
的提交策略，同时避开这四个问题。

## 📊 实验结果

<div align="center">
  <img src="assets/readme_results.png" width="1000" alt="12 个测试集上的平均发射延迟，以及中英文平均错误率与延迟的对比">
</div>

- **12 个测试集上延迟全部最低。** 平均发射延迟为 12–109 ms，各流式基线为 194–735 ms。
- **6 个测试集上识别效果最优。** AISHELL-1、AISHELL-3、WenetSpeech Net、
  GigaSpeech clean、VoxPopuli 与 TED-LIUM。
- **中英文整体错误率均为最低。** 按全部测试语料汇总，X2Streaming-ASR 中文 CER 为
  5.65%（115 万字），英文 WER 为 8.35%（57.7 万词），优于次优的 Confucius4-R2T2
  (quality)（5.79% / 8.43%），而后者延迟高 8–12 倍。汇总结果按测试集规模加权，
  规模大、难度高的测试集占主导：GigaSpeech 约占英文总词数的 70%（WER 10.01%），
  两个 WenetSpeech 子集约占中文总字数的 55%。若对各测试集简单平均，则为 4.65% CER
  和 5.64% WER。
- **自适应等待优于固定前瞻。** 在 AISHELL-1 上，固定前瞻需要 396 ms 平均延迟才能达到
  1.58% CER；X2Streaming-ASR 仅用 32 ms 就达到 1.53%。

所有系统均在单次硬提交、不可回退的设定下评测。延迟为字/词强制对齐结束时刻到模型发射
该字/词的时间差。

中文使用 CER（%），英文使用 WER（%），延迟单位均为毫秒。

### 中文

<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th rowspan="2">数据集</th>
      <th colspan="3" align="center">X2Streaming-ASR</th>
      <th colspan="3" align="center">X-ASR (160 ms)</th>
      <th colspan="3" align="center">Confucius4-R2T2 (Speed)</th>
      <th colspan="3" align="center">Confucius4-R2T2 (Quality)</th>
      <th colspan="3" align="center">Zipformer</th>
      <th colspan="3" align="center">Paraformer</th>
    </tr>
    <tr>
      <th align="right">CER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">CER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">CER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">CER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">CER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">CER</th><th align="right">Mean</th><th align="right">P95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AISHELL-1</td>
      <td align="right"><strong>1.53</strong></td><td align="right"><strong>32</strong></td><td align="right"><strong>320</strong></td>
      <td align="right">4.81</td><td align="right">651</td><td align="right">880</td>
      <td align="right">3.55</td><td align="right">308</td><td align="right">560</td>
      <td align="right">2.29</td><td align="right">735</td><td align="right">1280</td>
      <td align="right">1.93</td><td align="right">472</td><td align="right">800</td>
      <td align="right">3.09</td><td align="right">586</td><td align="right">880</td>
    </tr>
    <tr>
      <td>AISHELL-2 android</td>
      <td align="right">4.27</td><td align="right"><strong>81</strong></td><td align="right"><strong>400</strong></td>
      <td align="right">4.87</td><td align="right">657</td><td align="right">926</td>
      <td align="right">4.59</td><td align="right">318</td><td align="right">560</td>
      <td align="right"><strong>3.56</strong></td><td align="right">628</td><td align="right">1040</td>
      <td align="right">4.19</td><td align="right">447</td><td align="right">800</td>
      <td align="right">4.05</td><td align="right">567</td><td align="right">840</td>
    </tr>
    <tr>
      <td>AISHELL-2 ios</td>
      <td align="right">3.94</td><td align="right"><strong>75</strong></td><td align="right"><strong>400</strong></td>
      <td align="right">4.77</td><td align="right">654</td><td align="right">922</td>
      <td align="right">4.64</td><td align="right">316</td><td align="right">560</td>
      <td align="right"><strong>3.58</strong></td><td align="right">620</td><td align="right">1040</td>
      <td align="right">4.12</td><td align="right">450</td><td align="right">800</td>
      <td align="right">3.81</td><td align="right">572</td><td align="right">840</td>
    </tr>
    <tr>
      <td>AISHELL-2 mic</td>
      <td align="right">4.14</td><td align="right"><strong>79</strong></td><td align="right"><strong>480</strong></td>
      <td align="right">4.89</td><td align="right">658</td><td align="right">923</td>
      <td align="right">4.59</td><td align="right">322</td><td align="right">560</td>
      <td align="right"><strong>3.61</strong></td><td align="right">632</td><td align="right">1040</td>
      <td align="right">4.21</td><td align="right">453</td><td align="right">800</td>
      <td align="right">3.91</td><td align="right">566</td><td align="right">840</td>
    </tr>
    <tr>
      <td>AISHELL-3</td>
      <td align="right"><strong>2.08</strong></td><td align="right"><strong>57</strong></td><td align="right"><strong>400</strong></td>
      <td align="right">5.14</td><td align="right">660</td><td align="right">899</td>
      <td align="right">3.98</td><td align="right">321</td><td align="right">560</td>
      <td align="right">2.85</td><td align="right">672</td><td align="right">1120</td>
      <td align="right">2.95</td><td align="right">465</td><td align="right">800</td>
      <td align="right">3.72</td><td align="right">576</td><td align="right">849</td>
    </tr>
    <tr>
      <td>WenetSpeech Meeting</td>
      <td align="right">8.67</td><td align="right"><strong>74</strong></td><td align="right"><strong>320</strong></td>
      <td align="right">11.28</td><td align="right">595</td><td align="right">810</td>
      <td align="right">8.63</td><td align="right">320</td><td align="right">560</td>
      <td align="right"><strong>7.37</strong></td><td align="right">685</td><td align="right">1200</td>
      <td align="right">7.88</td><td align="right">436</td><td align="right">800</td>
      <td align="right">10.06</td><td align="right">582</td><td align="right">880</td>
    </tr>
    <tr>
      <td>WenetSpeech Net</td>
      <td align="right"><strong>7.88</strong></td><td align="right"><strong>109</strong></td><td align="right"><strong>400</strong></td>
      <td align="right">8.63</td><td align="right">563</td><td align="right">800</td>
      <td align="right">9.52</td><td align="right">310</td><td align="right">480</td>
      <td align="right">8.48</td><td align="right">570</td><td align="right">960</td>
      <td align="right">8.53</td><td align="right">409</td><td align="right">720</td>
      <td align="right">8.36</td><td align="right">563</td><td align="right">880</td>
    </tr>
  </tbody>
</table>
</div>

### 英文

<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th rowspan="2">数据集</th>
      <th colspan="3" align="center">X2Streaming-ASR</th>
      <th colspan="3" align="center">X-ASR (160 ms)</th>
      <th colspan="3" align="center">Confucius4-R2T2 (Speed)</th>
      <th colspan="3" align="center">Confucius4-R2T2 (Quality)</th>
    </tr>
    <tr>
      <th align="right">WER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">WER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">WER</th><th align="right">Mean</th><th align="right">P95</th>
      <th align="right">WER</th><th align="right">Mean</th><th align="right">P95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LibriSpeech test-clean</td>
      <td align="right">2.20</td><td align="right"><strong>29</strong></td><td align="right"><strong>240</strong></td>
      <td align="right">3.86</td><td align="right">574</td><td align="right">800</td>
      <td align="right">2.83</td><td align="right">194</td><td align="right">320</td>
      <td align="right"><strong>2.11</strong></td><td align="right">472</td><td align="right">800</td>
    </tr>
    <tr>
      <td>LibriSpeech test-other</td>
      <td align="right">5.50</td><td align="right"><strong>85</strong></td><td align="right">480</td>
      <td align="right">9.64</td><td align="right">585</td><td align="right">800</td>
      <td align="right">6.38</td><td align="right">201</td><td align="right"><strong>320</strong></td>
      <td align="right"><strong>4.81</strong></td><td align="right">463</td><td align="right">720</td>
    </tr>
    <tr>
      <td>GigaSpeech (clean)</td>
      <td align="right"><strong>10.01</strong></td><td align="right"><strong>43</strong></td><td align="right"><strong>320</strong></td>
      <td align="right">10.47</td><td align="right">592</td><td align="right">847</td>
      <td align="right">10.69</td><td align="right">204</td><td align="right">400</td>
      <td align="right">10.21</td><td align="right">501</td><td align="right">880</td>
    </tr>
    <tr>
      <td>VoxPopuli</td>
      <td align="right"><strong>7.07</strong></td><td align="right"><strong>12</strong></td><td align="right"><strong>320</strong></td>
      <td align="right">11.25</td><td align="right">603</td><td align="right">880</td>
      <td align="right">8.97</td><td align="right">204</td><td align="right">400</td>
      <td align="right">7.26</td><td align="right">523</td><td align="right">880</td>
    </tr>
    <tr>
      <td>TED-LIUM</td>
      <td align="right"><strong>3.42</strong></td><td align="right"><strong>15</strong></td><td align="right"><strong>240</strong></td>
      <td align="right">3.81</td><td align="right">590</td><td align="right">880</td>
      <td align="right">3.98</td><td align="right">196</td><td align="right">320</td>
      <td align="right">3.45</td><td align="right">497</td><td align="right">880</td>
    </tr>
  </tbody>
</table>
</div>

完整的实验配置、基线对比与消融实验请参阅
[论文](https://arxiv.org/abs/2609.08672)。

## 🧠 方法

<div align="center">
  <img src="assets/streaming_asr.png" width="1000" alt="X2Streaming-ASR 在 LISTEN 与 DECODE 之间交替的 token 级架构">
  <p><em>每个 80 ms 音频 token 到来时，模型预测 Wait（继续聆听）或 Emit（开始解码）；Emit 之后写出新增音频对应的文本，并以 Eos 结束。</em></p>
</div>

X2Streaming-ASR 保留了 Voxtral Realtime 的因果音频编码器、Adapter 与 decoder-only
语言模型，但用学习得到的决策替代了固定延迟：

1. **LISTEN。** 每个新的 80 ms 音频 token 到来时，若上下文不足或存在歧义则预测
   *Wait*，信息充分则预测 *Emit*。
2. **DECODE。** *Emit* 之后生成新增音频对应的文本，直到 `Eos`，再回到 LISTEN。
3. **复用历史。** 编码器增量运行，语言模型保留 KV Cache，任何音频块都不会触发对历史
   音频的重算；在配置的上下文长度内，也不会被滑动窗口丢弃。

提交策略分三阶段训练：建立流式识别能力、基于探测提交时刻的监督热启动，以及“准确率
优先、时延次之”、按字分配信用的 GRPO。详见[论文](https://arxiv.org/abs/2609.08672)。

## 📦 模型范围

| 项目 | 说明 |
| --- | --- |
| 基座模型 | Voxtral Mini Realtime |
| 语言 | 中文、英文及中英混合 |
| 流式输出 | Append-only 硬提交 |
| 推理模式 | 自适应流式、全上下文离线 |
| 后端 | Transformers、vLLM |
| 模型权重 | [x-square-robot/X2Streaming-ASR-4B-1009](https://huggingface.co/x-square-robot/X2Streaming-ASR-4B-1009) |

## 🚀 Quick Start

### 1. 安装

建议使用 Linux、Python 3.11 和支持 CUDA 的 NVIDIA GPU。

```bash
git clone https://github.com/ZwlinZw/X2StreamingASR.git
cd X2StreamingASR

conda create -n x2streamingasr python=3.11 -y
conda activate x2streamingasr
pip install -r requirements.txt
```

按需安装 vLLM 和网页 Demo 依赖：

```bash
pip install -r requirements-vllm.txt -r demo/requirements.txt
```

### 2. 模型

模型权重见 Hugging Face：
[x-square-robot/X2Streaming-ASR-4B-1009](https://huggingface.co/x-square-robot/X2Streaming-ASR-4B-1009)。

下载并设置 `MODEL_PATH`：

```bash
pip install -U huggingface_hub

huggingface-cli download x-square-robot/X2Streaming-ASR-4B-1009 --local-dir ./X2Streaming-ASR-4B-1009
export MODEL_PATH="$(pwd)/X2Streaming-ASR-4B-1009"
```

或使用已有的本地 checkpoint 目录：

```bash
export MODEL_PATH=/path/to/X2Streaming-ASR-4B-1009
```

### 3. 流式与离线推理

```bash
# 自适应流式推理
python infer.py \
  --config configs/infer.yaml \
  --model_path "$MODEL_PATH" \
  --language zh \
  --audio /path/to/audio.wav

# 全上下文离线推理
python infer.py \
  --config configs/infer_offline.yaml \
  --model_path "$MODEL_PATH" \
  --language en \
  --audio /path/to/audio.wav
```

> [!IMPORTANT]
> **请用 `--language` 指定音频语言。** 模型以语言标签为条件：`zh` 为中文，`en` 为英文，
> `cs` 为中英混合。不传 `--language` 时会根据输入路径猜测，只有路径包含
> `librispeech`、`gigaspeech`、`tedlium` 或 `voxpopuli` 才判为英文，其余一律按 `zh`
> 处理，因此英文音频务必加上 `--language en`。一次运行只使用一个标签，中英文混合的
> JSONL 请按语言拆分，或使用 `cs`。

### 4. JSONL 批量推理

输入 JSONL 每行必须包含 `audio`、`wav` 或 `wav_path` 之一：

```json
{"audio": "/path/to/first.wav"}
{"wav": "/path/to/second.wav"}
```

```bash
python infer.py \
  --config configs/infer.yaml \
  --model_path "$MODEL_PATH" \
  --language en \
  --input_file audio.jsonl \
  --output_file results.jsonl
```

输出 JSONL 会包含音频路径、识别文本、推理模式，以及可选的时间戳或自适应
发射记录，无需提供参考文本。

### 5. 网页 Demo

```bash
BACKEND=transformers \
MODE=adaptive \
LANGUAGE=zh \
REPO_ID="$MODEL_PATH" \
bash demo/run.sh
```

服务启动后访问 `http://localhost:7860`。`LANGUAGE` 设置默认语言标签（`zh`、`en` 或
`cs`，默认 `zh`），英文语音请使用 `LANGUAGE=en`。

使用 vLLM 时，先转换并启动模型服务：

```bash
python tools/convert_hf_to_mistral.py "$MODEL_PATH"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --enforce-eager \
  --gpu-memory-utilization 0.9
```

再启动 Demo 代理：

```bash
BACKEND=vllm \
MODE=adaptive \
LANGUAGE=zh \
VLLM_HOST=127.0.0.1 \
VLLM_PORT=8000 \
REPO_ID="$MODEL_PATH" \
bash demo/run.sh
```

## 📝 引用

如果 X2Streaming-ASR 对你的研究有帮助，请引用：

```bibtex
@article{lin2026x2streaming,
  title={X2Streaming-ASR: wait when uncertain, emit when ready for streaming ASR},
  author={Lin, Zhiwei and Fu, Kaiqi and Wen, Rime and Liu, Zehan and Qin, Shawn and Gan, Roy and Wang, Hao and Wang, Qian},
  journal={arXiv preprint arXiv:2609.08672},
  year={2026}
}
```

## 📄 许可证

- 代码：[Apache License 2.0](LICENSE)
- 模型权重：见 [MODEL_LICENSE.md](MODEL_LICENSE.md)
- 上游模型：
  [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
