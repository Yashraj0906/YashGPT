# 🤖 YashGPT — Fine-Tuned LLM for YouTube Comment Responses

<p align="center">
  <img src="https://img.shields.io/badge/Model-Mistral--7B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Method-LoRA-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Quantization-GPTQ%204--bit-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-HuggingFace-yellow?style=for-the-badge" />
</p>

YashGPT is a **parameter-efficient fine-tuned LLM** that acts as a virtual data science consultant on YouTube. Given a viewer's comment, it generates a personalized, on-brand reply ending with its signature `–YashGPT`.

Built by fine-tuning **Mistral-7B-Instruct-v0.2** (GPTQ 4-bit quantized) using **LoRA**, training only **0.79% of the model's parameters** (~2.1M out of 264M).

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Training Results](#-training-results)
- [LoRA Experiment Comparison](#-lora-experiment-comparison)
- [Sample Outputs](#-sample-outputs)
- [Tech Stack](#-tech-stack)
- [Links](#-links)

---

## 🎯 Demo

Run the Gradio-powered interactive demo:

```bash
python app.py
```

This launches a web UI where you can type a YouTube comment and get a YashGPT-style reply.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                  YashGPT Pipeline                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  YouTube Comments ──► Mistral Prompt Template        │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────────────────────────┐            │
│  │   Mistral-7B-Instruct-v0.2 (GPTQ)  │ ◄─ Frozen  │
│  │         264M parameters             │            │
│  │  ┌───────────────────────────────┐  │            │
│  │  │   LoRA Adapter (q_proj)       │  │ ◄─ Trained │
│  │  │     2.1M parameters (0.79%)   │  │            │
│  │  └───────────────────────────────┘  │            │
│  └─────────────────────────────────────┘            │
│         │                                           │
│         ▼                                           │
│  Personalized Reply (ending with –YashGPT)          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
YashGPT/
├── README.md                       # This file
├── CHANGES.md                      # Changelog of all modifications
├── requirements.txt                # Python dependencies
├── finetuning_yt.ipynb             # Main fine-tuning notebook (Colab)
├── app.py                          # Gradio interactive demo
├── inference_comparison.py         # Before vs After inference comparison
├── lora_experiments.py             # LoRA hyperparameter experiments
├── data/
│   ├── file.ipynb                  # Data preparation notebook
│   ├── train.parquet               # Training set (50 examples)
│   └── test.parquet                # Test set (9 examples)
└── .gitignore
```

---

## ⚙ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (T4 or better recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run on Google Colab (Recommended)
1. Upload `finetuning_yt.ipynb` to Google Colab
2. Select **T4 GPU** runtime
3. Run all cells sequentially

---

## 🚀 Usage

### Fine-Tuning
Open and run `finetuning_yt.ipynb` on Google Colab with a T4 GPU.

### Inference Comparison
```bash
python inference_comparison.py
```
Shows base model vs fine-tuned model responses side by side.

### Interactive Demo
```bash
python app.py
```
Launches a Gradio web UI for real-time comment response generation.

### LoRA Experiments
```bash
python lora_experiments.py
```
Compares different LoRA configurations (rank, target modules, learning rate).

---

## 📊 Training Results

Training ran for **10 epochs** on a Tesla T4 GPU with the following results:

| Epoch | Training Loss | Validation Loss |
|:-----:|:------------:|:---------------:|
| 1     | 4.1411       | 3.7247          |
| 2     | 3.4155       | 3.0512          |
| 3     | 2.8326       | 2.5657          |
| 4     | 2.3749       | 2.1839          |
| 5     | 2.0718       | 1.8534          |
| 6     | 1.6055       | 1.6359          |
| 7     | 1.3957       | 1.5077          |
| 8     | 1.3841       | 1.4337          |
| 9     | 1.3273       | 1.3988          |
| 10    | 1.2134       | 1.3883          |

**Key metrics:**
- Training loss reduction: **4.14 → 1.21** (70.7% decrease)
- Validation loss reduction: **3.72 → 1.39** (62.6% decrease)
- Trainable parameters: **2,097,152 / 264,507,392** (0.79%)

---

## 🧪 LoRA Experiment Comparison

Different LoRA configurations were compared to find the optimal setup:

| Config | Rank (r) | Target Modules | Alpha | Trainable Params | Notes |
|--------|:--------:|---------------|:-----:|:----------------:|-------|
| A (Baseline) | 8 | q_proj | 32 | 2.1M | Current default |
| B | 4 | q_proj | 16 | 1.0M | Lighter, faster training |
| C | 16 | q_proj | 32 | 4.2M | Higher capacity |
| D | 8 | q_proj, v_proj | 32 | 4.2M | Multi-module adaptation |
| E | 8 | q_proj, k_proj, v_proj, o_proj | 32 | 8.4M | Full attention adaptation |

> Run `python lora_experiments.py` on Colab to reproduce the comparison.

---

## 💬 Sample Outputs

### Comment: *"Great content, thank you!"*

| Model | Response |
|-------|---------|
| **Base Mistral** | I'm glad you found the content helpful! If you have any specific questions or topics you'd like me to cover in the future, feel free to ask. I'm here to help... |
| **YashGPT (Fine-tuned)** | Thank you for your kind words! I'm glad you found the content helpful. If you have any specific questions or topics you'd like me to cover in the future, feel free to let me know. –YashGPT |

### Comment: *"you explained totally wrong, and waist my time."*

| Model | Response |
|-------|---------|
| **Base Mistral** | I apologize if my explanation was unclear or incorrect. Could you please specify which part... *(lengthy generic response)* |
| **YashGPT (Fine-tuned)** | I'm sorry, which parts are wrong? –YashGPT |

> Notice how YashGPT gives **concise, personality-driven** responses matching the comment's tone, while the base model gives generic, verbose answers.

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Mistral-7B-Instruct-v0.2 |
| Quantization | GPTQ 4-bit |
| Fine-tuning | LoRA (PEFT) |
| Training Framework | HuggingFace Transformers |
| Optimizer | Paged AdamW 8-bit |
| Precision | FP16 mixed precision |
| Compute | Google Colab (Tesla T4) |
| Demo UI | Gradio |
| Dataset Hosting | HuggingFace Hub |

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| GitHub Repository | [Yashraj0906/YashGPT](https://github.com/Yashraj0906/YashGPT) |
| HuggingFace Model | [yashrajkumar623/yashgpt-ft](https://huggingface.co/yashrajkumar623/yashgpt-ft) |
| HuggingFace Dataset | [yashrajkumar623/YashGPT-dataset](https://huggingface.co/datasets/yashrajkumar623/YashGPT-dataset) |
| Base Model | [TheBloke/Mistral-7B-Instruct-v0.2-GPTQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ) |

---

## 📄 License

This project is for educational and portfolio purposes.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/Yashraj0906">Yashraj Kumar</a>
</p>