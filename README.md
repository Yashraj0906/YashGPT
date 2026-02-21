# 🤖 YashGPT — Fine-Tuned Mistral 7B for YouTube Comment Responses

A LoRA fine-tuned version of **Mistral-7B-Instruct-v0.2** trained to generate intelligent, context-aware replies to YouTube comments — especially for data science and technical content creators.

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🧠 Model on Hugging Face | [yashrajkumar623/YashGPT](https://huggingface.co/yashrajkumar623/YashGPT) |
| 📦 Dataset on Hugging Face | [yashrajkumar623/yashgpt-ft](https://huggingface.co/yashrajkumar623/yashgpt-ft) |

---

## 📊 Training Results

The model was trained for **10 epochs** with consistent loss reduction across both training and validation sets:

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 4.2332       | 3.8036          |
| 2     | 3.4915       | 3.1289          |
| 3     | 2.9164       | 2.6292          |
| 4     | 2.4421       | 2.3352          |
| 5     | 2.2276       | 1.9750          |
| 6     | 1.7431       | 1.7546          |
| 7     | 1.5363       | 1.6134          |
| 8     | 1.4762       | 1.5037          |
| 9     | 1.3885       | 1.4563          |
| 10    | 1.2724       | 1.4421          |

---

## ⚙️ Training Configuration

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha | 32 |
| Target Modules | `["q_proj"]` |
| Dropout | 0.05 |
| Bias | None |
| Task Type | CAUSAL_LM |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-4 |
| Batch Size | 4 (per device) |
| Epochs | 10 |
| Weight Decay | 0.01 |
| Gradient Accumulation Steps | 4 |
| Warmup Steps | 2 |
| Optimizer | paged_adamw_8bit |
| Precision | FP16 |
| Max Sequence Length | 512 tokens |

---

## 🖥️ Hardware Requirements

### Training
- GPU with at least **16GB VRAM** (recommended)
- CUDA-compatible device
- Sufficient RAM for model loading

### Inference
- GPU with at least **8GB VRAM**
- Can run on CPU (slower performance)

---

## 🎯 Use Cases

- YouTube content creator comment responses
- Data science consultation chatbot
- Educational content engagement
- Automated community management
- Customer support for technical content

---

## 🧠 Model Behavior

- **Adaptive Length** — Matches response length to comment complexity
- **Technical Depth** — Escalates technical detail upon request
- **Signature** — Always ends with `–Engage-GPT`

---

## ⚠️ Model Limitations

- Optimized specifically for the **YouTube comment format**
- Requires a **specific prompt template** for best results
- Performance may vary with **out-of-domain** comments

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install transformers peft torch accelerate bitsandbytes
```

### 2. Load the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_model = "yashrajkumar623/YashGPT"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, lora_model)
```

### 3. Generate a Response

```python
prompt = """### Instruction:
Reply to this YouTube comment as a data science content creator.

### Comment:
Can you explain overfitting in simple terms?

### Response:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📁 Dataset

The fine-tuning dataset is available at [yashrajkumar623/yashgpt-ft](https://huggingface.co/yashrajkumar623/yashgpt-ft) on Hugging Face.

- **Split**: Train / Test evaluation
- **Domain**: YouTube comments from data science and technical content

---

## 📄 License

This project is for educational and research purposes. Please refer to the base model's license: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

---

*Made with ❤️ by [Yash Raj Kumar](https://huggingface.co/yashrajkumar623)*
