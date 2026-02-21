# 🤖 YashGPT — Fine-Tuned Mistral 7B for YouTube Comment Responses

A LoRA fine-tuned version of **TheBloke/Mistral-7B-Instruct-v0.2-GPTQ** trained to generate intelligent, context-aware replies to YouTube comments — especially for data science and technical content creators.

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🧠 Model on Hugging Face | [yashrajkumar623/YashGPT](https://huggingface.co/yashrajkumar623/YashGPT) |
| 📦 Dataset on Hugging Face | [yashrajkumar623/YashGPT-dataset](https://huggingface.co/yashrajkumar623/yashgpt-ft) |

---

## 📊 Training Results

The model was trained for **10 epochs** with consistent loss reduction across both training and validation sets:

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 4.141072     | 3.724720        |
| 2     | 3.415536     | 3.051170        |
| 3     | 2.832638     | 2.565673        |
| 4     | 2.374947     | 2.183850        |
| 5     | 2.071811     | 1.853381        |
| 6     | 1.605482     | 1.635897        |
| 7     | 1.395695     | 1.507709        |
| 8     | 1.384065     | 1.433673        |
| 9     | 1.327265     | 1.398790        |
| 10    | 1.213437     | 1.388279        |

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

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install transformers peft torch accelerate bitsandbytes optimum auto-gptq
```

### 2. Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
lora_model = "yashrajkumar623/YashGPT"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)
model = PeftModel.from_pretrained(model, lora_model)
```

### 3. Generate a Response

```python
instructions_string = """YashGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '-YashGPT'. \
YashGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""

prompt_template = lambda comment: f'''[INST] {instructions_string} \n{comment} \n[/INST]'''

comment = "Great content, thank you!"
prompt = prompt_template(comment)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)
print(tokenizer.batch_decode(outputs)[0])
```

---

## 🧠 Model Behavior

- **Adaptive Length** — Tailors response length to match the viewer's comment
- **Technical Depth** — Escalates to technical detail upon request
- **Signature** — Always ends with `-YashGPT`

---

## 📁 Dataset

The fine-tuning dataset is available at [yashrajkumar623/YashGPT-dataset](https://huggingface.co/yashrajkumar623/yashgpt-ft) on Hugging Face.

- **Split**: Train / Test evaluation
- **Domain**: YouTube comments from data science and technical content
- **Field used**: `example` column for tokenization

---

## 🎯 Use Cases

- YouTube content creator comment responses
- Data science consultation chatbot
- Educational content engagement
- Automated community management
- Customer support for technical content

---

## ⚠️ Model Limitations

- Optimized specifically for the **YouTube comment format**
- Requires the **specific prompt template** shown above for best results
- Performance may vary with **out-of-domain** comments
- Base model is GPTQ quantized — requires `auto-gptq` and `optimum` to load

---

## 📄 License

This project is for educational and research purposes. Please refer to the base model's license: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

---

*Made with ❤️ by [Yash Raj Kumar](https://huggingface.co/yashrajkumar623)*
