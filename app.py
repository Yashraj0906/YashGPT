"""
YashGPT — Interactive Gradio Demo
==================================
A Gradio-powered web UI for generating YouTube comment replies
using the fine-tuned YashGPT model (Mistral-7B + LoRA).

Usage:
    python app.py

Requirements:
    - CUDA-compatible GPU (T4 or better)
    - Dependencies from requirements.txt
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
LORA_ADAPTER = "yashrajkumar623/yashgpt-ft"
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """YashGPT, functioning as a virtual data science consultant on YouTube, \
communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–YashGPT'. \
YashGPT will tailor the length of its responses to match the viewer's comment, \
providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment."""

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
print("🔄 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)

print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()
print("✅ Model loaded successfully!")


def format_prompt(comment: str) -> str:
    """Format a YouTube comment into a Mistral instruction prompt."""
    return f"[INST] {SYSTEM_PROMPT} \n{comment} \n[/INST]"


def generate_reply(comment: str, max_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.7) -> str:
    """Generate a YashGPT reply for a given YouTube comment."""
    if not comment.strip():
        return "Please enter a comment to get a response."

    prompt = format_prompt(comment)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract only the generated response (after [/INST])
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in full_output:
        response = full_output.split("[/INST]")[-1].strip()
    else:
        response = full_output

    return response


# ──────────────────────────────────────────────
# Gradio Interface
# ──────────────────────────────────────────────
EXAMPLE_COMMENTS = [
    "Great content, thank you!",
    "This was a very thorough introduction to LLMs and answered many questions I had. Thank you.",
    "you explained totally wrong, and waist my time.",
    "Can you explain the difference between fine-tuning and RAG?",
    "How do I get started with machine learning? I'm a complete beginner.",
    "Love your videos! Can you do a tutorial on transformers?",
]

with gr.Blocks(
    title="YashGPT — YouTube Comment Responder",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
) as demo:
    gr.Markdown(
        """
        # 🤖 YashGPT — YouTube Comment Responder
        **A fine-tuned Mistral-7B model** that generates personalized YouTube comment replies
        in the style of a virtual data science consultant.

        *Fine-tuned with LoRA (0.79% trainable parameters) on a custom dataset.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            comment_input = gr.Textbox(
                label="💬 YouTube Comment",
                placeholder="Type a YouTube comment here...",
                lines=3,
            )
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=50, maximum=512, value=256, step=10,
                    label="Max Tokens"
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature"
                )
            submit_btn = gr.Button("🚀 Generate Reply", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="🤖 YashGPT Reply",
                lines=6,
                interactive=False,
            )

    gr.Examples(
        examples=[[c] for c in EXAMPLE_COMMENTS],
        inputs=[comment_input],
        label="📝 Try these example comments",
    )

    submit_btn.click(
        fn=generate_reply,
        inputs=[comment_input, max_tokens_slider, temperature_slider],
        outputs=output,
    )

    comment_input.submit(
        fn=generate_reply,
        inputs=[comment_input, max_tokens_slider, temperature_slider],
        outputs=output,
    )

    gr.Markdown(
        """
        ---
        **Model:** Mistral-7B-Instruct-v0.2 (GPTQ 4-bit) + LoRA adapter
        | **Built by:** [Yashraj Kumar](https://github.com/Yashraj0906)
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
