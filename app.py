"""
Gradio Web Application for Gemma 4 12B Claim Verification
AI Seekho Day 2026 — 1st Place Winning Solution
Author: Muhammad Umar Farooq
"""

try:
    import spaces
    HAS_SPACES = True
except Exception:
    HAS_SPACES = False

def gpu_decorator(func):
    if HAS_SPACES:
        return spaces.GPU(func)
    return func

import os
import re
import traceback
import torch
import gradio as gr
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "google/gemma-4-12B-it"
ADAPTER_ID = "omerfarooq223/gemma-4-12b-evidence-verification-qlora"
HF_TOKEN = os.environ.get("HF_TOKEN")

PROMPT_TEMPLATE = """Classify the claim using only the supplied evidence.

SUPPORTS: the evidence establishes the claim.
REFUTES: the evidence contradicts the claim.
NOT_ENOUGH_INFO: the evidence neither establishes nor contradicts the specific claim.

End your response exactly as:
FINAL: SUPPORTS
or
FINAL: REFUTES
or
FINAL: NOT_ENOUGH_INFO

Claim:
{claim}

Evidence:
{evidence_block}"""

# Global model cache
processor = None
model = None

def load_verification_model():
    global processor, model
    if model is not None:
        return model, processor

    print(f"Loading tokenizer for {MODEL_ID}...")
    try:
        processor = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=False)
    except Exception:
        processor = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

    print("Loading 4-bit NF4 quantized base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )

    print(f"Attaching LoRA adapter from {ADAPTER_ID}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=HF_TOKEN)
    model.eval()
    return model, processor

def format_evidence(evidence_text: str) -> str:
    lines = [line.strip() for line in evidence_text.strip().split("\n") if line.strip()]
    formatted = []
    for idx, line in enumerate(lines, start=1):
        if not re.match(r"^\[\d+\]", line):
            formatted.append(f"[{idx}] {line}")
        else:
            formatted.append(line)
    return "\n".join(formatted)

@gpu_decorator
def verify_claim(claim: str, evidence: str):
    if not claim.strip() or not evidence.strip():
        return (
            "<div class='result-card nei'><div class='badge-title'>⚠️ INPUT REQUIRED</div><div class='badge-desc'>Please provide both a Claim and Evidence passages.</div></div>",
            "",
            "Please fill in both input fields above."
        )

    formatted_ev = format_evidence(evidence)
    prompt = PROMPT_TEMPLATE.format(claim=claim.strip(), evidence_block=formatted_ev)

    try:
        m, p = load_verification_model()
        device = next(m.parameters()).device
        try:
            inputs = p(prompt, return_tensors="pt").to(device)
        except TypeError:
            inputs = p(text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = m.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
            )
        
        generated_text = p.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r"FINAL:\s*(SUPPORTS|REFUTES|NOT_ENOUGH_INFO)", generated_text, re.IGNORECASE)
        
        if match:
            prediction = match.group(1).upper()
        else:
            prediction = "PARSE_ERROR"

        badge_html = {
            "SUPPORTS": """
                <div class="result-card supports">
                    <div class="badge-icon">✅</div>
                    <div class="badge-body">
                        <div class="badge-title">SUPPORTS</div>
                        <div class="badge-desc">The supplied evidence directly establishes the truth of the claim.</div>
                    </div>
                </div>
            """,
            "REFUTES": """
                <div class="result-card refutes">
                    <div class="badge-icon">❌</div>
                    <div class="badge-body">
                        <div class="badge-title">REFUTES</div>
                        <div class="badge-desc">The supplied evidence directly contradicts the claim.</div>
                    </div>
                </div>
            """,
            "NOT_ENOUGH_INFO": """
                <div class="result-card nei">
                    <div class="badge-icon">❓</div>
                    <div class="badge-body">
                        <div class="badge-title">NOT_ENOUGH_INFO</div>
                        <div class="badge-desc">The supplied evidence neither establishes nor contradicts the claim.</div>
                    </div>
                </div>
            """,
        }.get(
            prediction,
            f"<div class='result-card nei'><div class='badge-title'>{prediction}</div><div class='badge-desc'>Raw: {generated_text}</div></div>"
        )

        summary_md = f"**Prediction Label**: `{prediction}`\n\n**Evaluated Evidence Passages**:\n```text\n{formatted_ev}\n```"
        return badge_html, prompt, summary_md

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ MODEL INFERENCE ERROR:\n{err_msg}")
        error_html = f"""
        <div class="result-card refutes">
            <div class="badge-icon">⚠️</div>
            <div class="badge-body">
                <div class="badge-title">INFERENCE ERROR</div>
                <div class="badge-desc">{str(e)}</div>
            </div>
        </div>
        """
        return error_html, prompt, f"```text\n{err_msg}\n```"

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #0f172a !important;
    color: #f8fafc !important;
}

.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(67, 56, 202, 0.4);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #c7d2fe;
    font-weight: 600;
}

.champ-badge {
    display: inline-block;
    background: linear-gradient(90deg, #f59e0b, #d97706);
    color: #ffffff;
    font-weight: 800;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
}

.result-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px;
    border-radius: 14px;
    transition: transform 0.2s ease;
}

.result-card.supports {
    background: rgba(16, 185, 129, 0.15);
    border: 2px solid #10b981;
    color: #34d399;
}

.result-card.refutes {
    background: rgba(239, 68, 68, 0.15);
    border: 2px solid #ef4444;
    color: #f87171;
}

.result-card.nei {
    background: rgba(245, 158, 11, 0.15);
    border: 2px solid #f59e0b;
    color: #fbbf24;
}

.badge-icon {
    font-size: 2.8rem;
}

.badge-title {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.badge-desc {
    font-size: 0.95rem;
    opacity: 0.9;
    margin-top: 4px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Gemma 4 12B Claim Verification — 1st Place Winner") as demo:
    gr.HTML(
        """
        <div class="hero-banner">
            <div class="champ-badge">🏆 1st PLACE WINNER — AI SEEKHO DAY 2026</div>
            <div class="hero-title">Gemma 4 12B Evidence Claim Verifier</div>
            <div class="hero-subtitle">Fine-Tuned 12B QLoRA Model by <b>Muhammad Umar Farooq</b></div>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            claim_input = gr.Textbox(
                label="🔍 Natural Language Claim",
                placeholder="e.g., Omer is a high achiever in his university.",
                lines=2,
            )
            evidence_input = gr.Textbox(
                label="📜 Supplied Evidence Passages (One passage per line)",
                placeholder="[1] Omer won the hackathon in recent days.\n[2] Omer has scored 3 awards in his university and has always been a high achiever.",
                lines=5,
            )
            submit_btn = gr.Button("⚡ Verify Claim Relation", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_html = gr.HTML(
                value="""
                <div class="result-card nei" style="opacity: 0.7;">
                    <div class="badge-icon">💡</div>
                    <div class="badge-body">
                        <div class="badge-title">READY TO EVALUATE</div>
                        <div class="badge-desc">Enter a Claim and Evidence to generate instant evidence-grounded verification.</div>
                    </div>
                </div>
                """,
                label="Verification Result",
            )
            summary_output = gr.Markdown(label="Evaluation Summary")

    with gr.Accordion("🔍 View Formatted Prompt Structure (Sent to Gemma 4 12B)", open=False):
        prompt_preview = gr.Code(label="Formatted Prompt Structure", language="markdown")

    submit_btn.click(
        fn=verify_claim,
        inputs=[claim_input, evidence_input],
        outputs=[result_html, prompt_preview, summary_output],
    )

    gr.Examples(
        examples=[
            [
                "Omer is a high achiever in his university.",
                "[1] Omer won the hackathon in recent days.\n[2] Omer has scored 3 awards in his university and has always been a high achiever.",
            ],
            [
                "Global renewable energy capacity additions increased in 2023.",
                "[1] According to the IEA, global renewable energy capacity additions increased by 50% in 2023.\n[2] Solar energy accounted for the majority of new capacity additions.",
            ],
            [
                "The company's annual revenue declined in 2023 compared to 2022.",
                "[1] The official financial audit confirms annual revenue rose from $1.2B in 2022 to $1.5B in 2023.",
            ],
            [
                "NASA launched the Artemis III mission to Mars in 2025.",
                "[1] NASA announced new astronaut training programs for future lunar exploration.\n[2] Deep space exploration budgets were expanded.",
            ]
        ],
        inputs=[claim_input, evidence_input],
        outputs=[result_html, prompt_preview, summary_output],
        fn=verify_claim,
        cache_examples=False,
        label="⚡ Try Sample Claim Verification Scenarios",
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
