"""
Gradio Web Application for Gemma 4 12B Claim Verification
AI Seekho Day 2026 — 1st Place Winning Solution
Author: Muhammad Umar Farooq
"""

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

def gpu_decorator(func):
    if HAS_SPACES:
        return spaces.GPU(func)
    return func

import os
from html import escape
import re
import traceback
import torch
import gradio as gr
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4UnifiedForConditionalGeneration
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

READY_HTML = """
<div class="result-card ready">
    <div class="result-kicker">VERDICT</div>
    <div class="result-mark">—</div>
    <div class="badge-title">Ready when you are</div>
    <div class="badge-desc">Add a claim and the evidence you want it checked against.</div>
</div>
"""

INPUT_REQUIRED_HTML = """
<div class="result-card warning">
    <div class="result-kicker">CHECK YOUR INPUT</div>
    <div class="result-mark">!</div>
    <div class="badge-title">Claim and evidence are required</div>
    <div class="badge-desc">Complete both fields, then run the verification again.</div>
</div>
"""

# Global model cache
processor = None
base_model = None
model = None

def load_base_model_and_processor():
    global processor, base_model
    if base_model is not None:
        return base_model, processor

    print(f"Loading processor for {MODEL_ID}...")
    loaded_processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

    print("Loading 4-bit NF4 quantized base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    loaded_base_model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )

    base_model, processor = loaded_base_model, loaded_processor
    return base_model, processor


def load_verification_model():
    global model
    if model is not None:
        return model, processor

    loaded_base_model, loaded_processor = load_base_model_and_processor()

    print(f"Attaching LoRA adapter from {ADAPTER_ID}...")
    loaded_model = PeftModel.from_pretrained(
        loaded_base_model, ADAPTER_ID, token=HF_TOKEN,
    )
    loaded_model.eval()
    model = loaded_model
    return model, loaded_processor

# ZeroGPU prepares CUDA state during startup, before allocating a GPU to requests.
if os.environ.get("SPACES_ZERO_GPU") == "1":
    load_base_model_and_processor()


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
            INPUT_REQUIRED_HTML,
            "",
            ""
        )

    formatted_ev = format_evidence(evidence)
    prompt = PROMPT_TEMPLATE.format(claim=claim.strip(), evidence_block=formatted_ev)

    try:
        m, p = load_verification_model()
        device = next(m.parameters()).device
        chat_prompt = p.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = p(
            text=chat_prompt, return_tensors="pt", add_special_tokens=False,
        ).to(device)
        tokenizer = p.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            outputs = m.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
            )
        
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        matches = re.findall(
            r"\bFINAL\s*:\s*(SUPPORTS|REFUTES|NOT_ENOUGH_INFO)\b",
            generated_text, re.IGNORECASE,
        )
        prediction = matches[-1].upper() if matches else "PARSE_ERROR"

        badge_html = {
            "SUPPORTS": """
                <div class="result-card supports">
                    <div class="result-kicker">VERDICT</div>
                    <div class="result-mark">✓</div>
                    <div class="badge-title">Supports</div>
                    <div class="badge-desc">The supplied evidence establishes the claim.</div>
                </div>
            """,
            "REFUTES": """
                <div class="result-card refutes">
                    <div class="result-kicker">VERDICT</div>
                    <div class="result-mark">×</div>
                    <div class="badge-title">Refutes</div>
                    <div class="badge-desc">The supplied evidence contradicts the claim.</div>
                </div>
            """,
            "NOT_ENOUGH_INFO": """
                <div class="result-card nei">
                    <div class="result-kicker">VERDICT</div>
                    <div class="result-mark">?</div>
                    <div class="badge-title">Not enough information</div>
                    <div class="badge-desc">The evidence does not establish or contradict the claim.</div>
                </div>
            """,
        }.get(
            prediction,
            f"<div class='result-card warning'><div class='result-kicker'>MODEL OUTPUT</div><div class='result-mark'>!</div><div class='badge-title'>Could not read a verdict</div><div class='badge-desc'>{escape(generated_text)}</div></div>"
        )

        passage_count = len(formatted_ev.splitlines())
        passage_word = "passage" if passage_count == 1 else "passages"
        summary_md = (
            f"**Model label:** `{prediction}`  \n"
            f"**Evidence reviewed:** {passage_count} {passage_word}"
        )
        return badge_html, prompt, summary_md

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ MODEL INFERENCE ERROR:\n{err_msg}")
        error_html = f"""
        <div class="result-card warning">
            <div class="result-kicker">INFERENCE ERROR</div>
            <div class="result-mark">!</div>
            <div class="badge-title">The model could not finish this request</div>
            <div class="badge-desc">{escape(str(e))}</div>
        </div>
        """
        return error_html, prompt, f"```text\n{err_msg}\n```"


def reset_form():
    return "", "", READY_HTML, "", ""

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

:root {
    --page: #08101f;
    --surface: rgba(17, 27, 48, 0.88);
    --surface-soft: rgba(24, 36, 61, 0.78);
    --line: rgba(148, 163, 184, 0.16);
    --text: #f7f9fc;
    --muted: #a7b2c7;
    --accent: #8b83ff;
    --accent-bright: #a9a4ff;
}

html, body, .gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--page) !important;
    color: var(--text) !important;
}

body {
    background-image:
        radial-gradient(circle at 12% 5%, rgba(99, 102, 241, 0.16), transparent 28rem),
        radial-gradient(circle at 88% 14%, rgba(14, 165, 233, 0.10), transparent 24rem) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    width: calc(100% - 48px) !important;
    max-width: 1540px !important;
    margin: 0 auto !important;
    padding: 34px 22px 56px !important;
}

.hero-banner {
    padding: 28px 20px;
    margin-bottom: 12px;
    border: 1px solid rgba(129, 140, 248, 0.30);
    border-radius: 16px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    box-shadow: 0 10px 30px -10px rgba(67, 56, 202, 0.40);
    text-align: center;
}

.champ-badge {
    display: inline-block;
    padding: 6px 16px;
    margin-bottom: 10px;
    border-radius: 20px;
    background: linear-gradient(90deg, #f59e0b, #d97706);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.30);
    color: #ffffff;
    font-size: 0.9rem;
    font-weight: 800;
}

.hero-title {
    margin: 0 0 6px;
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    margin: 0;
    color: #c7d2fe;
    font-size: 1.05rem;
    font-weight: 600;
}

.identity-bar {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin: 0 0 20px;
}

.identity-link {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 13px;
    border: 1px solid rgba(167, 162, 255, 0.24);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.055);
    color: #f1f3ff !important;
    font-size: 0.79rem;
    font-weight: 700;
    text-decoration: none !important;
    transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
}

.identity-link:hover {
    transform: translateY(-1px);
    border-color: rgba(167, 162, 255, 0.48);
    background: rgba(255, 255, 255, 0.09);
}

.identity-link svg {
    width: 16px;
    height: 16px;
    flex: none;
}

.workspace {
    gap: 18px !important;
    align-items: stretch !important;
}

.panel {
    min-width: 0 !important;
    padding: 24px !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    background: var(--surface) !important;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.16) !important;
}

.section-heading {
    margin: 0 0 18px;
}

.section-heading h2 {
    margin: 0 0 5px !important;
    color: var(--text) !important;
    font-size: 1.12rem !important;
    letter-spacing: -0.02em;
}

.section-heading p {
    margin: 0 !important;
    color: var(--muted) !important;
    font-size: 0.84rem !important;
    line-height: 1.55 !important;
}

.input-field {
    margin-bottom: 14px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.panel .form {
    gap: 14px !important;
    border: 0 !important;
    background: transparent !important;
}

.input-field > label.container {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.input-field [data-testid="block-info"] {
    color: #dfe5f2 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
}

.input-field textarea {
    border: 1px solid rgba(148, 163, 184, 0.20) !important;
    border-radius: 12px !important;
    background: #0d1729 !important;
    color: #f8fafc !important;
    box-shadow: none !important;
    font-size: 0.91rem !important;
    line-height: 1.55 !important;
}

.input-field textarea::placeholder {
    color: #66748d !important;
}

.input-field textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139, 131, 255, 0.12) !important;
}

.action-row {
    gap: 10px !important;
    margin-top: 4px !important;
}

#verify-button {
    min-height: 46px !important;
    border: 0 !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #756cf3, #635bdb) !important;
    color: white !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 28px rgba(99, 91, 219, 0.26) !important;
}

#verify-button:hover {
    background: linear-gradient(135deg, #8981ff, #7067ec) !important;
}

#clear-button {
    min-height: 46px !important;
    border: 1px solid rgba(148, 163, 184, 0.20) !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.035) !important;
    color: #c5cede !important;
}

.result-shell {
    height: 100%;
}

.result-card {
    display: flex;
    min-height: 260px;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    position: relative;
    overflow: hidden;
    padding: 30px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: var(--surface-soft);
}

.result-card::after {
    content: '';
    position: absolute;
    width: 180px;
    height: 180px;
    right: -75px;
    bottom: -95px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.06;
}

.result-kicker {
    margin-bottom: 19px;
    color: currentColor;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.16em;
}

.result-mark {
    display: grid;
    width: 48px;
    height: 48px;
    place-items: center;
    margin-bottom: 17px;
    border: 1px solid currentColor;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.055);
    font-size: 1.5rem;
    font-weight: 800;
}

.badge-title {
    max-width: 440px;
    color: #ffffff;
    font-size: clamp(1.45rem, 3vw, 2rem);
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -0.035em;
}

.badge-desc {
    max-width: 440px;
    margin-top: 9px;
    color: #bdc7da;
    font-size: 0.88rem;
    line-height: 1.6;
}

.result-card.ready { color: #8b95aa; }
.result-card.supports { color: #37d39a; background: rgba(13, 55, 48, 0.64); border-color: rgba(55, 211, 154, 0.30); }
.result-card.refutes { color: #ff7b89; background: rgba(66, 25, 37, 0.66); border-color: rgba(255, 123, 137, 0.30); }
.result-card.nei { color: #f7bd58; background: rgba(65, 46, 22, 0.66); border-color: rgba(247, 189, 88, 0.30); }
.result-card.warning { color: #f7bd58; background: rgba(65, 46, 22, 0.66); border-color: rgba(247, 189, 88, 0.30); }

.summary-box {
    min-height: 46px !important;
    margin-top: 12px !important;
    padding: 10px 13px !important;
    border: 1px solid var(--line) !important;
    border-radius: 11px !important;
    background: rgba(255, 255, 255, 0.025) !important;
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

.summary-box:empty { display: none !important; }
.summary-box:has(.prose:empty) { display: none !important; }

.details-accordion, .examples-block {
    margin-top: 16px !important;
    border: 1px solid var(--line) !important;
    border-radius: 15px !important;
    background: rgba(17, 27, 48, 0.66) !important;
    overflow: hidden !important;
}

.details-accordion > button {
    color: #cbd4e4 !important;
    font-size: 0.83rem !important;
}

.examples-block {
    padding: 18px !important;
}

.examples-block .gallery,
.examples-block table {
    border-color: var(--line) !important;
    background: transparent !important;
}

.footer-note {
    margin: 22px 4px 0;
    color: #77849b;
    font-size: 0.75rem;
    line-height: 1.6;
    text-align: center;
}

.footer-note a { color: #aaa5ff !important; text-decoration: none !important; }

@media (max-width: 760px) {
    .gradio-container { width: 100% !important; padding: 18px 12px 36px !important; }
    .hero-banner { padding: 24px 18px; }
    .hero-title { font-size: 1.8rem; }
    .panel { padding: 18px !important; border-radius: 16px !important; }
    .result-card { min-height: 230px; padding: 24px; }
    .action-row { flex-direction: column !important; }
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, css=custom_css, title="Gemma Claim Verification") as demo:
    gr.HTML(
        """
        <div class="hero-banner">
            <div class="champ-badge">🏆 1st PLACE WINNER — AI SEEKHO DAY 2026</div>
            <div class="hero-title">Gemma 4 12B Evidence Claim Verifier</div>
            <div class="hero-subtitle">Fine-Tuned 12B QLoRA Model by <b>Muhammad Umar Farooq</b></div>
        </div>
        <div class="identity-bar">
                <a class="identity-link" href="https://omerfarooq223.github.io/" target="_blank" rel="noopener noreferrer" aria-label="Visit Muhammad Umar Farooq's portfolio">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="8" r="3.5"></circle><path d="M5.5 20c.7-4.1 3-6 6.5-6s5.8 1.9 6.5 6"></path></svg>
                    Muhammad Umar Farooq
                </a>
                <a class="identity-link" href="https://github.com/omerfarooq223/gemma-claim-verification" target="_blank" rel="noopener noreferrer" aria-label="Open the Gemma claim verification repository">
                    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .7A11.3 11.3 0 0 0 8.4 22.8c.6.1.8-.2.8-.6v-2.2c-3.4.7-4.1-1.4-4.1-1.4-.5-1.4-1.4-1.8-1.4-1.8-1.1-.8.1-.8.1-.8 1.2.1 1.9 1.3 1.9 1.3 1.1 1.9 2.9 1.4 3.6 1.1.1-.8.4-1.4.8-1.7-2.7-.3-5.5-1.4-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.6.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.6.2 2.9.1 3.2.8.9 1.2 1.9 1.2 3.2 0 4.5-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .4.2.7.8.6A11.3 11.3 0 0 0 12 .7Z"></path></svg>
                    GitHub repository
                </a>
        </div>
        """
    )

    with gr.Row(equal_height=True, elem_classes=["workspace"]):
        with gr.Column(scale=6, elem_classes=["panel"]):
            gr.HTML(
                """
                <div class="section-heading">
                    <h2>What should I verify?</h2>
                    <p>Write one claim, then add one evidence passage per line.</p>
                </div>
                """
            )
            claim_input = gr.Textbox(
                label="Claim",
                placeholder="Example: The company's revenue declined in 2023.",
                lines=2,
                elem_classes=["input-field"],
            )
            evidence_input = gr.Textbox(
                label="Evidence",
                placeholder="Example: The annual report states that revenue rose from $1.2B in 2022 to $1.5B in 2023.",
                lines=5,
                elem_classes=["input-field"],
            )
            with gr.Row(elem_classes=["action-row"]):
                submit_btn = gr.Button(
                    "Verify claim", variant="primary", size="lg", scale=4,
                    elem_id="verify-button",
                )
                clear_btn = gr.Button(
                    "Clear", variant="secondary", size="lg", scale=1,
                    elem_id="clear-button",
                )

        with gr.Column(scale=5, elem_classes=["panel", "result-shell"]):
            gr.HTML(
                """
                <div class="section-heading">
                    <h2>Result</h2>
                    <p>The model returns one of three evidence relationships.</p>
                </div>
                """
            )
            result_html = gr.HTML(
                value=READY_HTML,
                label="Verification Result",
            )
            summary_output = gr.Markdown(elem_classes=["summary-box"])

    with gr.Accordion("See the exact model prompt", open=False, elem_classes=["details-accordion"]):
        prompt_preview = gr.Code(label="Prompt sent to Gemma", language="markdown")

    submit_btn.click(
        fn=verify_claim,
        inputs=[claim_input, evidence_input],
        outputs=[result_html, prompt_preview, summary_output],
    )

    clear_btn.click(
        fn=reset_form,
        inputs=[],
        outputs=[claim_input, evidence_input, result_html, prompt_preview, summary_output],
        queue=False,
    )

    with gr.Column(elem_classes=["examples-block"]):
        gr.Examples(
            examples=[
                [
                    "Global renewable energy capacity additions increased in 2023.",
                    "According to the IEA, global renewable energy capacity additions increased by 50% in 2023.\nSolar accounted for most new capacity.",
                ],
                [
                    "The company's annual revenue declined in 2023.",
                    "The audited report confirms annual revenue rose from $1.2B in 2022 to $1.5B in 2023.",
                ],
                [
                    "The Artemis III mission launched in 2025.",
                    "NASA announced new astronaut training programs for future lunar exploration.\nDeep-space exploration budgets were expanded.",
                ],
            ],
            inputs=[claim_input, evidence_input],
            outputs=[result_html, prompt_preview, summary_output],
            fn=verify_claim,
            cache_examples=False,
            label="Try an example",
        )

    gr.HTML(
        """
        <div class="footer-note">
            Built by <a href="https://omerfarooq223.github.io/" target="_blank">Muhammad Umar Farooq</a>
            · <a href="https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora" target="_blank">Model card</a>
            · The verifier evaluates supplied evidence; it does not search the web.
        </div>
        """
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
