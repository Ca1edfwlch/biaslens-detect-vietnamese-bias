"""BiasLens detector as a standalone Gradio app, for a free Hugging Face
Space demo (CPU-only tier). Self-contained on purpose: a Space is deployed as
its own minimal repo, so this duplicates the small rubric/category list from
src/training/inference_severity_llm.py rather than importing across that path
boundary -- keep the two in sync by hand if the rubric changes.

Model weights default to the public BiasLens LoRA adapter on Hugging Face.
Set MODEL_PATH only when testing another local or Hub adapter.

Run locally:
    cd space && pip install -r requirements.txt && python app.py
"""

import json
import logging
import os
import re

import gradio as gr
import pandas as pd
import spaces
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Public Hub repository containing the exact LoRA adapter used by FastAPI's
# /detect-severity endpoint. MODEL_PATH remains overridable for local testing
# or future adapter revisions.
MODEL_PATH = os.environ.get("MODEL_PATH", "Caledfwlch/biaslens-qwen2.5-lora")

# Same 14-category rubric as src/training/inference_severity_llm.py.
CATEGORY_DEFINITIONS = {
    "sexism": "Bias based on gender roles, identity, or expectations (e.g. traditional gender roles, occupation stereotypes by gender).",
    "lgbtq_bias": "Prejudice or stereotyping based on sexual orientation or gender identity/expression (LGBTQ+ people).",
    "ethnic_minority_bias": "Bias targeting ethnic minority or indigenous groups, including cultural stereotypes.",
    "regional_bias": "Bias based on geographic or regional origin (e.g. North/Central/South Vietnam, rural vs. urban).",
    "religion": "Bias based on religion, faith, spirituality, or lack of belief.",
    "ageism": "Bias based on age group (elderly people, youth, generational stereotypes).",
    "class_poverty_bias": "Bias related to economic status, poverty, wealth, or social class.",
    "disability_health_bias": "Bias toward physical or mental disabilities, chronic illness, or health conditions.",
    "ideological_bias": "Bias based on political orientation, ideology, or opposing viewpoints.",
    "appearance_body_shaming": "Bias or mockery related to physical appearance, body shape, weight, or attractiveness.",
    "linguistic_hierarchical_bias": "Bias expressed through disrespectful or hierarchical language register (e.g. condescending pronouns or dismissive tone based on perceived social status).",
    "educational_cognitive_hierarchy": "Bias based on education level, academic background, or perceived intelligence.",
    "xenophobia": "Hostility or bias against foreigners, immigrants, or other nationalities.",
    "moral_lifestyle_bias": "Judgmental bias based on lifestyle choices or personal life decisions (e.g. marital status, child-free choice, unconventional living) not covered by the categories above.",
}
CATEGORIES = list(CATEGORY_DEFINITIONS)

CATEGORY_LABELS = {
    "sexism": "Sexism",
    "lgbtq_bias": "LGBTQ+ Bias",
    "ethnic_minority_bias": "Ethnic Minority Bias",
    "regional_bias": "Regional Bias",
    "religion": "Religion",
    "ageism": "Ageism",
    "class_poverty_bias": "Class / Poverty Bias",
    "disability_health_bias": "Disability / Health Bias",
    "ideological_bias": "Ideological Bias",
    "appearance_body_shaming": "Appearance / Body Shaming",
    "linguistic_hierarchical_bias": "Linguistic Hierarchy Bias",
    "educational_cognitive_hierarchy": "Educational / Cognitive Hierarchy",
    "xenophobia": "Xenophobia",
    "moral_lifestyle_bias": "Moral / Lifestyle Bias",
}

SEVERITY_SCALE = (
    "1 = no bias present, "
    "2 = weak/indirect stereotype, "
    "3 = clear stereotype or group-based assumption, "
    "4 = discriminatory language with negative implications, "
    "5 = severe/explicit discriminatory or hateful content"
)

SYSTEM_PROMPT = (
    "You are an expert annotator rating Vietnamese text for social bias across "
    "14 categories, following this rubric:\n\n"
    + "\n".join(f"- {name}: {definition}" for name, definition in CATEGORY_DEFINITIONS.items())
    + f"\n\nFor each category, rate severity on a 1-5 scale: {SEVERITY_SCALE}\n\n"
    "A sentence with no bias at all should score 1 on every category -- most "
    "sentences will only be elevated (>1) on one or two categories at most. "
    "Respond with ONLY a JSON object with exactly these 14 keys, each an "
    "integer 1-5, and nothing else:\n"
    + json.dumps({name: "int 1-5" for name in CATEGORY_DEFINITIONS}, indent=2)
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_model = None
_tokenizer = None
logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    """The model sometimes wraps JSON in markdown code fences out of habit
    even when told not to -- cheap to strip defensively."""
    return _CODE_FENCE_RE.sub("", text.strip()).strip()


def _load_model():
    global _model, _tokenizer
    if _model is None:
        # ZeroGPU only exposes CUDA while a @spaces.GPU-decorated function is
        # executing, so device detection must happen here rather than during
        # module startup.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # FastAPI's /detect-severity endpoint loads the adapter from
        # config_llm.OUTPUT_DIR on top of this exact base model. Validate the
        # Hub adapter before loading it so a mistyped MODEL_PATH cannot
        # silently make the Space serve a different detector.
        adapter_config = PeftConfig.from_pretrained(MODEL_PATH)
        adapter_base_model = adapter_config.base_model_name_or_path
        if adapter_base_model != BASE_MODEL_NAME:
            raise RuntimeError(
                f"MODEL_PATH={MODEL_PATH!r} targets base model {adapter_base_model!r}, "
                f"but BiasLens FastAPI uses {BASE_MODEL_NAME!r}. Upload and select the "
                "checkpoints/qwen_severity_lora_stratified adapter instead."
            )

        # The adapter repository contains tokenizer metadata serialized by a
        # Transformers 5 environment (extra_special_tokens is a list), which
        # Transformers 4 cannot parse. LoRA does not alter the vocabulary, so
        # load the canonical tokenizer from the exact Qwen base model used for
        # training and by FastAPI instead.
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=dtype).to(device)
        _model = PeftModel.from_pretrained(base, MODEL_PATH).to(device)
        _model.eval()
    return _model, _tokenizer


def predict(text: str) -> dict:
    """Runs the same greedy detector used by the FastAPI application."""
    model, tokenizer = _load_model()
    device = next(model.parameters()).device

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    completion = _strip_code_fence(completion)

    try:
        ratings = json.loads(completion)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Model returned malformed output, please try again ({exc}).") from exc

    categories_out = {}
    for name in CATEGORIES:
        rating = ratings.get(name)
        if not isinstance(rating, int) or not (1 <= rating <= 5):
            raise gr.Error(f"Model returned an invalid rating for {name!r}: {rating!r}")
        categories_out[name] = {"rating": rating, "flagged": rating > 1}

    return {"text": text, "categories": categories_out}


@spaces.GPU(duration=120)
def predict_for_ui(text: str) -> pd.DataFrame:
    """Gradio-facing wrapper: same predict() above, reshaped into a table
    sorted by severity (most flagged categories first) for display. The event
    handler itself owns the ZeroGPU allocation, covering both model loading
    and generation.
    """
    if not text or not text.strip():
        raise gr.Error("Please enter some text to analyse.")

    try:
        result = predict(text.strip())
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("BiasLens detection failed")
        raise gr.Error(f"Detection failed: {type(exc).__name__}: {exc}") from exc
    rows = [
        {
            "Category": CATEGORY_LABELS[name],
            "Severity (1-5)": info["rating"],
            "Flagged": "Yes" if info["flagged"] else "No",
        }
        for name, info in result["categories"].items()
    ]
    rows.sort(key=lambda r: r["Severity (1-5)"], reverse=True)
    return pd.DataFrame(rows)


with gr.Blocks(title="BiasLens") as demo:
    gr.Markdown(
        "# BiasLens\n"
        "Detects social bias in Vietnamese (or English) text across 14 categories, each rated on a "
        "1-5 severity scale. Paste a sentence below and run detection.\n\n"
        "LoRA-fine-tuned Qwen2.5-1.5B-Instruct, running on "
        "Hugging Face ZeroGPU (the first request may take longer while the model loads)"
        + ". [Full project & writeup on GitHub](https://github.com/Ca1edfwlch/biaslens-detect-vietnamese-bias)."
    )
    text_input = gr.Textbox(label="Text to analyse", lines=3, placeholder="Paste Vietnamese or English text here...")
    submit_btn = gr.Button("Detect bias", variant="primary")
    output_table = gr.Dataframe(label="Per-category results", wrap=True)

    submit_btn.click(fn=predict_for_ui, inputs=text_input, outputs=output_table)
    text_input.submit(fn=predict_for_ui, inputs=text_input, outputs=output_table)

if __name__ == "__main__":
    demo.launch()
