---
title: BiasLens
emoji: 🔎
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# BiasLens

Detects social bias in Vietnamese (and English) generated text across 14 categories, each rated on a
1-5 severity scale, using a LoRA-fine-tuned Qwen2.5-1.5B-Instruct model.

Full project, training pipeline, and writeup:
https://github.com/Baoho113/End-to-End-LLM-Pipeline-for-Vietnamese-generated-text-bias-

## Deploying this Space

1. Upload the LoRA adapter (`checkpoints/qwen_severity_lora_stratified/`) to the Hugging Face Hub model
   repo (`hf upload Caledfwlch/biaslens-qwen2.5-lora checkpoints/qwen_severity_lora_stratified .`).
2. Create a new Space (SDK: Gradio, hardware: free CPU tier is enough) and push this `space/` folder's
   contents (`app.py`, `requirements.txt`, this `README.md`) as the Space's repo root.
3. The app defaults to `Caledfwlch/biaslens-qwen2.5-lora`; an optional `MODEL_PATH` variable can override it.

CPU inference takes a few seconds per request -- there is no GPU on the free Spaces tier.
