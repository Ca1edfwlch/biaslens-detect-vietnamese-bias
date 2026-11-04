import os

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Overridable via env vars so run_cv.py can point a training/eval subprocess
# at one fold's data/checkpoint dir without touching the defaults every other
# script (serve.py included) relies on.
TRAIN_PATH = os.environ.get("BIASLENS_TRAIN_PATH", "dataset/processed_v2/train.csv")

VAL_PATH = os.environ.get("BIASLENS_VAL_PATH", "dataset/processed_v2/val.csv")

TEST_PATH = os.environ.get("BIASLENS_TEST_PATH", "dataset/processed_v2/test.csv")

CATEGORY_MAPPING_PATH = "dataset/metadata/category_mapping_v2.json"

OUTPUT_DIR = os.environ.get("BIASLENS_OUTPUT_DIR", "checkpoints/qwen_severity_lora_stratified")

# The 14-category rubric (system prompt) alone is ~675 tokens; a short
# sentence + compact JSON completion adds another ~100-150. 1024 gives
# headroom for longer real sentences without silently dropping rows.
MAX_LENGTH = 1024

# LoRA hyperparameters. Targeting all attention + MLP projection layers (not
# just attention) gives the adapter more capacity to shift the model's
# judgments without touching the frozen base weights.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Small per-step batch with gradient accumulation to fit 6GB VRAM while still
# getting a reasonable effective batch size.
BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 8

# LoRA adapters typically want a higher LR than full fine-tuning since far
# fewer parameters are updating.
LEARNING_RATE = 2e-4
EPOCHS = 3
WARMUP_RATIO = 0.1

# Assumes a 1-5 raw rating scale in the processed CSVs -- verify against the
# real dataset/raw_v2/ batch; fix this constant if the range differs.
NUM_SEVERITY_LEVELS = 5

SEED = 42

# Per-example oversampling weight cap for train_llm_lora.py's WeightedTrainer.
# A row's weight is the inverse frequency (within train.csv) of its rarest
# flagged category, capped here -- uncapped inverse-frequency weighting was
# tried on the earlier from-scratch PhoBERT classifier and caused training to
# collapse (a handful of ultra-rare-category rows dominating every batch).
# Capping keeps rare categories oversampled without letting any one row
# dominate.
SAMPLE_WEIGHT_CAP = 10.0
