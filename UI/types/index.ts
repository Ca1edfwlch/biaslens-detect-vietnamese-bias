// --- Severity detection module (src/training/inference_llm_lora.py, served
// via FastAPI's /detect-severity) ---
// Real API response shape from inference_llm_lora.py:predict() -- a
// 14-category multi-label detector (LoRA-fine-tuned Qwen2.5), replacing the
// old 13-category single-label classifier this UI used to call.

export interface SeverityCategoryResult {
  rating: number
  flagged: boolean
}

export interface DetectionResult {
  text: string
  categories: Record<string, SeverityCategoryResult>
}

// --- Mitigation module (src/training/mitigate.py, OpenAI rewrite) ---

export interface MitigationResult {
  original_text: string
  safer_text: string
  label: string | null
  model: string
}

// --- Evaluation snapshot (src/training/evaluate_llm_lora.py, served via
// FastAPI's /metrics) ---
// Real shape written by evaluate_llm_lora.py: two views of the same test-set
// predictions -- exact 1-5 severity match, and the easier-to-read binary
// "was this category flagged at all" view the Evaluate stage displays.

export interface BinaryCategoryMetric {
  category: string
  precision: number
  recall: number
  f1: number
  support: number
}

export interface SeverityCategoryMetric {
  category: string
  macro_precision: number
  macro_recall: number
  macro_f1: number
  exact_match_rate: number
  support: number
}

export interface EvalMetrics {
  generated_at: string
  model: string
  dataset: { train: number; val: number; test: number }
  severity: {
    overall: { macro_f1_across_categories: number; all_categories_exact_match: number }
    per_category: SeverityCategoryMetric[]
  }
  binary_presence: {
    overall: { macro_f1_across_categories: number; all_categories_exact_match: number }
    per_category: BinaryCategoryMetric[]
  }
}
