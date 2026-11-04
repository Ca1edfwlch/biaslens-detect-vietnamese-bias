"""AI rewriting for flagged text, using an OpenAI-compatible chat API.

Task 4 from docs/task_definition.md ("Bias Mitigation Generation" / safer_text)
done with a hosted LLM instead of a fine-tuned ViT5 -- that's the v2/severity
track's job. Takes a sentence plus its detected category from
inference_test.predict() and asks a chat model to rewrite it into neutral
Vietnamese while preserving the original meaning.

Talks to RMIT's VAL gateway (an OpenAI-compatible proxy) by default, not
OpenAI's own api.openai.com -- that's what the "VAL-Balam-Key3" Bitwarden item
is scoped to. Override OPENAI_BASE_URL to point at a different OpenAI-
compatible endpoint (e.g. unset it, or set it to OpenAI's own API, if you're
using a personal OpenAI account key instead).

The OpenAI key isn't stored in .env -- it's pulled from the local Bitwarden
vault via the `bw` CLI. `bw` must already be logged in and unlocked (BW_SESSION
set in this process's environment); this deliberately doesn't attempt an
interactive `bw unlock`, since a server process has no way to satisfy a master
password prompt. See README for the one-time `bw login` / `bw unlock` steps.

CLI usage (manual testing, independent of the FastAPI service):
    Just check the key resolves (no OpenAI call, no tokens spent):
        python mitigate.py

    Full rewrite round trip:
        python mitigate.py "Người già thường khó tiếp thu." --label "Age Bias"
"""

import argparse
import json
import os
import shutil
import subprocess
from functools import lru_cache

from openai import OpenAI

BW_ITEM_NAME = os.environ.get("OPENAI_BW_ITEM", "VAL-Balam-Key3")
MITIGATION_MODEL = os.environ.get("MITIGATION_MODEL", "openai-gpt-4o")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://val.rmit.edu.au/api/")

# The openai SDK's own default timeout is 600s -- far longer than a UI button
# should ever sit spinning. A short rewrite normally comes back in a few
# seconds; if the gateway is degraded, fail fast with a clear error instead
# of hanging silently for up to 10 minutes.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("MITIGATION_TIMEOUT_SECONDS", "30"))

SYSTEM_PROMPT = (
    "Bạn là một biên tập viên trung lập. Nhiệm vụ của bạn là viết lại câu "
    "tiếng Việt do người dùng cung cấp để loại bỏ định kiến/thiên vị, trong "
    "khi vẫn giữ nguyên ý nghĩa và thông tin cốt lõi của câu gốc. Chỉ trả về "
    "câu đã viết lại, không thêm giải thích, không thêm dấu ngoặc kép."
)

# Per-category rewrite guidance, keyed by the same category names used by the
# detector (inference_severity_llm.CATEGORIES) -- not a separate prompt per
# category (that breaks down for multi-category text and 14x's the prompt
# maintenance), but specific instructions folded into the one shared rewrite
# prompt so the model knows exactly what to strip for each flagged category
# instead of just seeing a bare label name.
CATEGORY_MITIGATION_GUIDANCE = {
    "sexism": "Loại bỏ giả định hoặc kỳ vọng dựa trên giới tính (vai trò, nghề nghiệp, tính cách); giữ nguyên nội dung không liên quan đến giới.",
    "lgbtq_bias": "Loại bỏ định kiến hoặc ngôn từ miệt thị về xu hướng tính dục hay bản dạng giới; không thay đổi thông tin thực tế khác trong câu.",
    "ethnic_minority_bias": "Loại bỏ khuôn mẫu văn hoá hoặc định kiến gán cho một nhóm dân tộc thiểu số cụ thể; tránh khái quát hoá cho cả nhóm.",
    "regional_bias": "Loại bỏ định kiến vùng miền (Bắc/Trung/Nam, thành thị/nông thôn); không gán đặc điểm tiêu cực cho cả một vùng.",
    "religion": "Loại bỏ định kiến hoặc giả định tiêu cực về một tôn giáo, tín ngưỡng cụ thể; giữ thái độ trung lập với niềm tin của người khác.",
    "ageism": "Loại bỏ định kiến dựa trên độ tuổi (người già, người trẻ); không gán năng lực hay giá trị dựa trên tuổi tác.",
    "class_poverty_bias": "Loại bỏ định kiến về hoàn cảnh kinh tế, giàu nghèo, hay tầng lớp xã hội; không đánh giá năng lực dựa trên thu nhập.",
    "disability_health_bias": "Loại bỏ định kiến về khuyết tật, bệnh mãn tính hay tình trạng sức khoẻ tâm thần; không coi đó là điểm yếu hay gánh nặng.",
    "ideological_bias": "Loại bỏ định kiến về quan điểm chính trị hay tư tưởng; giữ câu trung lập, không công kích phe nào.",
    "appearance_body_shaming": "Loại bỏ lời chê bai hay định kiến về ngoại hình, cân nặng, vóc dáng; không đánh giá con người qua vẻ bề ngoài.",
    "linguistic_hierarchical_bias": "Loại bỏ giọng điệu bề trên, xưng hô miệt thị hay coi thường dựa trên cách nói năng; chuyển sang ngôn ngữ tôn trọng, ngang hàng.",
    "educational_cognitive_hierarchy": "Loại bỏ định kiến về trình độ học vấn hay năng lực trí tuệ; không đánh giá con người qua bằng cấp.",
    "xenophobia": "Loại bỏ định kiến hay thái độ thù địch với người nước ngoài, người nhập cư; giữ thái độ trung lập, tôn trọng.",
    "moral_lifestyle_bias": "Loại bỏ phán xét về lựa chọn lối sống cá nhân (hôn nhân, sinh con, cách sống); tôn trọng quyền lựa chọn của mỗi người.",
}


@lru_cache(maxsize=1)
def _get_api_key() -> str:
    """Resolves the OpenAI key: OPENAI_API_KEY env var first (explicit
    override / CI), otherwise `bw get password <BW_ITEM_NAME>`.
    """
    env_override = os.environ.get("OPENAI_API_KEY")
    if env_override:
        return env_override

    bw_path = shutil.which("bw")
    if bw_path is None:
        raise RuntimeError(
            "Bitwarden CLI ('bw') not found on PATH. Install it "
            "(https://bitwarden.com/help/cli/) and `bw login`, or set "
            "OPENAI_API_KEY directly to bypass Bitwarden."
        )

    try:
        result = subprocess.run(
            [bw_path, "get", "password", BW_ITEM_NAME],
            capture_output=True,
            text=True,
            check=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"`bw get password {BW_ITEM_NAME}` didn't respond within {REQUEST_TIMEOUT_SECONDS}s "
            "-- this usually means the vault is locked and `bw` is stuck instead of failing "
            "cleanly. Run `bw unlock` in this shell, export the BW_SESSION it prints, and "
            "restart uvicorn from that same shell."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"`bw get password {BW_ITEM_NAME}` failed -- is the vault unlocked "
            "(BW_SESSION set in this shell)? Run `bw unlock` and export the "
            f"session key it prints. stderr: {exc.stderr.strip()}"
        ) from exc

    key = result.stdout.strip()
    if not key:
        raise RuntimeError(f"Bitwarden item '{BW_ITEM_NAME}' returned an empty password.")
    return key


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    if OPENAI_BASE_URL:
        return OpenAI(
            api_key=_get_api_key(),
            base_url=OPENAI_BASE_URL.rstrip("/"),
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=1,
        )
    return OpenAI(api_key=_get_api_key(), timeout=REQUEST_TIMEOUT_SECONDS, max_retries=1)


def _build_category_hint(label: str | None, categories: list[str] | None) -> str:
    """Prefers `categories` (raw detector keys, e.g. ["sexism", "ageism"]) to
    build one specific rewrite instruction per flagged category -- this is
    what lets a single shared prompt handle a multi-category sentence
    correctly, since every flagged category's guidance goes into the same
    rewrite call instead of requiring one call per category. Falls back to
    the older bare `label` string (human-readable display names, comma
    joined by the caller) when `categories` isn't provided, so existing
    callers keep working unchanged.
    """
    if categories:
        lines = [CATEGORY_MITIGATION_GUIDANCE[c] for c in categories if c in CATEGORY_MITIGATION_GUIDANCE]
        if lines:
            bullets = "\n".join(f"- {line}" for line in lines)
            return f" Các định kiến đã phát hiện và cách xử lý:\n{bullets}"

    if label and label != "Non-bias":
        return f" Loại định kiến đã phát hiện: {label}."

    return ""


def mitigate(text: str, label: str | None = None, categories: list[str] | None = None) -> dict:
    """Rewrites `text` into a safer, non-biased version.

    `label` is a human-readable bias category (or comma-joined categories,
    e.g. "Sexism, Age Bias") kept for display/back-compat. `categories` is
    the detector's raw category keys (e.g. ["sexism", "ageism"]) and, when
    given, is preferred: it looks up CATEGORY_MITIGATION_GUIDANCE so the
    model gets a specific instruction per flagged category instead of just
    the bare label name.
    """
    client = _get_client()

    category_hint = _build_category_hint(label, categories)
    user_prompt = (
        "Viết lại câu sau để loại bỏ định kiến, giữ nguyên ý nghĩa cốt lõi."
        f"{category_hint}\n\nCâu gốc: {text}"
    )

    try:
        response = client.chat.completions.create(
            model=MITIGATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI mitigation request failed: {exc}") from exc

    safer_text = response.choices[0].message.content.strip()

    return {
        "original_text": text,
        "safer_text": safer_text,
        "label": label,
        "model": MITIGATION_MODEL,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "text", nargs="?", help="Sentence to rewrite; omit to only test that the API key resolves"
    )
    parser.add_argument("--label", help="Detected bias category to pass as context, e.g. 'Gender Bias'")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Raw detector category keys for specific per-category guidance, "
        "e.g. --categories sexism ageism (takes priority over --label)",
    )
    args = parser.parse_args()

    key = _get_api_key()
    masked = f"{key[:7]}...{key[-4:]}" if len(key) > 11 else "***"
    print(f"Resolved OpenAI key: {masked} ({len(key)} chars) -- source: "
          f"{'OPENAI_API_KEY env var' if os.environ.get('OPENAI_API_KEY') else f'Bitwarden item {BW_ITEM_NAME!r}'}")

    if args.text:
        print(f"Base URL: {OPENAI_BASE_URL or '(OpenAI default: api.openai.com)'}")
        print(f"Model: {MITIGATION_MODEL}")
        result = mitigate(args.text, label=args.label, categories=args.categories)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
