"""Tier-1 bias/severity classifier: a prompted LLM instead of a fine-tuned
model.

With only a few hundred to a couple thousand labeled Vietnamese examples
spread across 14 categories, training a classifier from scratch (the deep
PhoBERT fine-tune in train_severity.py, or the frozen-embedding shallow
classifier in train_severity_shallow.py) can't reliably learn the rarer
categories -- several ended up with single-digit or zero support in the real
eval runs. This dataset's own gold labels were themselves bootstrapped by an
LLM judge (google/gemma-4-26b-a4b, see DATASET_CARD.md) that human annotators
accepted 95.99% of the time -- strong evidence that a well-prompted general
LLM already performs close to human-level on this exact task, with no
training data required at all.

Reuses mitigate.py's OpenAI-compatible client / API-key resolution (RMIT VAL
gateway by default, Bitwarden-backed key) rather than duplicating it.

CLI usage:
    Single line:
        python inference_severity_llm.py "Nguoi gia thuong kho tiep thu."

    Mass score over a file (one sentence per line, or a CSV with a "text" column):
        python inference_severity_llm.py --file sentences.txt
        python inference_severity_llm.py --file input.csv --out results.json
"""

import argparse
import csv
import json
import os

from mitigate import _get_client

DETECTION_MODEL = os.environ.get("DETECTION_MODEL", "openai-gpt-4o")

# The 14 categories this dataset rates every human_input/model_output on
# (confirmed from the real dataset's eval_metrics.json category list -- see
# conversation history, not read from the dataset files directly). Definitions
# are adapted from docs/annotation_guideline.md where a category carries over
# from the original 13-class taxonomy, and written fresh for categories only
# introduced in the new VNFairness schema (lgbtq_bias, disability_health_bias,
# xenophobia, moral_lifestyle_bias, linguistic_hierarchical_bias).
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


def predict(text: str) -> dict:
    return predict_batch([text])[0]


def predict_batch(texts: list[str]) -> list[dict]:
    """One chat completion per text (matching mitigate.py's per-text call
    pattern) rather than batching multiple texts into one prompt, since a
    single malformed JSON response would otherwise sink the whole batch.
    """
    client = _get_client()
    results = []

    for text in texts:
        try:
            response = client.chat.completions.create(
                model=DETECTION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            ratings = json.loads(response.choices[0].message.content)
        except Exception as exc:
            raise RuntimeError(f"LLM detection request failed for text={text!r}: {exc}") from exc

        categories_out = {}
        for name in CATEGORIES:
            rating = ratings.get(name)
            if not isinstance(rating, int) or not (1 <= rating <= 5):
                raise ValueError(f"Model returned an invalid rating for {name!r}: {rating!r} (text={text!r})")
            categories_out[name] = {"rating": rating, "flagged": rating > 1}

        results.append({"text": text, "categories": categories_out})

    return results


def _read_texts(path: str) -> list[str]:
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "text" not in (reader.fieldnames or []):
                raise ValueError(f"{path} has no 'text' column")
            return [row["text"] for row in reader if row["text"].strip()]

    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="A single sentence to score")
    parser.add_argument("--file", help="Path to a .txt (one sentence per line) or .csv (with a 'text' column) to mass-score")
    parser.add_argument("--out", help="Write results to this JSON file instead of printing them (only with --file)")
    args = parser.parse_args()

    if args.file:
        texts = _read_texts(args.file)
        results = predict_batch(texts)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Wrote {len(results)} results -> {args.out}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        text = args.text or "Nguoi gia thuong kho tiep thu."
        print(json.dumps(predict(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
