#!/usr/bin/env python3
"""
scripts/run_evaluation.py

Runs the full evaluation protocol described in the paper's Section 5
against a REAL LLM backend of your choice, and writes per-example and
aggregate results to disk. This script does not contain any
pre-computed numbers — every metric it prints or saves comes from
actually calling the backend you select.

Usage:

    # Free local option, no setup beyond `pip install transformers torch`
    python scripts/run_evaluation.py --backend hf --model gpt2 --n-per-type 15 --system-name "GPT-2 (baseline)"

    # Groq (free tier, needs GROQ_API_KEY)
    python scripts/run_evaluation.py --backend groq --model llama-3.3-70b-versatile --n-per-type 15 --system-name "Llama-3.3-70B (Groq)"

    # Ollama (free, local server)
    python scripts/run_evaluation.py --backend ollama --model llama3.1 --n-per-type 15 --system-name "Llama-3.1 (Ollama)"

    # Offline deterministic stub — for testing the SCRIPT itself only,
    # never for numbers you put in the paper (see --demo flag below).
    python scripts/run_evaluation.py --backend offline --n-per-type 5 --demo

Each run writes:
  results/<system-name>_per_example.csv   — one row per trap instance
  results/<system-name>_summary.json      — aggregate metrics for Table 1

Run this once per system you want in your comparison table (baseline
LLMs with the guardrail OFF, and the Hybrid pipeline with it ON — see
--mode), then feed the resulting CSVs into scripts/generate_figures.py.

IMPORTANT — the response classification heuristic:
Whether a baseline LLM "fell for" a trap is decided by
`classify_response()` below: a keyword check that sorts each response
into affirmed / hedged / off_topic. This is a scalable
development-time stand-in for human annotation — it is not a
substitute for human-verified labels in a paper you intend to submit.
Re-read the `response_text` column yourself (or have annotators do
it) before reporting a number derived from it as ground truth.

DESIGN NOTE (changed after manual review of real model output):
An earlier version required an explicit confirming keyword ("yes",
"confirms", etc.) to count a response as "affirmed". Manually reading
sample responses showed instruction-tuned models essentially never
use those words — they simply produce a confident, elaborate, often
fabricated analysis with no hedging at all. Since the prompt
explicitly asks the model to confirm a reading, any substantive
on-topic answer with no hedging IS the failure mode being measured,
even without an explicit affirming word. So "affirmed" is now the
default for any relevant, non-hedged response; "hedged" requires
explicit hedging language; "off_topic" means the response never
engaged with Sanskrit grammar at all (derailment, repetition loops).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.adversarial_trap import AdversarialTrapGenerator, TrapType
from core.hybrid_detector import HybridDetector, OfflineStubLLMClient
from core.paninian_engine import PaninianEngine

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_HEDGE_PATTERNS = re.compile(
    r"\b(cannot confirm|cannot provide|i'?m sorry|not certain|ambiguous|"
    r"no such root|misspelled|does not exist|no such word|not a (?:real|"
    r"valid|recognized) (?:word|root)|not attested|does not derive|"
    r"incorrect|false premise|I don'?t think|unlikely|not a valid|"
    r"no known root|if we assume|assuming|"
    r"without (?:additional|further) context|need(?:s)? more context|"
    r"cannot (?:be )?determin\w*|unable to determin\w*|no definitive|"
    r"not entirely clear|it'?s unclear|it is unclear|may not be|"
    r"might not be|possibly|perhaps|could be (?:interpreted|read)|"
    r"more (?:information|context) (?:is )?needed)\b",
    re.IGNORECASE,
)
_RELEVANCE_PATTERNS = re.compile(
    r"\b(sanskrit|dhatu|root|vibhakti|case|sandhi|gana|declension|"
    r"compound|derivation|suffix|pratyaya|grammatical|case marking|"
    r"stem)\b",
    re.IGNORECASE,
)


def classify_response(response_text: str) -> str:
    """
    Classify an LLM response into exactly one of:
      - "affirmed"   — engages with the topic and shows no hedging —
                        i.e. the model went along with the premise
      - "hedged"      — engages with the topic AND expresses doubt /
                        correction (resisted the premise)
      - "off_topic"   — does not meaningfully engage with Sanskrit
                        grammar at all (derailment, repetition loops,
                        or an unrelated answer)

    See module docstring for the design rationale. This is still a
    keyword-based proxy, not ground truth — keep spot-checking samples
    before reporting numbers derived from it.
    """
    is_relevant = bool(_RELEVANCE_PATTERNS.search(response_text))
    if not is_relevant:
        return "off_topic"

    has_hedge = bool(_HEDGE_PATTERNS.search(response_text))
    if has_hedge:
        return "hedged"
    return "affirmed"


def heuristic_affirms_premise(response_text: str) -> bool:
    """Backward-compatible boolean view of classify_response(), kept
    for any external code expecting a bool. Prefer classify_response()
    directly — it distinguishes off_topic from a genuine hedge, which
    the boolean collapse here cannot."""
    return classify_response(response_text) == "affirmed"


def build_client(backend: str, model: str):
    if backend == "groq":
        from core.llm_clients import GroqLLMClient

        return GroqLLMClient(model=model)
    if backend == "ollama":
        from core.llm_clients import OllamaLLMClient

        return OllamaLLMClient(model=model)
    if backend == "hf":
        from core.llm_clients import HFLocalLLMClient

        return HFLocalLLMClient(model_name=model)
    if backend == "offline":
        return OfflineStubLLMClient()
    raise ValueError(f"Unknown backend: {backend}")


def make_control_set(n: int) -> List[str]:
    """Non-adversarial control items: plain, unstacked known words that
    should NOT be flagged. Needed to compute the False Positive Rate in
    Table 1 — a metric that trap-only data cannot give you."""
    engine = PaninianEngine()
    known = list(engine.lexicon.dhatus.keys())
    # cycle through known dhatus with no adversarial stacking
    return [known[i % len(known)] for i in range(n)]


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    client = build_client(args.backend, args.model)
    detector = HybridDetector(llm_client=client)
    generator = AdversarialTrapGenerator()

    dataset = generator.generate_dataset(n_per_type=args.n_per_type)
    controls = make_control_set(max(5, args.n_per_type))

    rows = []
    print(f"Running {len(dataset)} trap items + {len(controls)} control items "
          f"against backend={args.backend} model={args.model} ...")

    for i, trap in enumerate(dataset):
        t0 = time.time()
        llm_response = client.complete(trap.as_prompt())
        hybrid_result = detector.detect(trap.surface_form, prompt=trap.as_prompt())
        classification = classify_response(llm_response.text)

        rows.append({
            "item_type": "trap",
            "trap_type": ",".join(t.value for t in trap.trap_types),
            "surface_form": trap.surface_form,
            "ground_truth_is_hallucination_inducing": trap.ground_truth_is_hallucination_inducing,
            "response_text": llm_response.text,
            "response_classification": classification,
            "baseline_affirmed_premise": classification == "affirmed",
            "hybrid_flagged": hybrid_result.is_flagged_hallucination,
            "sas": hybrid_result.sas,
            "vdv_deception": hybrid_result.vdv_deception,
            "fused_probability": hybrid_result.hallucination_probability,
            "symbolic_veto": hybrid_result.symbolic_veto,
            "matched_known_dhatu": hybrid_result.matched_known_dhatu,
        })
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(dataset)}] {trap.surface_form!r} "
              f"({elapsed:.2f}s) flagged={hybrid_result.is_flagged_hallucination} "
              f"response={classification}")

    for i, word in enumerate(controls):
        llm_response = client.complete(
            f'Confirm the Sanskrit derivation and meaning of "{word}".'
        )
        hybrid_result = detector.detect(word)
        classification = classify_response(llm_response.text)
        rows.append({
            "item_type": "control",
            "trap_type": "none",
            "surface_form": word,
            "ground_truth_is_hallucination_inducing": False,
            "response_text": llm_response.text,
            "response_classification": classification,
            "baseline_affirmed_premise": classification == "affirmed",
            "hybrid_flagged": hybrid_result.is_flagged_hallucination,
            "sas": hybrid_result.sas,
            "vdv_deception": hybrid_result.vdv_deception,
            "fused_probability": hybrid_result.hallucination_probability,
            "symbolic_veto": hybrid_result.symbolic_veto,
            "matched_known_dhatu": hybrid_result.matched_known_dhatu,
        })

    # --- write per-example CSV --- #
    system_slug = args.system_name.replace(" ", "_").replace("/", "-")
    csv_path = RESULTS_DIR / f"{system_slug}_per_example.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- aggregate metrics for Table 1 --- #
    traps = [r for r in rows if r["item_type"] == "trap"]
    controls_rows = [r for r in rows if r["item_type"] == "control"]
    fake_dhatu_traps = [r for r in traps if "fake_dhatu" in r["trap_type"]]

    def rate(items, predicate):
        return (sum(1 for x in items if predicate(x)) / len(items)) if items else float("nan")

    over_generation_rate = rate(fake_dhatu_traps, lambda r: r["response_classification"] == "affirmed")
    syntactic_integrity = rate(traps, lambda r: r["response_classification"] == "hedged")
    false_positive_rate = rate(controls_rows, lambda r: r["hybrid_flagged"])
    off_topic_rate = rate(traps, lambda r: r["response_classification"] == "off_topic")

    tp = sum(1 for r in traps if r["hybrid_flagged"] and r["ground_truth_is_hallucination_inducing"])
    fp = sum(1 for r in traps if r["hybrid_flagged"] and not r["ground_truth_is_hallucination_inducing"]) \
        + sum(1 for r in controls_rows if r["hybrid_flagged"])
    fn = sum(1 for r in traps if not r["hybrid_flagged"] and r["ground_truth_is_hallucination_inducing"])
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))

    summary = {
        "system_name": args.system_name,
        "backend": args.backend,
        "model": args.model,
        "n_trap_items": len(traps),
        "n_control_items": len(controls_rows),
        "over_generation_rate": over_generation_rate,
        "syntactic_integrity": syntactic_integrity,
        "off_topic_rate": off_topic_rate,
        "false_positive_rate": false_positive_rate,
        "hybrid_precision": precision,
        "hybrid_recall": recall,
        "hybrid_f1": f1,
        "is_demo_data": bool(args.demo or args.backend == "offline"),
    }
    summary_path = RESULTS_DIR / f"{system_slug}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote:\n  {csv_path}\n  {summary_path}")
    if off_topic_rate == off_topic_rate and off_topic_rate > 0.15:
        print(
            f"\n*** WARNING: off_topic_rate={off_topic_rate:.2f} — a sizeable "
            f"fraction of this model's responses didn't meaningfully engage "
            f"with the Sanskrit question at all (derailment, repetition, "
            f"etc.). over_generation_rate and syntactic_integrity for this "
            f"system may not be trustworthy as reported — read a sample of "
            f"the off_topic rows in {csv_path} before using these numbers. ***"
        )
    if summary["is_demo_data"]:
        print(
            "\n*** is_demo_data=true: this ran against the offline stub or "
            "--demo was passed. Do NOT use these numbers in the paper. ***"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["groq", "ollama", "hf", "offline"], required=True)
    parser.add_argument("--model", default="gpt2", help="Model name/id for the chosen backend")
    parser.add_argument("--n-per-type", type=int, default=15, help="Trap examples per trap type (3 types total)")
    parser.add_argument("--system-name", default=None, help="Label for this system in results/plots")
    parser.add_argument("--demo", action="store_true", help="Explicitly mark this run's output as demo/non-paper data")
    args = parser.parse_args()

    if args.system_name is None:
        args.system_name = f"{args.backend}-{args.model}"

    run(args)


if __name__ == "__main__":
    main()