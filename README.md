# Sanskrit Hallucination Detector

A hybrid neural + symbolic pipeline for detecting **human-induced adversarial
hallucinations** in Sanskrit LLM outputs — cases where a user exploits
Sanskrit's morphosyntactic ambiguity (overlapping Vibhakti case markers,
Sandhi phonetics, Samasa compounds) to push a model into confirming a
non-existent root (dhatu) or a false grammatical reading.

> **Scope note:** This repository ships a small, illustrative seed lexicon
> (a handful of dhatus, pratyayas, and sandhi rules — see
> `core/paninian_engine.py`) and an offline deterministic LLM stub, so the
> full pipeline is runnable and testable out of the box with no API keys.
> It is **not** a research-grade Paninian grammar and does not ship any
> pre-computed evaluation numbers. To produce real results you need to (a)
> supply a vetted dhatu/pratyaya/sandhi lexicon, and (b) point
> `HybridDetector` at a real `LLMClient` implementation for the provider(s)
> you want to evaluate.

## Architecture

```
Adversarial Generator  ──▶  Surface Form  ──▶  LLM under test (LLMClient)
(core/adversarial_trap)                    └─▶  Symbolic Paninian Validator
                                                 (core/paninian_engine)
                                                       │
                                                       ▼
                                            Hybrid fusion (core/hybrid_detector)
                                                       │
                                                       ▼
                                          Hallucination Probability + verdict
```

### Sandhi Ambiguity Score (SAS)

For a surface form `w`:

```
SAS(w) = α · H_split(w) + β · O_boundary(w) + γ · C_phonetic(w)
```

- `H_split(w)` — normalized entropy over how "plausible" each candidate
  binary split of `w` is (plausibility approximated by whether each side
  resolves to a known dhatu/suffix)
- `O_boundary(w)` — fraction of known sandhi merge patterns found in `w`
- `C_phonetic(w)` — fraction of candidate splits where *both* sides
  independently resolve to a known dhatu (the strongest ambiguity signal)

Default weights: `α=0.45, β=0.35, γ=0.20` (see `config/settings.py`,
`SASWeights`) — starting points, not the result of a hyperparameter
search.

### Vibhakti Deception Vector (VDV)

An 8-dimensional vector over the eight Sanskrit cases (Prathama …
Sambodhana), populated from which case(s) a word's matched suffix is
compatible with, normalized to sum to 1. Its scalar **deception score**
is the normalized Shannon entropy of that distribution: 0 = an
unambiguous single-case spike, 1 = maximal spread across cases (i.e. a
suffix that's genuinely compatible with several different case
readings — fertile ground for a deceptive prompt asserting only one).

### Fusion

```
fused = λ · neural_component + (1 − λ) · symbolic_component
```

with `λ = fusion_weights.lambda_neural` (default 0.4), plus:
- a **symbolic veto**: if `symbolic_component ≥ thresholds.symbolic_veto_threshold`,
  the item is flagged regardless of the fused score
- a **fabricated-root floor**: if no candidate root matches the known
  dhatu lexicon, `fused` is floored at 0.5

## Repository layout

```
sanskrit_hallucination_detector/
├── config/settings.py        # weights, thresholds, LLM + trap-generator config
├── core/
│   ├── adversarial_trap.py   # synthetic trap generator (3 trap types)
│   ├── paninian_engine.py    # symbolic dhatu/pratyaya/sandhi validator
│   └── hybrid_detector.py    # neural + symbolic fusion pipeline
├── tests/test_pipeline.py    # pytest suite (offline, no API keys needed)
├── requirements.txt
└── git_push.sh
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the tests

```bash
pytest tests/ -v
```

All tests run against `OfflineStubLLMClient`, a deterministic offline
stand-in — no network access or API key required. They verify pipeline
*wiring and behavior* (score ranges, veto logic, fabricated-root
handling), not real-world accuracy, which depends on a live model and a
labeled gold dataset you construct.

## Running against a real LLM

Three free-to-use `LLMClient` implementations ship in
`core/llm_clients.py` — pick whichever fits your setup:

### Option A — Groq (hosted, free tier, Llama models)

```bash
pip install groq --break-system-packages
export GROQ_API_KEY="your-key-from-console.groq.com"
```

```python
from core.llm_clients import GroqLLMClient
from core.hybrid_detector import HybridDetector

detector = HybridDetector(llm_client=GroqLLMClient(model="llama-3.3-70b-versatile"))
result = detector.detect("your-sanskrit-surface-form")
print(result.hallucination_probability, result.explanation)
```

Note: Groq doesn't reliably return token logprobs for every
model/version. When it doesn't, `GroqLLMClient` falls back to a
neutral placeholder confidence (flagged in code, never fabricated) —
the symbolic (SAS/VDV) signal still drives detection in that case.

### Option B — Ollama (fully local, free, no API key)

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull a model:
ollama pull llama3.1
# Ollama's server runs automatically; if not, start it:
ollama serve
```

```python
from core.llm_clients import OllamaLLMClient
from core.hybrid_detector import HybridDetector

detector = HybridDetector(llm_client=OllamaLLMClient(model="llama3.1"))
result = detector.detect("your-sanskrit-surface-form")
```

Same logprob caveat as Groq — Ollama's logprob support is
version/model-dependent; the client falls back cleanly when absent.

### Option C — Local Hugging Face model (GPT-2, Qwen2.5) — real logprobs

```bash
pip install transformers torch --break-system-packages
```

```python
from core.llm_clients import HFLocalLLMClient
from core.hybrid_detector import HybridDetector

# GPT-2: small, fast, runs fine on CPU
detector = HybridDetector(llm_client=HFLocalLLMClient(model_name="gpt2"))

# or Qwen2.5 (better quality, still CPU-runnable at the 0.5B/1.5B sizes)
detector = HybridDetector(
    llm_client=HFLocalLLMClient(model_name="Qwen/Qwen2.5-1.5B-Instruct")
)

result = detector.detect("your-sanskrit-surface-form")
```

This option computes **real** per-token log-probabilities directly
from the model's own output logits (via `output_scores=True` in
`generate()`), since there's no API layer hiding them — the most
directly grounded neural-confidence signal of the three options.

### Writing your own client

Any provider can be added by subclassing `LLMClient`:

```python
from core.hybrid_detector import LLMClient, SimpleLLMResponse

class YourProviderLLMClient(LLMClient):
    def complete(self, prompt: str) -> SimpleLLMResponse:
        # call your provider, return its text + per-token logprobs
        ...
        return SimpleLLMResponse(text=text, token_logprobs=token_logprobs)
```

## Running the full evaluation and generating figures

Once you have a backend set up (Groq, Ollama, or a local HF model —
see above), run the evaluation and figure scripts:

```bash
# 1. Run the evaluation against each system you want in your comparison.
#    Repeat once per system — this is what fills Table 1 in the paper.
python scripts/run_evaluation.py --backend hf --model gpt2 \
    --n-per-type 20 --system-name "GPT-2"

python scripts/run_evaluation.py --backend groq --model llama-3.3-70b-versatile \
    --n-per-type 20 --system-name "Llama-3.3-70B"

# 2. Generate all figures from the accumulated results/ directory.
python scripts/generate_figures.py --results-dir results --out-dir figures
```

This writes, per system, `results/<system>_per_example.csv` (every
trap instance, its scores, and the raw model response — for manual
verification, since the affirmation heuristic is a scalable proxy, not
ground truth) and `results/<system>_summary.json` (the aggregate
metrics for Table 1).

`scripts/generate_figures.py` then produces, for every system found in
`results/`:
- `bar_comparison` — Over-generation Rate / False Positive Rate / F1
  across systems
- `sas_distribution`, `vdv_distribution` — score distributions by trap
  type
- `confusion_matrix_<system>` — flagged vs. ground truth
- `threshold_curve_<system>` — precision/recall/F1 vs. decision
  threshold

Each figure is written in three formats:
- **PDF and SVG** (vector) — use these in the actual Springer LNCS
  submission; vector graphics scale losslessly to any print
  resolution, which is what conference typesetters want
- **PNG** at true 3840×2160+ pixel density — for slides, the README,
  or anywhere raster is required

If a results file has `is_demo_data: true` in its summary (produced by
`--backend offline` or `--demo`), every figure generated from it is
watermarked "DEMO DATA — NOT FOR PUBLICATION" so a placeholder run
can't accidentally end up in a submission.

**On the affirmation heuristic:** `run_evaluation.py` decides whether
a baseline model "fell for" a trap using a keyword-based heuristic
(confirming vs. hedging language in the response). This is the same
scalable proxy named in the paper's Section 5 protocol — read the
`response_text` column yourself (or have annotators do it) before
reporting numbers derived from it in the paper itself.

## Publishing to GitHub

```bash
./git_push.sh git@github.com:yourname/sanskrit_hallucination_detector.git main
```

This initializes the repo (if needed), writes `.gitignore` if missing,
commits everything, and pushes. The repo also ships a GitHub Actions
workflow (`.github/workflows/tests.yml`) that runs the full test suite
on every push/PR across Python 3.10–3.12, using only the offline stub
— no API keys or secrets required in CI.



Populate `data/resources/dhatu_list.json`, `pratyaya_list.json`, and
`sandhi_rules.json` (paths configurable in `config/settings.py`) with a
vetted lexicon — e.g. digitized from the Dhatupatha and a standard
suffix table — and `PaninianEngine` will load them automatically,
falling back to the small illustrative seed set (with a logged warning)
if they're absent.

## What this repo deliberately does not include

- Pre-computed evaluation numbers (F1, over-generation rate, etc.)
  against GPT-4 / Claude / Llama-3 — these require live API calls and a
  labeled gold set that only you can run and construct honestly
- A complete, authoritative Paninian grammar — the seed lexicon is for
  testing the pipeline's wiring, not for citing as linguistic ground truth
