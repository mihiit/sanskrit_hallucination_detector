"""
core/hybrid_detector.py

Main pipeline: fuses the neural LLM's own confidence signal (derived
from token logprobs on its claimed derivation/translation) with the
Symbolic Paninian Validator's constraint-violation scores (SAS, VDV) to
produce a final Hallucination Probability for a given (prompt, LLM
response) pair.

Design notes:
  - The LLM client is a thin, provider-agnostic wrapper
    (`LLMClient`) so the same pipeline can be pointed at different
    providers for comparison. It is intentionally NOT hardwired to
    fabricate comparative results — running it against multiple
    providers and reporting real numbers is left to the user, because
    those numbers depend on live API responses this scaffold cannot
    generate on its own.
  - `symbolic_component` combines the Sandhi Ambiguity Score and the
    Vibhakti Deception Score into one normalized [0, 1] value.
  - `neural_component` inverts and normalizes the LLM's mean
    logprob on the tokens of its own claimed derivation: low
    confidence (more negative logprob) contributes toward a *higher*
    hallucination probability only when combined with symbolic
    disagreement — a low-confidence-but-symbolically-valid answer is
    treated differently from a high-confidence-but-symbolically-invalid
    one. See `_neural_score` and `HybridDetector.detect` docstrings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from config.settings import get_settings
from core.paninian_engine import PaninianEngine


# --------------------------------------------------------------------------- #
# LLM client abstraction
# --------------------------------------------------------------------------- #


class LLMResponse(Protocol):
    """Minimal shape expected back from an LLM client call."""

    text: str
    token_logprobs: List[float]


@dataclass
class SimpleLLMResponse:
    text: str
    token_logprobs: List[float]


class LLMClient:
    """
    Provider-agnostic LLM client interface.

    This scaffold ships a deterministic offline stub
    (`OfflineStubLLMClient`) so the pipeline and tests run without any
    network access or API keys. To evaluate a real provider, implement
    `complete()` against that provider's SDK (Anthropic, OpenAI, a
    local HF model, etc.) using the settings in config.settings.LLMConfig,
    and pass an instance of your subclass into HybridDetector.
    """

    def complete(self, prompt: str) -> SimpleLLMResponse:  # pragma: no cover
        raise NotImplementedError(
            "LLMClient.complete() must be implemented by a concrete "
            "provider-backed subclass. See docstring for guidance."
        )


class OfflineStubLLMClient(LLMClient):
    """
    A deterministic, offline stand-in LLM client used for local
    development and unit tests where no live API call should occur.

    It does NOT call any external API. It produces a syntactically
    valid SimpleLLMResponse whose confidence is a simple deterministic
    function of the prompt length and character content, purely so
    that HybridDetector's fusion logic is exercised end-to-end without
    network access. Do not use this stub's output to report real
    evaluation numbers — swap in a genuine provider-backed LLMClient
    subclass for that.
    """

    def complete(self, prompt: str) -> SimpleLLMResponse:
        # Deterministic pseudo-confidence: purely mechanical, not a real
        # model judgment. Longer / more varied prompts get a slightly
        # lower synthetic confidence, bounded to a plausible logprob range.
        base = -0.05 - (len(set(prompt)) % 7) * 0.08
        n_tokens = max(3, len(prompt.split()) // 2)
        token_logprobs = [base - 0.01 * i for i in range(n_tokens)]
        text = f"[offline-stub completion for prompt of length {len(prompt)}]"
        return SimpleLLMResponse(text=text, token_logprobs=token_logprobs)


# --------------------------------------------------------------------------- #
# Score fusion
# --------------------------------------------------------------------------- #


@dataclass
class DetectionResult:
    surface_form: str
    prompt: str
    llm_text: str
    sas: float
    vdv_deception: float
    symbolic_component: float
    neural_component: float
    hallucination_probability: float
    symbolic_veto: bool
    is_flagged_hallucination: bool
    matched_known_dhatu: bool
    explanation: str


class HybridDetector:
    """Main entry point: given a surface form (and optionally a prompt
    wrapping it), runs the LLM under test, scores it symbolically via
    PaninianEngine, fuses the two signals, and returns a DetectionResult."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        engine: Optional[PaninianEngine] = None,
    ):
        self.llm_client = llm_client or OfflineStubLLMClient()
        self.engine = engine or PaninianEngine()
        self.settings = get_settings()

    # ------------------------------------------------------------------ #

    def _neural_score(self, token_logprobs: List[float]) -> float:
        """
        Convert a list of token logprobs into a normalized [0, 1]
        "neural uncertainty" score, where 0 = maximally confident and
        1 = maximally uncertain.

        Uses mean logprob, then squashes with a logistic function
        centered at a configurable reference point (-2.0 nats, a
        reasonable "getting shaky" reference for short completions —
        calibrate against your specific model/provider's logprob scale
        before treating this as meaningful in isolation).
        """
        if not token_logprobs:
            return 0.5
        mean_logprob = sum(token_logprobs) / len(token_logprobs)
        reference = -2.0
        steepness = 1.5
        # logistic squash: uncertainty rises as mean_logprob drops below reference
        uncertainty = 1.0 / (1.0 + math.exp(steepness * (mean_logprob - reference)))
        return max(0.0, min(1.0, uncertainty))

    def _symbolic_score(self, surface_form: str) -> Dict[str, float]:
        sas = self.engine.sandhi_ambiguity_score(surface_form)
        vdv = self.engine.vibhakti_deception_vector(surface_form)
        vdv_score = self.engine.vdv_deception_score(vdv)
        combined = max(sas, vdv_score)  # either signal alone can indicate risk
        return {"sas": sas, "vdv_deception": vdv_score, "symbolic_component": combined}

    # ------------------------------------------------------------------ #

    def detect(self, surface_form: str, prompt: Optional[str] = None) -> DetectionResult:
        """
        Run the full hybrid pipeline on a single surface form.

        final_hallucination_probability =
            lambda_neural * neural_component
            + (1 - lambda_neural) * symbolic_component

        with a hard symbolic veto: if symbolic_component exceeds
        thresholds.symbolic_veto_threshold, the result is flagged as a
        hallucination regardless of the fused probability, on the
        principle that a clear grammatical impossibility should not be
        overridable by LLM confidence alone.
        """
        effective_prompt = prompt or (
            f'Confirm the Sanskrit derivation and meaning of "{surface_form}".'
        )
        llm_response = self.llm_client.complete(effective_prompt)

        neural_component = self._neural_score(llm_response.token_logprobs)
        symbolic = self._symbolic_score(surface_form)
        validation = self.engine.validate_word(surface_form)

        lam = self.settings.fusion_weights.lambda_neural
        fused = lam * neural_component + (1 - lam) * symbolic["symbolic_component"]

        # A fabricated (non-existent) root is itself strong symbolic
        # evidence, independent of the SAS/VDV scores above.
        if not validation["matched_known_dhatu"]:
            fused = max(fused, 0.5)

        symbolic_veto = (
            symbolic["symbolic_component"]
            >= self.settings.thresholds.symbolic_veto_threshold
        )
        is_flagged = symbolic_veto or (
            fused >= self.settings.thresholds.hallucination_probability_threshold
        )

        explanation_parts = [
            f"SAS={symbolic['sas']:.3f}",
            f"VDV_deception={symbolic['vdv_deception']:.3f}",
            f"neural_uncertainty={neural_component:.3f}",
            f"fused_probability={fused:.3f}",
        ]
        if not validation["matched_known_dhatu"]:
            explanation_parts.append(
                "no candidate root matched the known dhatu lexicon"
            )
        if symbolic_veto:
            explanation_parts.append("symbolic veto triggered")

        return DetectionResult(
            surface_form=surface_form,
            prompt=effective_prompt,
            llm_text=llm_response.text,
            sas=symbolic["sas"],
            vdv_deception=symbolic["vdv_deception"],
            symbolic_component=symbolic["symbolic_component"],
            neural_component=neural_component,
            hallucination_probability=fused,
            symbolic_veto=symbolic_veto,
            is_flagged_hallucination=is_flagged,
            matched_known_dhatu=validation["matched_known_dhatu"],
            explanation="; ".join(explanation_parts),
        )

    def detect_batch(self, surface_forms: List[str]) -> List[DetectionResult]:
        return [self.detect(sf) for sf in surface_forms]
