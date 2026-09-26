"""
config/settings.py

Central configuration for the Sanskrit Hallucination Detector.

Holds:
  - Weighting coefficients for the Sandhi Ambiguity Score (SAS)
    and the Vibhakti Deception Vector (VDV)
  - Thresholds used by the hybrid decision rule
  - LLM client configuration (model name, temperature, logprob settings)
  - Paths to the symbolic linguistic resource files used by the
    Paninian engine (dhatu list, pratyaya list, sandhi rule table)

All values here are defaults that can be overridden via environment
variables or a local `settings_local.py` (not tracked in git) — see
README.md for details.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESOURCES_DIR = DATA_DIR / "resources"

# Symbolic resource files consumed by core/paninian_engine.py.
# These are plain-text / JSON word lists that YOU must populate with a
# real Paninian lexicon (e.g. derived from the Dhatupatha and a
# pratyaya table). Shipping a complete, authoritative dhatu/pratyaya
# database is outside what this scaffold can responsibly fabricate —
# the engine ships with a small illustrative seed set (see
# core/paninian_engine.py::_SEED_DHATUS) and falls back to that seed
# set if these files are not found, so the pipeline is runnable
# out of the box but should be extended with a vetted lexicon before
# any research claims are made on top of it.
DHATU_LIST_PATH = RESOURCES_DIR / "dhatu_list.json"
PRATYAYA_LIST_PATH = RESOURCES_DIR / "pratyaya_list.json"
SANDHI_RULES_PATH = RESOURCES_DIR / "sandhi_rules.json"


# --------------------------------------------------------------------------- #
# Sandhi Ambiguity Score (SAS) weighting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SASWeights:
    """
    Coefficients for the Sandhi Ambiguity Score.

    SAS(w) = alpha * split_entropy(w) + beta * boundary_overlap(w)
             + gamma * phonetic_collision(w)

    See core/paninian_engine.py for the exact computation. Values below
    are reasonable defaults for a first pass; they are NOT the result
    of a hyperparameter search and should be tuned once you have a
    labeled validation set of genuine vs. adversarial sandhi splits.
    """

    alpha: float = 0.45   # weight on split-entropy term
    beta: float = 0.35    # weight on boundary-overlap term
    gamma: float = 0.20   # weight on phonetic-collision term


# --------------------------------------------------------------------------- #
# Vibhakti Deception Vector (VDV) weighting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VDVWeights:
    """
    Coefficients for the Vibhakti Deception Vector.

    VDV(w) is a length-8 vector (one entry per Vibhakti / case, plus
    Sambodhana) where each entry measures how strongly the surface
    form is compatible with a case reading that is NOT the one the
    adversarial prompt is implying. See core/paninian_engine.py.
    """

    case_ambiguity_weight: float = 0.6     # weight on cross-case overlap
    number_ambiguity_weight: float = 0.25  # weight on singular/dual/plural overlap
    gender_ambiguity_weight: float = 0.15  # weight on gender-marking overlap


# --------------------------------------------------------------------------- #
# Hybrid decision thresholds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecisionThresholds:
    """
    Thresholds used by core/hybrid_detector.py to convert the fused
    score into a discrete hallucination / not-hallucination decision.

    These are starting points, not calibrated values. Calibrate on a
    held-out dev set (see tests/test_pipeline.py for the expected
    interface) before reporting any evaluation numbers.
    """

    hallucination_probability_threshold: float = 0.5
    symbolic_veto_threshold: float = 0.85  # SAS/VDV fusion score above which
                                            # the symbolic engine can veto the
                                            # LLM regardless of its confidence


# --------------------------------------------------------------------------- #
# Fusion weighting (neural vs. symbolic)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FusionWeights:
    """
    How much the final Hallucination Probability trusts the neural
    LLM's own (inverted, normalized) logprob-derived confidence versus
    the symbolic engine's constraint-violation score.

    final_score = lambda_neural * neural_component
                  + (1 - lambda_neural) * symbolic_component
    """

    lambda_neural: float = 0.4


# --------------------------------------------------------------------------- #
# LLM client configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for the "LLM under test" client used inside
    core/hybrid_detector.py. The client itself is a thin wrapper you
    point at whichever provider you're evaluating (OpenAI, Anthropic,
    a local HF model, etc.) — this project does not hardcode a single
    provider's SDK so that the same pipeline can be run against
    multiple baselines for a fair comparison.
    """

    provider: str = os.environ.get("SHD_LLM_PROVIDER", "anthropic")
    model_name: str = os.environ.get("SHD_LLM_MODEL", "claude-sonnet-4-6")
    temperature: float = 0.0
    max_tokens: int = 256
    request_logprobs: bool = True
    top_logprobs: int = 5
    timeout_seconds: int = 60


# --------------------------------------------------------------------------- #
# Adversarial trap generator configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrapGeneratorConfig:
    """
    Controls how aggressive core/adversarial_trap.py is when
    synthesizing human-induced Sanskrit hallucination traps.
    """

    # Probability of injecting a sandhi-boundary ambiguity into a
    # generated trap sentence.
    sandhi_injection_rate: float = 0.6

    # Probability of injecting a Vibhakti (case-marker) collision.
    vibhakti_injection_rate: float = 0.5

    # Probability of injecting a fabricated (non-existent) dhatu root
    # disguised as a plausible compound.
    fake_dhatu_injection_rate: float = 0.3

    # Maximum number of simultaneous trap types stacked in a single
    # generated example (keeps traps legible rather than pure noise).
    max_stacked_traps: int = 2

    random_seed: int = 42


# --------------------------------------------------------------------------- #
# Aggregated settings object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    sas_weights: SASWeights = field(default_factory=SASWeights)
    vdv_weights: VDVWeights = field(default_factory=VDVWeights)
    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trap_generator: TrapGeneratorConfig = field(default_factory=TrapGeneratorConfig)


SETTINGS = Settings()


def get_settings() -> Settings:
    """Accessor used throughout the codebase instead of importing SETTINGS
    directly, so tests can monkeypatch configuration easily."""
    return SETTINGS
