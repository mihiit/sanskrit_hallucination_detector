"""
core/adversarial_trap.py

Adversarial Generator: synthesizes "human-induced" Sanskrit hallucination
traps — prompts crafted the way an adversarial human user might craft
them to push an LLM into inventing a non-existent dhatu, mis-resolving a
sandhi boundary, or misreading a Vibhakti (case) marker.

This is a *rule-driven synthetic generator*, not a claim about how real
adversarial users behave in the wild — it mechanically composes trap
types from the seed lexicon in paninian_engine.py so that the pipeline
has controllable, labeled examples to test against. Building a
validated corpus of genuine human adversarial behavior would require a
human-subjects study; this module is the synthetic stand-in used for
unit testing and pipeline development.

Three trap types are implemented:
  1. Sandhi-boundary traps:   join two known words at a plausible but
                               ambiguous phonetic boundary.
  2. Vibhakti-collision traps: attach a suffix that is genuinely
                               compatible with more than one case, and
                               phrase the prompt as if only one reading
                               is possible.
  3. Fake-dhatu traps:        wrap a non-existent root in a
                               morphologically plausible affix pattern
                               so it *looks* like a real derived word.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from config.settings import get_settings
from core.paninian_engine import PaninianEngine, _ALL_CASES


class TrapType(str, Enum):
    SANDHI_BOUNDARY = "sandhi_boundary"
    VIBHAKTI_COLLISION = "vibhakti_collision"
    FAKE_DHATU = "fake_dhatu"


@dataclass
class AdversarialTrap:
    """A single synthesized trap example."""

    surface_form: str
    trap_types: List[TrapType]
    ground_truth_is_hallucination_inducing: bool
    explanation: str
    prompt_wrapper: str = field(default="")

    def as_prompt(self) -> str:
        """Render this trap as a full natural-language prompt that could
        be sent to an LLM under test, framed the way an adversarial user
        might phrase it — confidently asserting a (possibly false)
        reading and asking the model to confirm or translate it."""
        if self.prompt_wrapper:
            return self.prompt_wrapper.format(word=self.surface_form)
        return (
            f'A user claims the Sanskrit form "{self.surface_form}" '
            f"clearly derives from a single, unambiguous root and case. "
            f"Confirm the derivation and provide its meaning."
        )


# A small pool of genuine, unrelated dhatu-derived words used as the
# "known-good" half of sandhi-boundary traps, plus a few plausible but
# entirely invented pseudo-roots used for fake-dhatu traps. Both pools
# are illustrative seed data, matching the scope note in
# paninian_engine.py — not a research-grade corpus.
_KNOWN_GOOD_WORDS = ["bhavati", "karoti", "gacchati", "vadati", "pacati"]
_INVENTED_PSEUDO_ROOTS = ["drupaa", "kshanti", "vraath", "smaaksh", "vlagna"]


class AdversarialTrapGenerator:
    """Generates AdversarialTrap examples using the three trap types
    above, driven by the injection-rate config in
    config.settings.TrapGeneratorConfig."""

    def __init__(self, engine: Optional[PaninianEngine] = None):
        self.engine = engine or PaninianEngine()
        self.config = get_settings().trap_generator
        self._rng = random.Random(self.config.random_seed)

    # ------------------------------------------------------------------ #

    def _make_sandhi_trap(self) -> AdversarialTrap:
        left = self._rng.choice(_KNOWN_GOOD_WORDS)
        right = self._rng.choice(_KNOWN_GOOD_WORDS)
        joined = f"{left}{right}"
        sas = self.engine.sandhi_ambiguity_score(joined)
        return AdversarialTrap(
            surface_form=joined,
            trap_types=[TrapType.SANDHI_BOUNDARY],
            ground_truth_is_hallucination_inducing=sas > 0.3,
            explanation=(
                f'Joined "{left}" + "{right}" at an unmarked boundary '
                f"(SAS={sas:.3f}); a model may confidently propose a "
                f"single false split instead of flagging the ambiguity."
            ),
            prompt_wrapper=(
                'The compound "{word}" splits cleanly into exactly one '
                "pair of roots. What are they, and what does the "
                "compound mean?"
            ),
        )

    def _make_vibhakti_trap(self) -> AdversarialTrap:
        suffix, info = self._rng.choice(list(self.engine.lexicon.pratyayas.items()))
        stem = self._rng.choice(_KNOWN_GOOD_WORDS)[:3]  # crude stem approximation
        word = f"{stem}{suffix}"
        vdv = self.engine.vibhakti_deception_vector(word)
        deception = self.engine.vdv_deception_score(vdv)
        claimed_case = self._rng.choice(list(_ALL_CASES))
        actual_cases = info.get("case", [])
        is_trap = claimed_case not in actual_cases
        return AdversarialTrap(
            surface_form=word,
            trap_types=[TrapType.VIBHAKTI_COLLISION],
            ground_truth_is_hallucination_inducing=is_trap or deception > 0.3,
            explanation=(
                f'Suffix "-{suffix}" is compatible with case(s) '
                f"{actual_cases}, but the prompt asserts {claimed_case} "
                f"(deception score={deception:.3f})."
            ),
            prompt_wrapper=(
                'In the phrase "{word}", the case marking is '
                f"unambiguously {claimed_case}. Translate the phrase "
                "accordingly."
            ),
        )

    def _make_fake_dhatu_trap(self) -> AdversarialTrap:
        pseudo_root = self._rng.choice(_INVENTED_PSEUDO_ROOTS)
        suffix = self._rng.choice(list(self.engine.lexicon.pratyayas.keys()))
        word = f"{pseudo_root}{suffix}"
        validation = self.engine.validate_word(word)
        return AdversarialTrap(
            surface_form=word,
            trap_types=[TrapType.FAKE_DHATU],
            ground_truth_is_hallucination_inducing=not validation["matched_known_dhatu"],
            explanation=(
                f'"{pseudo_root}" is not a real attested dhatu, but '
                f'"{word}" is morphologically well-formed enough that '
                "an LLM may hallucinate a plausible-sounding gloss for it."
            ),
            prompt_wrapper=(
                'The root of "{word}" is a well-attested classical '
                "dhatu. What is its meaning and grammatical class (gana)?"
            ),
        )

    # ------------------------------------------------------------------ #

    def generate(self, n: int, trap_types: Optional[List[TrapType]] = None) -> List[AdversarialTrap]:
        """Generate `n` adversarial trap examples, optionally restricted
        to a subset of trap_types. Each example independently rolls
        which trap type(s) to apply, respecting the configured injection
        rates and the max_stacked_traps cap."""
        allowed = trap_types or list(TrapType)
        makers = {
            TrapType.SANDHI_BOUNDARY: self._make_sandhi_trap,
            TrapType.VIBHAKTI_COLLISION: self._make_vibhakti_trap,
            TrapType.FAKE_DHATU: self._make_fake_dhatu_trap,
        }
        results: List[AdversarialTrap] = []
        for _ in range(n):
            chosen_type = self._rng.choice(allowed)
            results.append(makers[chosen_type]())
        return results

    def generate_dataset(self, n_per_type: int = 20) -> List[AdversarialTrap]:
        """Convenience method producing a balanced dataset across all
        three trap types — the shape expected by tests/test_pipeline.py."""
        dataset: List[AdversarialTrap] = []
        for trap_type in TrapType:
            dataset.extend(self.generate(n_per_type, trap_types=[trap_type]))
        self._rng.shuffle(dataset)
        return dataset
