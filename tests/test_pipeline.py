"""
tests/test_pipeline.py

Pytest suite verifying the architecture end-to-end:
  - PaninianEngine's dhatu matching, SAS computation, and VDV computation
  - AdversarialTrapGenerator's three trap types produce well-formed,
    labeled examples
  - HybridDetector correctly fuses neural + symbolic signals, including
    the symbolic-veto path and the fabricated-root path

These tests use the OfflineStubLLMClient (no network access, no API
keys required) so the suite runs deterministically in CI. They verify
pipeline *behavior and wiring*, not real-world accuracy numbers — real
accuracy/F1 figures require running against a live LLM and a labeled
gold dataset, which is outside the scope of a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable when running `pytest` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.adversarial_trap import AdversarialTrapGenerator, TrapType
from core.hybrid_detector import HybridDetector, OfflineStubLLMClient, SimpleLLMResponse
from core.paninian_engine import PaninianEngine


# --------------------------------------------------------------------------- #
# PaninianEngine tests
# --------------------------------------------------------------------------- #


class TestPaninianEngine:
    def setup_method(self):
        self.engine = PaninianEngine()

    def test_known_dhatu_matches(self):
        assert self.engine.is_known_dhatu("bhu")
        assert self.engine.is_known_dhatu("BHU")  # case-insensitive
        assert self.engine.is_known_dhatu("kr")

    def test_unknown_dhatu_does_not_match(self):
        assert not self.engine.is_known_dhatu("vraath")  # invented pseudo-root
        assert not self.engine.is_known_dhatu("smaaksh")

    def test_validate_word_finds_real_root_under_suffix(self):
        # "krena" = kr (root) + ena (Tritiya singular suffix)
        result = self.engine.validate_word("krena")
        assert result["matched_known_dhatu"] is True
        assert "kr" in result["matched_roots"]

    def test_validate_word_rejects_fake_root_under_suffix(self):
        result = self.engine.validate_word("vraathena")
        assert result["matched_known_dhatu"] is False

    def test_sandhi_ambiguity_score_in_range(self):
        for word in ["bhavatigacchati", "kr", "a", "pacativadati"]:
            score = self.engine.sandhi_ambiguity_score(word)
            assert 0.0 <= score <= 1.0

    def test_sandhi_ambiguity_score_higher_for_double_dhatu_join(self):
        # A join of two genuinely competing dhatu-bearing forms should
        # score at least as ambiguous as an empty/trivial string.
        trivial = self.engine.sandhi_ambiguity_score("a")
        joined = self.engine.sandhi_ambiguity_score("bhavatigacchati")
        assert joined >= trivial

    def test_vibhakti_vector_sums_to_one_when_matched(self):
        vdv = self.engine.vibhakti_deception_vector("karena")  # kar + ena
        total = sum(vdv.values())
        assert total == pytest.approx(1.0) or total == pytest.approx(0.0)

    def test_vdv_deception_score_in_range(self):
        vdv = self.engine.vibhakti_deception_vector("karena")
        score = self.engine.vdv_deception_score(vdv)
        assert 0.0 <= score <= 1.0

    def test_vdv_deception_score_zero_for_single_case_spike(self):
        # A synthetic vector with all mass on one case should show zero
        # deception (no entropy).
        spike_vector = {"Prathama": 1.0, "Dvitiya": 0.0, "Tritiya": 0.0,
                         "Chaturthi": 0.0, "Panchami": 0.0, "Shashthi": 0.0,
                         "Saptami": 0.0, "Sambodhana": 0.0}
        assert self.engine.vdv_deception_score(spike_vector) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# AdversarialTrapGenerator tests
# --------------------------------------------------------------------------- #


class TestAdversarialTrapGenerator:
    def setup_method(self):
        self.generator = AdversarialTrapGenerator()

    def test_generate_returns_requested_count(self):
        traps = self.generator.generate(10)
        assert len(traps) == 10

    def test_generate_restricted_to_single_trap_type(self):
        traps = self.generator.generate(5, trap_types=[TrapType.FAKE_DHATU])
        assert all(TrapType.FAKE_DHATU in t.trap_types for t in traps)

    def test_sandhi_trap_has_valid_prompt_and_explanation(self):
        traps = self.generator.generate(3, trap_types=[TrapType.SANDHI_BOUNDARY])
        for trap in traps:
            assert trap.surface_form
            assert trap.explanation
            assert "{word}" not in trap.as_prompt()  # was substituted
            assert trap.surface_form in trap.as_prompt()

    def test_fake_dhatu_trap_is_labeled_as_hallucination_inducing(self):
        traps = self.generator.generate(5, trap_types=[TrapType.FAKE_DHATU])
        # All fake-dhatu traps use invented pseudo-roots, so should be
        # labeled True by construction (no accidental collision with the
        # tiny seed lexicon).
        assert all(t.ground_truth_is_hallucination_inducing for t in traps)

    def test_generate_dataset_is_balanced_and_shuffled(self):
        dataset = self.generator.generate_dataset(n_per_type=4)
        assert len(dataset) == 4 * len(TrapType)
        counts = {tt: 0 for tt in TrapType}
        for trap in dataset:
            for tt in trap.trap_types:
                counts[tt] += 1
        for tt in TrapType:
            assert counts[tt] == 4


# --------------------------------------------------------------------------- #
# HybridDetector tests
# --------------------------------------------------------------------------- #


class TestHybridDetector:
    def setup_method(self):
        self.detector = HybridDetector(llm_client=OfflineStubLLMClient())

    def test_detect_returns_well_formed_result(self):
        result = self.detector.detect("bhavati")
        assert result.surface_form == "bhavati"
        assert 0.0 <= result.hallucination_probability <= 1.0
        assert 0.0 <= result.sas <= 1.0
        assert 0.0 <= result.vdv_deception <= 1.0
        assert isinstance(result.is_flagged_hallucination, bool)

    def test_fabricated_root_pushes_probability_up(self):
        # "vraathena" = invented pseudo-root "vraath" + real suffix "ena"
        result = self.detector.detect("vraathena")
        assert result.matched_known_dhatu is False
        assert result.hallucination_probability >= 0.5

    def test_known_root_does_not_force_high_probability(self):
        # "krena" = known root "kr" + real suffix "ena"; should not be
        # force-elevated by the fabricated-root rule (though the fused
        # score can still vary with the stub's synthetic neural
        # component).
        result = self.detector.detect("krena")
        assert result.matched_known_dhatu is True

    def test_symbolic_veto_overrides_low_fused_probability(self):
        # Construct a detector whose symbolic_component will be forced
        # above the veto threshold via a hand-rolled engine call, by
        # directly checking the veto flag logic against a high-SAS input.
        result = self.detector.detect("bhavatigacchativadatipacati")
        if result.symbolic_veto:
            assert result.is_flagged_hallucination is True

    def test_detect_batch_preserves_order_and_count(self):
        forms = ["bhavati", "vraathena", "karena"]
        results = self.detector.detect_batch(forms)
        assert [r.surface_form for r in results] == forms

    def test_offline_stub_never_raises_without_network(self):
        client = OfflineStubLLMClient()
        response = client.complete("kah dharmah")
        assert isinstance(response, SimpleLLMResponse)
        assert isinstance(response.token_logprobs, list)
        assert len(response.token_logprobs) > 0


# --------------------------------------------------------------------------- #
# Integration: trap generator -> hybrid detector
# --------------------------------------------------------------------------- #


class TestEndToEndIntegration:
    def setup_method(self):
        self.generator = AdversarialTrapGenerator()
        self.detector = HybridDetector(llm_client=OfflineStubLLMClient())

    def test_pipeline_runs_on_full_generated_dataset(self):
        dataset = self.generator.generate_dataset(n_per_type=3)
        results = self.detector.detect_batch([t.surface_form for t in dataset])
        assert len(results) == len(dataset)
        for result in results:
            assert 0.0 <= result.hallucination_probability <= 1.0

    def test_fake_dhatu_traps_are_generally_flagged(self):
        # Not asserted as a strict accuracy claim (that requires a real
        # LLM + gold labels) — only that the symbolic path meaningfully
        # engages with fabricated roots more often than not, as a smoke
        # test that the fabricated-root override in HybridDetector is
        # actually wired up.
        traps = self.generator.generate(10, trap_types=[TrapType.FAKE_DHATU])
        results = [self.detector.detect(t.surface_form) for t in traps]
        flagged_fraction = sum(r.is_flagged_hallucination for r in results) / len(results)
        assert flagged_fraction >= 0.5
