"""
core/paninian_engine.py

Symbolic Paninian Validator.

This module implements a small, transparent, rule-based state machine
that checks a Sanskrit surface form (word or short phrase) against a
seed lexicon of verbal roots (dhatus) and suffixes (pratyayas), and
computes two structural scores used elsewhere in the pipeline:

  - Sandhi Ambiguity Score (SAS): how ambiguous the possible sandhi
    splits of a string are, i.e. how many phonetically-valid ways it
    could be decomposed into two or more legitimate words, weighted
    by how divergent their meanings/cases are.

  - Vibhakti Deception Vector (VDV): an 8-dimensional vector (one
    entry per case: Prathama..Sambodhana) scoring how compatible a
    word's suffix is with each case reading, so that a "deceptive"
    form (one that superficially looks like a different case than it
    is) produces a flat/ambiguous vector instead of a single spike.

IMPORTANT — Scope and honesty about the lexicon:
This module ships with a small, illustrative SEED lexicon
(`_SEED_DHATUS`, `_SEED_PRATYAYAS`, `_SEED_SANDHI_RULES`) sufficient to
make the pipeline runnable and testable end-to-end. It is NOT a
complete or authoritative Paninian grammar. Real research use requires
plugging in a vetted lexicon (e.g. digitized Dhatupatha +
Siddhanta Kaumudi suffix tables) via the paths in config/settings.py.
The engine will transparently fall back to the seed set if the
external resource files are not found, and will log a warning when it
does so — it will never silently pretend to be more authoritative than
it is.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from config.settings import (
    DHATU_LIST_PATH,
    PRATYAYA_LIST_PATH,
    SANDHI_RULES_PATH,
    get_settings,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Seed lexicon (illustrative, not exhaustive — see module docstring)
# --------------------------------------------------------------------------- #

# A small set of genuine, well-attested verbal roots (dhatus), transliterated
# in IAST-ish ASCII for portability, each tagged with its gana (class) and a
# gloss. This is a teaching/testing seed, not a research-grade lexicon.
_SEED_DHATUS: Dict[str, Dict[str, str]] = {
    "bhu": {"gana": "1", "gloss": "to be, to become"},
    "kr": {"gana": "8", "gloss": "to do, to make"},
    "gam": {"gana": "1", "gloss": "to go"},
    "drs": {"gana": "1", "gloss": "to see"},  # drś, transliterated without diacritic
    "vad": {"gana": "1", "gloss": "to speak"},
    "pac": {"gana": "1", "gloss": "to cook"},
    "path": {"gana": "1", "gloss": "to read"},
    "likh": {"gana": "6", "gloss": "to write"},
    "sthaa": {"gana": "1", "gloss": "to stand"},
    "ii": {"gana": "2", "gloss": "to go"},
    "as": {"gana": "2", "gloss": "to be"},
    "han": {"gana": "2", "gloss": "to kill, to strike"},
    "daa": {"gana": "3", "gloss": "to give"},
    "labh": {"gana": "1", "gloss": "to obtain"},
    "budh": {"gana": "1", "gloss": "to know, to awaken"},
}

# Common primary/secondary suffixes (pratyayas), each with the case(s) they
# canonically mark for a-stem masculine nouns as a simple illustrative
# example (real Paninian suffix behavior varies enormously by stem class,
# gender, and sandhi context — this is deliberately a simplification).
_SEED_PRATYAYAS: Dict[str, Dict[str, object]] = {
    "aha": {"case": ["Prathama"], "number": ["singular"]},
    "au": {"case": ["Prathama", "Dvitiya"], "number": ["dual"]},
    "aah": {"case": ["Prathama"], "number": ["plural"]},
    "am": {"case": ["Dvitiya"], "number": ["singular"]},
    "aan": {"case": ["Dvitiya"], "number": ["plural"]},
    "ena": {"case": ["Tritiya"], "number": ["singular"]},
    "aabhyaam": {"case": ["Tritiya", "Panchami"], "number": ["dual"]},
    "aih": {"case": ["Tritiya"], "number": ["plural"]},
    "aaya": {"case": ["Chaturthi"], "number": ["singular"]},
    "aat": {"case": ["Panchami"], "number": ["singular"]},
    "asya": {"case": ["Shashthi"], "number": ["singular"]},
    "aanaam": {"case": ["Shashthi"], "number": ["plural"]},
    "e": {"case": ["Saptami"], "number": ["singular"]},
    "eshu": {"case": ["Saptami"], "number": ["plural"]},
}

_ALL_CASES: Tuple[str, ...] = (
    "Prathama",
    "Dvitiya",
    "Tritiya",
    "Chaturthi",
    "Panchami",
    "Shashthi",
    "Saptami",
    "Sambodhana",
)

# A minimal set of sandhi merge/split rules: (left_final, right_initial) ->
# merged_form. Used symmetrically to propose candidate splits of a string.
_SEED_SANDHI_RULES: Dict[str, str] = {
    "a+a": "aa",
    "a+i": "e",
    "a+u": "o",
    "aa+a": "aa",
    "i+i": "ii",
    "u+u": "uu",
    "as+a": "o'",   # visarga sandhi (simplified)
    "ah+a": "o",
}


@dataclass(frozen=True)
class Lexicon:
    dhatus: Dict[str, Dict[str, str]]
    pratyayas: Dict[str, Dict[str, object]]
    sandhi_rules: Dict[str, str]


def load_lexicon() -> Lexicon:
    """Load the Paninian lexicon from config-specified files, falling back
    to the illustrative seed set if the files are absent. Never raises on
    missing external resources — always returns a usable lexicon."""
    dhatus = _SEED_DHATUS
    pratyayas = _SEED_PRATYAYAS
    sandhi_rules = _SEED_SANDHI_RULES

    for path, label in (
        (DHATU_LIST_PATH, "dhatu list"),
        (PRATYAYA_LIST_PATH, "pratyaya list"),
        (SANDHI_RULES_PATH, "sandhi rules"),
    ):
        if not path.exists():
            logger.warning(
                "%s not found at %s — falling back to the illustrative "
                "seed lexicon shipped with this repo. Do not treat "
                "evaluation results as research-grade until a vetted "
                "lexicon is supplied.",
                label,
                path,
            )

    if DHATU_LIST_PATH.exists():
        with open(DHATU_LIST_PATH, "r", encoding="utf-8") as f:
            dhatus = json.load(f)
    if PRATYAYA_LIST_PATH.exists():
        with open(PRATYAYA_LIST_PATH, "r", encoding="utf-8") as f:
            pratyayas = json.load(f)
    if SANDHI_RULES_PATH.exists():
        with open(SANDHI_RULES_PATH, "r", encoding="utf-8") as f:
            sandhi_rules = json.load(f)

    return Lexicon(dhatus=dhatus, pratyayas=pratyayas, sandhi_rules=sandhi_rules)


class PaninianEngine:
    """Symbolic validator combining a dhatu/pratyaya lexicon lookup with
    sandhi-split enumeration to produce SAS and VDV scores for a given
    surface form."""

    def __init__(self, lexicon: Lexicon | None = None):
        self.lexicon = lexicon or load_lexicon()
        self.weights = get_settings().sas_weights
        self.vdv_weights = get_settings().vdv_weights

    # ----------------------------- Dhatu checks --------------------------- #

    def is_known_dhatu(self, root: str) -> bool:
        """Return True iff `root` (already stripped of inflection) matches
        a known dhatu in the loaded lexicon (case-insensitive)."""
        return root.strip().lower() in {k.lower() for k in self.lexicon.dhatus}

    def strip_common_suffixes(self, word: str) -> List[str]:
        """Return candidate root forms: the bare word itself (a real
        dhatu can be a complete word with no suffix, e.g. "gam"), plus
        every result of stripping a known pratyaya suffix from `word`.

        BUGFIX: an earlier version only fell back to the bare word when
        no suffix matched at all. That silently misclassified any real,
        unmodified dhatu that happens to end in a string identical to a
        known suffix — e.g. "gam" ends in "am" (a real Dvitiya suffix),
        so it was being stripped to "g" and never checked as "gam"
        itself, producing a false "unknown/fabricated root" verdict on
        a genuine word. Always including the bare form as a candidate
        fixes this without removing any previously-found candidate."""
        lw = word.strip().lower()
        candidates: List[str] = [lw]
        for suffix in self.lexicon.pratyayas:
            if lw.endswith(suffix.lower()) and len(lw) > len(suffix):
                candidates.append(lw[: -len(suffix)])
        return candidates

    def validate_word(self, word: str) -> Dict[str, object]:
        """Core symbolic check: does `word` reduce (after stripping a
        plausible suffix) to a known dhatu? Returns a structured verdict
        used by hybrid_detector.py to penalize fabricated roots."""
        candidates = self.strip_common_suffixes(word)
        matches = [c for c in candidates if self.is_known_dhatu(c)]
        return {
            "word": word,
            "candidate_roots": candidates,
            "matched_known_dhatu": bool(matches),
            "matched_roots": matches,
        }

    # ----------------------------- Sandhi (SAS) ---------------------------- #

    def _candidate_splits(self, s: str) -> List[Tuple[str, str]]:
        """Enumerate naive binary split points of a joined string `s` as
        (left, right) candidate word pairs. This treats every internal
        character boundary as a hypothetical sandhi junction — a
        simplification standing in for full sandhi-reversal rules, which
        require a much larger rule table (see SANDHI_RULES_PATH)."""
        s = s.strip()
        splits = []
        for i in range(1, len(s)):
            left, right = s[:i], s[i:]
            if len(left) >= 1 and len(right) >= 1:
                splits.append((left, right))
        return splits

    def sandhi_ambiguity_score(self, word: str) -> float:
        """
        Compute the Sandhi Ambiguity Score (SAS) for `word`:

            SAS(w) = alpha * H_split(w) + beta * O_boundary(w)
                     + gamma * C_phonetic(w)

        where
          H_split(w)   = normalized Shannon entropy over the distribution
                         of "plausibility" across all candidate binary
                         splits of w (plausibility approximated here by
                         whether each side reduces to a known dhatu or
                         known suffix — a symbolic proxy, not a full
                         phonological sandhi-reversal model),
          O_boundary(w)= fraction of character boundaries in w that
                         coincide with a known sandhi merge pattern from
                         the loaded sandhi_rules table,
          C_phonetic(w)= fraction of candidate splits whose left-hand
                         side and right-hand side BOTH resolve to a known
                         dhatu root (i.e. genuinely competing valid
                         parses — the strongest ambiguity signal).

        Returns a float in [0, 1], where higher = more ambiguous /
        more exploitable by an adversarial sandhi trap.
        """
        splits = self._candidate_splits(word)
        if not splits:
            return 0.0

        # --- H_split: entropy over plausibility of each split --- #
        plausibilities = []
        for left, right in splits:
            left_known = self.is_known_dhatu(left) or left.lower() in {
                k.lower() for k in self.lexicon.pratyayas
            }
            right_known = self.is_known_dhatu(right) or right.lower() in {
                k.lower() for k in self.lexicon.pratyayas
            }
            score = 0.0
            if left_known:
                score += 0.5
            if right_known:
                score += 0.5
            plausibilities.append(score)

        total = sum(plausibilities)
        if total <= 0:
            h_split = 0.0
        else:
            probs = [p / total for p in plausibilities if p > 0]
            h_split = -sum(p * math.log(p + 1e-12) for p in probs)
            h_split /= math.log(len(probs) + 1e-12) if len(probs) > 1 else 1.0
            h_split = max(0.0, min(1.0, h_split))

        # --- O_boundary: known sandhi merge patterns present --- #
        boundary_hits = 0
        for pattern in self.lexicon.sandhi_rules.values():
            if pattern in word:
                boundary_hits += 1
        o_boundary = min(1.0, boundary_hits / max(1, len(self.lexicon.sandhi_rules)))

        # --- C_phonetic: genuinely competing valid dhatu parses --- #
        competing = sum(
            1
            for left, right in splits
            if self.is_known_dhatu(left) and self.is_known_dhatu(right)
        )
        c_phonetic = min(1.0, competing / max(1, len(splits)))

        sas = (
            self.weights.alpha * h_split
            + self.weights.beta * o_boundary
            + self.weights.gamma * c_phonetic
        )
        return max(0.0, min(1.0, sas))

    # ----------------------------- Vibhakti (VDV) --------------------------- #

    def vibhakti_deception_vector(self, word: str) -> Dict[str, float]:
        """
        Compute the Vibhakti Deception Vector (VDV) for `word`: an entry
        per case in _ALL_CASES giving how compatible the word's suffix is
        with that case reading, according to the loaded pratyaya table.

        A "deceptive" word — one an adversarial prompt claims is
        unambiguously case X — produces a VDV with mass spread across
        multiple cases rather than concentrated on a single one. The
        scalar deception magnitude is exposed via `vdv_deception_score`.
        """
        lw = word.strip().lower()
        raw: Dict[str, float] = {case: 0.0 for case in _ALL_CASES}

        matched_any = False
        for suffix, info in self.lexicon.pratyayas.items():
            if lw.endswith(suffix.lower()):
                matched_any = True
                cases = info.get("case", [])
                for case in cases:
                    if case in raw:
                        raw[case] += 1.0

        if matched_any:
            total = sum(raw.values())
            if total > 0:
                raw = {k: v / total for k, v in raw.items()}

        return raw

    def vdv_deception_score(self, vdv: Dict[str, float]) -> float:
        """
        Collapse a VDV into a single scalar "deception magnitude" in
        [0, 1]: 0 means the vector is a clean spike on one case (no
        ambiguity), 1 means mass is maximally spread across cases
        (maximally deceptive / ambiguous).

        Implemented as normalized entropy of the VDV distribution. The
        `case_ambiguity_weight` from VDVWeights is applied by callers that
        fuse this scalar with number/gender-ambiguity terms (reserved for
        future extension); this method itself returns the unweighted
        normalized entropy so that weighting policy lives in one place
        (hybrid_detector.py) rather than being silently baked in here.
        """
        values = [v for v in vdv.values() if v > 0]
        if len(values) <= 1:
            return 0.0
        entropy = -sum(v * math.log(v + 1e-12) for v in values)
        entropy /= math.log(len(values))
        return max(0.0, min(1.0, entropy))