"""Shared structures: lenses, readings, comparison."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

LENS_KEYS: dict[str, list[str]] = {
    "polarity": ["affirmative", "negative"],
    "temporality": ["past", "present", "future"],
    "intent": ["information", "question", "directive", "opinion"],
    "domain": ["health", "finance", "tech", "transport", "humanitarian", "environment", "law", "education"],
    "trend": ["rise", "fall", "stable"],
}
LENSES = LENS_KEYS  # display labels are the same in English

LENS_TITLES = {
    "polarity": "Polarity",
    "temporality": "Temporality",
    "intent": "Intent",
    "domain": "Domain",
    "trend": "Trend",
}

# Cosine similarity below which two readings are judged different.
THRESHOLDS = {
    "polarity": 0.7,
    "temporality": 0.7,
    "intent": 0.7,
    "domain": 0.55,
    "trend": 0.7,
}


@dataclass
class Reading:
    """How one text reads under one lens: a distribution over the lens labels."""

    lens: str
    vector: list[float]
    determined: bool = True
    evidence: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.determined:
            return "— (undetermined)"
        parts = sorted(zip(LENSES[self.lens], self.vector), key=lambda p: -p[1])
        return " · ".join(f"{lab} {w:.2f}" for lab, w in parts if w > 0.01)

    def to_dict(self) -> dict:
        return {
            "lens": self.lens,
            "vector": [round(x, 4) for x in self.vector],
            "labels": LENSES[self.lens],
            "determined": self.determined,
            "evidence": self.evidence,
            "text": self.describe(),
        }


Profile = dict  # lens -> Reading


def cosine(u: list[float], v: list[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return 0.0
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


@dataclass
class Comparison:
    per_lens: dict  # lens -> similarity (float) or None when undetermined
    differing: list  # lenses on which the two texts diverge
    same: bool
    profile_a: dict
    profile_b: dict

    def to_dict(self) -> dict:
        return {
            "per_lens": self.per_lens,
            "differing": self.differing,
            "same": self.same,
            "profile_a": {k: v.to_dict() for k, v in self.profile_a.items()},
            "profile_b": {k: v.to_dict() for k, v in self.profile_b.items()},
        }


def compare(pa: Profile, pb: Profile, lenses: list[str] | None = None) -> Comparison:
    """Compare two profiles lens by lens.

    A lens that is undetermined for either text is ignored: we do not conclude
    anything from a dimension the text does not carry.
    """
    selected = list(lenses) if lenses else list(LENS_KEYS)
    per_lens: dict = {}
    differing: list = []
    for lens in selected:
        ra, rb = pa[lens], pb[lens]
        if not (ra.determined and rb.determined):
            per_lens[lens] = None
            continue
        sim = cosine(ra.vector, rb.vector)
        per_lens[lens] = round(sim, 4)
        if sim < THRESHOLDS[lens]:
            differing.append(lens)
    return Comparison(per_lens, differing, not differing, pa, pb)
