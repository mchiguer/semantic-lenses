"""Small text-normalisation helpers (English)."""
from __future__ import annotations

import re
import unicodedata


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def lower(s: str) -> str:
    """Lower-case, straight apostrophes, normalised spaces."""
    return s.lower().replace("’", "'").replace("‘", "'").replace(" ", " ")


def norm(s: str) -> str:
    return strip_accents(lower(s))


def words(s: str) -> list[str]:
    """Word tokens, keeping contractions together (isn't, they'll)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", lower(s))
