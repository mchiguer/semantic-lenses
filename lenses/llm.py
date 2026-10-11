"""Optional backend: the same lenses, scored by an LLM (Anthropic API).

⚠️ Response parsing is tested (see tests/), but a real API call could NOT be
run while building this project (no key, no network in the build environment).
Try it with your own key:

    export ANTHROPIC_API_KEY=...
    python evaluate.py --backend llm
"""
from __future__ import annotations

import json
import os
import re

from .base import LENS_KEYS, Reading

SYSTEM = """You are a linguistic annotator. For the given English text, score each lens with a \
distribution of weights between 0 and 1 over the labels listed.
Lenses and labels:
{lenses}

Rules:
- Reply with a single JSON object only: {{"polarity": {{...}}, "temporality": {{...}}, ...}}.
- For "domain" and "trend", set every weight to 0 if the text carries neither.
- For the other lenses the weights sum to 1.
- Add no text outside the JSON."""


def build_system_prompt() -> str:
    lines = [f"- {lens}: {', '.join(keys)}" for lens, keys in LENS_KEYS.items()]
    return SYSTEM.format(lenses="\n".join(lines))


def parse_response(raw: str) -> dict:
    """Turn the model's JSON reply into a profile of readings."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("LLM reply contains no JSON object")
    data = json.loads(m.group(0))
    profile: dict = {}
    for lens, keys in LENS_KEYS.items():
        scores = data.get(lens, {}) or {}
        vec = [max(0.0, float(scores.get(k, 0.0))) for k in keys]
        total = sum(vec)
        if total == 0:
            profile[lens] = Reading(lens, vec, False, ["not carried by the text (per the LLM)"])
        else:
            profile[lens] = Reading(lens, [x / total for x in vec], True, ["scored by the LLM"])
    return profile


class LLMLenses:
    name = "Claude (API)"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.environ.get("LENSES_LLM_MODEL", "claude-sonnet-5-5")
        if client is None:
            import anthropic  # lazy import: the rest of the project does not need it

            client = anthropic.Anthropic()
        self.client = client
        self._cache: dict = {}

    def profile(self, text: str) -> dict:
        if text in self._cache:
            return self._cache[text]
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=build_system_prompt(),
            messages=[{"role": "user", "content": text}],
        )
        raw = "".join(getattr(b, "text", "") for b in msg.content)
        prof = parse_response(raw)
        self._cache[text] = prof
        return prof
