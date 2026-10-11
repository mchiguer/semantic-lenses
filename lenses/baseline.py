"""Baseline: TF-IDF cosine similarity (words + bigrams), no labels used.

Neural sentence embeddings could not be downloaded in the build environment
(HuggingFace access was blocked), so the baseline is lexical. This is the main
limitation of the comparison and the README says so. To plug in real
embeddings, replace ``TfidfBaseline.similarity`` with a cosine over
sentence-transformers vectors; nothing else needs to change.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .textutil import norm


class TfidfBaseline:
    name = "TF-IDF (lexical baseline)"

    def __init__(self, corpus_texts: list[str] | None = None):
        self.corpus = list(corpus_texts or [])

    def similarity(self, a: str, b: str) -> float:
        texts = self.corpus + [a, b]
        vec = TfidfVectorizer(
            preprocessor=norm,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        m = vec.fit_transform(texts)
        return float(cosine_similarity(m[-2], m[-1])[0, 0])
