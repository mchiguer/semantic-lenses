"""Evaluation: lexical baseline vs lenses, on hand-labelled pairs."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

from .base import LENS_TITLES, compare
from .baseline import TfidfBaseline

DATA = Path(__file__).resolve().parent.parent / "data"


def load_pairs(name: str) -> list[dict]:
    return json.loads((DATA / f"pairs_{name}.json").read_text(encoding="utf-8"))


def all_texts(*sets: list[dict]) -> list[str]:
    out: list[str] = []
    for pairs in sets:
        for p in pairs:
            out += [p["a"], p["b"]]
    return out


def tune_threshold(sims: list[float], gold_same: list[bool]) -> float:
    """Baseline threshold, chosen on the DEV set only (never on test)."""
    uniq = sorted(set(sims))
    cands = [uniq[0] - 1e-6] + [(a + b) / 2 for a, b in zip(uniq, uniq[1:])] + [uniq[-1] + 1e-6]
    best_acc, best = -1.0, []
    for t in cands:
        acc = sum((s >= t) == g for s, g in zip(sims, gold_same)) / len(sims)
        if acc > best_acc + 1e-12:
            best_acc, best = acc, [t]
        elif abs(acc - best_acc) <= 1e-12:
            best.append(t)
    return sorted(best)[len(best) // 2]


def baseline_sims(pairs: list[dict], baseline: TfidfBaseline) -> list[float]:
    return [baseline.similarity(p["a"], p["b"]) for p in pairs]


def run(pairs: list[dict], lens_backend, baseline: TfidfBaseline, thr: float) -> list[dict]:
    rows = []
    for p in pairs:
        sim = baseline.similarity(p["a"], p["b"])
        cmp = compare(lens_backend.profile(p["a"]), lens_backend.profile(p["b"]))
        rows.append({
            **p,
            "base_sim": sim,
            "base_same": sim >= thr,
            "lens_same": cmp.same,
            "lens_differing": cmp.differing,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_same = sum(r["same"] for r in rows)
    out = {
        "n": n,
        "acc_base": sum(r["base_same"] == r["same"] for r in rows),
        "acc_lens": sum(r["lens_same"] == r["same"] for r in rows),
        "majority": max(n_same, n - n_same),  # score of a classifier that always gives the same answer
        # Threshold-free AUC of the TF-IDF similarity for "same meaning" (0.5 = chance, < 0.5 = ranks backwards)
        "auc_base": float(roc_auc_score([r["same"] for r in rows], [r["base_sim"] for r in rows]))
        if 0 < n_same < n else None,
        "by_type": {},
    }
    for t in ("hard_negative", "paraphrase", "out_of_scope"):
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        # hard_negative / out_of_scope: the right move is NOT to merge; paraphrase: the right move is to merge
        want_same = t == "paraphrase"
        out["by_type"][t] = {
            "n": len(sub),
            "base_ok": sum(r["base_same"] == want_same for r in sub),
            "lens_ok": sum(r["lens_same"] == want_same for r in sub),
        }
    flagged = [r for r in rows if r["type"] == "hard_negative" and not r["lens_same"]]
    out["explain"] = {
        "flagged": len(flagged),
        "exact": sum(set(r["lens_differing"]) == set(r["differs_on"]) for r in flagged),
        "contains_gold": sum(set(r["differs_on"]) <= set(r["lens_differing"]) for r in flagged),
    }
    return out


TYPE_LABELS = {
    "hard_negative": "Near-identical, different meaning (should not merge)",
    "paraphrase": "Paraphrases, different words (should merge)",
    "out_of_scope": "Out of the lenses' reach (should not merge)",
}


def _fmt_set(names: list[str]) -> str:
    return ", ".join(LENS_TITLES.get(n, n) for n in names) or "—"


def markdown_report(backend_name: str, thr: float, dev_rows: list[dict], test_rows: list[dict]) -> str:
    lines = [f"# Results — {backend_name}", ""]
    lines += [
        f"TF-IDF baseline threshold (tuned on DEV only): **{thr:.3f}**.",
        "The lenses have no threshold tuned on this data (fixed thresholds in `lenses/base.py`).",
        "",
    ]
    for title, rows in (("DEV (used to write the rules)", dev_rows), ("TEST (not used to tune anything)", test_rows)):
        s = summarize(rows)
        lines += [f"## {title} — {s['n']} pairs", "",
                  "| Family | n | TF-IDF baseline | Lenses |", "|---|---|---|---|"]
        for t, d in s["by_type"].items():
            lines.append(f"| {TYPE_LABELS[t]} | {d['n']} | {d['base_ok']}/{d['n']} | {d['lens_ok']}/{d['n']} |")
        lines.append(f"| **Total** | {s['n']} | **{s['acc_base']}/{s['n']}** | **{s['acc_lens']}/{s['n']}** |")
        lines += ["", f"Reference: always giving the same answer would score {s['majority']}/{s['n']}. "
                  f"AUC of the TF-IDF similarity for \"same meaning\": {s['auc_base']:.2f} "
                  "(0.5 = chance; below that, similarity ranks the pairs backwards)."]
        e = s["explain"]
        lines += ["",
                  f"Explanation: of {e['flagged']} near-identical pairs flagged as different by the lenses, the right "
                  f"dimension is found exactly in {e['exact']} cases (included in {e['contains_gold']}).", ""]
        errs = [r for r in rows if r["lens_same"] != r["same"] or r["base_same"] != r["same"]]
        lines += ["<details><summary>Error details</summary>", "",
                  "| id | a | b | gold | baseline (sim) | lenses |", "|---|---|---|---|---|---|"]
        for r in errs:
            gold = "same meaning" if r["same"] else "different meaning"
            b = f"{'same' if r['base_same'] else 'different'} ({r['base_sim']:.2f})"
            l = "same" if r["lens_same"] else "differs: " + _fmt_set(r["lens_differing"])
            lines.append(f"| {r['id']} | {r['a']} | {r['b']} | {gold} | {b} | {l} |")
        lines += ["", "</details>", ""]
    lines += [
        "## Honest reading",
        "",
        "- The corpora are small and hand-written: read differences as numbers of pairs, not percentages.",
        "- The rules were written with DEV in view; TEST was not used to tune anything but was written by the same author "
        "at the same time, so it measures generalisation, not independence. The real test is pairs you write yourself.",
        "- The baseline is lexical (TF-IDF), not neural: it is a reference point, not an upper bound.",
        "- The \"out of reach\" families (swapped roles, changed quantities) show what the lenses cannot see.",
        "",
    ]
    return "\n".join(lines)
