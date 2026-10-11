"""Run the baseline-vs-lenses evaluation and write RESULTS.md.

    python evaluate.py                 # rule-based lenses (offline)
    python evaluate.py --backend llm   # lenses scored by an LLM (API key required)
    python evaluate.py --dev-only      # print DEV numbers only (use this while tuning rules)
"""
import argparse
from pathlib import Path

from lenses import RuleLenses, TfidfBaseline
from lenses.evaluation import (all_texts, baseline_sims, load_pairs, markdown_report, run, summarize,
                               tune_threshold)


def show(label: str, rows: list[dict]) -> None:
    s = summarize(rows)
    print(f"{label}: baseline {s['acc_base']}/{s['n']} | lenses {s['acc_lens']}/{s['n']}")
    for t, d in s["by_type"].items():
        print(f"   {t:14s} baseline {d['base_ok']}/{d['n']} | lenses {d['lens_ok']}/{d['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rules", "llm"], default="rules")
    ap.add_argument("--dev-only", action="store_true", help="do not evaluate or write anything for the TEST set")
    args = ap.parse_args()

    dev, test = load_pairs("dev"), load_pairs("test")
    baseline = TfidfBaseline(all_texts(dev, test))  # unsupervised IDF: no label is used
    thr = tune_threshold(baseline_sims(dev, baseline), [p["same"] for p in dev])

    if args.backend == "llm":
        from lenses.llm import LLMLenses
        backend = LLMLenses()
        out = "RESULTS_llm.md"
    else:
        backend = RuleLenses()
        out = "RESULTS.md"

    dev_rows = run(dev, backend, baseline, thr)
    show("DEV", dev_rows)
    if args.dev_only:
        for r in dev_rows:
            if r["lens_same"] != r["same"] or (not r["lens_same"] and set(r["lens_differing"]) != set(r["differs_on"])):
                print("   ✗", r["id"], "predicted:", "same" if r["lens_same"] else r["lens_differing"], "| gold:", r["differs_on"] or "same")
        return

    test_rows = run(test, backend, baseline, thr)
    show("TEST", test_rows)
    Path(out).write_text(markdown_report(backend.name, thr, dev_rows, test_rows), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
