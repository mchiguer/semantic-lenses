# Semantic lens selector

> Two texts that look alike can say different things; two texts that look nothing alike can say the same thing.
> **It depends on the dimension you read them through.**

This prototype is my reading of one problem: reading the same piece of information through different spaces (“lenses”).
Each text becomes a small, **explainable** multidimensional profile; two texts are compared lens by lens, and the tool says
*which dimension* they diverge on.

## The five lenses

| Lens | Labels | Example of divergence |
|---|---|---|
| Polarity | affirmative / negative | “The camp has medicines” vs “The camp no longer has medicines” |
| Temporality | past / present / future | “Flights will resume” vs “Flights resumed” |
| Intent | information / question / directive / opinion | “The vaccine is available” vs “Is the vaccine available?” |
| Domain | health, finance, tech, transport, humanitarian, environment, law, education | “**mortality** rate” vs “**interest** rate” |
| Trend | rise / fall / stable | “cases are rising” vs “cases are falling” |

Design principles:

- **An undetermined lens is ignored.** If a text says nothing about a trend, we conclude nothing about the trend.
- **Context resolves ambiguity.** “rate”, “case”, “virus” and “network” split their weight across several domains,
  and fixed expressions (“blood bank”, “computer network”, “interest rate”) settle the reading.
- **Everything is justified.** Each reading keeps its evidence (words, patterns); the interface shows it.

## Run it

```bash
pip install -r requirements.txt      # scikit-learn, numpy (anthropic only for the optional LLM backend)
python app.py                        # interface: http://127.0.0.1:8000
python evaluate.py                   # baseline vs lenses → RESULTS.md
python evaluate.py --dev-only        # DEV numbers only: use this while tuning rules, keep TEST untouched
python -m unittest discover -s tests # unit tests
```

The LLM backend (same lenses, scored by Claude) switches on with `ANTHROPIC_API_KEY`; see “Limits” below.

## Evaluation

Corpus (`data/`): 50 hand-written English pairs in three families.

- **Near-identical, different meaning** (29): exactly one dimension changes (negation, tense, direction…).
- **Paraphrases** (17): different words, same meaning.
- **Out of reach** (4): swapped roles, changed quantities. They are here to show what the lenses *cannot* see.

Two sets: **DEV** (30 pairs, in view while the rules were written) and **TEST** (20 pairs, no threshold or rule tuned on it).

| | TF-IDF baseline | Lenses |
|---|---|---|
| DEV: near-identical | 17/17 | 17/17 |
| DEV: paraphrases | 0/11 | 11/11 |
| DEV: out of reach | 2/2 | 0/2 |
| **DEV total** | **19/30** | **28/30** |
| TEST: near-identical | 12/12 | 12/12 |
| TEST: paraphrases | 0/6 | 6/6 |
| TEST: out of reach | 1/2 | 0/2 |
| **TEST total** | **13/20** | **18/20** |

### How to read these numbers (and what they do not prove)

1. **The baseline is not “bad”, it is backwards.** Its DEV-tuned threshold (0.805) ends up answering “different meaning” to everything on DEV:
   19/30 is exactly the score of a constant answer. On TEST it scores 13/20, one pair *below* the constant answer (14/20).
   On this corpus TF-IDF similarity ranks pairs **backwards** (AUC 0.11 on DEV, 0.19 on TEST; 0.5 = chance): near-identical pairs share more words than
   paraphrases do, by construction. That illustrates the problem, but it is not a measurement of what neural embeddings would do, which I could not test (see Limits).
2. **TEST is not independent.** It was written by the same author, at the same time as the rules, and **this English corpus mirrors the French one**
   (same themes and the same kinds of trap, rewritten as natural English), so it is not fresh evidence that the approach holds up on arbitrary English.
   What I did do: no rule was changed after seeing TEST results (one generic fix, “river bank”, was made beforehand from fresh probe sentences).
   The real test is **yours**: add your own pairs to `data/` (or type them in the interface) and see what breaks.
3. **Small corpus:** 20 to 30 pairs, so read differences as numbers of pairs, not percentages. A perfect score on the near-identical family
   mostly says the corpus is easy for rules written with those traps in mind.
4. **Explanations:** when the lenses flag a divergence, the dimension they name is the right one in 17/17 (DEV) and 12/12 (TEST) cases.
   That is what separates the approach from a single similarity score.

## Limits (what I did not solve)

- **Swapped roles and changed quantities** (“the doctor examined the patient” / “the patient examined the doctor”; “5%” / “50%”): no lens sees them,
  so the four matching pairs are wrong, on purpose. A “who did what to whom” or a “quantity” lens would be the natural next step.
- **Rule-based and brittle.** About 160 lexicon words (plus 16 ambiguous ones and 6 phrases) and no syntactic parsing. Known English traps it does not handle:
  exclamations without a question mark (“What a result” reads as a question), idioms (“fell short” reads as a fall), conditionals and “would”,
  adjectival “-ed” words outside a short exception list, and double negation. Present perfect (“has left”) is deliberately read as past.
  Outside the covered themes the Domain lens answers “undetermined”.
- **Lexical baseline.** HuggingFace access was blocked in the build environment, so no neural embeddings. To plug some in, replace
  `TfidfBaseline.similarity` with a cosine over `sentence-transformers` vectors; `evaluate.py` does not change.
  Expect them to do much better than TF-IDF on paraphrases while staying sensitive to negation and tense.
- **LLM backend not tested against the real API.** The prompt, response parsing and caching are tested with a fake client (`tests/`),
  but no real call was made (no key, no network). The default model name can be changed with `LENSES_LLM_MODEL`.
- **Interface tested server-side only.** The API was exercised end to end and the page's JavaScript was syntax-checked,
  but the page was not opened in a real browser. If something looks odd on screen, that is the first place to look.

## Next steps

1. Replace the rules with **learned projections** (a small linear head per lens on top of embeddings) and compare with the rules on your own pairs.
2. Have an LLM score the lenses and measure its **agreement** with the rules: where do they disagree?
3. Move from a synthetic corpus to **real text** from your domain (indicator labels, report sentences).
4. Add lenses: quantity, scale (local / national), source, certainty.

## Questions this opens for me

- Who defines the lenses: fixed in advance, or should they emerge from the data?
- How should data be handled when it is only readable under some lenses (here: “undetermined”)?
- When two readings diverge on one lens but not another, which one wins, and in what context?

## Layout

```
app.py               web interface (standard library) + JSON API
evaluate.py          evaluation → RESULTS.md
lenses/rules.py      the five rule-based lenses (offline)
lenses/llm.py        optional LLM backend
lenses/baseline.py   TF-IDF baseline
lenses/base.py       readings, thresholds, comparison
lenses/evaluation.py corpus loading, metrics, report
data/                dev and test pairs
tests/               unit tests
```
