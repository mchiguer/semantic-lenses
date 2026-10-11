"""Rule-based lenses for English, 100% offline.

Each lens turns a text into a distribution over a few labels (polarity,
temporality, intent, domain, trend) and keeps the evidence behind the reading,
so every verdict can be explained.
"""
from __future__ import annotations

import re

from .base import LENS_KEYS, Reading
from .textutil import lower, words


def _normalize(v: list[float]) -> list[float]:
    s = sum(v)
    return [x / s for x in v] if s else v


def _onehot(n: int, i: int) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


# --------------------------------------------------------------------------- #
# Polarity
# --------------------------------------------------------------------------- #
NEG_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "without",
            "cannot", "hardly", "barely", "scarcely"}
NT_RE = re.compile(r"\b\w+n't\b")


def polarity(text: str) -> Reading:
    low = lower(text)
    ev: list[str] = []
    for w in re.findall(r"[a-z]+(?:'[a-z]+)?", low):
        if w in NEG_WORDS and w not in ev:
            ev.append(w)
    for m in NT_RE.finditer(low):
        if m.group(0) not in ev:
            ev.append(m.group(0))
    if ev:
        return Reading("polarity", [0.0, 1.0], True, ev)
    return Reading("polarity", [1.0, 0.0], True, ["no negation"])


# --------------------------------------------------------------------------- #
# Temporality
# --------------------------------------------------------------------------- #
IRREG_PAST = {"was", "were", "had", "did", "went", "came", "took", "made", "said", "got", "saw", "gave", "found",
              "told", "knew", "left", "began", "ran", "brought", "bought", "sent", "held", "kept", "led", "lost",
              "met", "paid", "sold", "shot", "stood", "understood", "wrote", "fell", "rose", "grew", "fought",
              "drew", "drove", "ate", "spoke", "broke", "chose", "wore", "won", "built", "spent", "taught",
              "thought", "caught", "felt", "heard", "meant", "sat", "slept", "swam", "threw", "flew", "sank",
              "became", "sought", "fled", "hid", "rang", "sang", "shook", "stole", "woke"}
IRREG_PP = {"been", "gone", "done", "seen", "taken", "given", "written", "begun", "broken", "chosen", "spoken",
            "fallen", "risen", "grown", "driven", "eaten", "shown", "known", "become", "come", "run", "flown",
            "shaken", "stolen", "woken", "sold", "paid", "left", "lost", "made", "kept", "held", "built", "sent",
            "brought", "bought", "found", "met", "won"}
ADV_SKIP = {"not", "also", "still", "already", "currently", "now", "often", "usually", "just", "only", "even",
            "always", "never", "ever", "really", "simply"}
PASSIVE_AUX = {"is", "are", "am", "be", "being", "isn't", "aren't"}
ED_EXC = {"speed", "indeed", "hundred", "sacred", "naked", "wicked", "bleed", "breed", "proceed", "exceed",
          "succeed", "hatred", "kindred", "displaced", "united", "limited", "advanced", "related", "concerned",
          "interested", "required", "skilled", "experienced"}

PAST_RE = re.compile(
    r"\b(?:yesterday|ago|previously|formerly|earlier|in the past|the day before"
    r"|last (?:night|week|month|year|decade|season|quarter|time|monday|tuesday|wednesday|thursday|friday"
    r"|saturday|sunday))\b"
)
FUT_RE = re.compile(
    r"\b(?:tomorrow|soon|in the future|from now on|about to"
    r"|next (?:week|month|year|decade|season|quarter|time|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|in (?:\d+|a|an|one|two|three|four|five|six|seven|ten|few|several|couple of) (?:more )?"
    r"(?:days?|weeks?|months?|years?|hours?|minutes?))\b"
)
GOING_RE = re.compile(r"\bgoing to (?!the\b|a\b|an\b|my\b|our\b|his\b|her\b|their\b)\w+")
PRES_RE = re.compile(r"\b(?:today|currently|now|at the moment|right now|nowadays)\b")


def temporality(text: str) -> Reading:
    low = lower(text)
    ws = words(text)
    past: list[str] = []
    fut: list[str] = []
    pres: list[str] = []

    past += [m.group(0) for m in PAST_RE.finditer(low)]
    fut += [m.group(0) for m in FUT_RE.finditer(low)]
    fut += [m.group(0) for m in GOING_RE.finditer(low)]
    pres += [m.group(0) for m in PRES_RE.finditer(low)]

    for i, w in enumerate(ws):
        if w in IRREG_PAST:
            past.append(w)
        elif w.endswith("ed") and len(w) >= 5 and w not in ED_EXC:
            j = i - 1
            while j >= 0 and ws[j] in ADV_SKIP:
                j -= 1
            prev = ws[j] if j >= 0 else ""
            if prev not in PASSIVE_AUX:  # "is supplied" is a present state, "supplied X" is past
                past.append(w)
        if w in {"has", "have"}:
            j = i + 1
            while j < len(ws) and ws[j] in ADV_SKIP:
                j += 1
            if j < len(ws) and ws[j] in IRREG_PP:
                past.append(f"{w} {ws[j]}")
        if w in {"will", "shall", "won't"} or w.endswith("'ll"):
            fut.append(w)

    p, pr, f = len(past), len(pres), len(fut)
    if p + f == 0:
        ev = pres or ["present by default (no past or future marker)"]
        return Reading("temporality", [0.0, 1.0, 0.0], True, ev)
    ev = [f"past: {x}" for x in past] + [f"present: {x}" for x in pres] + [f"future: {x}" for x in fut]
    return Reading("temporality", _normalize([float(p), float(pr), float(f)]), True, ev)


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #
WH_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}
AUX_Q = {"is", "are", "was", "were", "does", "did", "can", "could", "will", "would", "should", "shall", "may",
        "might", "has", "do"}
DIRECTIVE_RE = re.compile(
    r"\b(?:must|have to|has to|need to|needs to|required to|ought to|you should|we should|please|kindly"
    r"|make sure|be sure to)\b|^(?:do not|don't)\b"
)
IMPERATIVES = {"send", "check", "verify", "complete", "sign", "contact", "call", "read", "make", "put", "add",
            "remove", "delete", "validate", "submit", "share", "review", "update", "ensure", "confirm",
            "fill", "open", "close", "provide", "forward"}
OPINION_RE = re.compile(
    r"\b(?:i think|i believe|i feel|i guess|i suppose|i reckon|i consider|i doubt|in my opinion|in my view"
    r"|to me|it seems to me|i'd say|i would say|personally|as i see it|from my point of view)\b"
)


def intent(text: str) -> Reading:
    low = lower(text).strip()
    ws = words(text)
    first = ws[0] if ws else ""
    second = ws[1] if len(ws) > 1 else ""
    ev_q: list[str] = []
    if "?" in text:
        ev_q.append("?")
    if first in WH_WORDS:
        ev_q.append(f"question word: {first}")
    elif first in AUX_Q and not (first == "do" and second == "not"):
        ev_q.append(f"auxiliary-first order: {first}")
    if ev_q:
        return Reading("intent", _onehot(4, 1), True, ev_q)
    ev_d = [m.group(0) for m in DIRECTIVE_RE.finditer(low)]
    if first in IMPERATIVES:
        ev_d.append(f"imperative: {first}")
    if ev_d:
        return Reading("intent", _onehot(4, 2), True, ev_d)
    ev_o = [m.group(0) for m in OPINION_RE.finditer(low)]
    if ev_o:
        return Reading("intent", _onehot(4, 3), True, ev_o)
    return Reading("intent", _onehot(4, 0), True, ["plain statement by default"])


# --------------------------------------------------------------------------- #
# Domain: lexicon + phrases that resolve ambiguity
# --------------------------------------------------------------------------- #
DOMAIN_KEYS = LENS_KEYS["domain"]

# Entries ending in * match any word starting with that stem; others match the exact word (and its plural).
SINGLE = {
    "health": "patient* clinic* hospital* doctor* nurse* medic* vaccin* epidemic* pandemic* malaria dengue cholera "
            "measles fever blood mortality surgery surgeon* health* symptom* diagnos* dhis2 infect* pharma* "
              "nutrition disease* illness* treatment* therap* outbreak* dispensar* operated",
    "finance": "budget* price euro dollar* franc chf bank* credit* invest investment* investor* inflation cost "
               "invoice* profit* subsid* fund funding funded donor* donation* payment* salary salaries expense* "
               "revenue* saving* insurance loan* currency",
    "tech": "server* api model* code software app application* data algorithm* bug crash* outage* computer* "
            "android ios cloud file cyber* hack* database* password* malware online website*",
    "transport": "flight* airline* airport* aircraft* plane train road roadway* traffic baggage luggage passenger* "
                 "ferry bus railway* runway takeoff",
    "humanitarian": "refugee* camp rescue* rescuer* relief humanitarian* ngo disaster* evacuat* aid displaced "
                    "conflict* war shelter* icrc civilian*",
    "environment": "rain* rainfall climate* flood* drought* harvest* weather storm* earthquake* wildfire* "
                   "pollution agricultur* heatwave snow crop",
    "law": "court* judge* law lawsuit* contract* clause* lawyer* attorney* trial* parliament* lawmaker* "
           "legislat* prosecut* convict* legal*",
    "education": "course coursework student* exam examination* universit* school* professor* teacher* "
                 "lecture* curriculum classroom*",
}

# Ambiguous words: weight split across several domains; context decides.
AMBIG = {
    "rate": {"finance": 0.3, "health": 0.3},
    "case": {"health": 0.4, "law": 0.3},
    "virus": {"health": 0.35, "tech": 0.35},
    "network": {"tech": 0.7, "transport": 0.3},
    "intervention": {"health": 0.5, "humanitarian": 0.5},
    "operation": {"health": 0.5, "finance": 0.3},
    "bridge": {"transport": 0.6},
    "load": {"tech": 0.4},
    "mission": {"humanitarian": 0.4},
    "population": {"humanitarian": 0.3, "health": 0.3},
    "village": {"humanitarian": 0.3},
    "water": {"humanitarian": 0.4, "environment": 0.4},
    "update": {"tech": 0.4},
    "termination": {"law": 0.5},
    "temperature": {"environment": 0.5, "health": 0.2},
    "pilot": {"transport": 0.5},
}

# Fixed expressions: read first and removed from the text before the word-by-word count.
PHRASES: list[tuple[str, dict]] = [
    (r"data bank|computer network|software update|system update|operating system|data cent(?:er|re)", {"tech": 1.0}),
    (r"blood bank|health system|health care|healthcare|(?:mortality|birth|death|infection|vaccination|incidence"
    r"|fatality) rate|network of (?:clinics|dispensaries|hospitals|health cent(?:er|re)s)|health network"
    "|clinic network", {"health": 1.0}),
    (r"(?:interest|exchange|inflation|policy|base) rate", {"finance": 1.0}),
    (r"river ?bank|bank of the river", {"environment": 1.0}),
    (r"(?:drinking|clean|safe) water", {"humanitarian": 0.7, "health": 0.3}),
    (r"of course", {}),
]

_EXACT: dict[str, dict] = {}
_STEMS: list[tuple[str, dict]] = []
for _dom, _words in SINGLE.items():
    for _w in _words.split():
        if _w.endswith("*"):
            _STEMS.append((_w[:-1], {_dom: 1.0}))
        else:
            _EXACT[_w] = {_dom: 1.0}
_EXACT.update(AMBIG)
_STEMS.sort(key=lambda s: -len(s[0]))


def _lookup(tok: str) -> dict | None:
    if tok in _EXACT:
        return _EXACT[tok]
    if tok.endswith("s") and tok[:-1] in _EXACT:
        return _EXACT[tok[:-1]]
    for stem, w in _STEMS:
        if tok.startswith(stem):
            return w
    return None


def domain(text: str) -> Reading:
    low = lower(text)
    score = {k: 0.0 for k in DOMAIN_KEYS}
    ev: list[str] = []
    for pat, weights in PHRASES:
        for m in re.finditer(pat, low):
            for k, x in weights.items():
                score[k] += x
            if weights:
                ev.append(f"“{m.group(0)}” → " + ", ".join(weights))
        low = re.sub(pat, " ", low)
    for tok in re.findall(r"[a-z0-9]+", low):
        w = _lookup(tok)
        if not w:
            continue
        for k, x in w.items():
            score[k] += x
        ev.append(f"{tok} → " + ", ".join(w))
    vec = [score[k] for k in DOMAIN_KEYS]
    if sum(vec) == 0:
        return Reading("domain", vec, False, ["no word from the lexicon"])
    return Reading("domain", _normalize(vec), True, ev)


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
RISE_RE = re.compile(
    r"\b(?:increas\w*|ris(?:e|es|ing)|rose|risen|grow\w*|grew|climb\w*|surge[sd]?|surging|soar\w*|jump\w*"
    r"|spik\w*|escalat\w*|worsen\w*|expand\w*|(?:go|goes|going|went|gone) up)\b"
)
FALL_RE = re.compile(
    r"\b(?:decreas\w*|declin\w*|drop\w*|fall|falls|falling|fell|fallen|plung\w*|plummet\w*|collaps\w*|shrink\w*"
    r"|shrank|shrunk|reduc\w*|diminish\w*|slump\w*|reced\w*|dwindl\w*|slows|slowed|slowing|slowdown"
    r"|(?:go|goes|going|went|gone) down)\b"
)
STABLE_RE = re.compile(
    r"\b(?:stable|stabili[sz]\w*|steady|steadily|stagnat\w*|stagnant|constant\w*|unchanged|flat|plateau\w*"
    r"|levell?ed off)\b"
)


def trend(text: str) -> Reading:
    low = lower(text)
    r = [m.group(0) for m in RISE_RE.finditer(low)]
    f = [m.group(0) for m in FALL_RE.finditer(low)]
    s = [m.group(0) for m in STABLE_RE.finditer(low)]
    if not (r or f or s):
        return Reading("trend", [0.0, 0.0, 0.0], False, ["no word of change"])
    ev = [f"rise: {x}" for x in r] + [f"fall: {x}" for x in f] + [f"stable: {x}" for x in s]
    return Reading("trend", _normalize([float(len(r)), float(len(f)), float(len(s))]), True, ev)


class RuleLenses:
    """Offline backend: no dependency, no network call."""

    name = "Rules (offline)"

    def profile(self, text: str) -> dict:
        return {
            "polarity": polarity(text),
            "temporality": temporality(text),
            "intent": intent(text),
            "domain": domain(text),
            "trend": trend(text),
        }
