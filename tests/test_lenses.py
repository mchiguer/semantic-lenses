"""Unit tests (unittest, no dependency): python -m unittest discover -s tests"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenses import LENS_KEYS, RuleLenses, compare  # noqa: E402
from lenses.llm import LLMLenses, parse_response  # noqa: E402
from lenses.rules import domain, intent, polarity, temporality, trend  # noqa: E402


def top(reading):
    labels = LENS_KEYS[reading.lens]
    return labels[max(range(len(reading.vector)), key=lambda i: reading.vector[i])]


class TestRules(unittest.TestCase):
    def test_polarity(self):
        self.assertEqual(top(polarity("The road is open.")), "affirmative")
        for t in ("The road isn't open.", "No road is open.", "We can't confirm the figure.",
                  "She cannot attend.", "He knows nothing.", "It was done without permission."):
            self.assertEqual(top(polarity(t)), "negative", t)
        self.assertEqual(top(polarity("A noble cause.")), "affirmative")  # "no" inside a word

    def test_temporality(self):
        cases = {
            "The shipment arrived last week.": "past",
            "Schools reopened in May.": "past",
            "They have already delivered the kits.": "past",
            "The patient has left the hospital.": "past",
            "The shipment arrives next week.": "future",
            "Flights will resume on Monday.": "future",
            "We're going to expand the clinic.": "future",
            "The shipment is delayed.": "present",       # "delayed" after "is" is a state, not a past verb
            "Doctors are expected to arrive.": "present",
            "The displaced families live in tents.": "present",
            "We're going to the clinic.": "present",     # movement, not future
        }
        for text, want in cases.items():
            self.assertEqual(top(temporality(text)), want, text)

    def test_intent(self):
        self.assertEqual(top(intent("Is the vaccine available?")), "question")
        self.assertEqual(top(intent("Why does the server crash")), "question")
        self.assertEqual(top(intent("Do the figures match?")), "question")
        self.assertEqual(top(intent("Do not enter the ward.")), "directive")
        self.assertEqual(top(intent("Please confirm receipt.")), "directive")
        self.assertEqual(top(intent("Check the logs.")), "directive")
        self.assertEqual(top(intent("You must send the report.")), "directive")
        self.assertEqual(top(intent("I doubt it will work.")), "opinion")
        self.assertEqual(top(intent("In my opinion, it is fine.")), "opinion")
        self.assertEqual(top(intent("The model is reliable.")), "information")
        self.assertEqual(top(intent("He asked what happened.")), "information")

    def test_domain_resolves_ambiguity(self):
        self.assertEqual(top(domain("The mortality rate rose.")), "health")
        self.assertEqual(top(domain("The interest rate rose.")), "finance")
        self.assertEqual(top(domain("The data bank is overloaded.")), "tech")
        self.assertEqual(top(domain("The blood bank is overloaded.")), "health")
        self.assertEqual(top(domain("The river bank flooded.")), "environment")
        self.assertEqual(top(domain("The exam results are out.")), "education")  # not tripped by "example"
        self.assertFalse(domain("Examples are listed below.").determined)
        self.assertFalse(domain("It is nice outside.").determined)

    def test_trend(self):
        self.assertEqual(top(trend("Cases are increasing.")), "rise")
        self.assertEqual(top(trend("Prices surged.")), "rise")
        self.assertEqual(top(trend("Cases are falling.")), "fall")
        self.assertEqual(top(trend("Demand levelled off.")), "stable")
        self.assertFalse(trend("Surgery was postponed.").determined)  # "surgery" is not "surge"
        self.assertFalse(trend("The traffic exists.").determined)


class TestCompare(unittest.TestCase):
    R = RuleLenses()

    def cmp(self, a, b, lenses=None):
        return compare(self.R.profile(a), self.R.profile(b), lenses)

    def test_negation_detected(self):
        c = self.cmp("The flight is cancelled.", "The flight isn't cancelled.")
        self.assertFalse(c.same)
        self.assertEqual(c.differing, ["polarity"])

    def test_paraphrase_merged(self):
        self.assertTrue(self.cmp("In my opinion, the model is reliable.", "I think the model is reliable.").same)

    def test_unselected_lens_ignored(self):
        c = self.cmp("Cases are rising.", "Cases are falling.", ["polarity"])
        self.assertTrue(c.same)  # the divergence is on "trend", which is not selected

    def test_undetermined_lens_ignored(self):
        c = self.cmp("Cases are rising.", "Cases are numerous.", ["trend"])
        self.assertIsNone(c.per_lens["trend"])
        self.assertTrue(c.same)

    def test_known_limits(self):
        # Documented in the README: neither swapped roles nor changed quantities are seen.
        self.assertTrue(self.cmp("The doctor examined the patient.", "The patient examined the doctor.").same)
        self.assertTrue(self.cmp("The price rose by 5%.", "The price rose by 50%.").same)


class FakeClient:
    def __init__(self, text):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)
        self._text = text

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


class TestLLMParsing(unittest.TestCase):
    RAW = json.dumps({
        "polarity": {"affirmative": 0.1, "negative": 0.9},
        "temporality": {"past": 0, "present": 1, "future": 0},
        "intent": {"information": 1, "question": 0, "directive": 0, "opinion": 0},
        "domain": {k: 0 for k in LENS_KEYS["domain"]},
        "trend": {"rise": 0.5, "fall": 0.5, "stable": 0},
    })

    def test_parse(self):
        prof = parse_response("Here:\n" + self.RAW + "\nend")
        self.assertEqual(top(prof["polarity"]), "negative")
        self.assertFalse(prof["domain"].determined)
        self.assertAlmostEqual(sum(prof["trend"].vector), 1.0)

    def test_parse_without_json(self):
        with self.assertRaises(ValueError):
            parse_response("no json here")

    def test_cache_and_fake_client(self):
        client = FakeClient(self.RAW)
        backend = LLMLenses(client=client)
        backend.profile("text")
        backend.profile("text")
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
