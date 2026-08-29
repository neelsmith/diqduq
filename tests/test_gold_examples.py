"""
Every GOLD_EXAMPLES entry, run through DummyLM, should validate cleanly --
proving each hand-written canned_answer is at least internally consistent
(every id resolves, every implied token is well-formed) per
hebrew_syntax_dspy.validate(). This does NOT check that the canned answer
is *linguistically correct* against syntax_model.md -- that's a human
judgment call made when each fixture was written (see gold_examples.py's
own module docstring).
"""

import pytest

from conftest import run_gold_example
from diqduq.hebrew_syntax_dspy import validate
from fixtures.gold_examples import GOLD_EXAMPLES


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda ex: ex.slug)
def test_gold_example_validates_cleanly(example):
    tokens, result = run_gold_example(example)
    problems = validate(tokens, result)
    assert problems == []


@pytest.mark.parametrize("example", GOLD_EXAMPLES, ids=lambda ex: ex.slug)
def test_gold_example_tokengraph_matches_canned_answer(example):
    tokens, result = run_gold_example(example)
    got_ids = [tok.id for tok in result.tokengraph]
    expected_ids = [entry["id"] for entry in example.canned_answer["tokengraph"]]
    assert got_ids == expected_ids


def test_no_duplicate_slugs():
    slugs = [ex.slug for ex in GOLD_EXAMPLES]
    assert len(slugs) == len(set(slugs))
