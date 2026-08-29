"""
Shared test helpers. Modeled directly on arsgrammatica's tests/conftest.py.

Kept deliberately small: every DummyLM-backed test reconfigures
dspy.settings.lm before it runs, so there's no cross-test state to tear
down here (yet). If that changes, this is the place for a pytest fixture
that resets dspy.settings between tests.
"""

import os
from pathlib import Path

import dspy
import pytest
from dotenv import load_dotenv
from dspy.utils.dummies import DummyLM

from diqduq import analyze
from diqduq.models import IMPLIED_TOKENTYPES, Token
from diqduq.segmentation_dspy import segment_sources


def tokens_from_canned_answer(canned_answer):
    """Build a Token list directly from a canned_answer's own tokengraph,
    rather than from a separate tokenizer.

    Gold fixtures already specify each token's id and surface text via
    their tokengraph entries (id, token) -- that's the authoritative
    source now that there's no deterministic tokenize() to derive tokens
    from independently. Reusing it here also guarantees the token list
    handed to analyze() always agrees with the canned answer, by
    construction.

    Implied/elided tokengraph entries (tokentype in IMPLIED_TOKENTYPES --
    currently just 'implied sum'; see models.py's TokenAnalysis) are
    excluded: they were never part of the original, pre-analysis token
    list -- analyze() itself is what adds them to its OUTPUT tokengraph --
    and Token.text is required, non-None, which an implied entry's
    token=None couldn't satisfy anyway.
    """
    return [
        Token(id=entry["id"], text=entry["token"])
        for entry in canned_answer["tokengraph"]
        if entry.get("tokentype") not in IMPLIED_TOKENTYPES
    ]


def run_gold_example(example):
    """Run a GoldExample's passage through analyze(), with DummyLM standing
    in for the real LM and returning that example's canned_answer.

    Returns (tokens, result) -- the same pair analyze_passage() returns.
    """
    dspy.configure(lm=DummyLM([example.canned_answer]))
    tokens = tokens_from_canned_answer(example.canned_answer)
    result = analyze(passage=example.passage, tokens=tokens)
    return tokens, result


def run_segmentation_example(example):
    """Run a SegmentationExample's sources through segment_sources(), with
    DummyLM returning that example's canned_sentences. Returns the
    resulting list of Sentence."""
    dspy.configure(lm=DummyLM([example.canned_sentences]))
    return segment_sources(example.sources)


@pytest.fixture
def real_lm():
    """Configure dspy against the actual model from .env, for tests marked
    `live` -- e.g. `@pytest.mark.live` plus `def test_x(real_lm): ...`.
    No DummyLM-backed test should request this fixture; it's the one place
    a test actually calls out to the configured LM instead of a canned
    answer. Skips (not errors) if no API key is configured, so `pytest -m
    live` degrades gracefully rather than failing on missing credentials.
    """
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("no API_KEY in .env -- skipping live test")

    api_base = os.getenv("API_BASE")
    model = os.getenv("MODEL")
    if not model:
        pytest.skip("no MODEL in .env -- skipping live test")

    lm_kwargs = dict(model=model, api_key=api_key)
    if api_base:
        lm_kwargs["api_base"] = api_base
    dspy.configure(lm=dspy.LM(**lm_kwargs))
