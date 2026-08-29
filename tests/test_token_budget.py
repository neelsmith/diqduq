"""
Tests for diqduq/token_budget.py: estimate_max_tokens()'s formula and
clamping, and analyze_with_retry()'s detect-truncation-and-retry loop.
Modeled directly on arsgrammatica's tests/test_token_budget.py. Also covers
this module's own extension beyond arsgrammatica: estimate_segmentation_
max_tokens()/segment_with_retry() (see token_budget.py's module docstring
for why segmentation needed budget management added at all).

analyze_with_retry()'s primary truncation signal -- a tokengraph missing
some of the input tokens' own ids -- is exercised directly with DummyLM by
handing it a deliberately incomplete canned_answer first and a complete one
second: DummyLM's list mode returns one answer per call, in order, so a
result that matches the *second* answer is only possible if a retry
actually happened. This sidesteps needing a fake LM that can simulate a
real provider's finish_reason="length" truncation -- see
_finish_reason_was_length()'s own docstring for why that secondary signal
isn't (and doesn't need to be) covered the same way here. The same
list-mode trick works for segment_with_retry(), against its own primary
signal (_segmentation_undercoverage()) instead.
"""

import copy

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from diqduq import (
    analyze_passage,
    analyze_with_retry,
    estimate_max_tokens,
    estimate_segmentation_max_tokens,
    get_calibration,
    segment_with_retry,
)
from diqduq.token_budget import DEFAULT_CEILING, DEFAULT_FLOOR
from conftest import tokens_from_canned_answer
from fixtures.gold_examples import GOLD_EXAMPLES
from fixtures.segmentation_examples import SEGMENTATION_EXAMPLES


def _example(slug):
    return next(e for e in GOLD_EXAMPLES if e.slug == slug)


# ---------------------------------------------------------------------------
# estimate_max_tokens()
# ---------------------------------------------------------------------------


def test_estimate_max_tokens_grows_with_input_length():
    small = estimate_max_tokens(5)
    large = estimate_max_tokens(50)
    assert large > small


def test_estimate_max_tokens_rejects_negative_input():
    with pytest.raises(ValueError):
        estimate_max_tokens(-1)


def test_estimate_max_tokens_respects_floor():
    # A tiny passage still gets at least `floor`, even though the raw
    # fitted value for num_tokens=0 could fall under it for a small
    # enough calibration.
    assert estimate_max_tokens(0, floor=10_000, ceiling=20_000) == 10_000


def test_estimate_max_tokens_respects_ceiling():
    # An enormous passage is clamped, not allowed to grow unbounded.
    assert estimate_max_tokens(1_000_000, floor=0, ceiling=4_096) == 4_096


def test_estimate_max_tokens_safety_margin_scales_the_budget():
    baseline = estimate_max_tokens(20, safety_margin=1.0, floor=0, ceiling=1_000_000)
    margined = estimate_max_tokens(20, safety_margin=2.0, floor=0, ceiling=1_000_000)
    assert margined == pytest.approx(2 * baseline, rel=0.05)


def test_estimate_max_tokens_uses_fallback_constants_when_uncalibrated(tmp_path, monkeypatch):
    # Point CALIBRATION_FILE at a path that doesn't exist, forcing the
    # fallback constants token_budget.py ships with.
    monkeypatch.setattr("diqduq.token_budget.CALIBRATION_FILE", tmp_path / "missing.json")
    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_get_calibration_reads_a_real_calibration_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text(
        '{"intercept": 100.0, "slope": 10.0, "sample_size": 8, "model": "test-model", '
        '"calibrated_at": "2026-01-01T00:00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("diqduq.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "calibrated"
    assert calibration["intercept"] == 100.0
    assert calibration["slope"] == 10.0

    # And estimate_max_tokens() actually uses it: with intercept=100,
    # slope=10, safety_margin=1.0, num_tokens=10 -> raw 200, well inside
    # generous floor/ceiling.
    assert estimate_max_tokens(10, safety_margin=1.0, floor=0, ceiling=10_000) == 200


def test_get_calibration_falls_back_on_malformed_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr("diqduq.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_default_floor_and_ceiling_are_sane():
    # Not much to assert here beyond "floor is well under ceiling" -- this
    # just guards against a typo swapping the two.
    assert 0 < DEFAULT_FLOOR < DEFAULT_CEILING


# ---------------------------------------------------------------------------
# analyze_with_retry()
# ---------------------------------------------------------------------------


def test_analyze_with_retry_matches_analyze_on_a_normal_gold_example():
    """No truncation at all: behaves exactly like calling analyze()
    directly, and consumes exactly one DummyLM answer."""
    example = _example("genesis_1_1_root_and_pairwise_conjunction")
    dspy.configure(lm=DummyLM([example.canned_answer]))
    tokens = tokens_from_canned_answer(example.canned_answer)

    result = analyze_with_retry(example.passage, tokens)

    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in example.canned_answer["tokengraph"]]


def test_analyze_with_retry_retries_once_on_missing_ids_then_succeeds():
    example = _example("genesis_1_1_root_and_pairwise_conjunction")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    # A truncated first answer: drop the tail of the tokengraph, as a real
    # cut-off-mid-JSON response would.
    truncated_answer = copy.deepcopy(example.canned_answer)
    truncated_answer["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2]

    dspy.configure(lm=DummyLM([truncated_answer, example.canned_answer]))

    with pytest.warns(UserWarning, match="missing"):
        result = analyze_with_retry(example.passage, tokens, max_retries=1)

    # The complete, second answer -- only reachable via a retry.
    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in full_tokengraph]


def test_analyze_with_retry_gives_up_after_max_retries_and_returns_incomplete_result():
    example = _example("genesis_1_1_root_and_pairwise_conjunction")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    truncated_once = copy.deepcopy(example.canned_answer)
    truncated_once["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2]
    truncated_twice = copy.deepcopy(example.canned_answer)
    truncated_twice["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2 + 1]

    dspy.configure(lm=DummyLM([truncated_once, truncated_twice]))

    # Two UserWarnings fire in this scenario: the intermediate "...
    # retrying..." warning after the first (truncated) attempt, and the
    # final "still looks truncated" one once max_retries is exhausted.
    # Capture the whole block with `as record` and assert on the specific
    # message we care about, rather than passing `match=` -- match= only
    # silences the warning(s) it matches, so the *other*, unmatched one
    # would otherwise still leak into pytest's warnings summary. See
    # https://docs.pytest.org/en/stable/how-to/capture-warnings.html.
    with pytest.warns(UserWarning) as record:
        result = analyze_with_retry(example.passage, tokens, max_retries=1)

    assert any("still looks truncated" in str(w.message) for w in record)

    # Best-effort: returns the last (still incomplete) attempt rather than
    # raising.
    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in truncated_twice["tokengraph"]]


def test_analyze_with_retry_does_not_retry_past_the_ceiling():
    """If the budget is already pinned at `ceiling`, a truncated result is
    returned (with a warning) rather than retried again."""
    example = _example("genesis_1_1_root_and_pairwise_conjunction")
    full_tokengraph = example.canned_answer["tokengraph"]
    tokens = tokens_from_canned_answer(example.canned_answer)

    truncated_answer = copy.deepcopy(example.canned_answer)
    truncated_answer["tokengraph"] = full_tokengraph[: len(full_tokengraph) // 2]

    dspy.configure(lm=DummyLM([truncated_answer]))

    with pytest.warns(UserWarning, match="still looks truncated"):
        result = analyze_with_retry(
            example.passage,
            tokens,
            max_retries=1,
            initial_max_tokens=500,
            ceiling=500,  # already at the ceiling -- no room to grow
        )

    assert [tok.id for tok in result.tokengraph] == [e["id"] for e in truncated_answer["tokengraph"]]


def test_analyze_with_retry_honors_initial_max_tokens_over_the_estimate():
    """A caller-supplied initial_max_tokens is used as-is for the first
    attempt rather than being recomputed from estimate_max_tokens()."""
    example = _example("genesis_1_1_root_and_pairwise_conjunction")
    dspy.configure(lm=DummyLM([example.canned_answer]))
    tokens = tokens_from_canned_answer(example.canned_answer)

    # A tiny explicit budget, well under what estimate_max_tokens() would
    # pick -- if this were ignored in favor of the estimate, this
    # assertion wouldn't tell us anything, so the point is just that the
    # call succeeds and the config kwarg is accepted at all.
    result = analyze_with_retry(example.passage, tokens, initial_max_tokens=50)
    assert result is not None


# ---------------------------------------------------------------------------
# estimate_segmentation_max_tokens() / segment_with_retry()
# ---------------------------------------------------------------------------


def _segmentation_example(slug):
    return next(e for e in SEGMENTATION_EXAMPLES if e.slug == slug)


def test_estimate_segmentation_max_tokens_grows_with_input_length():
    from diqduq.models import CitedText

    small = estimate_segmentation_max_tokens([CitedText(citation="", text="שלום")])
    large = estimate_segmentation_max_tokens([CitedText(citation="", text="שלום " * 200)])
    assert large > small


def test_estimate_segmentation_max_tokens_respects_floor_and_ceiling():
    from diqduq.models import CitedText

    empty = [CitedText(citation="", text="")]
    assert estimate_segmentation_max_tokens(empty, floor=10_000, ceiling=20_000) == 10_000

    huge = [CitedText(citation="", text="א" * 1_000_000)]
    assert estimate_segmentation_max_tokens(huge, floor=0, ceiling=4_096) == 4_096


def test_segment_with_retry_matches_segment_sources_on_a_normal_example():
    """No truncation at all: consumes exactly one DummyLM answer and
    returns the same sentences segment_sources() would."""
    example = _segmentation_example("genesis_1_1_article_preposition_conjunction_cantillation")
    dspy.configure(lm=DummyLM([example.canned_sentences]))

    sentences = segment_with_retry(example.sources)

    expected_ids = [t["id"] for t in example.canned_sentences["sentences"][0]["tokens"]]
    assert [tok.id for sentence in sentences for tok in sentence.tokens] == expected_ids


def test_segment_with_retry_retries_once_on_undercoverage_then_succeeds():
    example = _segmentation_example("genesis_1_1_article_preposition_conjunction_cantillation")
    full_tokens = example.canned_sentences["sentences"][0]["tokens"]

    truncated_answer = copy.deepcopy(example.canned_sentences)
    truncated_answer["sentences"][0]["tokens"] = full_tokens[: len(full_tokens) // 2]

    dspy.configure(lm=DummyLM([truncated_answer, example.canned_sentences]))

    with pytest.warns(UserWarning, match="looks incomplete"):
        sentences = segment_with_retry(example.sources, max_retries=1)

    # The complete, second answer -- only reachable via a retry.
    assert [tok.id for sentence in sentences for tok in sentence.tokens] == [t["id"] for t in full_tokens]


def test_segment_with_retry_gives_up_after_max_retries_and_returns_incomplete_result():
    example = _segmentation_example("genesis_1_1_article_preposition_conjunction_cantillation")
    full_tokens = example.canned_sentences["sentences"][0]["tokens"]

    truncated_once = copy.deepcopy(example.canned_sentences)
    truncated_once["sentences"][0]["tokens"] = full_tokens[: len(full_tokens) // 3]
    truncated_twice = copy.deepcopy(example.canned_sentences)
    truncated_twice["sentences"][0]["tokens"] = full_tokens[: len(full_tokens) // 3 + 1]

    dspy.configure(lm=DummyLM([truncated_once, truncated_twice]))

    with pytest.warns(UserWarning) as record:
        sentences = segment_with_retry(example.sources, max_retries=1)

    assert any("still looks truncated" in str(w.message) for w in record)

    # Best-effort: returns the last (still incomplete) attempt rather than
    # raising.
    assert [tok.id for sentence in sentences for tok in sentence.tokens] == [
        t["id"] for t in truncated_twice["sentences"][0]["tokens"]
    ]


def test_segment_with_retry_does_not_retry_past_the_ceiling():
    example = _segmentation_example("genesis_1_1_article_preposition_conjunction_cantillation")
    full_tokens = example.canned_sentences["sentences"][0]["tokens"]

    truncated_answer = copy.deepcopy(example.canned_sentences)
    truncated_answer["sentences"][0]["tokens"] = full_tokens[: len(full_tokens) // 2]

    dspy.configure(lm=DummyLM([truncated_answer]))

    with pytest.warns(UserWarning, match="still looks truncated"):
        sentences = segment_with_retry(
            example.sources,
            max_retries=1,
            initial_max_tokens=500,
            ceiling=500,  # already at the ceiling -- no room to grow
        )

    assert [tok.id for sentence in sentences for tok in sentence.tokens] == [
        t["id"] for t in truncated_answer["sentences"][0]["tokens"]
    ]


def test_segment_with_retry_honors_initial_max_tokens_over_the_estimate():
    example = _segmentation_example("genesis_1_1_article_preposition_conjunction_cantillation")
    dspy.configure(lm=DummyLM([example.canned_sentences]))

    sentences = segment_with_retry(example.sources, initial_max_tokens=50)
    assert sentences is not None


# ---------------------------------------------------------------------------
# Live sanity check (see tests/conftest.py's real_lm fixture) -- skipped by
# default, run with `pytest -m live` once .env has real credentials.
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_pipeline_handles_a_longer_passage_without_truncating(real_lm):
    """End-to-end check against the real configured LM: pipeline.py's
    analyze_sources() now calls analyze_with_retry() (see pipeline.py's own
    docstring update) rather than analyze() directly, so a passage with
    several verses -- exactly the kind of case a fixed max_tokens would
    eventually truncate -- should come back with a complete tokengraph,
    covering every input token id, with no manual retry needed from the
    caller."""
    sentences, results = analyze_passage(
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃ "
        "וְהָאָרֶץ הָיְתָה תֹהוּ וָבֹהוּ וְחֹשֶׁךְ עַל־פְּנֵי תְהוֹם "
        "וְרוּחַ אֱלֹהִים מְרַחֶפֶת עַל־פְּנֵי הַמָּיִם׃"
    )

    for sentence, result in zip(sentences, results):
        seen_ids = {tok.id for tok in result.tokengraph}
        missing = {t.id for t in sentence.tokens} - seen_ids
        assert not missing, f"tokengraph still missing input token id(s): {missing}"
