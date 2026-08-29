"""
End-to-end pipeline tests (pipeline.py's analyze_passage()/analyze_sources()/
combined_tokengraph()), driving both segmentation and analysis with
DummyLM in sequence -- modeled on arsgrammatica's test_analyze_passage.py.
"""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from diqduq import analyze_passage, combined_tokengraph, print_analysis
from diqduq.hebrew_syntax_dspy import validate
from fixtures.gold_examples import GOLD_EXAMPLES
from fixtures.segmentation_examples import SEGMENTATION_EXAMPLES

_CITATION = "urn:cts:compnov:bible.genesis.masoretic:1.1"


def _find(slug_prefix, examples):
    for ex in examples:
        if ex.slug.startswith(slug_prefix):
            return ex
    raise LookupError(slug_prefix)


def test_analyze_passage_runs_segmentation_then_analysis():
    seg_example = _find("genesis_1_1", SEGMENTATION_EXAMPLES)
    gold_example = _find("genesis_1_1", GOLD_EXAMPLES)

    dspy.configure(lm=DummyLM([seg_example.canned_sentences, gold_example.canned_answer]))
    sentences, results = analyze_passage(gold_example.passage, citation=_CITATION)

    assert len(sentences) == 1
    assert len(results) == 1
    problems = validate(sentences[0].tokens, results[0])
    assert problems == []

    tokengraph = combined_tokengraph(results)
    expected_ids = [entry["id"] for entry in gold_example.canned_answer["tokengraph"]]
    assert [t.id for t in tokengraph] == expected_ids

    # Should not raise for a well-formed result.
    print_analysis(sentences[0].tokens, results[0])


def test_analyze_passage_defaults_citation_to_empty_string():
    seg_example = _find("genesis_1_1", SEGMENTATION_EXAMPLES)
    gold_example = _find("genesis_1_1", GOLD_EXAMPLES)

    # Re-derive canned_sentences with no citation, matching the no-citation
    # default analyze_passage() uses when `citation` is omitted.
    canned = {
        "reasoning": seg_example.canned_sentences["reasoning"],
        "sentences": [
            {"tokens": [{**tok, "citation": None} for tok in sent["tokens"]]}
            for sent in seg_example.canned_sentences["sentences"]
        ],
    }
    dspy.configure(lm=DummyLM([canned, gold_example.canned_answer]))
    sentences, results = analyze_passage(gold_example.passage)

    assert sentences[0].tokens[0].citation is None


@pytest.mark.live
def test_analyze_passage_against_the_real_configured_lm(real_lm):
    sentences, results = analyze_passage(
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃"
    )
    assert len(sentences) >= 1
    for sentence, result in zip(sentences, results):
        # A live model may not get the linguistics right -- this only
        # checks the call completes and returns a well-formed shape.
        problems = validate(sentence.tokens, result)
        assert isinstance(problems, list)
