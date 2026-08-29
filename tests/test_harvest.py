"""
Tests for fixtures/harvest.py -- modeled on arsgrammatica's
tests/test_harvest.py (condensed).
"""

from conftest import run_gold_example
from diqduq.models import Sentence, Token
from fixtures.gold_examples import GOLD_EXAMPLES
from fixtures.harvest import format_gold_example_source, gold_example_from_analysis


def test_gold_example_from_analysis_round_trips_a_clean_result():
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("genesis_1_1"))
    tokens, result = run_gold_example(example)
    sentence = Sentence(tokens=[Token(id=t.id, text=t.text) for t in tokens])

    harvested = gold_example_from_analysis(
        slug="harvested_genesis_1_1",
        tags=["harvested"],
        sentences=[sentence],
        verbalunits=result.verbalunits,
        tokengraph=result.tokengraph,
        reasoning="test reasoning",
    )
    assert harvested.slug == "harvested_genesis_1_1"
    assert harvested.canned_answer["reasoning"] == "test reasoning"
    assert [t["id"] for t in harvested.canned_answer["tokengraph"]] == [t.id for t in result.tokengraph]
    # Should reconstruct the original surface text via tokengraph_to_text().
    assert harvested.passage == example.passage


def test_gold_example_from_analysis_rejects_a_broken_analysis():
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("genesis_1_1"))
    tokens, result = run_gold_example(example)
    sentence = Sentence(tokens=[Token(id=t.id, text=t.text) for t in tokens])

    broken_tokengraph = list(result.tokengraph)
    broken_tokengraph[0] = broken_tokengraph[0].model_copy(update={"relatedtoken1": "does-not-exist"})

    try:
        gold_example_from_analysis(
            slug="broken",
            tags=[],
            sentences=[sentence],
            verbalunits=result.verbalunits,
            tokengraph=broken_tokengraph,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a referentially broken analysis")


def test_format_gold_example_source_is_valid_python():
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("genesis_1_1"))
    source = format_gold_example_source(example, "_TEST_ANSWER")
    namespace = {"GoldExample": type(example)}
    exec(source, namespace)  # noqa: S102 -- exercising our own generated source
    assert namespace["_TEST_ANSWER"]["tokengraph"][0]["id"] == "t0"
