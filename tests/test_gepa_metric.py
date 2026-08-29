"""
Tests for gepa_metric.py's syntax_metric() -- fully offline, no dependency
on GOLD_EXAMPLES or dspy.GEPA itself. Modeled on arsgrammatica's
test_gepa_metric.py.
"""

import dspy

from diqduq.gepa_metric import syntax_metric
from diqduq.models import TokenAnalysis, VerbalExpression


def _gold_example():
    return dspy.Example(
        passage="בָּרָא אֱלֹהִים",
        tokens=[],
        verbalunits=[VerbalExpression(id="t0", syntactic_type="independent", semantic_type="transitive active")],
        tokengraph=[
            TokenAnalysis(id="t0", token="בָּרָא", tokentype="lexical", verbalunitid="t0",
                          relatedtoken1="root", relationship1="unit verb"),
            TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical",
                          relatedtoken1="t0", relationship1="subject"),
        ],
    ).with_inputs("passage", "tokens")


def test_perfect_match_scores_one():
    gold = _gold_example()
    pred = dspy.Prediction(verbalunits=gold.verbalunits, tokengraph=gold.tokengraph)
    result = syntax_metric(gold, pred)
    assert result.score == 1.0
    assert "Perfect match" in result.feedback


def test_relation_slot_swap_still_scores_perfectly():
    """relatedtoken1/relationship1 vs. relatedtoken2/relationship2 is
    (except for 'coordinating conjunction') an overflow slot -- a
    prediction that puts the same relation in the other slot from the gold
    answer should still score as a perfect match."""
    gold = _gold_example()
    swapped_tok = TokenAnalysis(
        id="t1", token="אֱלֹהִים", tokentype="lexical",
        relatedtoken2="t0", relationship2="subject",
    )
    pred = dspy.Prediction(verbalunits=gold.verbalunits, tokengraph=[gold.tokengraph[0], swapped_tok])
    result = syntax_metric(gold, pred)
    assert result.score == 1.0


def test_missing_relation_is_penalized_and_named_in_feedback():
    gold = _gold_example()
    pred = dspy.Prediction(
        verbalunits=gold.verbalunits,
        tokengraph=[gold.tokengraph[0], TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical")],
    )
    result = syntax_metric(gold, pred)
    assert result.score < 1.0
    assert result.relation_score < 1.0
    assert "subject" in result.feedback


def test_missing_verbal_expression_is_penalized():
    gold = _gold_example()
    pred = dspy.Prediction(verbalunits=[], tokengraph=gold.tokengraph)
    result = syntax_metric(gold, pred)
    assert result.vu_score < 1.0
    assert "missing from verbalunits" in result.feedback


def test_empty_gold_relations_score_full_marks_when_prediction_also_empty():
    gold = dspy.Example(
        passage="x",
        tokens=[],
        verbalunits=[],
        tokengraph=[TokenAnalysis(id="t0", token="x", tokentype="lexical")],
    ).with_inputs("passage", "tokens")
    pred = dspy.Prediction(verbalunits=[], tokengraph=gold.tokengraph)
    result = syntax_metric(gold, pred)
    assert result.score == 1.0
