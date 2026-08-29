"""
Deliberately-broken tokengraphs, exercising hebrew_syntax_dspy.validate()'s
own checks in isolation -- unlike test_gold_examples.py, which only ever
feeds validate() something already known to be clean.
"""

import dspy

from diqduq.hebrew_syntax_dspy import validate
from diqduq.models import Token, TokenAnalysis, VerbalExpression


def _tokens():
    return [Token(id="t0", text="בָּרָא"), Token(id="t1", text="אֱלֹהִים")]


def test_clean_result_has_no_problems():
    result = dspy.Prediction(
        verbalunits=[VerbalExpression(id="t0", syntactic_type="independent", semantic_type="transitive active")],
        tokengraph=[
            TokenAnalysis(id="t0", token="בָּרָא", tokentype="lexical", verbalunitid="t0",
                           relatedtoken1="root", relationship1="unit verb"),
            TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical",
                           relatedtoken1="t0", relationship1="subject"),
        ],
    )
    assert validate(_tokens(), result) == []


def test_unknown_related_token_id_is_flagged():
    result = dspy.Prediction(
        verbalunits=[],
        tokengraph=[
            TokenAnalysis(id="t0", token="בָּרָא", tokentype="lexical",
                           relatedtoken1="does-not-exist", relationship1="subject"),
            TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical"),
        ],
    )
    problems = validate(_tokens(), result)
    assert any("does-not-exist" in p for p in problems)


def test_reserved_root_id_on_an_actual_token_is_flagged():
    tokens = [Token(id="root", text="בָּרָא")]
    result = dspy.Prediction(verbalunits=[], tokengraph=[
        TokenAnalysis(id="root", token="בָּרָא", tokentype="lexical"),
    ])
    problems = validate(tokens, result)
    assert any("reserved" in p for p in problems)


def test_implied_token_reusing_a_real_id_is_flagged():
    result = dspy.Prediction(
        verbalunits=[],
        tokengraph=[
            TokenAnalysis(id="t0", token=None, tokentype="implied sum"),  # t0 is a real id
            TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical"),
        ],
    )
    problems = validate(_tokens(), result)
    assert any("reuses an id already in the input" in p for p in problems)


def test_implied_token_with_non_none_text_is_flagged():
    result = dspy.Prediction(
        verbalunits=[],
        tokengraph=[
            TokenAnalysis(id="t0_implied", token="הוּא", tokentype="implied sum"),
            TokenAnalysis(id="t0", token="בָּרָא", tokentype="lexical"),
            TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical"),
        ],
    )
    problems = validate(_tokens(), result)
    assert any("must be left unset" in p for p in problems)


def test_non_implied_token_with_none_text_is_flagged():
    result = dspy.Prediction(
        verbalunits=[],
        tokengraph=[
            TokenAnalysis(id="t0", token=None, tokentype="lexical"),
            TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical"),
        ],
    )
    problems = validate(_tokens(), result)
    assert any("only" in p and "may omit surface text" in p for p in problems)


def test_verbal_expression_with_unknown_id_is_flagged():
    result = dspy.Prediction(
        verbalunits=[VerbalExpression(id="ghost", syntactic_type="independent", semantic_type="linking verb")],
        tokengraph=[
            TokenAnalysis(id="t0", token="בָּרָא", tokentype="lexical"),
            TokenAnalysis(id="t1", token="אֱלֹהִים", tokentype="lexical"),
        ],
    )
    problems = validate(_tokens(), result)
    assert any("ghost" in p for p in problems)
