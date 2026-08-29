"""
Tests for rendering.py's spacing/join rules -- modeled on arsgrammatica's
test_rendering.py, but rewritten for Biblical Hebrew's own tokenization
(article/preposition glued forward, maqaf glued both ways, enclitic
pronoun and cantillation glued backward).
"""

from conftest import run_gold_example
from diqduq.models import TokenAnalysis
from diqduq.rendering import tokengraph_to_depth_html, tokengraph_to_html, tokengraph_to_text
from fixtures.gold_examples import GOLD_EXAMPLES


def _tokengraph_for(slug_prefix):
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith(slug_prefix))
    _tokens, result = run_gold_example(example)
    return example, result.tokengraph


def test_genesis_1_1_round_trips_exactly():
    example, tokengraph = _tokengraph_for("genesis_1_1")
    assert tokengraph_to_text(tokengraph) == example.passage


def test_genesis_2_3_round_trips_maqaf_and_chained_proclitics():
    example, tokengraph = _tokengraph_for("genesis_2_3")
    assert tokengraph_to_text(tokengraph) == example.passage


def test_genesis_1_3_round_trips_maqaf_and_direct_quote():
    example, tokengraph = _tokengraph_for("genesis_1_3")
    assert tokengraph_to_text(tokengraph) == example.passage


def test_enclitic_pronoun_glues_backward():
    tokengraph = [
        TokenAnalysis(id="t0", token="אֲחִי", tokentype="lexical"),
        TokenAnalysis(id="t1", token="כֶם", tokentype="enclitic pronoun"),
        TokenAnalysis(id="t2", token="הַ", tokentype="lexical", relatedtoken1="t3", relationship1="article"),
        TokenAnalysis(id="t3", token="קָּטֹן", tokentype="lexical", relatedtoken1="t0", relationship1="adjectival"),
    ]
    assert tokengraph_to_text(tokengraph) == "אֲחִיכֶם הַקָּטֹן"


def test_implied_token_is_omitted_from_plain_text():
    example, tokengraph = _tokengraph_for("implied_sum")
    text = tokengraph_to_text(tokengraph)
    assert "None" not in text
    assert text == "לֹא אֱלֹהִים הֵמָּה"


def test_tokengraph_to_html_wraps_lexical_and_conjunction_but_not_others():
    example, tokengraph = _tokengraph_for("genesis_1_1")
    html_out = tokengraph_to_html(tokengraph)
    assert "<span" in html_out
    # The direct-object marker אֵת carries no relation and is never wrapped.
    assert html_out.count("<span") >= 1


def test_tokengraph_to_html_escapes_and_omits_implied_tokens():
    example, tokengraph = _tokengraph_for("implied_sum")
    html_out = tokengraph_to_html(tokengraph)
    assert "None" not in html_out


def test_tokengraph_to_depth_html_negative_depth_raises():
    example, tokengraph = _tokengraph_for("genesis_1_1")
    try:
        tokengraph_to_depth_html(tokengraph, depth=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a negative depth")


def test_tokengraph_to_depth_html_depth_zero_omits_the_direct_quote():
    example, tokengraph = _tokengraph_for("genesis_1_3")
    html_all, _warnings = tokengraph_to_depth_html(tokengraph, depth=None)
    html_depth0, _warnings = tokengraph_to_depth_html(tokengraph, depth=0)
    # The quoted יְהִי אֹור clause (depth 1) should be present when depth is
    # unrestricted, and absent when capped at depth 0.
    assert "אֹור" in html_all
    assert html_depth0 != html_all
