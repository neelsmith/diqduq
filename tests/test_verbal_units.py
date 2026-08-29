"""
Tests for verbal_units.py -- assign_verbal_units(), compute_subordination_
depths(), and find_unanchored_coordinated_verbs() -- modeled on
arsgrammatica's test_verbal_units.py.
"""

from conftest import run_gold_example
from diqduq.models import TokenAnalysis
from diqduq.verbal_units import (
    assign_verbal_units,
    compute_subordination_depths,
    find_unanchored_coordinated_verbs,
    max_subordination_depth,
)
from fixtures.gold_examples import GOLD_EXAMPLES


def _tokengraph_for(slug_prefix):
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith(slug_prefix))
    _tokens, result = run_gold_example(example)
    return result.tokengraph


def test_genesis_1_1_every_token_resolves_to_the_one_verbal_unit_or_none():
    tokengraph = _tokengraph_for("genesis_1_1")
    assignment = assign_verbal_units(tokengraph)
    # אֵת (the direct-object marker, twice) has no relation and resolves to
    # None; every other lexical/proclitic token resolves to t2 (בָּרָא).
    assert assignment["t2"] == "t2"
    assert assignment["t3"] == "t2"  # אֱלֹהִים, subject
    assert assignment["t6"] == "t2"  # שָּׁמַיִם, direct object
    assert assignment["t10"] == "t2"  # אָרֶץ, direct object (via the article)
    assert assignment["t4"] is None  # אֵת, unrelated


def test_genesis_2_3_chained_conjunction_resolves_each_verb_to_itself():
    tokengraph = _tokengraph_for("genesis_2_3")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t1"] == "t1"
    assert assignment["t9"] == "t9"
    # Each proclitic conjunction resolves to the verb it introduces.
    assert assignment["t0"] == "t1"
    assert assignment["t8"] == "t9"


def test_genesis_1_3_direct_quote_is_its_own_unit_at_depth_1():
    tokengraph = _tokengraph_for("genesis_1_3")
    depths, warnings = compute_subordination_depths(tokengraph)
    assert warnings == []
    assert depths["t1"] == 0  # יֹּאמֶר, independent
    assert depths["t3"] == 1  # יְהִי, direct quote subordinate to יֹּאמֶר
    assert depths["t6"] == 0  # second יְהִי, independent, coordinated with יֹּאמֶר
    assert max_subordination_depth(tokengraph) == 1


def test_implied_sum_token_anchors_its_own_unit():
    tokengraph = _tokengraph_for("implied_sum")
    assignment = assign_verbal_units(tokengraph)
    assert assignment["t2_implied"] == "t2_implied"
    assert assignment["t1"] == "t2_implied"  # אֱלֹהִים, predicate
    assert assignment["t2"] == "t2_implied"  # הֵמָּה, subject


def test_no_verbal_expressions_at_all_gives_an_all_none_assignment():
    tokengraph = _tokengraph_for("adjectival_achikhem")
    assignment = assign_verbal_units(tokengraph)
    assert set(assignment.values()) <= {None}
    depths, warnings = compute_subordination_depths(tokengraph)
    assert depths == {}
    assert max_subordination_depth(tokengraph) is None


def test_find_unanchored_coordinated_verbs_flags_a_lopsided_pair():
    tokengraph = [
        TokenAnalysis(id="t0", token="וַ", tokentype="proclitic conjunction",
                      relatedtoken1="t1", relationship1="coordinating conjunction",
                      relatedtoken2="t2", relationship2="coordinating conjunction"),
        TokenAnalysis(id="t1", token="בָּרָא", tokentype="lexical",
                      verbalunitid="t1", relatedtoken1="root", relationship1="unit verb"),
        # t2 is coordinated with t1 but is missing its own verbalunitid/root
        # entry -- exactly the asymmetry this heuristic looks for.
        TokenAnalysis(id="t2", token="קָדַשׁ", tokentype="lexical"),
    ]
    warnings = find_unanchored_coordinated_verbs(tokengraph)
    assert len(warnings) == 1
    assert "t2" in warnings[0]


def test_find_unanchored_coordinated_verbs_is_clean_for_two_ordinary_nouns():
    tokengraph = [
        TokenAnalysis(id="t0", token="וְ", tokentype="proclitic conjunction",
                      relatedtoken1="t1", relationship1="coordinating conjunction",
                      relatedtoken2="t2", relationship2="coordinating conjunction"),
        TokenAnalysis(id="t1", token="שָׁמַיִם", tokentype="lexical"),
        TokenAnalysis(id="t2", token="אֶרֶץ", tokentype="lexical"),
    ]
    assert find_unanchored_coordinated_verbs(tokengraph) == []


def test_find_unanchored_coordinated_verbs_ignores_a_chained_series():
    # The chained-connector pattern (genesis_2_3) should never trip this
    # heuristic, since relatedtoken2 there points at a fellow connector,
    # not a second conjunct.
    tokengraph = _tokengraph_for("genesis_2_3")
    assert find_unanchored_coordinated_verbs(tokengraph) == []
