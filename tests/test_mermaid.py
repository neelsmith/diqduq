"""
Tests for mermaid.py -- modeled on arsgrammatica's test_mermaid_coloring.py/
test_mermaid_ranking.py, condensed into one file since diqduq's scheme is
considerably smaller.
"""

from conftest import run_gold_example
from diqduq.mermaid import tokengraph_to_mermaid
from fixtures.gold_examples import GOLD_EXAMPLES


def _tokengraph_for(slug_prefix):
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith(slug_prefix))
    _tokens, result = run_gold_example(example)
    return result.tokengraph


def test_diagram_has_a_node_per_substantive_token_and_no_editorial_nodes():
    tokengraph = _tokengraph_for("genesis_1_1")
    diagram, warnings = tokengraph_to_mermaid(tokengraph)
    assert diagram.startswith("graph BT")
    # The cantillation token (sof pasuq) should not become a node.
    assert 't11["' not in diagram
    assert 't0["' in diagram  # the preposition בְּ IS a node
    assert warnings == []


def test_root_relation_produces_no_edge_but_no_warning():
    tokengraph = _tokengraph_for("genesis_1_1")
    diagram, warnings = tokengraph_to_mermaid(tokengraph)
    assert "-->|unit verb| root" not in diagram
    assert warnings == []


def test_implied_token_gets_its_own_shape_and_color_class():
    tokengraph = _tokengraph_for("implied_sum")
    diagram, warnings = tokengraph_to_mermaid(tokengraph)
    # Implied tokens render as rounded-rectangle nodes with the dedicated
    # "implied" classDef, keyed by label "elided sum" rather than raw text.
    assert '"elided sum"' in diagram
    assert "classDef implied" in diagram
    assert "class t2_implied implied;" in diagram


def test_rank_by_depth_chains_the_two_coordinated_root_verbs():
    tokengraph = _tokengraph_for("genesis_2_3")
    diagram, warnings = tokengraph_to_mermaid(tokengraph, rank_by_depth=True)
    assert "t1 ~~~ t9" in diagram or "t9 ~~~ t1" in diagram


def test_color_by_verbal_unit_false_skips_colors():
    tokengraph = _tokengraph_for("genesis_1_1")
    diagram, _warnings = tokengraph_to_mermaid(tokengraph, color_by_verbal_unit=False)
    assert "classDef" not in diagram
