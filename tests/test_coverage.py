"""
Coverage check, modeled on arsgrammatica's tests/test_coverage.py: every
documented RelationLabel value, every TokenAnalysis.tokentype value
(including IMPLIED_TOKENTYPES), and every VerbalExpression syntactic_type/
semantic_type value should be exercised by at least one entry in
GOLD_EXAMPLES. This keeps the gold corpus honest as syntax_model.md grows
-- a newly documented relation/type with no fixture at all is a gap this
test will catch immediately.
"""

from typing import get_args

from diqduq.models import (
    IMPLIED_TOKENTYPES,
    RelationLabel,
    TokenAnalysis,
    VerbalExpression,
)
from fixtures.gold_examples import GOLD_EXAMPLES


def _all_relationship_values():
    values = set()
    for example in GOLD_EXAMPLES:
        for entry in example.canned_answer["tokengraph"]:
            for field in ("relationship1", "relationship2"):
                if field in entry:
                    values.add(entry[field])
    return values


def _all_tokentype_values():
    return {
        entry["tokentype"]
        for example in GOLD_EXAMPLES
        for entry in example.canned_answer["tokengraph"]
    }


def _all_syntactic_types():
    return {
        vu["syntactic_type"]
        for example in GOLD_EXAMPLES
        for vu in example.canned_answer["verbalunits"]
    }


def _all_semantic_types():
    return {
        vu["semantic_type"]
        for example in GOLD_EXAMPLES
        for vu in example.canned_answer["verbalunits"]
    }


def test_every_relation_label_is_exercised():
    documented = set(get_args(RelationLabel))
    exercised = _all_relationship_values()
    missing = documented - exercised
    assert not missing, f"RelationLabel value(s) with no gold example: {sorted(missing)}"


def test_every_tokentype_is_exercised():
    tokentype_field = TokenAnalysis.model_fields["tokentype"].annotation
    documented = set(get_args(tokentype_field))
    exercised = _all_tokentype_values()
    missing = documented - exercised
    assert not missing, f"tokentype value(s) with no gold example: {sorted(missing)}"


def test_implied_tokentypes_is_a_subset_of_documented_tokentypes():
    tokentype_field = TokenAnalysis.model_fields["tokentype"].annotation
    documented = set(get_args(tokentype_field))
    assert IMPLIED_TOKENTYPES <= documented


def test_every_syntactic_type_is_exercised():
    documented = set(get_args(VerbalExpression.model_fields["syntactic_type"].annotation))
    exercised = _all_syntactic_types()
    missing = documented - exercised
    assert not missing, f"syntactic_type value(s) with no gold example: {sorted(missing)}"


def test_every_semantic_type_is_exercised():
    documented = set(get_args(VerbalExpression.model_fields["semantic_type"].annotation))
    exercised = _all_semantic_types()
    missing = documented - exercised
    assert not missing, f"semantic_type value(s) with no gold example: {sorted(missing)}"
