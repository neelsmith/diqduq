"""
Basic sanity checks on models.py -- pydantic validation of the documented
Literal values, and the IMPLIED_TOKENTYPES/NON_SUBSTANTIVE_TOKENTYPES
constants.
"""

import pytest
from pydantic import ValidationError

from diqduq.models import (
    IMPLIED_TOKENTYPES,
    NON_SUBSTANTIVE_TOKENTYPES,
    TokenAnalysis,
    VerbalExpression,
)


def test_implied_tokentypes_is_a_frozenset_of_one():
    assert IMPLIED_TOKENTYPES == frozenset({"implied sum"})


def test_non_substantive_tokentypes_excludes_implied_and_word_bearing_types():
    assert NON_SUBSTANTIVE_TOKENTYPES.isdisjoint(IMPLIED_TOKENTYPES)
    assert "lexical" not in NON_SUBSTANTIVE_TOKENTYPES
    assert "enclitic pronoun" not in NON_SUBSTANTIVE_TOKENTYPES
    assert "proclitic conjunction" not in NON_SUBSTANTIVE_TOKENTYPES


def test_verbal_expression_rejects_an_undocumented_syntactic_type():
    with pytest.raises(ValidationError):
        VerbalExpression(id="t0", syntactic_type="dependent", semantic_type="linking verb")


def test_token_analysis_rejects_an_undocumented_relationship_label():
    with pytest.raises(ValidationError):
        TokenAnalysis(id="t0", token="x", tokentype="lexical", relationship1="ablative absolute")


def test_token_analysis_allows_an_implied_token_with_no_surface_text():
    tok = TokenAnalysis(id="t0_implied", token=None, tokentype="implied sum")
    assert tok.token is None
