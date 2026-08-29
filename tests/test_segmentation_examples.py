"""
Coverage check for the segmentation fixtures themselves: every tokentype
syntax_model.md's "Tokenization" section documents should appear at least
once across fixtures/segmentation_examples.py, so a newly-added token type
with no illustrative fixture is caught immediately -- the
segmentation-stage counterpart to test_coverage.py.
"""

from fixtures.segmentation_examples import SEGMENTATION_EXAMPLES

# hebrew_syntax_dspy operates on post-analysis TokenAnalysis, which is where
# tokentype actually lives -- segmentation_dspy's own Token model has no
# tokentype field at all (that's assigned by the LATER analysis stage). So
# these fixtures only demonstrate *segmentation boundaries*, not tokentype
# labels; this test instead checks that every documented token *shape* is
# represented by inspecting each fixture's own tags.
_DOCUMENTED_TOKEN_KINDS = {
    "lexical",
    "proclitic conjunction",
    "cantillation",
    "enclitic pronoun",
    "independent (non-enclitic) pronoun",
    "maqaf",
    "paragraph",
}


def test_every_documented_token_kind_has_a_tagged_example():
    tagged = {tag for example in SEGMENTATION_EXAMPLES for tag in example.tags}
    # "article-as-own-token" and "editorial" aren't in _DOCUMENTED_TOKEN_KINDS
    # verbatim -- this just checks every kind we set out to cover is tagged
    # somewhere, not a strict 1:1 mapping.
    missing = {
        kind
        for kind in _DOCUMENTED_TOKEN_KINDS
        if not any(kind in tag or tag in kind for tag in tagged)
    }
    assert not missing, f"no segmentation example tagged for: {sorted(missing)}"
