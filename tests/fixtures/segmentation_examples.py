"""
Gold-annotated examples for the segmentation stage (segmentation_dspy.py) --
the segmentation-stage counterpart to fixtures/gold_examples.py. Modeled
directly on arsgrammatica's tests/fixtures/segmentation_examples.py.

Each SegmentationExample pairs a list of CitedText sources with the
sentences/tokens a *correctly working* segmenter should produce, per
syntax_model.md's "Tokenization" section. Like fixtures/gold_examples.py,
these are structural fixtures: running them through DummyLM proves the
code (models, segment_sources() plumbing) handles a well-formed, correct
answer properly. It does NOT prove a real LM produces that answer -- see
test_segmentation_live.py for that half of the picture.

A NOTE ON A GENUINE TENSION IN syntax_model.md: its own tokenization
walkthrough treats מִכָּל (מִן + כָּל fused, "from all") as a SINGLE lexical
token, but its "object of preposition" relation example (בְּאֶרֶץ -> אֶרֶץ
relates to בְּ) only makes sense if an inseparable preposition is its OWN
token, distinct from the noun that follows. This suite resolves that
tension in favor of always splitting an inseparable preposition from its
noun (matching gold_examples.py's own choice, for internal consistency,
and because the relation examples are the more load-bearing half of the
scheme) -- but this is a genuine first-draft inconsistency in
syntax_model.md worth resolving explicitly rather than silently, per
notes/DEVELOPMENT.md's own convention. The examples below are chosen to avoid
מִכָּל itself, sidestepping the ambiguity rather than resolving it by
example.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from diqduq.models import CitedText


@dataclass
class SegmentationExample:
    slug: str
    sources: List[CitedText]
    tags: List[str]
    canned_sentences: Dict[str, Any]


_GEN_1_1 = "urn:cts:compnov:bible.genesis.masoretic:1.1"
_GEN_1_5 = "urn:cts:compnov:bible.genesis.masoretic:1.5"

SEGMENTATION_EXAMPLES = [
    SegmentationExample(
        slug="genesis_1_1_article_preposition_conjunction_cantillation",
        sources=[
            CitedText(citation=_GEN_1_1, text="בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"),
        ],
        tags=["lexical", "proclitic conjunction", "cantillation", "article-as-own-token"],
        canned_sentences={
            "reasoning": (
                "One sentence, ending at the sof pasuq ׃ (its own "
                "cantillation token). The inseparable preposition בְּ and "
                "the article הַ are each split from the noun/adjective "
                "they attach to, matching syntax_model.md's own 'object of "
                "preposition' and 'article' relation examples. The "
                "proclitic conjunction וְ before the second אֵת is its own "
                "token, tokentype 'proclitic conjunction'."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "בְּ", "citation": _GEN_1_1},
                    {"id": "t1", "text": "רֵאשִׁ֖ית", "citation": _GEN_1_1},
                    {"id": "t2", "text": "בָּרָ֣א", "citation": _GEN_1_1},
                    {"id": "t3", "text": "אֱלֹהִ֑ים", "citation": _GEN_1_1},
                    {"id": "t4", "text": "אֵ֥ת", "citation": _GEN_1_1},
                    {"id": "t5", "text": "הַ", "citation": _GEN_1_1},
                    {"id": "t6", "text": "שָּׁמַ֖יִם", "citation": _GEN_1_1},
                    {"id": "t7", "text": "וְ", "citation": _GEN_1_1},
                    {"id": "t8", "text": "אֵ֥ת", "citation": _GEN_1_1},
                    {"id": "t9", "text": "הָ", "citation": _GEN_1_1},
                    {"id": "t10", "text": "אָֽרֶץ", "citation": _GEN_1_1},
                    {"id": "t11", "text": "׃", "citation": _GEN_1_1},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="enclitic_pronoun_bo_and_independent_pronoun_oto",
        sources=[
            CitedText(citation="ex.1", text="וַיְקַדֵּשׁ אֹתֹו כִּי בֹו שָׁבַת"),
        ],
        tags=["enclitic pronoun", "independent (non-enclitic) pronoun"],
        canned_sentences={
            "reasoning": (
                "אֹתֹו is an independent direct-object pronoun, not an "
                "enclitic form -- one lexical token. בֹו, by contrast, is "
                "the preposition בְּ plus a bound 3ms object pronoun -- two "
                "tokens, the second tokentype 'enclitic pronoun'."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "וַ", "citation": "ex.1"},
                    {"id": "t1", "text": "יְקַדֵּשׁ", "citation": "ex.1"},
                    {"id": "t2", "text": "אֹתֹו", "citation": "ex.1"},
                    {"id": "t3", "text": "כִּי", "citation": "ex.1"},
                    {"id": "t4", "text": "בְּ", "citation": "ex.1"},
                    {"id": "t5", "text": "ו", "citation": "ex.1"},
                    {"id": "t6", "text": "שָׁבַת", "citation": "ex.1"},
                ]},
            ],
        },
    ),
    SegmentationExample(
        slug="paragraph_marker_after_cantillation",
        sources=[
            CitedText(citation=_GEN_1_5, text="וַֽיְהִי־עֶ֥רֶב וַֽיְהִי־בֹ֖קֶר יֹ֥ום אֶחָֽד׃ פ"),
        ],
        tags=["maqaf", "cantillation", "paragraph"],
        canned_sentences={
            "reasoning": (
                "וַֽיְהִי and עֶרֶב/בֹקֶר are joined by maqaf (־). The verse "
                "ends with the sof pasuq (cantillation), immediately "
                "followed by a standalone פ (petuhah), tokentype "
                "'paragraph', marking a section division -- still part of "
                "this same sentence's token list, not a new sentence of "
                "its own, since it has no verbal content."
            ),
            "sentences": [
                {"tokens": [
                    {"id": "t0", "text": "וַֽיְהִי", "citation": _GEN_1_5},
                    {"id": "t1", "text": "־", "citation": _GEN_1_5},
                    {"id": "t2", "text": "עֶ֥רֶב", "citation": _GEN_1_5},
                    {"id": "t3", "text": "וַֽיְהִי", "citation": _GEN_1_5},
                    {"id": "t4", "text": "־", "citation": _GEN_1_5},
                    {"id": "t5", "text": "בֹ֖קֶר", "citation": _GEN_1_5},
                    {"id": "t6", "text": "יֹ֥ום", "citation": _GEN_1_5},
                    {"id": "t7", "text": "אֶחָֽד", "citation": _GEN_1_5},
                    {"id": "t8", "text": "׃", "citation": _GEN_1_5},
                    {"id": "t9", "text": "פ", "citation": _GEN_1_5},
                ]},
            ],
        },
    ),
]
