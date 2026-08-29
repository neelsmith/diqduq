"""
Gold-annotated examples for the analysis stage (hebrew_syntax_dspy.py) --
each one a hand-built, syntax_model.md-faithful "correct answer" for a real
passage of Biblical Hebrew. Modeled directly on arsgrammatica's
tests/fixtures/gold_examples.py, but far smaller: every entry here is drawn
straight from one of syntax_model.md's own worked examples, rather than
from a broad, independently-curated corpus -- fitting for a first-draft
scheme that is still being written alongside this test suite (see
DEVELOPMENT.md).

Running these through DummyLM (see conftest.py's run_gold_example()) proves
the CODE (models.py's pydantic models, hebrew_syntax_dspy.validate(),
verbal_units.py, rendering.py, mermaid.py, serialization.py) correctly
*represents* a correct answer. It does NOT prove a real LM produces that
answer -- see tests/test_analyze_passage.py's `live`-marked test for that
half of the picture (skipped by default; run with `pytest -m live`).

A note on tokenization in these fixtures: syntax_model.md documents
cantillation marks (the te'amim) as their own token type, but most of them
are combining diacritics riding on a specific syllable within a word.
Decomposing every single accent mark by hand for each fixture below would
add a great deal of mechanical noise without exercising anything these
particular fixtures are meant to test (the RELATION scheme, not
cantillation segmentation itself -- see segmentation_examples.py for
fixtures that DO exercise tokenization, including cantillation, in
isolation). So each lexical token's surface text below keeps its own
niqqud and any cantillation marks embedded exactly as they appear in the
source (e.g. "בָּרָ֣א"), and only the standalone *sof pasuq* (׃) at a
verse's end gets its own separate `tokentype="cantillation"` entry. This is
a deliberate simplification, flagged here per this project's own
DEVELOPMENT.md convention of flagging judgment calls rather than silently
picking one.

A second note: inseparable prepositions (בְּ/כְּ/לְ) and the article (הַ)
are written in the Masoretic text with no space or maqaf at all before the
word they attach to, but syntax_model.md's own "object of preposition" and
"article" relation examples (בְּאֶרֶץ -> אֶרֶץ relates to בְּ; הַשָּׁמַיִם ->
the article relates to the noun) only make sense if each is its own
token, distinct from the noun/adjective that follows -- so segmentation
here splits both, matching those worked examples, even though nothing
in syntax_model.md's own "Tokenization" section calls this out explicitly
as a sub-case of "lexical". A possessive or object pronoun bound as a
suffix (e.g. -ָיו, -ְך) is the one case that IS explicitly its own token
type ("enclitic pronoun").
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GoldExample:
    slug: str
    passage: str
    tags: List[str]
    canned_answer: Dict[str, Any]


# ---------------------------------------------------------------------------
# Genesis 1.1 -- root/unit verb, subject, direct object, object marker,
# adverbial (prepositional phrase modifying the verb), single-pair
# coordinating conjunction, article.
# ---------------------------------------------------------------------------
_GENESIS_1_1_ANSWER = {
    "reasoning": (
        "בָּרָא is the sentence's one finite verb, an independent clause "
        "(relatedtoken1='root', relationship1='unit verb'), transitive "
        "active. The adverbial phrase בְּרֵאשִׁית ('in the beginning') "
        "modifies בָּרָא: the preposition בְּ itself has relatedtoken1 -> "
        "בָּרָא, relationship1='adverbial', while its own object רֵאשִׁית "
        "keeps its ordinary 'object of preposition' relation to בְּ. "
        "אֱלֹהִים is the subject of בָּרָא. אֵת (twice) is the untranslated "
        "definite direct-object marker; each instance now carries its own "
        "'object marker' relation pointing at the noun it marks (the first "
        "אֵת -> שָּׁמַיִם, the second -> אָרֶץ), in addition to (not instead "
        "of) that noun's own separate 'direct object' relation to בָּרָא. "
        "הַשָּׁמַיִם and הָאָרֶץ are both objects of בָּרָא; each noun's own "
        "article (הַ) relates back to it. The single proclitic וְ before "
        "the second אֵת joins the two nouns שָּׁמַיִם and אָרֶץ as a "
        "coordinating conjunction (using both relatedtoken1/relatedtoken2 "
        "at once, per syntax_model.md)."
    ),
    "verbalunits": [
        {"id": "t2", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "בְּ", "tokentype": "lexical", "lemma": "בְּ",
         "relatedtoken1": "t2", "relationship1": "adverbial"},
        {"id": "t1", "token": "רֵאשִׁ֖ית", "tokentype": "lexical", "lemma": "רֵאשִׁית",
         "relatedtoken1": "t0", "relationship1": "object of preposition"},
        {"id": "t2", "token": "בָּרָ֣א", "tokentype": "lexical", "lemma": "ברא",
         "verbalunitid": "t2", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t3", "token": "אֱלֹהִ֑ים", "tokentype": "lexical", "lemma": "אֱלֹהִים",
         "relatedtoken1": "t2", "relationship1": "subject"},
        {"id": "t4", "token": "אֵ֥ת", "tokentype": "lexical", "lemma": "אֵת",
         "relatedtoken1": "t6", "relationship1": "object marker"},
        {"id": "t5", "token": "הַ", "tokentype": "lexical", "lemma": "הַ",
         "relatedtoken1": "t6", "relationship1": "article"},
        {"id": "t6", "token": "שָּׁמַ֖יִם", "tokentype": "lexical", "lemma": "שָׁמַיִם",
         "relatedtoken1": "t2", "relationship1": "direct object"},
        {"id": "t7", "token": "וְ", "tokentype": "proclitic conjunction", "lemma": "וְ",
         "relatedtoken1": "t6", "relationship1": "coordinating conjunction",
         "relatedtoken2": "t10", "relationship2": "coordinating conjunction"},
        {"id": "t8", "token": "אֵ֥ת", "tokentype": "lexical", "lemma": "אֵת",
         "relatedtoken1": "t10", "relationship1": "object marker"},
        {"id": "t9", "token": "הָ", "tokentype": "lexical", "lemma": "הַ",
         "relatedtoken1": "t10", "relationship1": "article"},
        {"id": "t10", "token": "אָֽרֶץ", "tokentype": "lexical", "lemma": "אֶרֶץ",
         "relatedtoken1": "t2", "relationship1": "direct object"},
        {"id": "t11", "token": "׃", "tokentype": "cantillation"},
    ],
}


# ---------------------------------------------------------------------------
# Genesis 2.3 -- repeated/chained coordinating conjunction (two independent
# root verbs), maqaf, article, adjectival.
# ---------------------------------------------------------------------------
_GENESIS_2_3_ANSWER = {
    "reasoning": (
        "Two independent verbs, יְבָרֶךְ and יְקַדֵּשׁ, are each prefixed with "
        "their own proclitic וַ and coordinated as a chain (not a single "
        "pair): the first וַ has relatedtoken1 pointing at the first "
        "verb and relatedtoken2 at the NEXT connector; the second וַ has "
        "relatedtoken1 pointing at the second verb and relatedtoken2 at "
        "the PRECEDING connector. Each verb still gets its own ordinary "
        "relatedtoken1='root' entry, unaffected by the chain. אֶת־יֹום is "
        "joined by maqaf; הַשְּׁבִיעִי (article + adjective) modifies יֹום."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "independent", "semantic_type": "transitive active"},
        {"id": "t9", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "וַ", "tokentype": "proclitic conjunction", "lemma": "וְ",
         "relatedtoken1": "t1", "relationship1": "coordinating conjunction",
         "relatedtoken2": "t8", "relationship2": "coordinating conjunction"},
        {"id": "t1", "token": "יְבָרֶךְ", "tokentype": "lexical", "lemma": "ברך",
         "verbalunitid": "t1", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t2", "token": "אֱלֹהִים", "tokentype": "lexical", "lemma": "אֱלֹהִים",
         "relatedtoken1": "t1", "relationship1": "subject"},
        {"id": "t3", "token": "אֶת", "tokentype": "lexical", "lemma": "אֵת"},
        {"id": "t4", "token": "־", "tokentype": "maqaf"},
        {"id": "t5", "token": "יֹום", "tokentype": "lexical", "lemma": "יֹום",
         "relatedtoken1": "t1", "relationship1": "direct object"},
        {"id": "t6", "token": "הַ", "tokentype": "lexical", "lemma": "הַ",
         "relatedtoken1": "t7", "relationship1": "article"},
        {"id": "t7", "token": "שְּׁבִיעִי", "tokentype": "lexical", "lemma": "שְׁבִיעִי",
         "relatedtoken1": "t5", "relationship1": "adjectival"},
        {"id": "t8", "token": "וַ", "tokentype": "proclitic conjunction", "lemma": "וְ",
         "relatedtoken1": "t9", "relationship1": "coordinating conjunction",
         "relatedtoken2": "t0", "relationship2": "coordinating conjunction"},
        {"id": "t9", "token": "יְקַדֵּשׁ", "tokentype": "lexical", "lemma": "קדשׁ",
         "verbalunitid": "t9", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t10", "token": "אֹתֹו", "tokentype": "lexical", "lemma": "אֹתֹו",
         "relatedtoken1": "t9", "relationship1": "direct object"},
    ],
}


# ---------------------------------------------------------------------------
# Genesis 1.3 -- direct quote, coordinating conjunction pairing two ROOT
# verbs, linking verb / subject of a "to be" clause, maqaf.
# ---------------------------------------------------------------------------
_GENESIS_1_3_ANSWER = {
    "reasoning": (
        "יֹּאמֶר is the outer, independent verb of saying. The quoted verb "
        "יְהִי ('let there be') is direct speech subordinate to it "
        "(relatedtoken1 -> יֹּאמֶר, relationship1='direct quote'); it is a "
        "linking/existential verb whose own 'subject' is אֹור. The second, "
        "narrative יְהִי ('there was...') reports fulfillment as its own "
        "independent root clause, coordinated with יֹּאמֶר by the proclitic "
        "וַ prefixed to it (a single-pair coordination joining two root "
        "verbs, not two nouns: relatedtoken1 -> the first joined verb "
        "יֹּאמֶר, relatedtoken2 -> the second joined verb, this יְהִי). The "
        "leading וַ on יֹּאמֶר itself has nothing to its left within this "
        "isolated passage to pair with, so it is left unrelated, the same "
        "way a sentence-initial coordinator with no visible left partner "
        "is left unrelated in arsgrammatica's parallel Latin scheme."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "independent", "semantic_type": "transitive active"},
        {"id": "t3", "syntactic_type": "direct quote", "semantic_type": "linking verb"},
        {"id": "t6", "syntactic_type": "independent", "semantic_type": "linking verb"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "וַ", "tokentype": "proclitic conjunction", "lemma": "וְ"},
        {"id": "t1", "token": "יֹּאמֶר", "tokentype": "lexical", "lemma": "אמר",
         "verbalunitid": "t1", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t2", "token": "אֱלֹהִים", "tokentype": "lexical", "lemma": "אֱלֹהִים",
         "relatedtoken1": "t1", "relationship1": "subject"},
        {"id": "t3", "token": "יְהִי", "tokentype": "lexical", "lemma": "היה",
         "verbalunitid": "t3", "relatedtoken1": "t1", "relationship1": "direct quote"},
        {"id": "t4", "token": "אֹור", "tokentype": "lexical", "lemma": "אֹור",
         "relatedtoken1": "t3", "relationship1": "subject"},
        {"id": "t5", "token": "וַ", "tokentype": "proclitic conjunction", "lemma": "וְ",
         "relatedtoken1": "t1", "relationship1": "coordinating conjunction",
         "relatedtoken2": "t6", "relationship2": "coordinating conjunction"},
        {"id": "t6", "token": "יְהִי", "tokentype": "lexical", "lemma": "היה",
         "verbalunitid": "t6", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t7", "token": "־", "tokentype": "maqaf"},
        {"id": "t8", "token": "אֹור", "tokentype": "lexical", "lemma": "אֹור",
         "relatedtoken1": "t6", "relationship1": "subject"},
    ],
}


# ---------------------------------------------------------------------------
# An elided present of "to be" (implied sum): "לֹא אֱלֹהִים הֵמָּה", "they are
# not gods" -- syntax_model.md's own worked example for this construction.
# ---------------------------------------------------------------------------
_IMPLIED_SUM_LO_ELOHIM_HEMMAH_ANSWER = {
    "reasoning": (
        "No verb is present at all -- a bare predicate construction. הֵמָּה "
        "is the subject, אֱלֹהִים the predicate noun; לֹא (the negative "
        "particle) carries no relation under the current scheme. A new "
        "implied token (tokentype 'implied sum', token=None) is added, "
        "named by appending '_implied' to the last real token's id (t2), "
        "anchoring an 'independent'/'linking verb' verbal expression."
    ),
    "verbalunits": [
        {"id": "t2_implied", "syntactic_type": "independent", "semantic_type": "linking verb"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "לֹא", "tokentype": "lexical", "lemma": "לֹא"},
        {"id": "t1", "token": "אֱלֹהִים", "tokentype": "lexical", "lemma": "אֱלֹהִים",
         "relatedtoken1": "t2_implied", "relationship1": "predicate"},
        {"id": "t2", "token": "הֵמָּה", "tokentype": "lexical", "lemma": "הֵמָּה",
         "relatedtoken1": "t2_implied", "relationship1": "subject"},
        {"id": "t2_implied", "token": None, "tokentype": "implied sum",
         "verbalunitid": "t2_implied"},
    ],
}


# ---------------------------------------------------------------------------
# "אֲחִיכֶם הַקָּטֹן", "your younger brother" (Joseph narrative, Genesis 42/44)
# -- adjectival + article, enclitic pronoun, no verb at all.
# ---------------------------------------------------------------------------
_ADJECTIVAL_ACHIKHEM_HAQQATON_ANSWER = {
    "reasoning": (
        "A bare noun phrase, no verbal expression at all. אֲחִי ('brother "
        "of') takes the enclitic possessive pronoun כֶם ('your'), which "
        "carries no relation under the current scheme (possessive-suffix "
        "relations are not yet documented). קָּטֹן ('young/small') is "
        "adjectival, relating to אֲחִי; its own article הַ relates back to "
        "it."
    ),
    "verbalunits": [],
    "tokengraph": [
        {"id": "t0", "token": "אֲחִי", "tokentype": "lexical", "lemma": "אָח"},
        {"id": "t1", "token": "כֶם", "tokentype": "enclitic pronoun", "lemma": "אַתֶּם"},
        {"id": "t2", "token": "הַ", "tokentype": "lexical", "lemma": "הַ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "קָּטֹן", "tokentype": "lexical", "lemma": "קָטָן",
         "relatedtoken1": "t0", "relationship1": "adjectival"},
    ],
}


# ---------------------------------------------------------------------------
# Genesis 3.19 -- "בְּזֵעַת אַפֶּיךָ תֹּאכַל לֶחֶם" ("by the sweat of your brow
# you will eat bread") -- construct relation, object of preposition,
# enclitic pronoun, direct object, an implicit (pro-dropped) subject.
# ---------------------------------------------------------------------------
_CONSTRUCT_BEZEAT_APPEKHA_ANSWER = {
    "reasoning": (
        "זֵעַת ('sweat of') is the object of the preposition בְּ; אַפֶּי "
        "('brow/nostrils of'), governed by זֵעַת, relates to it via "
        "'construct' -- the governing noun (זֵעַת) is separately recorded "
        "by its own function (object of preposition), per syntax_model.md's "
        "own worked example. ךָ ('your') is an enclitic possessive pronoun, "
        "unrelated under the current scheme. תֹּאכַל ('you will eat') is "
        "the sentence's one finite verb, an independent root clause with "
        "no separate subject token (the subject is carried by verb "
        "morphology alone); לֶחֶם ('bread') is its direct object."
    ),
    "verbalunits": [
        {"id": "t4", "syntactic_type": "independent", "semantic_type": "transitive active"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "בְּ", "tokentype": "lexical", "lemma": "בְּ"},
        {"id": "t1", "token": "זֵעַת", "tokentype": "lexical", "lemma": "זֵעָה",
         "relatedtoken1": "t0", "relationship1": "object of preposition"},
        {"id": "t2", "token": "אַפֶּי", "tokentype": "lexical", "lemma": "אַף",
         "relatedtoken1": "t1", "relationship1": "construct"},
        {"id": "t3", "token": "ךָ", "tokentype": "enclitic pronoun", "lemma": "אַתָּה"},
        {"id": "t4", "token": "תֹּאכַל", "tokentype": "lexical", "lemma": "אכל",
         "verbalunitid": "t4", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t5", "token": "לֶחֶם", "tokentype": "lexical", "lemma": "לֶחֶם",
         "relatedtoken1": "t4", "relationship1": "direct object"},
    ],
}


_INTRANSITIVE_VAYYAMOT_ANSWER = {
    "reasoning": (
        "A short, partly-synthetic coverage fixture (see this module's own "
        "docstring): יָּמָת ('he died') is intransitive, its subject אִישׁ "
        "governed by the article הָ. The trailing '*' is a synthetic "
        "editorial mark (e.g. a critical-edition footnote reference) -- "
        "tokentype 'editorial' -- and פ is a paragraph marker; neither "
        "carries a relation. The leading וַ has nothing to its left within "
        "this isolated fixture to pair with, so it is left unrelated."
    ),
    "verbalunits": [
        {"id": "t1", "syntactic_type": "independent", "semantic_type": "intransitive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "וַ", "tokentype": "proclitic conjunction", "lemma": "וְ"},
        {"id": "t1", "token": "יָּמָת", "tokentype": "lexical", "lemma": "מות",
         "verbalunitid": "t1", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t2", "token": "הָ", "tokentype": "lexical", "lemma": "הַ",
         "relatedtoken1": "t3", "relationship1": "article"},
        {"id": "t3", "token": "אִישׁ", "tokentype": "lexical", "lemma": "אִישׁ",
         "relatedtoken1": "t1", "relationship1": "subject"},
        {"id": "t4", "token": "*", "tokentype": "editorial"},
        {"id": "t5", "token": "׃", "tokentype": "cantillation"},
        {"id": "t6", "token": "פ", "tokentype": "paragraph"},
    ],
}


_TRANSITIVE_PASSIVE_NIVREU_ANSWER = {
    "reasoning": (
        "A short coverage fixture for the 'transitive passive' semantic "
        "type (not otherwise exercised above): נִבְרְאוּ ('were created') is "
        "a Niphal passive verb, an independent root clause, with "
        "הַשָּׁמַיִם as its subject."
    ),
    "verbalunits": [
        {"id": "t0", "syntactic_type": "independent", "semantic_type": "transitive passive"},
    ],
    "tokengraph": [
        {"id": "t0", "token": "נִבְרְאוּ", "tokentype": "lexical", "lemma": "ברא",
         "verbalunitid": "t0", "relatedtoken1": "root", "relationship1": "unit verb"},
        {"id": "t1", "token": "הַ", "tokentype": "lexical", "lemma": "הַ",
         "relatedtoken1": "t2", "relationship1": "article"},
        {"id": "t2", "token": "שָּׁמַיִם", "tokentype": "lexical", "lemma": "שָׁמַיִם",
         "relatedtoken1": "t0", "relationship1": "subject"},
    ],
}


GOLD_EXAMPLES: List[GoldExample] = [
    GoldExample(
        slug="genesis_1_1_root_and_pairwise_conjunction",
        passage="בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃",
        tags=["unit verb", "root", "subject", "direct object", "article",
              "coordinating conjunction (single pair)", "object of preposition",
              "object marker", "adverbial"],
        canned_answer=_GENESIS_1_1_ANSWER,
    ),
    GoldExample(
        slug="genesis_2_3_chained_coordinating_conjunction",
        passage="וַיְבָרֶךְ אֱלֹהִים אֶת־יֹום הַשְּׁבִיעִי וַיְקַדֵּשׁ אֹתֹו",
        tags=["coordinating conjunction (chained)", "maqaf", "article", "adjectival",
              "direct object", "unit verb", "root"],
        canned_answer=_GENESIS_2_3_ANSWER,
    ),
    GoldExample(
        slug="genesis_1_3_direct_quote_and_verb_coordination",
        passage="וַיֹּאמֶר אֱלֹהִים יְהִי אֹור וַיְהִי־אֹור",
        tags=["direct quote", "coordinating conjunction (single pair, verbs)",
              "linking verb", "maqaf", "subject"],
        canned_answer=_GENESIS_1_3_ANSWER,
    ),
    GoldExample(
        slug="implied_sum_lo_elohim_hemmah",
        passage="לֹא אֱלֹהִים הֵמָּה",
        tags=["implied sum", "linking verb", "predicate", "subject"],
        canned_answer=_IMPLIED_SUM_LO_ELOHIM_HEMMAH_ANSWER,
    ),
    GoldExample(
        slug="adjectival_achikhem_haqqaton",
        passage="אֲחִיכֶם הַקָּטֹן",
        tags=["adjectival", "article", "enclitic pronoun", "no verbal expression"],
        canned_answer=_ADJECTIVAL_ACHIKHEM_HAQQATON_ANSWER,
    ),
    GoldExample(
        slug="construct_bezeat_appekha",
        passage="בְּזֵעַת אַפֶּיךָ תֹּאכַל לֶחֶם",
        tags=["construct", "object of preposition", "enclitic pronoun",
              "direct object", "unit verb", "root"],
        canned_answer=_CONSTRUCT_BEZEAT_APPEKHA_ANSWER,
    ),
    GoldExample(
        slug="coverage_intransitive_paragraph_editorial",
        passage="וַיָּמָת הָאִישׁ*׃ פ",
        tags=["intransitive", "editorial", "paragraph", "article", "subject",
              "coverage fixture (partly synthetic)"],
        canned_answer=_INTRANSITIVE_VAYYAMOT_ANSWER,
    ),
    GoldExample(
        slug="coverage_transitive_passive",
        passage="נִבְרְאוּ הַשָּׁמַיִם",
        tags=["transitive passive", "article", "subject", "coverage fixture"],
        canned_answer=_TRANSITIVE_PASSIVE_NIVREU_ANSWER,
    ),
]
