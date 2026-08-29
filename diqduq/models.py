"""
Pydantic models describing the two structures from syntax_model.md:

1. A list of VerbalExpression entries (the "table of verbal expressions").
2. A list of TokenAnalysis entries (the "token-level table of dependencies").

These need to be real pydantic BaseModel subclasses (not plain classes with
bare `=` assignments) for DSPy to generate and validate structured output
against them when used inside `List[...]` input/output fields.

`diqduq` is modeled closely on `arsgrammatica` (the same author's analyzer
for Latin syntax, https://github.com/neelsmith/arsgrammatica) -- the module
layout, the implied-token convention, and the validate()/rendering/
serialization architecture are all deliberately parallel. The *content* of
the scheme is different, and considerably less developed: syntax_model.md
is explicitly a "first draft" for Biblical Hebrew, with far fewer
documented relations and verbal-expression categories than the mature Latin
scheme, and its own "TBA" section names several constructions (the
functions of prepositions beyond "object of preposition", subordinating
conjunctions, and the relative pronoun אֲשֶׁר) that are not yet part of the
scheme at all. Where syntax_model.md is silent, this module stays silent
too rather than inventing an answer -- see "Extending the scheme" in
USAGE.md for the intended workflow when a real passage needs one of these.

syntax_model.md's own "TBA" section originally also listed the direct
object marker אֵת and the functions of prepositions beyond "object of
preposition" as not yet covered; both have since been added to the scheme
(RelationLabel's "object marker" and "adverbial" values below) -- only
subordinating conjunctions and the relative pronoun אֲשֶׁר remain listed as
"TBA" as of this writing.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CitedText(BaseModel):
    """One citable unit of source text -- e.g. one verse -- paired with its
    citation label. A sequence of these is segmentation_dspy.py's input:
    sentence boundaries do NOT need to respect CitedText boundaries (one
    sentence may span several units), but every resulting token still
    records which unit it came from via Token.citation."""

    citation: str = Field(description="Citation label for this unit, e.g. 'urn:cts:compnov:bible.genesis.masoretic:1.1'.")
    text: str = Field(description="This unit's raw text, exactly as written (including niqqud and cantillation).")


class Token(BaseModel):
    """A single pre-segmented token with a stable id.

    `citation` is optional so this model still works for citation-free
    callers -- e.g. a test fixture built directly from a canned tokengraph,
    with no CitedText source at all -- as well as for the citation-aware
    segmentation stage (segmentation_dspy.py), which is the only thing that
    actually populates it, knowing which CitedText source unit each token
    came from."""

    id: str = Field(description="Stable token id, globally unique and sequential across the whole input, e.g. 't0', 't1', ...")
    text: str = Field(description="The token's surface text, exactly as it appears in the source.")
    citation: Optional[str] = Field(
        default=None,
        description="Citation label of the source unit this token came from (e.g. 'urn:cts:compnov:bible.genesis.masoretic:1.1'), if known.",
    )


class Sentence(BaseModel):
    """One sentence's worth of tokens, in reading order, as produced by the
    LLM-driven segmentation stage (segmentation_dspy.py). Token ids are
    global across the whole passage -- numbering continues across sentence
    boundaries rather than restarting at t0 for each sentence -- so a
    Sentence is a contiguous slice of the passage's id sequence, not an
    independently-numbered unit."""

    tokens: List[Token] = Field(
        description="This sentence's tokens, in reading order, using the passage's global token ids."
    )


class VerbalExpression(BaseModel):
    """One entry in the table of verbal expressions (syntax_model.md,
    'Table of verbal expressions'). Two constructions count as a verbal
    expression, in this first draft of the scheme:

    1. Every finite verb.
    2. Every participle.

    Each verbal expression is classified along two independent axes:

    - `syntactic_type`: 'independent' (also called "main" or "principal" --
      a syntactically independent finite verb, whose clause is coherent by
      itself) or 'direct quote' (a verbal expression occurring in directly
      quoted speech -- see RelationLabel's 'direct quote' value for how it
      relates back to the verb that introduces the quotation). These are
      the only two values syntax_model.md currently documents; unlike
      arsgrammatica's mature Latin scheme, there is no 'dependent' category
      yet (subordinating conjunctions are listed under syntax_model.md's
      'TBA' section) and no separate category for an "aside" or an
      "indirect statement" -- Biblical Hebrew narrative marks reported
      speech directly (see 'direct quote' below) rather than through an
      accusative-and-infinitive construction the way Latin does.
    - `semantic_type`: 'transitive active', 'transitive passive',
      'intransitive', or 'linking verb'.

    A participle's own syntactic_type isn't pinned down by syntax_model.md
    beyond the two values above -- this codebase's convention (flagged
    here, matching the style of arsgrammatica's own documented judgment
    calls, since syntax_model.md itself doesn't say) is to classify a
    participle 'independent' unless it occurs within quoted speech, in
    which case 'direct quote' applies exactly as it would to a finite verb.
    Extend syntax_model.md first if a real passage needs a genuinely
    different category for a participial clause (e.g. something
    circumstantial/subordinate, as arsgrammatica's Latin scheme has for a
    circumstantial participle) -- see USAGE.md's "Extending the scheme".

    `diqduq` also recognizes one construction as understood or implied
    even though it has no surface realization: an elided present of "to
    be" (see IMPLIED_TOKENTYPES's 'implied sum' below, and TokenAnalysis's
    own docstring for the new-token convention this requires)."""

    id: str = Field(
        description=(
            "The token id (from the input `tokens` list) of the finite "
            "verb or participle that anchors this verbal expression. For "
            "an elided present of 'to be' (see TokenAnalysis's 'implied "
            "sum' tokentype, IMPLIED_TOKENTYPES), use the new implied "
            "token's id instead."
        )
    )
    syntactic_type: Literal["independent", "direct quote"] = Field(
        description=(
            "'independent' (main/principal -- a syntactically independent "
            "finite verb or participle) or 'direct quote' (occurring in "
            "directly quoted speech; see RelationLabel's 'direct quote' "
            "value). syntax_model.md documents no other values yet -- see "
            "this model's own docstring."
        )
    )
    semantic_type: Literal[
        "transitive active", "transitive passive", "intransitive", "linking verb"
    ] = Field(description="The verb's semantic/voice type.")


# The relation labels documented in syntax_model.md ("Syntactic relations
# among tokens"). Keep relationship1 and relationship2 restricted to the
# same set of labels. relation2/relationship2 is an overflow slot for a
# token that needs to record two relations at once, used only when
# relation1/relationship1 is already occupied by something else -- EXCEPT
# for "coordinating conjunction", which genuinely uses both relation1 and
# relation2 for the two sides of what it joins (see that value's own note
# below), the one relation syntax_model.md documents this way.
#
# This is a considerably smaller set than arsgrammatica's mature Latin
# RelationLabel: syntax_model.md itself still says two things are "TBA"
# (subordinating conjunctions and the relative pronoun אֲשֶׁר) -- don't
# invent labels for either of these. If a real passage needs one, extend
# syntax_model.md first, then add the label here, following "Extending the
# scheme" in USAGE.md.
#
# - "unit verb": every INDEPENDENT verb's own relation1 is the special
#   sentinel value 'root' (never an actual token id -- no real token may
#   be assigned the id 'root'), relationship1 = 'unit verb'. Example: in
#   בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ, the independent verb
#   בָּרָא has relation1 = 'root', relationship1 = 'unit verb'.
# - "direct quote": a verb (or participle) of directly quoted speech has
#   relation1 -> the id of the verb of the governing verbal expression (the
#   verb of saying that frames the quotation), relationship1 = 'direct
#   quote' -- matching its own syntactic_type, the same convention
#   VerbalExpression uses. Example: in וַיֹּאמֶר אֱלֹהִים יְהִי אֹור וַיְהִי־אֹור,
#   the verbal unit anchored at יְהִי is direct speech subordinate to
#   יֹּאמֶר: יְהִי has relation1 -> יֹּאמֶר's id, relationship1 = 'direct quote'.
# - "subject": a noun or pronoun serving as the subject of a verbal
#   expression has relation1 -> the id of the verb (or predicate-linking
#   implied-sum token), relationship1 = 'subject'.
# - "direct object": a noun or pronoun functioning as the direct object of
#   a verbal expression has relation1 -> the id of the verb, relationship1
#   = 'direct object'.
# - "object marker": the direct object marker אֵת itself has relation1 ->
#   the id of the direct object it marks, relationship1 = 'object marker'.
#   The marked noun keeps its OWN separate 'direct object' relation to the
#   verb -- this is an additional entry on the marker token, not a
#   replacement for that one. Example: in בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם
#   וְאֵת הָאָרֶץ, the first אֵת has relation1 -> הַשָּׁמַיִם's own noun's id
#   (שָּׁמַיִם), the second -> הָאָרֶץ's own noun's id (אָרֶץ), both relationship1
#   'object marker'.
# - "predicate": a noun or pronoun functioning as the predicate complement
#   of a LINKING verb (including an elided-sum implied token) has relation1
#   -> the id of that verb, relationship1 = 'predicate'.
# - "coordinating conjunction": when a coordinating conjunction (proclitic
#   וְ) joins a SINGLE pair of adjectives, nouns, prepositional phrases, or
#   verbs, it has relation1 -> the id of the first joined token, relation2
#   -> the id of the second, with BOTH relationship1 and relationship2 =
#   'coordinating conjunction' -- the one relation that genuinely uses both
#   slots for the two ends of the same relation at once, not an overflow
#   slot. Example: in בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ, the
#   conjunction וְ (before הָאָרֶץ) has relation1 -> הַשָּׁמַיִם's id, relation2
#   -> הָאָרֶץ's id, both relationship 'coordinating conjunction'.
#
#   The proclitic conjunction וְ can also be REPEATED, prefixed onto every
#   one of a series of two or more coordinated items rather than appearing
#   just once between a pair -- syntax_model.md annotates this differently
#   from the single-connector case above. Every connector's own relation1
#   -> the id of the item it immediately introduces (never another
#   connector), relationship1 = 'coordinating conjunction', same as above.
#   relation2 is what differs: the FIRST connector's relation2 -> the id
#   of the NEXT connector, while every connector AFTER the first has
#   relation2 -> the id of the PRECEDING connector instead (not the
#   following one); relationship2 = 'coordinating conjunction' for all of
#   them. Example: in וַיְבָרֶךְ אֱלֹהִים אֶת־יֹום הַשְּׁבִיעִי וַיְקַדֵּשׁ אֹתֹו, two
#   verbal expressions (בָרֶךְ and קַדֵּשׁ) are coordinated by two instances of
#   וְ: the first וְ (prefixed to יְבָרֶךְ) has relation1 -> בָרֶךְ's id, relation2
#   -> the second וְ's id; the second וְ (prefixed to יְקַדֵּשׁ) has relation1 ->
#   קַדֵּשׁ's id, relation2 -> the first וְ's id. Each connected item ALSO keeps
#   its own ordinary relation to the rest of the sentence, independent of
#   this chain (e.g. each coordinated verb still gets its own relation1 =
#   'root'/relationship1 = 'unit verb', unaffected by which connector
#   introduces it).
# - "object of preposition": a noun or pronoun functioning as the object of
#   a preposition has relation1 -> the id of the preposition, relationship1
#   = 'object of preposition'. Example: in בְּאֶרֶץ, אֶרֶץ has relation1 ->
#   בְּ's id, relationship1 = 'object of preposition'.
# - "article": when the article הַ relates to a noun or adjective, it has
#   relation1 -> the id of the noun or adjective, relationship1 =
#   'article'. Example: in הַשָּׁמַיִם, the article has relation1 ->
#   הַשָּׁמַיִם's own lexical-token id, relationship1 = 'article'.
# - "construct": when two nouns stand in a construct relation, the related
#   (governed) noun has relation1 -> the id of the governing noun,
#   relationship1 = 'construct'. The governing noun is recorded according
#   to its own function elsewhere in the sentence. Example: in בְּזֵעַת
#   אַפֶּיךָ תֹּאכַל לֶחֶם, אַפֶּי (governed by זֵעַת) has relation1 -> זֵעַת's id,
#   relationship1 = 'construct'; זֵעַת itself is separately recorded as the
#   object of the preposition בְּ.
# - "adjectival": an adjective has relation1 -> the id of the noun it
#   modifies, relationship1 = 'adjectival'. Example: in אֲחִיכֶם הַקָּטֹן,
#   קָּטֹן (modifying אֲחִי) has relation1 -> אֲחִי's id, relationship1 =
#   'adjectival'.
# - "adverbial": when a prepositional phrase modifies a verb adverbially,
#   the PREPOSITION ITSELF (not its object) has relation1 -> the id of the
#   verb, relationship1 = 'adverbial'. The preposition's own object is
#   still separately recorded as 'object of preposition', exactly as
#   usual -- this is an additional relation on the preposition, on top of
#   its object's own unaffected relation to it. Example: in בְּרֵאשִׁית בָּרָא
#   אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ, the preposition בְּ (of the adverbial
#   phrase בְּרֵאשִׁית) has relation1 -> בָּרָא's id, relationship1 =
#   'adverbial'; its own object רֵאשִׁית has relation1 -> בְּ's id,
#   relationship1 = 'object of preposition', unchanged.
RelationLabel = Literal[
    "unit verb",
    "direct quote",
    "subject",
    "direct object",
    "object marker",
    "predicate",
    "coordinating conjunction",
    "object of preposition",
    "article",
    "construct",
    "adjectival",
    "adverbial",
]


class TokenAnalysis(BaseModel):
    """One entry per token in the dependency graph (syntax_model.md,
    'Token-level table of dependencies'). Per syntax_model.md's own
    "TBA" section (some constructions have no documented relation at all
    yet), not every token will have a relation -- leave the
    relatedtoken*/relationship* fields unset when none of the documented
    relations apply.

    Every entry corresponds 1:1 to an entry in the input `tokens` list,
    EXCEPT for the one IMPLIED_TOKENTYPES value below: syntax_model.md's
    "understood or implied verbal expressions" section documents an elided
    present of "to be" ("to be" is often left out of a Hebrew nominal
    sentence entirely) as a VERBAL EXPRESSION that exists grammatically
    but has no surface realization at all. For that case, add a NEW entry
    here -- with a NEW id, not present in `tokens` -- rather than skipping
    the construction: tokentype 'implied sum', with `token` left unset
    (None). See hebrew_syntax_dspy.SyntaxAnalysis's docstring for the full
    rule and the id-naming convention."""

    id: str = Field(
        description=(
            "For an ordinary entry, must match the id of the corresponding "
            "entry in the input `tokens` list. For an implied token "
            "(tokentype 'implied sum'), a NEW id not used by any entry in "
            "`tokens` or elsewhere in this tokengraph -- see "
            "SyntaxAnalysis's docstring for the naming convention."
        )
    )
    token: Optional[str] = Field(
        default=None,
        description=(
            "The token's surface text; should match the `text` of the "
            "input token with this id. Leave as None ONLY for an implied "
            "token (tokentype 'implied sum') -- one with no surface "
            "realization in the passage at all; every other tokentype "
            "must have real text."
        ),
    )
    tokentype: Literal[
        "lexical",
        "enclitic pronoun",
        "proclitic conjunction",
        "maqaf",
        "cantillation",
        "paragraph",
        "editorial",
        "implied sum",
    ] = Field(
        description=(
            "Per syntax_model.md's 'Tokenization' section: 'cantillation' "
            "for any of the te'amim (e.g. sof pasuq, silluq, atnach); "
            "'paragraph' for a פ (petuhah) or ס (setumah) marking a "
            "semantic division of the text; 'enclitic pronoun' for a "
            "pronoun bound as the object of a preposition or verb, or as a "
            "possessive with a noun; 'proclitic conjunction' specifically "
            "for the conjunction וְ; 'maqaf' for the joining token ־; "
            "'lexical' for a continuous alphabetic sequence together with "
            "its own niqqud/dagesh/mappiq/sin-shin-dot (but never "
            "cantillation marks, which are their own token type); "
            "'editorial' for any Unicode punctuation character or other "
            "editorial mark, such as the masora circle. 'implied sum' "
            "marks a token with NO surface realization at all (an elided "
            "present of 'to be' -- see this model's own docstring) -- the "
            "only tokentype whose `token` field is None and whose `id` is "
            "not one of the input `tokens`' own ids; it always anchors its "
            "own entry in `verbalunits`, exactly like a real verb."
        )
    )

    lemma: Optional[str] = Field(default=None, description="Dictionary headword, for lexical tokens. Omit for cantillation/paragraph/editorial tokens.")
    verbalunitid: Optional[str] = Field(
        default=None,
        description="If this token anchors a verbal expression in `verbalunits`, repeat its own id here; otherwise omit.",
    )

    relatedtoken1: Optional[str] = Field(
        default=None,
        description=(
            "Token id this token relates to (primary relation). For an "
            "INDEPENDENT verb's own 'unit verb' relation, use the special "
            "sentinel string 'root' instead of a token id -- 'root' is "
            "reserved and must never be assigned as an actual token's id."
        ),
    )
    relationship1: Optional[RelationLabel] = Field(default=None, description="The primary relation type, if any.")

    relatedtoken2: Optional[str] = Field(default=None, description="Token id this token relates to (secondary relation -- an overflow slot, EXCEPT for 'coordinating conjunction', which uses both slots for its two sides at once).")
    relationship2: Optional[RelationLabel] = Field(default=None, description="The secondary relation type, if any.")


# The one tokentype value (see TokenAnalysis's own docstring) that marks a
# token with no surface realization at all -- an elided present of "to
# be". Every other module that needs to ask "is this an implied token"
# (validate(), rendering.py, serialization.py, conftest.py's
# tokens_from_canned_answer()) checks membership in this set rather than
# hardcoding the string itself, matching arsgrammatica's own convention --
# so a second implied-token category, if syntax_model.md ever grows one,
# only needs a change here.
IMPLIED_TOKENTYPES = frozenset({"implied sum"})

# tokentype values that are never meaningful participants in the dependency
# graph or a diagram of it -- they carry no relation of their own and, per
# syntax_model.md, never anchor or participate in a verbal unit. Centralized
# here (rather than duplicated) so verbal_units.py's first-appearance color
# ordering and mermaid.py's node-dropping always agree on exactly the same
# set, the same reason IMPLIED_TOKENTYPES is centralized above. Note this is
# a *different* set from the "glued" tokentypes rendering.py cares about
# (enclitic pronoun and proclitic conjunction DO carry real relations --
# 'subject'/'direct object' and 'coordinating conjunction' respectively --
# so they belong in a mermaid diagram and in color-ordering even though they
# also happen to render with no surrounding space).
NON_SUBSTANTIVE_TOKENTYPES = frozenset({"cantillation", "paragraph", "editorial", "maqaf"})
