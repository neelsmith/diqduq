"""
DSPy program that analyzes the syntax of a Biblical Hebrew passage according
to the scheme documented in syntax_model.md: a table of verbal expressions,
plus a token-level dependency graph.

This module covers only the analysis stage:
  1. SyntaxAnalysis -- a dspy.Signature that takes a sentence plus its
     pre-segmented token list and produces `verbalunits` and `tokengraph`,
     using the ids handed to it.
  2. validate()   -- a light sanity check that every id the LM refers to in
     its output actually exists in the input token list, so malformed
     output is easy to spot.

Tokens come from segmentation_dspy.py's LLM-driven, citation-aware
segmentation stage, not from a separate deterministic tokenizer -- see
pipeline.py for the module that ties the two stages together, including its
analyze_passage() convenience wrapper.

Modeled closely on arsgrammatica's latin_syntax_dspy.py, but syntax_model.md
documents a considerably smaller, first-draft scheme for Biblical Hebrew:
two verbal-expression syntactic types instead of five, ten relation labels
instead of two dozen, and no dependent-clause/ablative-absolute/gerund
machinery at all yet. Don't backport any of arsgrammatica's Latin-specific
categories here -- if a real passage needs something syntax_model.md
doesn't document, extend syntax_model.md first (see USAGE.md's "Extending
the scheme"), then this signature's docstring, then models.py.

Run this file directly for a quick smoke test against the configured LM:
    python -m diqduq.hebrew_syntax_dspy

For tests that don't need network access, see the `tests/` directory,
which drives this signature with dspy's DummyLM.
"""

from typing import List

import dspy

from .models import IMPLIED_TOKENTYPES, Token, VerbalExpression, TokenAnalysis


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class SyntaxAnalysis(dspy.Signature):
    """Analyze the syntax of a passage of Biblical Hebrew according to a
    two-part scheme:

    (1) a list of verbal expressions. Two constructions count as a verbal
        expression: every finite verb, and every participle.

        Classify each verbal expression's syntactic type as 'independent'
        (main/principal -- syntactically independent, its clause coherent
        by itself) or 'direct quote' (occurring in directly quoted speech
        introduced by a verb of saying). syntax_model.md documents no other
        syntactic_type values in this first draft of the scheme -- in
        particular there is no 'dependent' category yet for a subordinate
        clause (subordinating conjunctions are listed as "TBA"). A
        participle is classified the same way: 'independent' unless it
        occurs within quoted speech.

        Classify each verbal expression's semantic type too (transitive
        active/transitive passive/intransitive/linking verb).

    (2) a token-by-token dependency graph. For each token, record up to two
        relations to other tokens (by id), using only these relation
        labels:

        - unit verb (independent): every INDEPENDENT verb (or participle
          functioning as one) has relatedtoken1 = the special sentinel
          string 'root' -- never an actual token id; no real token may be
          assigned the id 'root' -- and relationship1 = 'unit verb'.
          Example: in בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ, the
          independent verb בָּרָא has relatedtoken1 = 'root', relationship1 =
          'unit verb'.
        - direct quote: a verb (or participle) of directly quoted speech
          has relatedtoken1 -> the id of the verb of the governing verbal
          expression (the verb of saying that introduces/frames the
          quotation), relationship1 = 'direct quote' -- matching its own
          syntactic_type. Example: in וַיֹּאמֶר אֱלֹהִים יְהִי אֹור וַיְהִי־אֹור,
          the verbal unit anchored at יְהִי is direct speech subordinate to
          יֹּאמֶר: יְהִי has relatedtoken1 -> יֹּאמֶר's id, relationship1 =
          'direct quote'.
        - subject / direct object / predicate: a noun or pronoun serving as
          the subject of a verbal expression has relatedtoken1 -> the id of
          the verb, relationship1 = 'subject'. One functioning as direct
          object has relatedtoken1 -> the verb's id, relationship1 =
          'direct object'. One functioning as the predicate complement of a
          LINKING verb (including an elided-'to be' implied token -- see
          (3) below) has relatedtoken1 -> that verb's id, relationship1 =
          'predicate'. Example: in Genesis 1.1 (as above), אֱלֹהִים has
          relatedtoken1 -> בָּרָא's id, relationship1 = 'subject'; שָׁמַיִם and
          אָרֶץ each have relatedtoken1 -> בָּרָא's id, relationship1 = 'direct
          object'.
        - coordinating conjunction (single pair): when a coordinating
          conjunction (the proclitic וְ) joins exactly ONE pair of
          adjectives, nouns, prepositional phrases, or verbal expressions,
          it has relatedtoken1 -> the id of the first joined token,
          relatedtoken2 -> the id of the second, with BOTH relationship1
          and relationship2 = 'coordinating conjunction' (not an overflow
          slot here -- this is the one relation that genuinely uses both
          ends at once). Example: in בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם
          וְאֵת הָאָרֶץ, the conjunction וְ (prefixed to אֵת before הָאָרֶץ) has
          relatedtoken1 -> הַשָּׁמַיִם's id, relatedtoken2 -> הָאָרֶץ's id, both
          relationship 'coordinating conjunction'.
        - coordinating conjunction (repeated, as a chain): the proclitic וְ
          can instead be repeated, prefixed onto EVERY one of a series of
          two or more coordinated items (most often a chain of narrative
          wayyiqtol verbs), not just used once between a pair. Annotate
          this differently from the single-pair case above. Every
          connector's own relatedtoken1 -> the id of the item it
          immediately introduces (a real noun, adjective, prepositional
          phrase, or verbal-expression anchor -- NEVER another connector),
          relationship1 = 'coordinating conjunction', exactly as in the
          single-pair case. relatedtoken2 is what differs: the FIRST
          connector's relatedtoken2 -> the id of the NEXT (second)
          connector, while every connector AFTER the first has
          relatedtoken2 -> the id of the PRECEDING connector instead (not
          the following one); relationship2 = 'coordinating conjunction'
          for all of them, same as relationship1 -- still not an overflow
          slot. Each connected item ALSO keeps its own ordinary relation to
          the rest of the sentence (subject, object of preposition, or
          whatever fits), completely independent of this chain. Example:
          in וַיְבָרֶךְ אֱלֹהִים אֶת־יֹום הַשְּׁבִיעִי וַיְקַדֵּשׁ אֹתֹו, two verbal
          expressions (anchored at בָרֶךְ and קַדֵּשׁ) are coordinated by two
          instances of וְ: the first וְ (prefixed to יְבָרֶךְ) has
          relatedtoken1 -> בָרֶךְ's id, relatedtoken2 -> the second וְ's id;
          the second וְ (prefixed to יְקַדֵּשׁ) has relatedtoken1 -> קַדֵּשׁ's
          id, relatedtoken2 -> the first וְ's id. בָרֶךְ and קַדֵּשׁ each ALSO
          have their own relatedtoken1 = 'root', relationship1 = 'unit
          verb' entries, unaffected by which connector introduces them --
          being coordinated by וְ does not exempt either verb from its own
          normal 'unit verb'/'root' entry.
        - object of preposition: a noun or pronoun functioning as the
          object of a preposition has relatedtoken1 -> the id of the
          preposition, relationship1 = 'object of preposition'. Example: in
          the phrase בְּאֶרֶץ, אֶרֶץ has relatedtoken1 -> בְּ's id,
          relationship1 = 'object of preposition'.
        - article: when the article הַ relates to a noun or adjective, it
          has relatedtoken1 -> the id of that noun or adjective,
          relationship1 = 'article'. Example: in הַשָּׁמַיִם, the article has
          relatedtoken1 -> הַשָּׁמַיִם's own lexical-token id, relationship1 =
          'article'.
        - construct: when two nouns stand in a construct relation, the
          governed (related) noun has relatedtoken1 -> the id of the
          governing noun, relationship1 = 'construct'; the governing noun
          is separately recorded according to its own function elsewhere
          in the sentence. Example: in בְּזֵעַת אַפֶּיךָ תֹּאכַל לֶחֶם, אַפֶּי
          (governed by זֵעַת) has relatedtoken1 -> זֵעַת's id, relationship1 =
          'construct'; זֵעַת itself is recorded as the object of the
          preposition בְּ (relatedtoken1 -> בְּ's id, relationship1 = 'object
          of preposition').
        - adjectival: an adjective has relatedtoken1 -> the id of the noun
          it modifies, relationship1 = 'adjectival'. Example: in אֲחִיכֶם
          הַקָּטֹן, קָּטֹן (modifying אֲחִי) has relatedtoken1 -> אֲחִי's id,
          relationship1 = 'adjectival'.

        Only assign relations described above. Leave relatedtoken/
        relationship fields unset for tokens with no relation of these
        kinds -- not every token will have one (syntax_model.md's own "TBA"
        section names several constructions -- the functions of
        prepositions beyond 'object of preposition', subordinating
        conjunctions, and the relative pronoun אֲשֶׁר -- that have no
        documented relation at all yet; leave a token unrelated rather than
        guessing a label for one of these). Use only the token ids given in
        the input `tokens` list, the sentinel 'root', or a NEW id you
        create for an implied token (see below), in your output; never
        invent an id for anything else.

    (3) implied/elided tokens. `diqduq` recognizes one situation where
        something exists grammatically but has no surface realization in
        the passage at all: an elided present tense of "to be", which
        Biblical Hebrew routinely omits from a nominal (verbless) sentence.
        When this happens, add a NEW entry to `tokengraph` with: a
        brand-new id, not used by any entry in `tokens` or elsewhere in
        your own output (see the naming rule below); tokentype 'implied
        sum'; and no `token` value (leave it unset/None). Also add a
        matching new entry to `verbalunits`, exactly like any other verbal
        expression, classified 'independent' (or 'direct quote', if the
        elided-copula clause is itself directly quoted speech) and
        'linking verb'. The subject and predicate each relate to this new
        token exactly as they would to any linking verb ('subject' /
        'predicate'). Example: in לֹא אֱלֹהִים הֵמָּה ("they are not gods"),
        the subject is הֵמָּה and the predicate noun is אֱלֹהִים; add a new
        implied token (tokentype 'implied sum', token=None) anchoring an
        'independent'/'linking verb' verbal expression, with הֵמָּה related
        to it as 'subject' and אֱלֹהִים as 'predicate'.

        Naming an implied token's id: append '_implied' to the id of the
        LAST real token in `tokens` that precedes where the elided "to be"
        would have stood (or, if the elided word would come before every
        real token in the sentence, the FIRST real token's id instead). If
        more than one implied token is ever needed in the same sentence,
        append '2', '3', ... after '_implied' to keep them unique (e.g.
        't5_implied', 't5_implied2'). Place the new `tokengraph` entry at
        the list position where the elided word would have appeared,
        among the tokens of its own clause.
    """

    passage: str = dspy.InputField(desc="The Hebrew passage to analyze, exactly as written.")
    tokens: List[Token] = dspy.InputField(
        desc="Pre-segmented tokens of the passage, in order, with fixed ids. Reference these ids in your output; do not create new ones."
    )
    verbalunits: List[VerbalExpression] = dspy.OutputField(
        desc="One entry per verbal expression (finite verb or participle) in the passage."
    )
    tokengraph: List[TokenAnalysis] = dspy.OutputField(
        desc=(
            "One entry per token in `tokens`, in the same order, with its "
            "type and any relations -- PLUS one additional entry for each "
            "implied/elided token you add (see this signature's docstring), "
            "positioned where that token's clause falls in reading order."
        )
    )


analyze = dspy.ChainOfThought(SyntaxAnalysis)


# ---------------------------------------------------------------------------
# Runner + validation
# ---------------------------------------------------------------------------

def validate(tokens: List[Token], result) -> List[str]:
    """Check that every id the LM produced actually exists among `tokens`
    -- OR is a legitimately new implied token (tokentype in
    IMPLIED_TOKENTYPES -- currently just 'implied sum'; see
    SyntaxAnalysis's docstring) -- and that implied tokens themselves are
    well-formed. Returns a list of human-readable problem descriptions
    (empty if clean).

    'root' is a special sentinel value for an independent verb's own
    relatedtoken1 (see SyntaxAnalysis's docstring) -- it is never treated as
    an unknown id, but syntax_model.md also requires that no actual token
    ever be assigned the id 'root', so that's checked here too.

    Implied tokens get their own, narrower checks: a tokengraph entry
    claiming an IMPLIED_TOKENTYPES value must use a genuinely NEW id (not
    one already in `tokens`) and must leave `token` unset (None). A
    non-implied entry, conversely, must use one of `tokens`' own ids and
    must NOT have `token=None`. This check is purely structural -- it does
    NOT also require an 'implied sum' token to have a matching
    `verbalunits` entry; that distinction is documented behavior (see
    SyntaxAnalysis's docstring), not something this function enforces."""
    valid_ids = {t.id for t in tokens}
    problems = []

    if "root" in valid_ids:
        problems.append(
            "token id 'root' is reserved as the sentinel relatedtoken1 "
            "value for independent verbs and must not be assigned to an "
            "actual token"
        )

    implied_ids = {tok.id for tok in result.tokengraph if tok.tokentype in IMPLIED_TOKENTYPES}
    known_ids = valid_ids | implied_ids

    for tok in result.tokengraph:
        if tok.tokentype in IMPLIED_TOKENTYPES:
            if tok.id in valid_ids:
                problems.append(
                    f"tokengraph entry {tok.id!r} is tokentype={tok.tokentype!r} but "
                    "reuses an id already in the input `tokens` list -- an "
                    "implied token must use a new id"
                )
            if tok.token is not None:
                problems.append(
                    f"tokengraph entry {tok.id!r} is tokentype={tok.tokentype!r} but "
                    f"has a non-None token value {tok.token!r} -- an implied "
                    "token's text must be left unset"
                )
        else:
            if tok.id not in valid_ids:
                problems.append(f"tokengraph entry has unknown id {tok.id!r}")
            if tok.token is None:
                allowed = "/".join(repr(t) for t in sorted(IMPLIED_TOKENTYPES))
                problems.append(
                    f"tokengraph entry {tok.id!r} has token=None but "
                    f"tokentype={tok.tokentype!r} -- only {allowed} may "
                    "omit surface text"
                )
        for field in ("relatedtoken1", "relatedtoken2"):
            val = getattr(tok, field)
            if val is not None and val != "root" and val not in known_ids:
                problems.append(f"token {tok.id!r} {field}={val!r} is not a known token id")

    for vu in result.verbalunits:
        if vu.id not in known_ids:
            problems.append(f"verbal expression id {vu.id!r} is not a known token id")

    return problems


def print_analysis(tokens: List[Token], result):
    print("Tokens:")
    for t in tokens:
        print(f"  {t.id:>4}  {t.text}")

    print("\nVerbal expressions:")
    for vu in result.verbalunits:
        print(f"  id={vu.id}  syntactic_type={vu.syntactic_type}  semantic_type={vu.semantic_type}")

    print("\nToken graph:")
    for tok in result.tokengraph:
        rels = []
        if tok.relationship1:
            rels.append(f"{tok.relationship1} -> {tok.relatedtoken1}")
        if tok.relationship2:
            rels.append(f"{tok.relationship2} -> {tok.relatedtoken2}")
        rel_str = "; ".join(rels) if rels else "-"
        vu_str = f" [verbal unit {tok.verbalunitid}]" if tok.verbalunitid else ""
        token_str = tok.token if tok.token is not None else f"({tok.tokentype})"
        print(f"  {tok.id:>4}  {token_str:<15} type={tok.tokentype:<19} lemma={tok.lemma or '-':<15} {rel_str}{vu_str}")
