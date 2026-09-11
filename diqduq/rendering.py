"""
Reconstructs readable text from a tokengraph (a list of TokenAnalysis, as
produced by hebrew_syntax_dspy.analyze/pipeline.py), in two forms:
`tokengraph_to_text()` for plain text, and `tokengraph_to_html()` for HTML
with substantive tokens wrapped in verbal-unit-colored spans. Modeled
directly on arsgrammatica's rendering.py, with the join rules rewritten
for Biblical Hebrew's own tokenization scheme (syntax_model.md's
"Tokenization" section) instead of Latin's.

This module needs `tokentype` to decide spacing, so it operates on
TokenAnalysis (post-SyntaxAnalysis), not the plain pre-analysis Token list
segmentation produces -- tokentype isn't known yet at that earlier stage.

Every token is classified as one of:

- **normal** (the default -- lexical and paragraph tokens): gets a space
  before it, UNLESS the immediately preceding token was right-joining or
  glued, in which case it attaches directly with no space.
- **right-joining** (the proclitic conjunction וְ; the article הַ; an
  inseparable preposition, e.g. בְּ): always preceded by a space, but
  attaches with NO space to whatever follows it -- the mirror image of
  Latin's enclitic, since each of these is a *prefix* rather than a
  suffix. The article is detected by its own 'article' relation
  (relationship1/2); a preposition is detected indirectly instead, from
  whichever OTHER token relates to it via 'object of preposition' (see
  `_preposition_ids()` below) -- even though a preposition heading an
  adverbial phrase now also carries its own outgoing 'adverbial' relation
  (see models.py's RelationLabel comment), that relation is optional (not
  every preposition in a passage modifies a verb adverbially), where
  'object of preposition' is the reliable, universal signal, so spacing
  detection keeps using the latter.
- **left-joining** (an enclitic pronoun, or a cantillation mark): attaches
  directly to whatever precedes it, no space, ever -- these are both
  suffix-like in orthography (an enclitic pronoun suffixed to its host
  word; a cantillation mark riding on the end of the word it marks), so
  both are treated as backward-attaching. This is a modeling convention
  for cantillation specifically (syntax_model.md documents cantillation as
  its own token type but doesn't specify its rendering behavior); attaching
  it to the preceding word matches how it actually appears in the
  Masoretic text (e.g. sof pasuq ׃ glued to the last word of a verse).
- **glued** (maqaf ־): attaches with NO space on EITHER side -- unlike every
  other category, a glued token forces the token that follows it to attach
  with no space too, exactly the way it visually joins two words into one
  hyphenated unit in the source text (e.g. מִכָּל־מְלַאכְתֹּו).
- **editorial** (any other punctuation/editorial mark, e.g. the masora
  circle): defaults to left-joining, same as ordinary punctuation in
  arsgrammatica's Latin scheme, with the same opening/closing-bracket
  carve-out for the rare case an edition uses bracket-style editorial
  marks.

An **implied/elided token** (models.py's IMPLIED_TOKENTYPES -- currently
just "implied sum") has no surface realization at all and is skipped
entirely, exactly as if it weren't in the list, in both
tokengraph_to_text() and tokengraph_to_html() -- mermaid.py's diagram is
the one place its presence is worth showing (see that module's own
docstring).
"""

import html
from typing import Dict, List, Optional, Tuple

from .models import IMPLIED_TOKENTYPES, TokenAnalysis
from .verbal_units import (
    assign_verbal_units,
    assign_verbal_unit_colors,
    compute_subordination_depths,
)

_OPENING_BRACKETS = {"(", "[", "{"}
_CLOSING_BRACKETS = {")", "]", "}"}

_LEFT = "left"
_RIGHT = "right"
_GLUED = "glued"
_NORMAL = "normal"

# Tokentypes that attach directly to a neighboring word with no space --
# used below both for join-classification and (in
# tokengraph_to_depth_html()) to keep a glued function word/mark from
# fragmenting a block of tokens away from the word it's orthographically
# attached to. A different concern from models.NON_SUBSTANTIVE_TOKENTYPES:
# an enclitic pronoun and a proclitic conjunction DO carry real relations
# and DO belong in color-ordering/mermaid nodes (see that constant's own
# docstring) even though they also render glued to a neighbor.
_GLUED_TOKENTYPES = {"enclitic pronoun", "proclitic conjunction", "maqaf", "cantillation"}


def _preposition_ids(tokengraph: List[TokenAnalysis]) -> set:
    """Token ids that some OTHER token relates to via 'object of
    preposition' -- i.e., the ids of the preposition tokens themselves.
    Needed because a preposition's own outgoing relation (relationship1=
    'adverbial', when the phrase it heads modifies a verb -- see models.py's
    RelationLabel comment) is optional, not universal: not every
    preposition in a passage functions adverbially, and syntax_model.md
    doesn't document any other outward function for one yet. So detecting
    ANY preposition (for spacing purposes) still goes through the reliable,
    universal signal instead, unlike the article, which carries its own
    'article' relation and so is detected directly in `_classify()`."""
    ids = set()
    for tok in tokengraph:
        for related_field, label_field in (
            ("relatedtoken1", "relationship1"),
            ("relatedtoken2", "relationship2"),
        ):
            if getattr(tok, label_field) == "object of preposition":
                related = getattr(tok, related_field)
                if related is not None:
                    ids.add(related)
    return ids


def _classify(tok: TokenAnalysis, preposition_ids: set) -> str:
    """Return this token's join behavior: one of _LEFT, _RIGHT, _GLUED, or
    _NORMAL."""
    text = tok.token

    if tok.tokentype in ("enclitic pronoun", "cantillation"):
        return _LEFT

    if tok.tokentype == "proclitic conjunction":
        return _RIGHT

    if tok.tokentype == "maqaf":
        return _GLUED

    if tok.relationship1 == "article" or tok.relationship2 == "article":
        return _RIGHT

    if tok.id in preposition_ids:
        return _RIGHT

    if tok.tokentype == "editorial":
        if text in _OPENING_BRACKETS:
            return _RIGHT
        if text in _CLOSING_BRACKETS:
            return _LEFT
        # Any other punctuation/editorial mark (e.g. the masora circle)
        # defaults to left-joining.
        return _LEFT

    # lexical (an ordinary word), paragraph, and any future tokentype not
    # covered above all get the same "normal" spacing rule.
    return _NORMAL


def tokengraph_to_text(tokengraph: List[TokenAnalysis]) -> str:
    """Join `tokengraph`'s tokens into one continuous plain-text string,
    per this module's docstring. Tokens are read in list order (the same
    order tokengraph_to_mermaid() and validate() assume)."""
    preposition_ids = _preposition_ids(tokengraph)
    pieces: List[str] = []
    previous_class = None

    for tok in tokengraph:
        if tok.tokentype in IMPLIED_TOKENTYPES:
            continue
        cls = _classify(tok, preposition_ids)
        text = tok.token

        if not pieces:
            pieces.append(text)
        elif cls in (_LEFT, _GLUED):
            pieces.append(text)
        elif cls == _RIGHT:
            pieces.append(" " + text)
        else:  # _NORMAL
            if previous_class in (_RIGHT, _GLUED):
                pieces.append(text)
            else:
                pieces.append(" " + text)

        previous_class = cls

    return "".join(pieces)


def tokengraph_to_html(tokengraph: List[TokenAnalysis], *, include_cantillation: bool = True) -> str:
    """Render `tokengraph` as an HTML string: the same continuous text
    `tokengraph_to_text()` produces -- identical spacing rules -- except
    every **lexical** token, and every **proclitic conjunction** carrying a
    'coordinating conjunction' relation (relationship1 or relationship2),
    has its text wrapped in a `<span style="...">` colored by the verbal
    unit it belongs to. Colors come from `verbal_units.assign_verbal_units()`
    / `assign_verbal_unit_colors()` -- the same assignment and the same
    first-appearance palette ordering `tokengraph_to_mermaid()` uses for its
    node coloring -- so a passage rendered here and the same passage's
    Mermaid diagram color each verbal unit identically.

    The coordinating-conjunction carve-out exists because the proclitic וְ
    is tokentype "proclitic conjunction", not "lexical", but
    `assign_verbal_units()` still resolves it to one of the units it
    coordinates (see that module's docstring). Leaving it unwrapped would
    visually hide that assignment even though it's a real one.

    Every other non-lexical, non-conjunction token -- paragraph, editorial,
    maqaf, and a non-conjunction enclitic pronoun -- is still emitted as
    plain (escaped) text even though `assign_verbal_units()` assigns every
    token to whichever unit its relations resolve to; this function just
    doesn't turn that assignment into a span for anything else.

    `include_cantillation` (default `True`, matching arsgrammatica's own
    tokengraph_to_html(), which has no equivalent exclusion) controls
    whether cantillation tokens (te'amim -- e.g. the verse-final sof pasuq
    ׃) are rendered at all. Pass `include_cantillation=False` to omit them
    from the output entirely -- not just leave them uncolored, the way the
    other non-lexical tokentypes above are -- for a reading view that
    foregrounds the lexical/relational content without the accent marks
    interspersed. This is a deliberate Hebrew-specific divergence from
    arsgrammatica: Latin punctuation is sparse enough to leave visible by
    default, but cantillation is dense enough (in principle -- most marks
    stay embedded in a lexical token's own niqqud under this project's
    fixtures; see gold_examples.py's own note on that simplification) that
    a caller may want it gone rather than merely unhighlighted.

    An **implied/elided token** (models.py's IMPLIED_TOKENTYPES) is omitted
    entirely -- same as tokengraph_to_text() -- rather than rendered with
    any span: it has no surface text, and unlike
    `tokengraph_to_mermaid()`'s diagram (which DOES show these), inserting
    placeholder text into the middle of reconstructed prose here would
    misrepresent what the passage actually says.

    Every token's text is HTML-escaped (`&`, `<`, `>`, and quote characters)
    before being emitted, spans or not.
    """
    assignment = assign_verbal_units(tokengraph)
    colors, _warnings = assign_verbal_unit_colors(tokengraph, assignment=assignment)
    return _tokens_to_html(tokengraph, assignment, colors, include_cantillation=include_cantillation)


def _tokens_to_html(
    tokens: List[TokenAnalysis],
    assignment: Dict[str, Optional[str]],
    colors: Dict[str, Tuple[str, str, str]],
    *,
    include_cantillation: bool = True,
) -> str:
    """Shared rendering core behind tokengraph_to_html() and
    tokengraph_to_depth_html(): join `tokens` into one HTML string with the
    same spacing/escaping rules as tokengraph_to_text(), plus color spans
    for lexical tokens and coordinating-conjunction proclitics, given an
    already-computed verbal-unit `assignment` and `colors` mapping. Taking
    these as parameters (rather than deriving them from `tokens` itself) is
    what lets tokengraph_to_depth_html() render one depth-block's tokens at
    a time while every block still uses the exact same unit-to-color
    mapping as the whole passage.

    `include_cantillation=False` drops cantillation tokens from the output
    entirely, exactly like an implied/elided token -- see
    tokengraph_to_html()'s own docstring for why. tokengraph_to_depth_html()
    doesn't expose this itself (it always passes the default, `True`) since
    dropping a cantillation token there could otherwise leave a subordination
    block's rendering looking identical to an adjacent one with a different
    depth; nothing currently stops a future caller from passing it through
    if that turns out to be wanted too.
    """
    preposition_ids = _preposition_ids(tokens)
    pieces: List[str] = []
    previous_class = None

    for tok in tokens:
        if tok.tokentype in IMPLIED_TOKENTYPES:
            continue
        if not include_cantillation and tok.tokentype == "cantillation":
            continue
        cls = _classify(tok, preposition_ids)
        rendered = html.escape(tok.token)

        is_coordinating_conjunction = (
            tok.relationship1 == "coordinating conjunction"
            or tok.relationship2 == "coordinating conjunction"
        )
        if tok.tokentype == "lexical" or is_coordinating_conjunction:
            unit_id = assignment.get(tok.id)
            color = colors.get(unit_id) if unit_id is not None else None
            if color is not None:
                fill, _stroke, text_color = color
                rendered = (
                    f'<span style="background-color: {fill}; color: {text_color};">'
                    f"{rendered}</span>"
                )

        if not pieces:
            pieces.append(rendered)
        elif cls in (_LEFT, _GLUED):
            pieces.append(rendered)
        elif cls == _RIGHT:
            pieces.append(" " + rendered)
        else:  # _NORMAL
            if previous_class in (_RIGHT, _GLUED):
                pieces.append(rendered)
            else:
                pieces.append(" " + rendered)

        previous_class = cls

    return "".join(pieces)


# Left-margin indent per level of subordination depth, in
# tokengraph_to_depth_html()'s default rendering -- purely a CSS layout
# value, tunable per call via that function's `indent_em` parameter.
_DEFAULT_DEPTH_INDENT_EM = 2.0


def tokengraph_to_depth_html(
    tokengraph: List[TokenAnalysis],
    indent_em: float = _DEFAULT_DEPTH_INDENT_EM,
    depth: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Render `tokengraph` as HTML illustrating each verbal expression's
    *depth of subordination* (see verbal_units.compute_subordination_
    depths()): tokens are assembled sequentially exactly as
    tokengraph_to_html() does -- same spacing, escaping, and verbal-unit
    color highlighting -- but grouped into consecutive-run "blocks" by
    which verbal unit each token belongs to (per assign_verbal_units()),
    each rendered as its own <div> indented by a CSS margin-left of
    `depth * indent_em` em -- 0 for an independent clause, 1 for a directly
    quoted clause it introduces, and so on. All layout is CSS -- no table
    or nested-list structure is used to produce the indentation.

    `depth`, if given, caps how deep the rendering goes: ONLY blocks whose
    own depth of subordination is <= `depth` are included in the output.
    `depth=0` shows root/independent clauses only; omit `depth` (or pass
    `None`, the default) to show every block. Valid values run from 0 up
    to verbal_units.max_subordination_depth()'s own return value for this
    `tokengraph`; a negative `depth` raises ValueError.

    Block boundaries follow assign_verbal_units()'s token-to-unit
    assignment, with one adjustment: a token whose own tokentype is one of
    `_GLUED_TOKENTYPES` (enclitic pronoun, proclitic conjunction, maqaf,
    cantillation) never starts a new block, even when its own assignment
    differs from the block currently open -- these all attach with no
    space to a neighboring word (see this module's own docstring), and
    starting a new block there would split a Hebrew word (or a
    maqaf-joined pair) across two <div>s. A token with no verbal-unit
    assignment at all (None) likewise never starts a new block; it folds
    into whichever block is currently open. Leading tokens before the
    first resolvable verbal-unit token (rare) default to depth 0.

    A verbal expression whose depth couldn't be resolved (see
    compute_subordination_depths()) renders at depth 0 rather than
    raising, with a warning explaining why.

    Returns (html, warnings), combining assign_verbal_unit_colors()'s
    warnings (colors repeating past 8 verbal units) and
    compute_subordination_depths()'s (an unresolved governing verbal
    expression) -- computed the same way, and returned in full, regardless
    of whether `depth` filters some blocks out of the rendered `html`
    itself.
    """
    if depth is not None and depth < 0:
        raise ValueError(f"depth must be >= 0 (root clauses only), got {depth!r}")

    assignment = assign_verbal_units(tokengraph)
    colors, color_warnings = assign_verbal_unit_colors(tokengraph, assignment=assignment)
    depths, depth_warnings = compute_subordination_depths(tokengraph)
    warnings = color_warnings + depth_warnings

    blocks = []
    for tok in tokengraph:
        unit_id = assignment.get(tok.id)
        starts_new_block = (
            unit_id is not None
            and tok.tokentype not in _GLUED_TOKENTYPES
            and (not blocks or blocks[-1][0] != unit_id)
        )
        if starts_new_block:
            blocks.append((unit_id, []))
        elif not blocks:
            blocks.append((None, []))
        blocks[-1][1].append(tok)

    lines = []
    for unit_id, block_tokens in blocks:
        block_depth = depths.get(unit_id) if unit_id is not None else 0
        if block_depth is None:
            block_depth = 0
        if depth is not None and block_depth > depth:
            continue
        block_html = _tokens_to_html(block_tokens, assignment, colors)
        margin_left = block_depth * indent_em
        lines.append(
            f'<div style="margin-left: {margin_left}em; margin-bottom: 0.35em;">'
            f"{block_html}</div>"
        )

    return "\n".join(lines), warnings
