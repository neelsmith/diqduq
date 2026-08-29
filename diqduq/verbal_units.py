"""
Partitions a `tokengraph` into the verbal units its own relations already
imply, so every token can be labelled "this token belongs to verbal unit
X" (or to none) -- e.g. for coloring a Mermaid diagram by clause (see
mermaid.py), or for rendering.py's colored HTML. Modeled directly on
arsgrammatica's verbal_units.py.

The key idea: a verbal unit's anchor token (the finite verb or participle
that owns an entry in `verbalunits` -- see models.py's VerbalExpression)
already marks itself in the tokengraph via `verbalunitid` (set to its own
id). Every OTHER token can be assigned to whichever anchor its own
relatedtoken1/relatedtoken2 chain eventually leads to.

Unlike arsgrammatica's mature Latin scheme, syntax_model.md's first-draft
Hebrew scheme does not (yet) document any relation whose OUTGOING pointer
would misclassify the pointing token's own clause membership -- there is
no "unit verb (dependent)" via a subordinating conjunction, and no
"ablative absolute" wrinkle, because subordinating conjunctions and the
relative pronoun אֲשֶׁר are both listed under syntax_model.md's own "TBA"
section. A 'direct quote' verb's own relatedtoken1 points straight at the
verb that governs it, but that verb itself is checked FIRST (via its own
`verbalunitid`) before any relation is followed at all, so it resolves to
its OWN verbal unit correctly with no special-casing needed. So this
module needs no reverse-index correction the way arsgrammatica's does --
plain forward-chasing of relatedtoken1/relatedtoken2 is enough. If
syntax_model.md later documents a relation with the same "points at the
outer clause but belongs to the inner one" shape (most likely once
subordinating conjunctions are added), add the same kind of reverse-index
wrinkle arsgrammatica's verbal_units.py uses for its own "unit verb
(dependent)" case.

The two relations added since ("object marker": the direct object marker
אֵת -> the noun it marks; "adverbial": a preposition -> the verb it
modifies) both point straight at another token in the SAME clause -- the
marked noun and the verb, respectively -- so plain forward-chasing
resolves both correctly too, with no reverse-index wrinkle needed for
either (see tests/test_verbal_units.py's genesis_1_1 coverage)."""

from typing import Dict, List, Optional, Tuple

from .models import NON_SUBSTANTIVE_TOKENTYPES, TokenAnalysis

# Categorical palette for coloring verbal units: 8 (fill, stroke, text)
# triples, in a fixed order chosen so adjacent slots stay distinguishable
# under color-vision deficiency as well as normal vision. Ported verbatim
# from arsgrammatica's own validated palette (see that project's
# verbal_units.py for the full rationale and validation notes) -- there is
# nothing Latin-specific about it, so both projects can share the same
# tuned ordering.
_VERBAL_UNIT_PALETTE = [
    ("#82bbff", "#2a78d6", "#000000"),  # blue
    ("#ffa682", "#eb6834", "#000000"),  # orange
    ("#70ffcc", "#1baf7a", "#000000"),  # aqua
    ("#ffd170", "#eda100", "#000000"),  # yellow
    ("#ff94bc", "#e87ba4", "#000000"),  # magenta
    ("#7aff7a", "#008300", "#000000"),  # green
    ("#a494ff", "#4a3aa7", "#000000"),  # violet
    ("#ff9594", "#e34948", "#000000"),  # red
]

# A dedicated "caution" color for implied/elided tokens (models.py's
# IMPLIED_TOKENTYPES: currently just "implied sum") -- a strong, saturated
# amber, deliberately NOT drawn from _VERBAL_UNIT_PALETTE above (whose
# pastel tints it would otherwise be confusable with) so it reads as "this
# marks something MISSING from the surface text" rather than as just
# another clause's color. Every consumer that renders an implied token
# (rendering.py's tokengraph_to_html()/tokengraph_to_depth_html() and
# mermaid.py's tokengraph_to_mermaid()) uses this SAME color for it.
_IMPLIED_TOKEN_COLOR = ("#ffc107", "#7a5200", "#000000")  # amber warning


def assign_verbal_units(tokengraph: List[TokenAnalysis]) -> Dict[str, Optional[str]]:
    """Return {token id: verbal unit id or None}, one entry per token in
    `tokengraph` (including cantillation/editorial/maqaf and any other
    unrelated token, so every id is accounted for -- callers that only
    care about assigned tokens can filter out the None values themselves).

    A verbal unit's own anchor token is assigned to itself (its
    `verbalunitid`). Every other token is assigned to the verbal unit its
    relations resolve to via a plain forward chase of relatedtoken1 (then
    relatedtoken2); a token with no resolvable relation (e.g. a
    preposition, whose own outward relation syntax_model.md doesn't
    document yet -- see this module's own docstring) gets None.
    """
    by_id = {tok.id: tok for tok in tokengraph}

    resolved: Dict[str, Optional[str]] = {}
    in_progress: set = set()

    def resolve(tid: str) -> Optional[str]:
        if tid in resolved:
            return resolved[tid]
        tok = by_id.get(tid)
        if tok is None:
            return None

        if tok.verbalunitid is not None:
            resolved[tid] = tok.verbalunitid
            return tok.verbalunitid

        if tid in in_progress:
            # A cycle in the relation graph (malformed LM output) -- bail
            # out on this token rather than recursing forever.
            return None
        in_progress.add(tid)

        result = None
        for related_field in ("relatedtoken1", "relatedtoken2"):
            related = getattr(tok, related_field)
            if related is None or related == "root":
                continue
            result = resolve(related)
            if result is not None:
                break

        in_progress.discard(tid)
        resolved[tid] = result
        return result

    for tid in by_id:
        resolve(tid)

    return resolved


def assign_verbal_unit_colors(
    tokengraph: List[TokenAnalysis],
    assignment: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Dict[str, Tuple[str, str, str]], List[str]]:
    """Assign each verbal unit found in `tokengraph` a stable (fill, stroke,
    text) triple from `_VERBAL_UNIT_PALETTE`, using the exact ordering rule
    `tokengraph_to_mermaid()` uses for its node coloring -- so any other
    caller wanting "the same colors as the mermaid graph" (rendering.py's
    `tokengraph_to_html()`) gets an identical mapping without re-deriving
    the rule itself.

    Order is by first appearance of each verbal unit among tokengraph's
    *substantive* tokens (tokentype not in models.NON_SUBSTANTIVE_TOKENTYPES
    -- cantillation, paragraph, editorial, and maqaf never become mermaid
    nodes at all).

    Pass `assignment` (the result of `assign_verbal_units(tokengraph)`) if
    the caller already computed it, to avoid re-deriving it here; otherwise
    it's computed internally.

    Returns `({verbal unit id: (fill, stroke, text)}, warnings)` --
    `warnings` holds one entry if there are more distinct verbal units than
    palette slots (colors repeat past the 8th unit). A verbal unit id
    absent from the returned dict was never assigned to any substantive
    token -- callers should treat that the same as "no verbal unit" (no
    coloring).
    """
    if assignment is None:
        assignment = assign_verbal_units(tokengraph)

    substantive_ids = {tok.id for tok in tokengraph if tok.tokentype not in NON_SUBSTANTIVE_TOKENTYPES}

    unit_order: List[str] = []
    seen_units = set()
    for tok in tokengraph:
        if tok.id not in substantive_ids:
            continue
        unit_id = assignment.get(tok.id)
        if unit_id is not None and unit_id not in seen_units:
            seen_units.add(unit_id)
            unit_order.append(unit_id)

    warnings: List[str] = []
    if len(unit_order) > len(_VERBAL_UNIT_PALETTE):
        warnings.append(
            f"{len(unit_order)} verbal units but only {len(_VERBAL_UNIT_PALETTE)} "
            "distinct colors -- colors repeat and may be ambiguous between units"
        )

    colors = {
        unit_id: _VERBAL_UNIT_PALETTE[i % len(_VERBAL_UNIT_PALETTE)]
        for i, unit_id in enumerate(unit_order)
    }
    return colors, warnings


def compute_subordination_depths(
    tokengraph: List[TokenAnalysis],
) -> Tuple[Dict[str, Optional[int]], List[str]]:
    """Compute each verbal expression's *depth of subordination*: the
    number of verbal expressions it is removed from an independent ("root")
    clause. An independent verb is depth 0; a directly-quoted verb it
    introduces is depth 1; a verbal expression quoted WITHIN that quote (if
    the scheme is ever asked to represent one) would be depth 2; and so on.

    A "verbal expression" here is any token that anchors one -- i.e. any
    token with `verbalunitid` set to its own id (the same convention
    `assign_verbal_units()` relies on). For each anchor, this function
    finds its *parent* anchor -- the verbal expression it's subordinate to
    -- by following the anchor's own relatedtoken1 (falling back to
    relatedtoken2) until it lands on another anchor:

    - unit verb (independent): relatedtoken1 == 'root' -> no parent, depth 0.
    - direct quote: relatedtoken1 -> the verb of the clause that introduces
      the quotation, directly (no intermediate token to hop through, unlike
      arsgrammatica's Latin scheme, which has no equivalent of a
      subordinating-conjunction intermediary for this case either).

    Returns `({anchor id: depth or None}, warnings)`. A depth of `None`
    means the chase from that anchor never reached another anchor (a
    malformed or genuinely disconnected verbal expression) or a cycle was
    detected; `warnings` names which anchor(s) and why, rather than
    raising.
    """
    by_id = {tok.id: tok for tok in tokengraph}
    anchor_ids = {tok.id for tok in tokengraph if tok.verbalunitid == tok.id}

    warnings: List[str] = []

    def chase(token_id: str, visited: set) -> Optional[str]:
        """Follow relatedtoken1 (then relatedtoken2) forward from
        `token_id`, returning the first anchor id reached, or None if the
        chain dead-ends or cycles before reaching one. `token_id` itself
        counts as a hit if it's already an anchor (the direct 'direct
        quote' case)."""
        if token_id in visited:
            return None
        visited.add(token_id)
        if token_id in anchor_ids:
            return token_id
        tok = by_id.get(token_id)
        if tok is None:
            return None
        for field in ("relatedtoken1", "relatedtoken2"):
            target = getattr(tok, field)
            if target is None or target == "root":
                continue
            result = chase(target, visited)
            if result is not None:
                return result
        return None

    def parent_of(anchor_id: str) -> Optional[str]:
        tok = by_id[anchor_id]
        for field in ("relatedtoken1", "relatedtoken2"):
            target = getattr(tok, field)
            if target is None or target == "root":
                continue
            result = chase(target, visited=set())
            if result is not None and result != anchor_id:
                return result
        return None

    depths: Dict[str, Optional[int]] = {}
    in_progress: set = set()

    def depth_of(anchor_id: str) -> Optional[int]:
        if anchor_id in depths:
            return depths[anchor_id]
        tok = by_id[anchor_id]
        if tok.relatedtoken1 == "root":
            depths[anchor_id] = 0
            return 0

        if anchor_id in in_progress:
            warnings.append(
                f"cycle detected resolving the governing verbal expression "
                f"for {anchor_id!r} -- leaving its depth (and its parent's) "
                f"unresolved"
            )
            return None
        in_progress.add(anchor_id)

        parent = parent_of(anchor_id)
        if parent is None:
            warnings.append(
                f"could not find a governing verbal expression for "
                f"{anchor_id!r} -- leaving its depth unresolved"
            )
            result = None
        else:
            parent_depth = depth_of(parent)
            result = None if parent_depth is None else parent_depth + 1

        in_progress.discard(anchor_id)
        depths[anchor_id] = result
        return result

    for anchor_id in anchor_ids:
        depth_of(anchor_id)

    return depths, warnings


def max_subordination_depth(
    tokengraph: List[TokenAnalysis],
    depths: Optional[Dict[str, Optional[int]]] = None,
) -> Optional[int]:
    """Return the deepest level of subordination reached anywhere in
    `tokengraph` -- the highest value `compute_subordination_depths()`
    assigns to any verbal expression. Root/independent clauses are depth
    0, so this is also the upper end of the valid `depth` range for
    `rendering.tokengraph_to_depth_html()`'s own `depth` parameter.

    Pass `depths` (the first element of `compute_subordination_depths()`'s
    return value) if the caller already computed it, to avoid re-deriving
    it here.

    Returns `None` if `tokengraph` has no verbal expressions at all, or if
    every anchor's own depth came back unresolved. Otherwise returns the
    maximum of every RESOLVED anchor's depth, ignoring unresolved ones
    rather than letting a single bad anchor blank out the whole result.
    """
    if depths is None:
        depths, _warnings = compute_subordination_depths(tokengraph)

    resolved = [d for d in depths.values() if d is not None]
    if not resolved:
        return None
    return max(resolved)


def find_unanchored_coordinated_verbs(tokengraph: List[TokenAnalysis]) -> List[str]:
    """Heuristic sanity check for a specific, plausible live-LM mistake: a
    coordinating conjunction that pairs two verbal expressions (see
    hebrew_syntax_dspy.py's docstring) is supposed to leave BOTH conjuncts
    anchoring their own verbal unit -- each with its own `verbalunitid`
    (and its own `verbalunits` entry). This is NOT the same kind of check
    as validate() (referential id integrity) or
    compute_subordination_depths()'s warnings (a resolvable-but-broken
    relation graph) -- both of those only catch a problem if the
    tokengraph is already self-inconsistent. This function catches a
    tokengraph that's perfectly well-formed and internally consistent, but
    still probably WRONG, by looking for an asymmetry a correct analysis
    should never produce.

    The heuristic: find every "coordinating conjunction" token that uses
    BOTH relatedtoken1 and relatedtoken2 (the two-conjunct, single-pair
    case -- see that relation's own note about the repeated-connector,
    series exception, which this deliberately ignores below). For each
    such pair, if EXACTLY ONE of the two joined tokens is a recognized
    verbal-unit anchor (`verbalunitid` set to its own id) and the other is
    not, that asymmetry is flagged: if the conjunction is genuinely pairing
    two nouns/adjectives/prepositional phrases, NEITHER side would be an
    anchor; if it's correctly pairing two verbal expressions, BOTH sides
    would be.

    This pairwise shape doesn't apply to a repeated connector coordinating
    a series of two or more items via the id-chaining convention (see
    hebrew_syntax_dspy.py's docstring): there, every connector's own
    relatedtoken2 points at a NEIGHBORING CONNECTOR, not at a second
    conjunct, so this heuristic's asymmetry check would misfire. A pair is
    therefore skipped whenever relatedtoken2 resolves to a token that is
    itself a coordinating-conjunction connector.

    Returns a list of warning strings (empty if nothing looks suspicious).
    This is a heuristic, not a guarantee.
    """
    by_id = {tok.id: tok for tok in tokengraph}
    anchor_ids = {tok.id for tok in tokengraph if tok.verbalunitid == tok.id}

    warnings: List[str] = []
    seen_pairs = set()

    for tok in tokengraph:
        if not (
            tok.relatedtoken1 is not None
            and tok.relatedtoken1 != "root"
            and tok.relationship1 == "coordinating conjunction"
            and tok.relatedtoken2 is not None
            and tok.relatedtoken2 != "root"
            and tok.relationship2 == "coordinating conjunction"
        ):
            continue

        pair = (tok.relatedtoken1, tok.relatedtoken2)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        first_id, second_id = pair
        second_tok = by_id.get(second_id)
        if second_tok is not None and (
            second_tok.relationship1 == "coordinating conjunction"
            or second_tok.relationship2 == "coordinating conjunction"
        ):
            # relatedtoken2 points at a FELLOW connector, not at a second
            # conjunct -- the signature of the repeated-connector/series
            # pattern (see this function's own docstring), where this
            # heuristic's pairwise-specific asymmetry check doesn't apply.
            continue

        first_anchored = first_id in anchor_ids
        second_anchored = second_id in anchor_ids
        if first_anchored == second_anchored:
            continue

        anchored_id, unanchored_id = (
            (first_id, second_id) if first_anchored else (second_id, first_id)
        )
        anchored_text = by_id[anchored_id].token if anchored_id in by_id else anchored_id
        unanchored_text = by_id[unanchored_id].token if unanchored_id in by_id else unanchored_id
        warnings.append(
            f"{tok.id} ({tok.token!r}) coordinates {anchored_id} "
            f"({anchored_text!r}), which anchors its own verbal unit, with "
            f"{unanchored_id} ({unanchored_text!r}), which does not -- if "
            "this conjunction is meant to join two verbal expressions "
            "(rather than a noun/adjective/prepositional-phrase pair), "
            f"{unanchored_id} is likely missing its own verbalunitid and "
            "'unit verb'/'root' relation."
        )

    return warnings
