"""
Render a `tokengraph` (a list of TokenAnalysis, as produced by
hebrew_syntax_dspy.analyze/pipeline.py) as a Mermaid flowchart. Modeled
directly on arsgrammatica's mermaid.py.

- Every substantive token (tokentype not in models.NON_SUBSTANTIVE_TOKENTYPES
  -- i.e. not cantillation, paragraph, editorial, or maqaf) becomes a node,
  labelled with the token's surface text.
- Every `relatedtoken1`/`relationship1` and `relatedtoken2`/`relationship2`
  pair on a token becomes a labelled edge from that token to the related
  token.
- By default (`color_by_verbal_unit=True`), every token is also colored by
  the verbal unit it belongs to (see verbal_units.py's assign_verbal_units())
  -- so the clauses a sentence breaks into are visually distinguishable at a
  glance. The one exception: an implied/elided token (models.py's
  IMPLIED_TOKENTYPES -- currently just "implied sum") always gets a
  dedicated "caution" amber (verbal_units._IMPLIED_TOKEN_COLOR) instead. Its
  label is "elided sum" (see `_IMPLIED_TOKEN_LABELS` below) rather than its
  own surface text, since it has none (`token` is `None`). It's also drawn
  as a rounded-corner rectangle (Mermaid's `(...)` node shape) rather than
  the plain `[...]` rectangle every other node uses. This is the ONE place
  these tokens are shown at all -- rendering.py's tokengraph_to_html()/
  tokengraph_to_depth_html() omit them entirely, same as
  tokengraph_to_text() does, since there's no real word to display in
  reconstructed prose.
- By default (`rank_by_depth=True`), every verbal-unit anchor node is also
  chained to every other anchor at the SAME depth of subordination (see
  verbal_units.py's compute_subordination_depths()) using Mermaid's
  invisible-link syntax (`~~~`) -- a layout nudge only, no visible edge and
  no new relation.

Non-substantive tokens (cantillation, paragraph, editorial, maqaf) are
dropped as nodes. Any edge that would point at a dropped or unrecognized
token id is skipped rather than emitted as a broken reference, and reported
back to the caller -- except the special sentinel target 'root' (an
independent verb's own relatedtoken1), which is skipped silently.
"""

from typing import List, Tuple

from .models import IMPLIED_TOKENTYPES, NON_SUBSTANTIVE_TOKENTYPES, TokenAnalysis
from .verbal_units import (
    _IMPLIED_TOKEN_COLOR,
    assign_verbal_units,
    assign_verbal_unit_colors,
    compute_subordination_depths,
)

_LABEL_ESCAPES = {
    '"': "&quot;",
    "<": "&lt;",
    ">": "&gt;",
}


def _escape_label(text: str) -> str:
    for char, replacement in _LABEL_ESCAPES.items():
        text = text.replace(char, replacement)
    return text


# The Mermaid node label for an implied/elided token (keyed by its own
# tokentype, since it has no surface text of its own to use instead).
_IMPLIED_TOKEN_LABELS = {
    "implied sum": "elided sum",
}


def tokengraph_to_mermaid(
    tokengraph: List[TokenAnalysis],
    orientation: str = "BT",
    color_by_verbal_unit: bool = True,
    rank_by_depth: bool = True,
) -> Tuple[str, List[str]]:
    """Build a Mermaid `graph` diagram from a tokengraph.

    `orientation` is Mermaid's own flowchart orientation code -- `BT`
    (bottom-to-top, the default here), `TB`, `LR`, or `RL` -- used verbatim
    in the diagram's opening line (`graph BT`, `graph LR`, etc.). See
    https://mermaid.js.org/syntax/flowchart.html for what each value looks
    like.

    `color_by_verbal_unit` (default True) colors every node by the verbal
    unit it belongs to, per verbal_units.assign_verbal_units(). Pass False
    to skip coloring and get a plain diagram.

    `rank_by_depth` (default True) makes the diagram's layout respect each
    verbal expression's own *depth of subordination* (see
    verbal_units.compute_subordination_depths()). Pass False to skip this
    and get the diagram's previous, unranked layout.

    Returns (diagram_text, warnings). `warnings` lists any edges that were
    skipped because they referenced a non-substantive token or an id not
    present in `tokengraph`, plus, if `color_by_verbal_unit` is True and the
    passage has more than 8 verbal units, one warning that colors are
    repeating, plus, if `rank_by_depth` is True, any of
    compute_subordination_depths()'s own warnings.
    """
    node_ids = {tok.id for tok in tokengraph if tok.tokentype not in NON_SUBSTANTIVE_TOKENTYPES}

    lines = [f"graph {orientation}"]
    for tok in tokengraph:
        if tok.id not in node_ids:
            continue
        label = (
            tok.token
            if tok.token is not None
            else _IMPLIED_TOKEN_LABELS.get(tok.tokentype, tok.tokentype)
        )
        open_bracket, close_bracket = (
            ("(", ")") if tok.tokentype in IMPLIED_TOKENTYPES else ("[", "]")
        )
        lines.append(f'    {tok.id}{open_bracket}"{_escape_label(label)}"{close_bracket}')

    warnings = []
    for tok in tokengraph:
        if tok.id not in node_ids:
            continue
        for related_field, label_field in (
            ("relatedtoken1", "relationship1"),
            ("relatedtoken2", "relationship2"),
        ):
            related_id = getattr(tok, related_field)
            label = getattr(tok, label_field)
            if related_id is None or label is None:
                continue
            if related_id == "root":
                continue
            if related_id not in node_ids:
                warnings.append(
                    f"skipped edge {tok.id} -[{label}]-> {related_id}: "
                    f"target is non-substantive or not in tokengraph"
                )
                continue
            lines.append(f'    {tok.id} -->|{_escape_label(label)}| {related_id}')

    if rank_by_depth:
        depths, depth_warnings = compute_subordination_depths(tokengraph)
        warnings.extend(depth_warnings)

        depth_groups: dict = {}
        for tok in tokengraph:
            if tok.id not in node_ids:
                continue
            depth = depths.get(tok.id)
            if depth is None:
                continue
            depth_groups.setdefault(depth, []).append(tok.id)

        rank_lines = [
            "    " + " ~~~ ".join(ids)
            for depth in sorted(depth_groups)
            for ids in (depth_groups[depth],)
            if len(ids) > 1
        ]
        if rank_lines:
            lines.append("")
            lines.extend(rank_lines)

    if color_by_verbal_unit:
        assignment = assign_verbal_units(tokengraph)
        colors, color_warnings = assign_verbal_unit_colors(tokengraph, assignment=assignment)
        warnings.extend(color_warnings)

        implied_ids = [
            tok.id
            for tok in tokengraph
            if tok.id in node_ids and tok.tokentype in IMPLIED_TOKENTYPES
        ]

        if colors or implied_ids:
            lines.append("")
            class_names = {}
            for i, (unit_id, (fill, stroke, text)) in enumerate(colors.items()):
                class_name = f"vu{i}"
                class_names[unit_id] = class_name
                lines.append(
                    f"    classDef {class_name} fill:{fill},stroke:{stroke},color:{text};"
                )
            for unit_id in colors:
                member_ids = [
                    tok.id
                    for tok in tokengraph
                    if tok.id in node_ids
                    and assignment.get(tok.id) == unit_id
                    and tok.id not in implied_ids
                ]
                if member_ids:
                    lines.append(f"    class {','.join(member_ids)} {class_names[unit_id]};")
            if implied_ids:
                fill, stroke, text = _IMPLIED_TOKEN_COLOR
                lines.append(
                    f"    classDef implied fill:{fill},stroke:{stroke},color:{text};"
                )
                lines.append(f"    class {','.join(implied_ids)} implied;")

    return "\n".join(lines), warnings


def save_mermaid(
    tokengraph: List[TokenAnalysis],
    path: str,
    orientation: str = "BT",
    color_by_verbal_unit: bool = True,
    rank_by_depth: bool = True,
) -> List[str]:
    """Write the diagram to `path` (e.g. 'analysis.mmd') and return any
    warnings from tokengraph_to_mermaid."""
    diagram, warnings = tokengraph_to_mermaid(
        tokengraph,
        orientation=orientation,
        color_by_verbal_unit=color_by_verbal_unit,
        rank_by_depth=rank_by_depth,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(diagram + "\n")
    return warnings
