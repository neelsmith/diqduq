"""
Turn a real analysis -- the same `sentences`/`verbalunits`/`tokengraph`
triple serialize_analyses()/write_analyses() take (diqduq/serialization.py)
-- into a `GoldExample` (see gold_examples.py), instead of hand-writing a
`canned_answer` dict from scratch. Modeled directly on arsgrammatica's
tests/fixtures/harvest.py.

Intended workflow: run analyze_passage()/analyze_sources() over new
passages, read over each result by hand against syntax_model.md, and for
the ones that come out correct, call gold_example_from_analysis() on that
sentence's own `sentences`/`result.verbalunits`/`result.tokengraph` to get
a `GoldExample` -- then format_gold_example_source() renders it as
ready-to-paste Python source.
"""

from typing import List, Optional

import dspy

from diqduq.hebrew_syntax_dspy import validate
from diqduq.models import Sentence, TokenAnalysis, VerbalExpression
from diqduq.rendering import tokengraph_to_text

from .gold_examples import GoldExample

_PLACEHOLDER_REASONING = (
    "(reasoning not supplied to gold_example_from_analysis() -- fill in a "
    "real explanation of this analysis before treating this as more than "
    "a draft fixture)"
)


def _verbalunit_to_dict(vu: VerbalExpression) -> dict:
    """VerbalExpression has exactly 3 required fields, none optional -- so
    this is just an ordinary full dump."""
    return {
        "id": vu.id,
        "syntactic_type": vu.syntactic_type,
        "semantic_type": vu.semantic_type,
    }


def _tokenanalysis_to_dict(tok: TokenAnalysis) -> dict:
    """TokenAnalysis -> the raw-dict shape gold_examples.py's
    canned_answer['tokengraph'] entries use: 'id'/'token'/'tokentype' are
    always present -- including 'token': None for an implied token, which
    is a meaningful value, not an absent one -- but every other optional
    field (lemma, verbalunitid, relatedtoken1/2, relationship1/2) is
    OMITTED entirely when None, matching how every hand-written entry in
    gold_examples.py already looks."""
    entry = {"id": tok.id, "token": tok.token, "tokentype": tok.tokentype}
    for field in (
        "lemma",
        "verbalunitid",
        "relatedtoken1",
        "relationship1",
        "relatedtoken2",
        "relationship2",
    ):
        value = getattr(tok, field)
        if value is not None:
            entry[field] = value
    return entry


def gold_example_from_analysis(
    slug: str,
    tags: List[str],
    sentences: List[Sentence],
    verbalunits: List[VerbalExpression],
    tokengraph: List[TokenAnalysis],
    *,
    passage: Optional[str] = None,
    reasoning: Optional[str] = None,
    skip_validation: bool = False,
) -> GoldExample:
    """Build a GoldExample from a real analysis's own objects -- the exact
    `sentences`/`verbalunits`/`tokengraph` triple serialize_analyses()/
    write_analyses() take, as produced by analyze_passage()/
    analyze_sources() (+ combined_tokengraph(), and concatenating every
    result.verbalunits, if the passage was more than one sentence).

    `slug`/`tags` are yours to choose -- see gold_examples.py's own
    convention.

    `passage` defaults to reconstructing the surface text directly from
    `tokengraph` via rendering.tokengraph_to_text() -- pass an explicit
    string instead if you'd rather preserve the exact original
    wording/whitespace you fed to analyze_passage().

    `reasoning` defaults to an obvious placeholder string if omitted.

    Unless `skip_validation` is True, this calls hebrew_syntax_dspy.
    validate() against every token in `sentences` before returning, and
    raises ValueError if it finds anything wrong. This does NOT check the
    analysis is *correct* (validate() never does); that's still on you to
    judge before calling this at all.

    `sentences` is used only for that validation and (when `passage` is
    omitted) is not otherwise consulted -- this function does NOT split a
    multi-sentence `sentences` list into multiple GoldExamples.
    """
    tokengraph = list(tokengraph)
    verbalunits = list(verbalunits)

    if not tokengraph:
        raise ValueError("gold_example_from_analysis() needs a non-empty tokengraph")

    if not skip_validation:
        all_tokens = [tok for sentence in sentences for tok in sentence.tokens]
        problems = validate(
            all_tokens, dspy.Prediction(tokengraph=tokengraph, verbalunits=verbalunits)
        )
        if problems:
            raise ValueError(
                "gold_example_from_analysis(): validate() found problems with "
                "this analysis -- fix them (or pass skip_validation=True if "
                f"you really mean to harvest it anyway): {problems}"
            )

    resolved_passage = passage if passage is not None else tokengraph_to_text(tokengraph)

    canned_answer = {
        "reasoning": reasoning if reasoning is not None else _PLACEHOLDER_REASONING,
        "verbalunits": [_verbalunit_to_dict(vu) for vu in verbalunits],
        "tokengraph": [_tokenanalysis_to_dict(tok) for tok in tokengraph],
    }

    return GoldExample(
        slug=slug, passage=resolved_passage, tags=list(tags), canned_answer=canned_answer
    )


def _pylit(value) -> str:
    """A Python source literal for `value` -- prefers double quotes for
    strings (matching gold_examples.py's own style), falling back to
    single quotes or repr() only if the string itself needs it."""
    if value is None:
        return "None"
    if isinstance(value, str):
        if '"' not in value:
            return f'"{value}"'
        if "'" not in value:
            return f"'{value}'"
        return repr(value)
    return repr(value)


def _dict_literal(d: dict) -> str:
    return "{" + ", ".join(f"{_pylit(k)}: {_pylit(v)}" for k, v in d.items()) + "}"


def _list_literal(items) -> str:
    return "[" + ", ".join(_pylit(item) for item in items) + "]"


def format_gold_example_source(example: GoldExample, answer_var_name: str) -> str:
    """Render `example` as ready-to-paste Python source: a `_SOME_ANSWER =
    {...}` module-level dict literal, followed by the `GoldExample(...)`
    entry that references it -- the same two-part shape every existing
    block in gold_examples.py already uses.

    `answer_var_name` is the module-level constant name to give the
    canned_answer dict (by convention, ALL_CAPS ending in `_ANSWER`).
    """
    answer_lines = [f"{answer_var_name} = {{"]
    answer_lines.append(f'    "reasoning": {_pylit(example.canned_answer["reasoning"])},')
    answer_lines.append('    "verbalunits": [')
    for vu in example.canned_answer["verbalunits"]:
        answer_lines.append(f"        {_dict_literal(vu)},")
    answer_lines.append("    ],")
    answer_lines.append('    "tokengraph": [')
    for tok in example.canned_answer["tokengraph"]:
        answer_lines.append(f"        {_dict_literal(tok)},")
    answer_lines.append("    ],")
    answer_lines.append("}")

    example_lines = [
        "GoldExample(",
        f"    slug={_pylit(example.slug)},",
        f"    passage={_pylit(example.passage)},",
        f"    tags={_list_literal(example.tags)},",
        f"    canned_answer={answer_var_name},",
        "),",
    ]

    return "\n".join(answer_lines) + "\n\n\n" + "\n".join(example_lines) + "\n"
