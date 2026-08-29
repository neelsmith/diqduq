"""
Orchestrates the two-stage pipeline: segmentation_dspy.py's citation-aware
sentence/token segmentation, feeding hebrew_syntax_dspy.py's unmodified
SyntaxAnalysis one sentence at a time.

Kept as its own module, separate from both stages, so neither stage needs
to know the other exists -- segmentation_dspy.py doesn't import
hebrew_syntax_dspy.py or vice versa. This is the only place that does.
Modeled directly on arsgrammatica's pipeline.py.
"""

from typing import List, Tuple

from .models import CitedText, Sentence
from .hebrew_syntax_dspy import validate
from .token_budget import analyze_with_retry, segment_with_retry


def _render_sentence_text(sentence: Sentence) -> str:
    """Reconstruct a surface string for a sentence from its tokens, to pass
    as SyntaxAnalysis's `passage` field.

    This is an approximation, not a faithful re-rendering: it puts a space
    before every token, including maqaf, proclitic conjunctions, enclitic
    pronouns, and cantillation marks. SyntaxAnalysis uses `passage` for
    readability alongside the authoritative `tokens` list, not for anything
    validate() checks, so exact fidelity isn't required -- use
    rendering.tokengraph_to_text() (post-analysis, once tokentype is known)
    for a faithfully-spaced reconstruction instead."""
    return " ".join(tok.text for tok in sentence.tokens)


def analyze_sources(sources: List[CitedText]) -> Tuple[List[Sentence], list]:
    """Segment `sources` into citation-aware sentences, run each sentence's
    tokens through SyntaxAnalysis, and validate each result.

    Returns (sentences, results): results[i] is the SyntaxAnalysis result
    for sentences[i], same order, one entry per sentence.

    Segmentation itself goes through `token_budget.segment_with_retry()`
    rather than calling `segmentation_dspy.segment()` directly, and each
    sentence's SyntaxAnalysis call goes through
    `token_budget.analyze_with_retry()` rather than calling `analyze()`
    directly -- both stages get an estimated, appropriately-sized `max_tokens`
    budget up front, and a retry with a larger one if either still comes
    back truncated -- see token_budget.py's module docstring for the full
    design (and for why segmentation needed this covered explicitly, unlike
    arsgrammatica's own pipeline.py).
    """
    sentences = segment_with_retry(sources)

    results = []
    for sentence in sentences:
        result = analyze_with_retry(passage=_render_sentence_text(sentence), tokens=sentence.tokens)

        problems = validate(sentence.tokens, result)
        if problems:
            first_id = sentence.tokens[0].id if sentence.tokens else "?"
            print(f"Validation warnings (sentence starting at {first_id}):")
            for p in problems:
                print(f"  - {p}")

        results.append(result)

    return sentences, results


def combined_tokengraph(results) -> list:
    """Concatenate every sentence result's tokengraph, in order, into one
    flat list spanning the whole input -- since token ids are global,
    tokengraph_to_mermaid() (mermaid.py) needs no changes at all to render
    this as one diagram for a multi-sentence, multi-citation passage."""
    combined = []
    for result in results:
        combined.extend(result.tokengraph)
    return combined


def analyze_passage(passage: str, citation: str = "") -> Tuple[List[Sentence], list]:
    """Convenience wrapper for the common case of a single string rather
    than a list of citation-labeled CitedText sources -- kept here so
    callers (diqduq_main.py) have a one-string entry point rather than
    needing to build a CitedText list themselves for the ordinary case of
    one passage from one source.

    Wraps `passage` as one CitedText (using `citation` if given, else an
    empty string -- fine for callers that don't track citations) and runs
    it through analyze_sources(). Returns (sentences, results) -- one entry
    per sentence segmentation finds in `passage`, in order (typically one
    per verse -- see segmentation_dspy.SegmentPassage's docstring).
    """
    return analyze_sources([CitedText(citation=citation, text=passage)])
