"""
Estimating and enforcing a `max_tokens` output budget for both LM stages of
the pipeline -- SyntaxAnalysis (per sentence) and SegmentPassage (per
passage) -- and retrying with a larger one when a call actually gets
truncated. Modeled directly on arsgrammatica's token_budget.py, which only
covers the analysis stage; the segmentation half below (`segment_with_retry()`
and friends) is a deliberate extension beyond what arsgrammatica does, added
because `segmentation_dspy.segment()` shipped with NO budget management at
all -- neither a calibrated estimate nor a retry -- so a `dspy.LM` configured
without an explicit `max_tokens` (see diqduq_main.py's `_configure_lm()`,
which never sets one) fell through to the provider's own default and could
truncate silently with `max_tokens=None` in the resulting warning. See
"Segmentation budget" below for the details.

Background: `hebrew_syntax_dspy.analyze` (a `dspy.ChainOfThought`) produces
a free-text `reasoning` field plus JSON-serialized `verbalunits` and
`tokengraph` lists -- one `TokenAnalysis` entry per input token, plus extra
entries for implied/elided tokens. That output's size scales with how long
and how syntactically complex the passage is, not with a fixed constant, so
any single hard-coded `max_tokens` value is eventually wrong for either a
short passage (wastes budget) or a long/complex one (truncates mid-output).

This module takes the same hybrid approach as arsgrammatica for the
analysis stage:

1. `estimate_max_tokens()` picks a per-call budget from a simple linear
   model (`completion_tokens ~= intercept + slope * num_input_tokens`),
   with a safety margin on top. Until `calibrate_max_tokens.py` (repo root)
   has been run against your own configured model, this falls back to a
   conservative, deliberately-generous untuned fit (`_FALLBACK_INTERCEPT`/
   `_FALLBACK_SLOPE` below) -- see `_load_calibration()`, and USAGE.md's
   "Estimating and enforcing a `max_tokens` budget".
2. `analyze_with_retry()` wraps `analyze()` and, if a call still comes back
   truncated despite that estimate, retries with a larger budget rather
   than silently returning an incomplete result or leaving the caller to
   guess a bigger number by hand.

Note that neither of the above -- nor the segmentation stage's own budget
below -- can protect against a *provider* rejecting a request outright for
requesting more `max_tokens` than that model actually allows (a "tokens
exceeded allowed length"-type error, as opposed to a truncated response);
that failure mode is about `ceiling` being set too high for your specific
model, not about the estimate being wrong. See USAGE.md's note on
`DEFAULT_CEILING` for that case -- pass a smaller `ceiling=` explicitly if
your model's real limit is below the default.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import List, Optional

import dspy
from dspy.utils.exceptions import AdapterParseError

from .hebrew_syntax_dspy import analyze
from .models import CitedText, Sentence, Token
from .segmentation_dspy import segment

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

# A future calibration script would write its fitted (intercept, slope)
# here -- kept next to this module (not under tests/) since it's runtime
# configuration, not test fixture data.
CALIBRATION_FILE = Path(__file__).with_name("token_budget_calibration.json")

# Untuned stand-ins, used unless/until a calibration file has been written
# for the model you've actually configured. Deliberately generous (60
# output tokens per input token, plus a 500-token allowance for the
# reasoning field and the verbalunits list) -- an overestimate here just
# spends a bit more of the model's output budget than necessary; an
# underestimate is what causes the truncation this module exists to avoid.
_FALLBACK_INTERCEPT = 500.0
_FALLBACK_SLOPE = 60.0

DEFAULT_SAFETY_MARGIN = 1.4
DEFAULT_FLOOR = 256
# Stand-in for "this model's real max output tokens". There's no single
# correct value across providers/models -- override this with whatever your
# configured MODEL actually allows (check its provider's documentation)
# rather than relying on this default for anything but a rough starting
# point.
DEFAULT_CEILING = 8192


def _load_calibration() -> dict:
    """Read a saved (intercept, slope) fit from CALIBRATION_FILE, if any.

    Returns a dict with at least "intercept", "slope", and "source" keys.
    "source" is "calibrated" when CALIBRATION_FILE was read successfully,
    or "fallback" when it's missing, unreadable, or malformed -- callers
    that want to know which one is active (or tests that want to force the
    fallback) can check that field rather than re-deriving it.
    """
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "intercept": float(data["intercept"]),
            "slope": float(data["slope"]),
            "source": "calibrated",
            "sample_size": data.get("sample_size"),
            "model": data.get("model"),
            "calibrated_at": data.get("calibrated_at"),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "intercept": _FALLBACK_INTERCEPT,
            "slope": _FALLBACK_SLOPE,
            "source": "fallback",
            "sample_size": None,
            "model": None,
            "calibrated_at": None,
        }


def get_calibration() -> dict:
    """Public introspection: what (intercept, slope) is estimate_max_tokens()
    currently using, and did it come from a real calibration fit or from
    this module's untuned fallback? See _load_calibration()'s docstring for
    the shape returned."""
    return _load_calibration()


def estimate_max_tokens(
    num_tokens: int,
    *,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    floor: int = DEFAULT_FLOOR,
    ceiling: int = DEFAULT_CEILING,
) -> int:
    """Estimate a `max_tokens` budget for a SyntaxAnalysis call over a
    sentence with `num_tokens` input tokens.

    `raw = intercept + slope * num_tokens` comes from the calibrated (or
    fallback) linear fit (see _load_calibration()); `safety_margin`
    multiplies that to leave room for the reasoning field's length being
    only roughly, not exactly, a function of passage length. The result is
    clamped to `[floor, ceiling]` -- `floor` guards against a degenerate
    tiny estimate for a 1-2 token sentence, `ceiling` is a hard cap you
    should set to your actual model's real max-output-tokens limit.

    Raises ValueError if `num_tokens` is negative.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

    calibration = _load_calibration()
    raw = calibration["intercept"] + calibration["slope"] * num_tokens
    budget = math.ceil(raw * safety_margin)
    return max(floor, min(ceiling, budget))


# ---------------------------------------------------------------------------
# Retry-on-truncation wrapper
# ---------------------------------------------------------------------------


def _finish_reason_was_length() -> bool:
    """Best-effort check of whether the most recent call made against the
    currently configured LM was cut off for hitting max_tokens, via the
    same `finish_reason == "length"` signal dspy.LM._check_truncation()
    itself warns on.

    This is a *secondary* corroborating signal, not the primary detector --
    see analyze_with_retry()'s docstring for why -- so it fails safe: any
    missing attribute, empty history, or non-dspy.LM configured LM (e.g.
    DummyLM in tests, which doesn't populate `.history` the same way)
    just returns False rather than raising.
    """
    try:
        lm = dspy.settings.lm
        entry = lm.history[-1]
        response = entry["response"]
        return any(getattr(c, "finish_reason", None) == "length" for c in response["choices"])
    except (AttributeError, IndexError, KeyError, TypeError):
        return False


def _missing_token_ids(tokens: List[Token], result) -> set:
    """Which of `tokens`' own ids never showed up in `result.tokengraph`.

    This is the primary truncation signal: unlike finish_reason (which
    depends on the LM class actually surfacing it, and isn't exercised at
    all by DummyLM-backed tests), a real truncation -- the model's
    per-token loop getting cut off partway through -- always shows up here,
    regardless of provider, and it's checked even when the JSON still
    happened to parse successfully. A non-empty result means the analysis
    is missing coverage for at least one real input token, which validate()
    (hebrew_syntax_dspy.py's own referential check) doesn't itself catch --
    validate() flags ids that shouldn't exist, not ids that should have.
    """
    seen_ids = {tok.id for tok in result.tokengraph}
    return {t.id for t in tokens} - seen_ids


def analyze_with_retry(
    passage: str,
    tokens: List[Token],
    *,
    max_retries: int = 1,
    growth_factor: float = 2.0,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    floor: int = DEFAULT_FLOOR,
    ceiling: int = DEFAULT_CEILING,
    initial_max_tokens: Optional[int] = None,
):
    """Call `analyze()`, detecting truncation and retrying with a larger
    `max_tokens` budget instead of either crashing or silently returning an
    incomplete result.

    The starting budget is `initial_max_tokens` if given, else
    `estimate_max_tokens(len(tokens), safety_margin=safety_margin,
    floor=floor, ceiling=ceiling)`.

    After each attempt, truncation is checked two ways: `_missing_token_ids`
    against the result (the primary, LM-independent signal -- works
    whenever a result exists at all, parsed or not, including under
    DummyLM in tests) and, if the call raised `AdapterParseError` instead
    of returning a result (the JSON was cut off badly enough to not parse
    at all), `_finish_reason_was_length()` as a corroborating check before
    deciding a retry is even worth trying -- a parse failure that ISN'T a
    length truncation is a real formatting bug a bigger budget won't fix,
    so it's re-raised immediately rather than retried.

    If truncation is detected and there's still a retry available (fewer
    than `max_retries` attempts so far, and the budget hasn't already hit
    `ceiling`), the budget is multiplied by `growth_factor` (capped at
    `ceiling`) and the call is retried. `max_tokens` is part of DSPy's own
    LM cache key, so a retry with a different budget always reaches the LM
    again rather than replaying a cached truncated response.

    Once retries are exhausted: if the last attempt raised, that exception
    propagates. If the last attempt returned a still-incomplete result,
    it's returned anyway -- with a `UserWarning` naming the missing token
    ids -- rather than raising, matching this codebase's convention of
    surfacing analysis problems as warnings rather than treating an
    imperfect LM result as fatal.
    """
    budget = initial_max_tokens if initial_max_tokens is not None else estimate_max_tokens(
        len(tokens), safety_margin=safety_margin, floor=floor, ceiling=ceiling
    )

    attempt = 0
    while True:
        old_budget = budget
        try:
            result = analyze(passage=passage, tokens=tokens, config={"max_tokens": budget})
        except AdapterParseError:
            if attempt < max_retries and budget < ceiling and _finish_reason_was_length():
                attempt += 1
                budget = min(ceiling, math.ceil(budget * growth_factor))
                warnings.warn(
                    f"SyntaxAnalysis call truncated at max_tokens={old_budget} before it "
                    f"could be parsed at all; retrying with max_tokens={budget} "
                    f"(attempt {attempt}/{max_retries}).",
                    stacklevel=2,
                )
                continue
            raise

        missing = _missing_token_ids(tokens, result)
        truncated = bool(missing) or _finish_reason_was_length()
        if truncated and attempt < max_retries and budget < ceiling:
            attempt += 1
            budget = min(ceiling, math.ceil(budget * growth_factor))
            warnings.warn(
                f"SyntaxAnalysis call at max_tokens={old_budget} returned a tokengraph "
                f"missing {len(missing)} input token id(s) ({sorted(missing)}); retrying "
                f"with a larger max_tokens={budget} (attempt {attempt}/{max_retries}).",
                stacklevel=2,
            )
            continue

        if truncated:
            missing_desc = sorted(missing) if missing else "(finish_reason indicated truncation, but no ids are directly missing)"
            warnings.warn(
                f"SyntaxAnalysis call still looks truncated after {attempt} retry(ies) "
                f"(max_tokens={old_budget}) -- returning it anyway. Missing input token "
                f"id(s): {missing_desc}.",
                stacklevel=2,
            )

        return result


# ---------------------------------------------------------------------------
# Segmentation budget
# ---------------------------------------------------------------------------

# Deliberately generous, deliberately UNCALIBRATED constants for the
# segmentation stage -- unlike SyntaxAnalysis's own fit, there is no
# calibrate_max_tokens.py-style script measuring these against a real model
# yet, because segmentation's output (a List[Sentence], each with its own
# List[Token]) doesn't have a natural "num_input_tokens" to fit against the
# way SyntaxAnalysis's tokengraph does -- segmentation is what PRODUCES
# token counts, not something that already has them. Input character count
# (summed across every CitedText in `sources`) is used as the proxy instead:
# cheap to compute up front, and roughly proportional to how much output
# segmentation has to produce (every input character ends up inside some
# token's own surface text, an implied/paragraph/cantillation entry, or is
# whitespace between tokens). If real usage shows this proxy is a poor fit,
# the right long-term fix is a calibration script analogous to
# calibrate_max_tokens.py's, fit specifically against character count
# instead of token count -- flagged here as a known simplification.
#
# _SEGMENTATION_FALLBACK_INTERCEPT started at 500.0 and proved too low in
# real use: a real ~60-character passage needed more than 1652 completion
# tokens (826 initial estimate, doubled once) before segmentation actually
# succeeded, meaning the `reasoning` field's own free-text length dominates
# for short passages far more than a 500-token baseline assumed. Raised to
# 2000.0 -- still a guess, not a real fit, but a guess grounded in an actual
# observed shortfall rather than an arbitrary starting point.
_SEGMENTATION_FALLBACK_INTERCEPT = 2000.0
_SEGMENTATION_FALLBACK_CHARS_PER_COMPLETION_TOKEN = 1.5

DEFAULT_SEGMENTATION_SAFETY_MARGIN = 1.4
DEFAULT_SEGMENTATION_FLOOR = 512
# Same placeholder-ceiling caveat as DEFAULT_CEILING above: override with
# your actual model's real max output tokens once you know it.
DEFAULT_SEGMENTATION_CEILING = DEFAULT_CEILING


def estimate_segmentation_max_tokens(
    sources: List[CitedText],
    *,
    safety_margin: float = DEFAULT_SEGMENTATION_SAFETY_MARGIN,
    floor: int = DEFAULT_SEGMENTATION_FLOOR,
    ceiling: int = DEFAULT_SEGMENTATION_CEILING,
) -> int:
    """Estimate a `max_tokens` budget for a SegmentPassage call over
    `sources`, from their combined input character count (see this
    section's own module-level comment for why character count, and why
    this is an uncalibrated proxy rather than a real fit).

    `raw = _SEGMENTATION_FALLBACK_INTERCEPT +
    _SEGMENTATION_FALLBACK_CHARS_PER_COMPLETION_TOKEN * num_chars`;
    `safety_margin` multiplies that, and the result is clamped to
    `[floor, ceiling]` -- same shape as estimate_max_tokens(), see its
    docstring for what each parameter guards against.
    """
    num_chars = sum(len(source.text) for source in sources)
    raw = _SEGMENTATION_FALLBACK_INTERCEPT + _SEGMENTATION_FALLBACK_CHARS_PER_COMPLETION_TOKEN * num_chars
    budget = math.ceil(raw * safety_margin)
    return max(floor, min(ceiling, budget))


def _segmentation_undercoverage(sources: List[CitedText], sentences: List[Sentence]) -> bool:
    """Primary, provider-independent truncation signal for segmentation,
    playing the same role `_missing_token_ids()` plays for analyze_with_retry():
    segmentation only ever SPLITS the input text into tokens -- it never
    drops characters -- so a complete result's tokens' own surface text,
    concatenated, should cover close to the full combined length of
    `sources`. A real truncation (the model's per-sentence/per-token loop
    cut off partway through) shows up here as a sharply undersized result,
    regardless of provider, even when the JSON still happened to parse.

    The 0.6 threshold is deliberately generous (not a tight round-trip
    check): tokenization legitimately drops the whitespace between words,
    and any given passage may have several sentences the model does render
    completely alongside a truncated tail, so this only needs to catch a
    result that's obviously, substantially incomplete -- not to verify
    exact spacing-and-content fidelity, which is rendering.py's job on the
    POST-analysis tokengraph, not raw pre-analysis tokens.
    """
    total_input_chars = sum(len(source.text) for source in sources)
    if total_input_chars == 0:
        return False
    total_output_chars = sum(
        len(tok.text) for sentence in sentences for tok in sentence.tokens if tok.text
    )
    return total_output_chars < total_input_chars * 0.6


def segment_with_retry(
    sources: List[CitedText],
    *,
    max_retries: int = 3,
    growth_factor: float = 2.0,
    safety_margin: float = DEFAULT_SEGMENTATION_SAFETY_MARGIN,
    floor: int = DEFAULT_SEGMENTATION_FLOOR,
    ceiling: int = DEFAULT_SEGMENTATION_CEILING,
    initial_max_tokens: Optional[int] = None,
) -> List[Sentence]:
    """Call `segmentation_dspy.segment()`, detecting truncation and
    retrying with a larger `max_tokens` budget instead of either crashing
    or silently returning an incomplete result -- the segmentation-stage
    counterpart to `analyze_with_retry()` above (see this module's own
    docstring for why segmentation needed this at all).

    `max_retries` defaults higher here than `analyze_with_retry()`'s `1`
    (three doublings from a truncated initial estimate reaches roughly
    8x that estimate before giving up) specifically because
    estimate_segmentation_max_tokens()'s budget is an uncalibrated
    character-count proxy, not a real fit the way estimate_max_tokens()'s
    is once calibrate_max_tokens.py has been run -- a rough guess deserves
    more retry headroom to self-correct, cheaply, rather than giving up
    after one doubling the way a properly-calibrated estimate can afford
    to.

    The starting budget is `initial_max_tokens` if given, else
    `estimate_segmentation_max_tokens(sources, safety_margin=safety_margin,
    floor=floor, ceiling=ceiling)`.

    After each attempt, truncation is checked two ways:
    `_segmentation_undercoverage()` against the result (the primary,
    LM-independent signal -- works whenever a result exists at all, parsed
    or not) and, if the call raised `AdapterParseError` instead of
    returning a result, `_finish_reason_was_length()` as a corroborating
    check before deciding a retry is worth trying -- same split as
    `analyze_with_retry()`'s own two-signal design; see its docstring for
    why a non-length parse failure is re-raised immediately rather than
    retried.

    If truncation is detected and a retry is still available (fewer than
    `max_retries` attempts so far, and the budget hasn't already hit
    `ceiling`), the budget is multiplied by `growth_factor` (capped at
    `ceiling`) and segmentation is retried. Once retries are exhausted: a
    raised exception propagates; an incomplete result is returned anyway,
    with a `UserWarning`, rather than treated as fatal -- matching
    `analyze_with_retry()`'s own warn-don't-raise convention.
    """
    budget = initial_max_tokens if initial_max_tokens is not None else estimate_segmentation_max_tokens(
        sources, safety_margin=safety_margin, floor=floor, ceiling=ceiling
    )

    attempt = 0
    while True:
        old_budget = budget
        try:
            result = segment(sources=sources, config={"max_tokens": budget})
        except AdapterParseError:
            if attempt < max_retries and budget < ceiling and _finish_reason_was_length():
                attempt += 1
                budget = min(ceiling, math.ceil(budget * growth_factor))
                warnings.warn(
                    f"SegmentPassage call truncated at max_tokens={old_budget} before it "
                    f"could be parsed at all; retrying with max_tokens={budget} "
                    f"(attempt {attempt}/{max_retries}).",
                    stacklevel=2,
                )
                continue
            raise

        sentences = result.sentences
        truncated = _segmentation_undercoverage(sources, sentences) or _finish_reason_was_length()
        if truncated and attempt < max_retries and budget < ceiling:
            attempt += 1
            budget = min(ceiling, math.ceil(budget * growth_factor))
            warnings.warn(
                f"SegmentPassage call at max_tokens={old_budget} returned a result that "
                f"looks incomplete; retrying with a larger max_tokens={budget} "
                f"(attempt {attempt}/{max_retries}).",
                stacklevel=2,
            )
            continue

        if truncated:
            warnings.warn(
                f"SegmentPassage call still looks truncated after {attempt} retry(ies) "
                f"(max_tokens={old_budget}) -- returning it anyway. If this keeps "
                "happening, call diqduq.segment_with_retry() directly with a larger "
                "initial_max_tokens/ceiling instead of going through analyze_sources()/"
                "analyze_passage()'s defaults.",
                stacklevel=2,
            )

        return sentences
