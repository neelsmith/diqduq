"""
A runnable command-line script to analyze a passage of Biblical Hebrew with
`diqduq`. Modeled directly on arsgrammatica's syntaxer_main.py.
"""

# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------
import argparse
from pathlib import Path
import os

import dspy
from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def _env(name: str, fallback_name: str, default: "str | None" = None) -> "str | None":
    value = os.getenv(name)
    if value:
        return value
    value = os.getenv(fallback_name)
    if value:
        return value
    return default


#: Fallback `max_tokens` for the `dspy.LM` itself when `.env` doesn't set
#: MAX_TOKENS -- see _configure_lm()'s own comment for why this exists at
#: all: leaving it unset (dspy.LM's own default) is `None`, and dspy's
#: truncation warning ("LM response was truncated due to exceeding
#: max_tokens=...") always reports the LM's OWN baseline value here, not
#: whatever value a per-call `config={"max_tokens": ...}` override (as
#: analyze_with_retry()/segment_with_retry() always use) actually sent to
#: the provider -- so an uncalibrated `None` baseline shows up in that
#: warning even when the real, per-call budget was a sensible number and
#: the retry machinery worked exactly as designed. This is purely a display
#: fix for that misleading text, not a substitute for token_budget.py's
#: real per-call estimates: the retry wrappers' own budget still always
#: wins for any call that goes through them (dspy.LM merges kwargs as
#: `{**self.kwargs, **per_call_kwargs}`, so a per-call override always
#: takes precedence over this baseline) -- this only matters as a floor
#: for a call that bypasses them entirely (a user's own direct analyze()/
#: segment() call, or the reflection LM in optimize_gepa.py).
_DEFAULT_MAX_TOKENS = 4096


def _configure_lm():
    api_base = _env("API_BASE", "API_BASE", None)
    model = _env("MODEL", "MODEL", None)

    if model is None:
        raise RuntimeError(
            "Missing MODEL. Set MODEL in your .env file, e.g. "
            "MODEL=litellm/anthropic/claude-opus-5 -- see USAGE.md."
        )

    # Distinguish "API_KEY isn't in .env at all" (a likely oversight -- keep
    # raising) from "API_KEY= is there but deliberately empty" (fine for a
    # local, unauthenticated model like Ollama). _env()'s own truthiness
    # check can't tell these apart (both look like "falsy"), so this checks
    # os.environ directly instead.
    if "API_KEY" not in os.environ:
        raise RuntimeError(
            "Missing API key. Set API_KEY in your .env file -- an empty "
            "value (API_KEY=) is fine for a local model that doesn't need "
            "one, e.g. Ollama; this only checks that the line exists at all."
        )
    api_key = os.environ["API_KEY"]

    # MAX_TOKENS is optional: falls back to _DEFAULT_MAX_TOKENS (see its own
    # comment above) rather than leaving dspy.LM's own max_tokens at None.
    # Set this in .env to your model's real max output tokens if you know
    # it -- see USAGE.md's "Estimating and enforcing a max_tokens budget".
    max_tokens_setting = _env("MAX_TOKENS", "MAX_TOKENS", None)
    max_tokens = int(max_tokens_setting) if max_tokens_setting else _DEFAULT_MAX_TOKENS

    # Only pass api_key through when it's actually non-empty. dspy.LM/litellm
    # don't need one at all for a local Ollama daemon -- passing api_key=""
    # explicitly is unnecessary and, depending on the provider, can behave
    # differently than omitting it outright.
    lm_kwargs = dict(model=model, max_tokens=max_tokens)
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    return lm


from diqduq import print_analysis, analyze_passage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Biblical Hebrew syntax analysis.")
    parser.add_argument(
        "--passage",
        default="בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃",
        help="Biblical Hebrew passage to analyze (defaults to the built-in sample, Genesis 1.1).",
    )
    parser.add_argument(
        "--citation",
        default="",
        help="Optional citation label for the passage (e.g. "
             "'urn:cts:compnov:bible.genesis.masoretic:1.1'), recorded on "
             "every token via Token.citation. Defaults to no citation.",
    )
    args = parser.parse_args()

    _configure_lm()
    sentences, results = analyze_passage(args.passage, citation=args.citation)

    for i, (sentence, result) in enumerate(zip(sentences, results), start=1):
        if len(sentences) > 1:
            print(f"\n=== Sentence {i} ===")
        print_analysis(sentence.tokens, result)
