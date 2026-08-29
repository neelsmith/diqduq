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

    # MAX_TOKENS is optional and, deliberately, has NO default injected here
    # (an earlier version of this function set one -- see git history/
    # DEVELOPMENT.md if curious -- but that only swapped dspy's own
    # "LM response was truncated due to exceeding max_tokens=None" warning
    # for an equally-inaccurate "...max_tokens=4096", since dspy's
    # `_check_truncation()` always reports the LM's OWN baseline kwargs,
    # never the real per-call value analyze_with_retry()/segment_with_retry()
    # actually sent (see token_budget.py's module docstring). A confident-
    # looking wrong number is worse than an obviously-wrong one, so this
    # reverts to leaving max_tokens unset unless you explicitly set it --
    # see USAGE.md's "Estimating and enforcing a max_tokens budget" for why
    # that raw dspy warning line can't be trusted either way, and what to
    # look at instead (this codebase's own UserWarnings, which do carry the
    # real numbers).
    max_tokens_setting = _env("MAX_TOKENS", "MAX_TOKENS", None)

    # Only pass api_key/max_tokens through when they're actually set.
    # dspy.LM/litellm don't need an api_key at all for a local Ollama daemon
    # -- passing api_key="" explicitly is unnecessary and, depending on the
    # provider, can behave differently than omitting it outright.
    lm_kwargs = dict(model=model)
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key
    if max_tokens_setting:
        lm_kwargs["max_tokens"] = int(max_tokens_setting)

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
