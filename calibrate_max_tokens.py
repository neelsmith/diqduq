"""
Calibrates diqduq/token_budget.py's max_tokens estimate against real LM
output, instead of the untuned fallback constants token_budget.py ships
with. Modeled directly on arsgrammatica's calibrate_max_tokens.py.

Why: SyntaxAnalysis's output (a `reasoning` field plus JSON-serialized
`verbalunits`/`tokengraph` lists) grows with how long and how syntactically
complex a passage is, not by a fixed amount -- so the right `max_tokens`
budget for a call is a function of the input token count, not a constant.
Until this script has been run once, `estimate_max_tokens()` falls back to
a deliberately generous, but untuned, placeholder fit (`_FALLBACK_INTERCEPT`/
`_FALLBACK_SLOPE` in token_budget.py) -- generous enough to avoid truncating
most short GOLD_EXAMPLES-sized sentences, but not measured against any real
model, and not a substitute for a provider-side "max_tokens exceeds this
model's real output limit" error, which only a correct, small-enough
DEFAULT_CEILING (or a `ceiling=...` you pass explicitly) can prevent.

This script measures the actual completion-length function directly: it
runs every GOLD_EXAMPLES passage (tests/fixtures/gold_examples.py) through
the real configured LM with a generous max_tokens ceiling so nothing
truncates, records how many completion tokens each one actually used, and
fits `completion_tokens ~ intercept + slope * num_input_tokens` by ordinary
least squares. The fitted (intercept, slope) is written to
diqduq/token_budget_calibration.json, where token_budget.estimate_max_tokens()
picks it up automatically.

Usage:

    python3 calibrate_max_tokens.py

Needs the same .env diqduq_main.py uses (see USAGE.md's "Running an
analysis from the command line"):

    API_BASE=https://localmodel/api
    MODEL=litellm/modelname
    API_KEY=your-key-here

This is a live-LM script with real API cost -- one call per GOLD_EXAMPLES
entry (8 as of this writing, considerably fewer than arsgrammatica's mature
gold corpus, since syntax_model.md is still a first draft). Re-run it
whenever the configured MODEL, the SyntaxAnalysis prompt, or the
TokenAnalysis/VerbalExpression schema changes substantially, since any of
those shifts how many output tokens a given passage actually needs.
GOLD_EXAMPLES itself is a corpus of short, illustrative single-construction
verses, not long real-world passages -- the fit is a genuine measurement
over that range, but treat max_tokens estimates for much longer passages as
an extrapolation, and lean on token_budget.py's safety_margin/ceiling/retry
machinery rather than trusting the raw line far past the calibrated range.

--calibration-ceiling controls the max_tokens used *during calibration
itself* (not the fitted result) -- generous by default so calibration runs
aren't the ones getting truncated; --limit runs a quick smoke test over
just the first N examples instead of the whole corpus.

Unlike arsgrammatica's own calibrate_max_tokens.py -- which deliberately
duplicates its own `_configure_lm()`/`_env()` helpers in every runnable
script -- this one reuses diqduq_main.py's copies (and tests/conftest.py's
tokens_from_canned_answer()) directly, matching optimize_gepa.py's own
established convention in this codebase of importing rather than
duplicating those helpers. A judgment call, flagged here per this
project's own convention for judgment calls, not a bug: diqduq has far
fewer runnable scripts than arsgrammatica, so there's no equivalent of
model_bakeoff.py's several near-identical _configure_*_lm() variants to
justify each script owning its own copy.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

# Reuse diqduq_main.py's own .env-loading + LM-config helpers -- see this
# file's own docstring for why this diverges from arsgrammatica's
# duplicate-per-script convention.
sys.path.insert(0, str(Path(__file__).parent))
from diqduq_main import _configure_lm  # noqa: E402

# tests/ isn't an installed package -- add it to sys.path the same way
# pytest does (see pytest.ini's own comment about this) so
# "from fixtures.gold_examples import GOLD_EXAMPLES" and
# "from conftest import tokens_from_canned_answer" resolve the same way
# they do under pytest, without duplicating either helper here.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from conftest import tokens_from_canned_answer  # noqa: E402
from fixtures.gold_examples import GOLD_EXAMPLES  # noqa: E402

from diqduq import analyze

CALIBRATION_FILE = Path(__file__).parent / "diqduq" / "token_budget_calibration.json"


def _fit_line(xs, ys):
    """Ordinary least squares for y = a + b*x, plain Python (no numpy
    dependency needed for a fit this simple). Returns (a, b). Raises
    ValueError if there are fewer than 2 distinct x values -- a line isn't
    identifiable from a single point."""
    n = len(xs)
    if len({x for x in xs}) < 2:
        raise ValueError(
            "Need at least 2 examples with different token counts to fit a "
            "line; every calibrated example had the same num_tokens."
        )

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    b = cov / var
    a = mean_y - b * mean_x
    return a, b


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate token_budget.py's max_tokens estimate against the real configured LM."
    )
    parser.add_argument(
        "--calibration-ceiling",
        type=int,
        default=8000,
        help="max_tokens used for calibration calls themselves (default: 8000) -- "
             "should comfortably exceed anything GOLD_EXAMPLES needs; raise it if "
             "examples are still getting skipped as truncated even at the default, "
             "or LOWER it if your provider rejects a request this large outright "
             "(see this script's module docstring, and USAGE.md's note on "
             "DEFAULT_CEILING, for why a too-large max_tokens can itself be the "
             "cause of a 'tokens exceeded allowed length' error rather than the fix).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only calibrate against the first N GOLD_EXAMPLES (for a quick smoke run).",
    )
    args = parser.parse_args()

    lm = _configure_lm()

    examples = GOLD_EXAMPLES[: args.limit] if args.limit else GOLD_EXAMPLES

    rows = []  # (slug, num_tokens, completion_tokens)
    skipped = []
    for example in examples:
        tokens = tokens_from_canned_answer(example.canned_answer)
        try:
            analyze(
                passage=example.passage,
                tokens=tokens,
                config={"max_tokens": args.calibration_ceiling},
            )
        except Exception as exc:  # noqa: BLE001 -- report and keep calibrating
            skipped.append((example.slug, f"raised {exc.__class__.__name__}: {exc}"))
            continue

        usage = lm.history[-1].get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            skipped.append((example.slug, "no completion_tokens in usage -- provider didn't report it"))
            continue

        choices = getattr(lm.history[-1].get("response"), "choices", [])
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            skipped.append((example.slug, f"still truncated even at max_tokens={args.calibration_ceiling}"))
            continue

        rows.append((example.slug, len(tokens), completion_tokens))

    print(f"Calibrated against {len(rows)}/{len(examples)} examples.")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for slug, reason in skipped:
            print(f"  - {slug}: {reason}")

    if len(rows) < 2:
        raise RuntimeError(
            f"Only {len(rows)} usable example(s) -- need at least 2 to fit a line. "
            "Check the skipped list above."
        )

    print("\nslug                                            num_tokens  completion_tokens")
    for slug, num_tokens, completion_tokens in rows:
        print(f"{slug:<48}  {num_tokens:>10}  {completion_tokens:>17}")

    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    intercept, slope = _fit_line(xs, ys)

    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    max_abs_residual = max(abs(r) for r in residuals)

    print(f"\nFitted: completion_tokens ~= {intercept:.1f} + {slope:.2f} * num_tokens")
    print(f"Largest residual over the calibration set: {max_abs_residual:.1f} tokens")
    print(
        "(token_budget.estimate_max_tokens() applies its own safety_margin on top "
        "of this fit -- the margin is what actually covers residual variance like "
        "this, not the fit itself.)"
    )

    payload = {
        "intercept": intercept,
        "slope": slope,
        "sample_size": len(rows),
        "model": lm.model,
        "calibrated_at": datetime.datetime.now().isoformat(),
    }

    CALIBRATION_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {CALIBRATION_FILE}")
    print(
        "\nIf you're calibrating specifically because a live run raised a "
        "'tokens exceeded allowed length' (or similar max_tokens-too-large) "
        "provider error rather than a truncation warning: that error comes from "
        "the *ceiling*, not the fit above. Check your provider's real max output "
        "tokens for MODEL and pass it explicitly, e.g.\n"
        "    from diqduq import analyze_with_retry\n"
        "    analyze_with_retry(passage, tokens, ceiling=4096)  # your model's real limit\n"
        "-- token_budget.py's DEFAULT_CEILING (8192) is only a placeholder stand-in "
        "(see USAGE.md's 'Estimating and enforcing a max_tokens budget')."
    )


if __name__ == "__main__":
    main()
