"""
Optimizes hebrew_syntax_dspy.py's SyntaxAnalysis prompt with dspy.GEPA,
using tests/fixtures/gold_examples.py's GOLD_EXAMPLES as the trainset and
diqduq/gepa_metric.py's syntax_metric as the scoring/feedback function.
Modeled directly on arsgrammatica's optimize_gepa.py.

This is a LIVE-LM script: unlike the pytest suite (entirely DummyLM-backed,
see TESTING.md), every trial here actually calls the configured task LM,
plus a reflection LM GEPA uses to read the metric's feedback and propose
better instructions. Budget is controlled by --auto (dspy's light/medium/
heavy presets -- default "light", the cheapest) or --max-metric-calls for
an exact call count. Expect this to run up real API usage against
whatever provider your .env configures; --auto light is meant as the "does
this all work" starting point before spending more on medium/heavy.

There is currently no separate valset: every gold example is used as the
trainset, and (per dspy.GEPA's own behavior when valset=None) also as the
Pareto-tracking set. That's a reasonable choice while the gold set is
still small, but it does mean GEPA is optimizing directly against the only
examples it's being judged on -- expect the result to fit these sentences
well without a guarantee it generalizes to new ones. Revisit this once
there are enough gold examples to hold some out.

Usage:
    python optimize_gepa.py                       # --auto light (default)
    python optimize_gepa.py --auto medium
    python optimize_gepa.py --max-metric-calls 40
    python optimize_gepa.py --skip-baseline        # skip the pre-GEPA scoring pass

Needs the same .env as diqduq_main.py (API_BASE/MODEL/API_KEY). Optionally
set REFLECTION_MODEL (and REFLECTION_API_BASE/REFLECTION_API_KEY, if they
differ) to use a different model for GEPA's own reflective step -- GEPA's
docs recommend a strong reasoning model specifically for reflection.
Without REFLECTION_MODEL set, the task model doubles as the reflection
model, which is a reasonable default for a first run but not a requirement.
"""

import argparse
import sys
from pathlib import Path

import dspy

# Reuse diqduq_main.py's own .env-loading + LM-config helpers rather than
# duplicating them.
sys.path.insert(0, str(Path(__file__).parent))
from diqduq_main import _configure_lm, _env, _DEFAULT_MAX_TOKENS  # noqa: E402

# tests/ isn't an installed package -- add it to sys.path the same way
# pytest does (see pytest.ini's own comment about this) so
# "from fixtures.gold_examples import GOLD_EXAMPLES" and
# "from conftest import tokens_from_canned_answer" resolve the same way
# they do under pytest, without duplicating either helper here.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from conftest import tokens_from_canned_answer  # noqa: E402
from fixtures.gold_examples import GOLD_EXAMPLES  # noqa: E402

from diqduq.gepa_metric import syntax_metric
from diqduq.hebrew_syntax_dspy import analyze
from diqduq.models import TokenAnalysis, VerbalExpression


def build_trainset():
    """Turn every GoldExample in GOLD_EXAMPLES into a dspy.Example GEPA can
    train against: `passage`/`tokens` as inputs (matching SyntaxAnalysis's
    own InputFields), `verbalunits`/`tokengraph` as the gold outputs
    diqduq.gepa_metric.syntax_metric compares predictions to."""
    trainset = []
    for example in GOLD_EXAMPLES:
        tokens = tokens_from_canned_answer(example.canned_answer)
        verbalunits = [VerbalExpression(**vu) for vu in example.canned_answer["verbalunits"]]
        tokengraph = [TokenAnalysis(**tok) for tok in example.canned_answer["tokengraph"]]
        trainset.append(
            dspy.Example(
                passage=example.passage,
                tokens=tokens,
                verbalunits=verbalunits,
                tokengraph=tokengraph,
            ).with_inputs("passage", "tokens")
        )
    return trainset


def _configure_reflection_lm(task_lm):
    """Build the LM GEPA uses to read syntax_metric's feedback and propose
    better instructions. Defaults to `task_lm` itself unless REFLECTION_MODEL
    is set in .env, in which case a separate dspy.LM is built for it (same
    API_BASE/API_KEY unless REFLECTION_API_BASE/REFLECTION_API_KEY override
    those too)."""
    reflection_model = _env("REFLECTION_MODEL", "REFLECTION_MODEL", None)
    if not reflection_model:
        return task_lm

    api_base = _env("REFLECTION_API_BASE", "REFLECTION_API_BASE", None) or _env(
        "API_BASE", "API_BASE", None
    )
    api_key = _env("REFLECTION_API_KEY", "REFLECTION_API_KEY", None) or _env("API_KEY", "API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key for the reflection LM. Set REFLECTION_API_KEY or API_KEY in .env."
        )
    # REFLECTION_MAX_TOKENS, falling back to MAX_TOKENS, falling back to
    # _DEFAULT_MAX_TOKENS -- same reasoning as diqduq_main._configure_lm()'s
    # own MAX_TOKENS handling: leaving dspy.LM's own max_tokens at None
    # produces a misleadingly-labeled truncation warning later even when a
    # per-call budget override was actually used (see that function's
    # comment for the full explanation).
    max_tokens_setting = _env("REFLECTION_MAX_TOKENS", "REFLECTION_MAX_TOKENS", None) or _env(
        "MAX_TOKENS", "MAX_TOKENS", None
    )
    max_tokens = int(max_tokens_setting) if max_tokens_setting else _DEFAULT_MAX_TOKENS
    lm_kwargs = dict(model=reflection_model, api_key=api_key, max_tokens=max_tokens)
    if api_base:
        lm_kwargs["api_base"] = api_base
    return dspy.LM(**lm_kwargs)


def _evaluate(program, trainset, label):
    """Run `program` over every trainset example, score each with
    syntax_metric, and print a min/mean/max summary."""
    scores = []
    for example in trainset:
        pred = program(passage=example.passage, tokens=example.tokens)
        scores.append(syntax_metric(example, pred).score)
    mean = sum(scores) / len(scores)
    print(f"{label}: mean={mean:.3f}  min={min(scores):.3f}  max={max(scores):.3f}  (n={len(scores)})")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Optimize SyntaxAnalysis's prompt with GEPA.")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="dspy's auto budget preset (default: %(default)s -- fewest LM calls).",
    )
    budget.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Exact LM-call budget instead of an --auto preset.",
    )
    parser.add_argument(
        "--out",
        default="optimized_syntax_analysis.json",
        help="Where to save the optimized program (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the pre-GEPA scoring pass (saves one live LM call per gold example).",
    )
    args = parser.parse_args()

    task_lm = _configure_lm()
    reflection_lm = _configure_reflection_lm(task_lm)

    trainset = build_trainset()
    print(f"Built a trainset of {len(trainset)} gold examples from tests/fixtures/gold_examples.py.")
    print(
        "No separate valset -- GEPA will use the trainset for both reflective updates "
        "and Pareto-score tracking. See this script's module docstring for why."
    )

    if not args.skip_baseline:
        print()
        _evaluate(analyze, trainset, "Baseline (before GEPA)")

    optimizer_kwargs = dict(
        metric=syntax_metric,
        reflection_lm=reflection_lm,
        track_stats=True,
        log_dir=str(Path(__file__).parent / "gepa_logs"),
    )
    if args.max_metric_calls is not None:
        optimizer_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        optimizer_kwargs["auto"] = args.auto

    gepa = dspy.GEPA(**optimizer_kwargs)

    print("\nRunning GEPA -- this makes many real LM calls through the configured provider.")
    optimized = gepa.compile(student=analyze, trainset=trainset)

    print()
    _evaluate(optimized, trainset, "Optimized (after GEPA)")

    optimized.save(args.out)
    print(f"\nSaved the optimized program to {args.out}.")
    print(
        "To use it, right after `from diqduq.hebrew_syntax_dspy import analyze`, call:\n"
        f"    analyze.load({args.out!r})\n"
        "before running analyze_passage()/analyze_sources() -- see "
        "OPTIMIZING.md."
    )


if __name__ == "__main__":
    main()
