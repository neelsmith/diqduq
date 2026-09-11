# Optimizing with GEPA

`optimize_gepa.py` uses [dspy.GEPA](https://dspy.ai) -- a reflective prompt optimizer -- to improve `SyntaxAnalysis`'s instructions against the gold examples in `tests/fixtures/gold_examples.py`. Unlike `pytest` (entirely `DummyLM`-backed), this is a **live-LM script**: every trial makes a real call to the configured task model, plus a reflection model GEPA uses to read scoring feedback and propose better instructions. Expect it to use real API usage against the configured provider.

```bash
python optimize_gepa.py                    # --auto light (cheapest; default)
python optimize_gepa.py --auto medium       # more thorough, more expensive
python optimize_gepa.py --auto heavy        # most thorough, most expensive
python optimize_gepa.py --max-metric-calls 40   # exact call budget instead of a preset
python optimize_gepa.py --skip-baseline     # skip the pre-GEPA scoring pass (saves N calls)
```

Needs the same `.env` as `diqduq_main.py` (`API_BASE`/`MODEL`/`API_KEY`). Optionally set `REFLECTION_MODEL` (and `REFLECTION_API_BASE`/`REFLECTION_API_KEY`, if they differ) to use a different model specifically for GEPA's reflective step -- GEPA's own docs recommend a strong reasoning model for this. Without `REFLECTION_MODEL` set, the task model doubles as the reflection model, a reasonable default for a first run. `MAX_TOKENS`/`REFLECTION_MAX_TOKENS` are optional and, like `diqduq_main.py`'s own `MAX_TOKENS`, have no default injected if you leave them unset -- see USAGE.md's note on why setting one doesn't make dspy's truncation-warning text any more trustworthy.

**Scope and data**: this optimizes only `SyntaxAnalysis` (the `analyze` module in `hebrew_syntax_dspy.py`), not the segmentation stage. It trains on all gold examples in `GOLD_EXAMPLES` (8 as of this writing) with no separate held-out valset -- per `dspy.GEPA`'s own behavior when no valset is given, it uses the trainset for both reflective updates and Pareto-score tracking. That's a reasonable starting point while the gold set is still small and the scheme itself is a first draft, but expect the optimized prompt to fit these exact sentences well without a guarantee it generalizes to new ones -- worth revisiting (holding out a few examples as a valset) once there are more gold examples to spare.

**Scoring**: `diqduq/gepa_metric.py`'s `syntax_metric` compares a prediction's `verbalunits`/`tokengraph` against the gold answer and returns a score in [0, 1] (a weighted blend: relations 50%, verbal-expression classification 30%, basic per-token fields 20% -- a judgment call, easy to retune in that file) plus specific, human-readable feedback naming every mismatched token/relation/classification, for GEPA's reflection model to read. Relations are compared as an unordered set, not by `relatedtoken1`/`relatedtoken2` position, since that pairing is documented as an interchangeable overflow slot (except for "coordinating conjunction", which genuinely uses both slots at once -- see `models.py`'s `RelationLabel` comment) -- see `tests/test_gepa_metric.py` for fully offline tests of the metric itself (including that a relation-slot swap scores as a perfect match, not an error).

**Using the result**: `optimize_gepa.py` saves the optimized program's instructions to `optimized_syntax_analysis.json` (configurable via `--out`).

To use it:

```python
from diqduq.hebrew_syntax_dspy import analyze
analyze.load("optimized_syntax_analysis.json")
```

right after import and before calling `analyze_passage()`/`analyze_sources()` -- `analyze` is the same module-level `ChainOfThought` instance the whole pipeline uses, so loading into it in place is enough; nothing else needs to change. `gepa_logs/` (GEPA's own run logs) is gitignored; `optimized_syntax_analysis.json` is not -- commit it once you're satisfied with a run, or gitignore it yourself if you'd rather treat it as a local, disposable artifact.
