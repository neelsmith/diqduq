# Developing `diqduq`

`USAGE.md` describes how to call the pipeline, `TESTING.md` how to run the offline test suite, and `OPTIMIZING.md` how to tune `SyntaxAnalysis`'s prompt with GEPA. Each of those describes one tool. This document describes how they fit together into a repeatable development loop, centered on testing the analyzer against real Biblical Hebrew text rather than just the hand-picked `GOLD_EXAMPLES` corpus.


## Why real-world testing, not just the gold-example suite

`tests/fixtures/gold_examples.py`'s `GOLD_EXAMPLES` is a deliberately curated corpus: `test_coverage.py` enforces that every documented relation label, verbal-expression classification, and token type in `syntax_model.md` has at least one example exercising it, and the whole `pytest` suite runs against `DummyLM`, not a real model. That combination is exactly right for what it's for -- proving the code (`models.py`'s pydantic models, `validate()`, `verbal_units.py`, `rendering.py`, `mermaid.py`, `serialization.py`) correctly *represents* a correct answer -- but it can't tell you whether a live model actually *produces* one, and it can't surface a construction nobody has thought to write a fixture for yet.

This matters more for `diqduq` than it did even at the start of `arsgrammatica`: `syntax_model.md` is explicitly a first draft, with its own "TBA" section naming two constructions (subordinating conjunctions, the relative pronoun אֲשֶׁר) that aren't part of the scheme yet at all -- the direct object marker אֵת and adverbial prepositional phrases were both added after that first draft (see `models.py`'s `RelationLabel` "object marker"/"adverbial" values). Running the analyzer against real Biblical Hebrew passages -- actual verses, not fixtures written to order -- is where you find the things the gold-example suite structurally can't show you:

- a construction `syntax_model.md` doesn't document at all yet;
- a construction the scheme already documents, but the current prompt still gets wrong;
- a genuinely ambiguous case that exposes a modeling choice worth deciding and flagging explicitly -- e.g. this codebase's own resolution of the tension between `syntax_model.md`'s tokenization walkthrough and its relation-scheme examples over whether the article/inseparable-preposition prefix gets its own token (see `hebrew_syntax_dspy.py` and `rendering.py`'s comments on this);
- an ordinary, everyday construction the model already handles correctly -- not new information about the scheme, but real evidence worth locking in as a regression guardrail so a future prompt or model change can't silently break it without anyone noticing.


## The core loop: analyze, check automatically, review by hand, triage, act

1. **Analyze a real passage.** `analyze_passage(passage)` for a single string, or `analyze_sources(sources)` for a list of citation-labeled `CitedText` (see USAGE.md's "Analyzing citable sources") -- either returns `(sentences, results)`, one `SyntaxAnalysis` result per sentence found.

2. **Run the automated checks first, before reading anything by hand.** `analyze_sources()` already calls `validate()` for you and prints any referential problems it finds (a token id that doesn't exist, a malformed implied token); `find_unanchored_coordinated_verbs(result.tokengraph)` (`verbal_units.py`) catches one more specific, self-consistent mistake (a coordinating-conjunction pair where only one side anchors its own verbal unit) that can slip past `validate()`. Both are cheap and mechanical -- let them rule out the "obviously broken" cases before you spend a human read on anything.

3. **Read the surviving result against `syntax_model.md` by hand.** This is the one step nothing in the codebase can do for you: `validate()` only checks referential integrity, never correctness, and neither `find_unanchored_coordinated_verbs()` nor any metric can substitute for actually knowing the Hebrew. `tokengraph_to_html()`/`tokengraph_to_depth_html()` (`rendering.py`) or `tokengraph_to_mermaid()` (`mermaid.py`) are worth rendering here -- seeing the verbal-unit coloring and subordination depth laid out is usually faster to check by eye than reading the raw `tokengraph` rows.

4. **Triage what you found into one of three outcomes, and act accordingly** (see the next section). Every outcome below can be turned into a `GoldExample` with `tests/fixtures/harvest.py`'s `gold_example_from_analysis()` -- the difference between them is what you do with the result afterward, not how you build it.


## The three outcomes, and what to do with each

### Outcome A: a failure

Something is referentially broken (`validate()`/`find_unanchored_coordinated_verbs()` caught it) or substantively wrong (you caught it by hand). Don't just note it and move on -- a failure is the most valuable signal this loop produces, because it's the main way the scheme and the prompt actually improve. Triage it further, in this order:

1. **Is `syntax_model.md` actually silent or ambiguous about this construction?** If so, this is a scheme gap, not a model mistake. Extend `syntax_model.md` first, then follow USAGE.md's "Extending the scheme" steps: add the new relation label / `tokentype` / `syntactic_type`/`semantic_type` value to the relevant `Literal` in `diqduq/models.py`, describe when to use it in `SyntaxAnalysis`'s docstring in `diqduq/hebrew_syntax_dspy.py`, and hand-write a `GoldExample` with a *correct* `canned_answer` exercising it (since the live model, by definition, didn't produce one) in `tests/fixtures/gold_examples.py`.
2. **Is the scheme already clear, but the prompt/model got it wrong anyway?** Hand-write a corrected `GoldExample` for the passage the same way, so the failure becomes a concrete, checkable trainset entry rather than an anecdote. If you're unsure between two defensible readings, say so explicitly in a comment above the fixture rather than silently committing to one.
3. Re-run `pytest` (fast, `DummyLM`-backed -- see TESTING.md) to confirm the new/corrected fixture actually validates and that `test_coverage.py` is satisfied, *before* spending any real API budget re-testing it against the live model.

Either way, the corrected fixture lands in `GOLD_EXAMPLES` -- see "How this feeds `OPTIMIZING.md`" below for what that means downstream.

### Outcome B: a success against a rare or tricky construction

The model got something genuinely uncommon or structurally hard right -- a deep subordination chain, a repeated-connector coordinating-conjunction pattern, anything you'd be nervous betting the model gets right consistently. This is worth *reinforcing*, not just recording: use `gold_example_from_analysis()` to build the `GoldExample` from the real `sentences`/`result.verbalunits`/`result.tokengraph` (passing `result.reasoning` too, since `dspy.ChainOfThought` gives you a real one for free here, not a placeholder), then `format_gold_example_source()` to get paste-ready source for `gold_examples.py`. Add it straight to `GOLD_EXAMPLES` as ordinary trainset material -- a correct demonstration of a rare case is precisely the kind of thing `optimize_gepa.py`'s trainset benefits from having more of.

### Outcome C: a success against a common, ordinary construction

The model got something right that it was already expected to get right -- a plain independent clause, an ordinary direct object, nothing structurally novel. This is real evidence, but low-value as *training* signal: `optimize_gepa.py` has no held-out split at all today (see below), so anything added to `GOLD_EXAMPLES` is immediately part of what GEPA both trains against and scores itself against, and an easy case the model already nails teaches the optimizer nothing new -- it just dilutes the trainset with redundant coverage. Harvest it the same way and add it to `GOLD_EXAMPLES` regardless -- it still grows the corpus GEPA optimizes against, and is worth keeping in mind as a candidate for a held-out valset once the corpus is large enough to afford one (see "How this feeds `OPTIMIZING.md`" below).

The dividing line, in short: rare-and-tricky successes are worth teaching the optimizer with; common-and-already-reliable successes mostly just confirm the status quo still holds. "Which bucket does this belong in" is a judgment call about how common the construction already is in `GOLD_EXAMPLES`, not about whether the analysis happened to be correct -- both outcomes started from a correct analysis.


## How this feeds `OPTIMIZING.md`

`optimize_gepa.py` (OPTIMIZING.md) trains `SyntaxAnalysis`'s prompt against *all* of `GOLD_EXAMPLES`, with no separate held-out valset at all -- per `dspy.GEPA`'s own behavior when none is given, the trainset doubles as the Pareto-tracking set GEPA scores itself against. That means every fixture this loop adds to `GOLD_EXAMPLES` -- a corrected failure (Outcome A) or a harvested rare-construction success (Outcome B) -- becomes real trainset material the next time `optimize_gepa.py` runs, which is the main way the shipped prompt actually improves over time: not synthetic examples, but real passages that either broke something or demonstrated something worth reinforcing.

This is a starting point, not a finished setup: with only 8 gold examples today, and the scheme itself still a first draft, there's no held-out evaluation set at all yet (unlike `arsgrammatica`, which grew one over time as its own gold corpus expanded). As `GOLD_EXAMPLES` grows, consider holding some examples out of `optimize_gepa.py`'s trainset -- following `USAGE.md`'s own note under "Scope and data" in OPTIMIZING.md -- so the optimized prompt's quality can be checked against passages it wasn't directly tuned on.


## Suggested cadence

- After any batch of real-world testing/harvesting, run `pytest` (TESTING.md) first -- it's fast and `DummyLM`-backed, and will immediately tell you if a new or corrected fixture doesn't actually validate or if `test_coverage.py` regressed.
- Run `optimize_gepa.py` (OPTIMIZING.md) periodically to refresh the shipped, production prompt against whatever `GOLD_EXAMPLES` has grown into since the last run -- this is a live-LM script with real API cost, so batch it rather than running it after every single new fixture.
- Extend `syntax_model.md` itself as new constructions come up in real text -- particularly the constructions its own "TBA" section already flags as not yet covered -- rather than letting the prompt paper over a genuine scheme gap.
