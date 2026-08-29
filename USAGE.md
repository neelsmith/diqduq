# Biblical Hebrew Syntax Analyzer — Usage Guide

A DSPy program that analyzes a Biblical Hebrew passage into two structures: a table of verbal expressions, and a token-by-token dependency graph. The analytic scheme itself is documented in `syntax_model.md`.


## Running an analysis from the command line

You can run an analysis from the command line with the wrapper script `diqduq_main.py`. It needs an `.env` file in this folder with your LM credentials, like this:

```
API_BASE=https://localmodel/api
MODEL=litellm/modelname
API_KEY=your-key-here
```

Then:

```bash
python3 diqduq_main.py --passage "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃"
```

Omitting `--passage` entirely runs the same built-in sample (Genesis 1.1).

`--citation` is an optional second argument giving a citation label for the passage (e.g. a CTS URN), recorded on every resulting token via `Token.citation`; it defaults to no citation if omitted:

```bash
python3 diqduq_main.py --passage "וַיִּקְרָא אֱלֹהִים לָאוֹר יוֹם" --citation "urn:cts:compnov:bible.genesis.masoretic:1.5"
```

`diqduq_main.py` reads `API_BASE` / `MODEL` / `API_KEY` from `.env`, configures the LM, and prints the analysis.

For a local, unauthenticated model (e.g. Ollama), leave `API_KEY` present but empty:

```
API_BASE=http://localhost:11434
MODEL=ollama_chat/llama3
API_KEY=
```

`diqduq_main.py` only raises "Missing API key" when `API_KEY` isn't in `.env` at all; an empty value is treated as "this model doesn't need one" and is left out of the LM call entirely, rather than sent through as an empty credential.

Optionally set `MAX_TOKENS` too, if you know your model's real max output tokens:

```
MAX_TOKENS=4096
```

Leave it unset unless you have a specific reason to set it -- see the note on `MAX_TOKENS` under "Estimating and enforcing a `max_tokens` budget" below for why setting it doesn't do what it might look like it does, and isn't needed for ordinary use.


## Using `diqduq` in a script

To call the pipeline from your own script or a REPL instead of the CLI, configure a `dspy.LM` yourself and use `diqduq` directly:

```python
import dspy
from diqduq import analyze_passage, print_analysis

dspy.configure(lm=dspy.LM(model="litellm_proxy/anthropic/Claude Opus 5",
                           api_base="https://api_url/litellm",
                           api_key="your-key-here"))

sentences, results = analyze_passage("בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃")
for sentence, result in zip(sentences, results):
    print_analysis(sentence.tokens, result)
```

Explanation:

- `analyze_passage()` returns `(sentences, results)`: that is, one `Sentence` and one `SyntaxAnalysis` result per sentence it finds in `passage`.
- `result.verbalunits` is a list of `VerbalExpression` objects
- `result.tokengraph` is a list of `TokenAnalysis` objects, one per token in that sentence, in order.

`analyze_passage()` also prints a warning if the LM refers to a token id that doesn't exist in its sentence's input tokens. (That's a sign that the output needs a re-run or a prompt tweak, and does not necessarily mean that your code is broken.)

`validate()` only catches referential problems like that one -- ids that don't exist. It can't tell you an otherwise well-formed analysis is probably still wrong. For one specific, observed failure mode (ported from `arsgrammatica`'s own experience) -- a coordinating conjunction correctly pairing two verbal expressions, but the second one silently missing its own `verbalunitid` -- call `find_unanchored_coordinated_verbs()` on the result:

```python
from diqduq import find_unanchored_coordinated_verbs

for sentence, result in zip(sentences, results):
    for warning in find_unanchored_coordinated_verbs(result.tokengraph):
        print(f"Possible mistake: {warning}")
```

It's a heuristic, not a guarantee -- see its own docstring -- but a clean result costs nothing to check, and a flagged one is worth a manual read before you trust the analysis.


## Analyzing citable sources

`diqduq` supports analyzing texts identified by some canonical citation. Under the hood, `analyze_passage()` wraps `passage` as a `CitedText` and hands this to `analyze_sources()`, which is what actually does the work. You can call `analyze_sources()` directly like this:

```python
from diqduq import analyze_sources, combined_tokengraph
from diqduq.models import CitedText

genesis = "urn:cts:compnov:bible.genesis.masoretic:"
sources = [
    CitedText(citation=f"{genesis}1.1", text="בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃"),
    CitedText(citation=f"{genesis}1.2", text="וְהָאָרֶץ הָיְתָה תֹהוּ וָבֹהוּ וְחֹשֶׁךְ עַל־פְּנֵי תְהוֹם׃"),
]
sentences, results = analyze_sources(sources)
tokengraph = combined_tokengraph(results)  # one flat list, spanning every sentence
```

`analyze_sources()` handles any number of sentences and citation units; sentence boundaries don't need to respect citation-unit boundaries (one sentence may span two source lines), and every token still records which source unit it came from via `Token.citation`.


## Saving and loading analyses

`write_analyses()`/`read_analyses()` (in `diqduq/serialization.py`) save and reload a full analysis -- `sentences`, `verbalunits` (concatenated across every sentence's result), and `tokengraph` (via `combined_tokengraph()`) -- as one deterministic, pipe-delimited plain-text file, so you can persist an analysis, diff it, hand-edit it, or reload it later without re-running the LM:

```python
from diqduq import write_analyses, read_analyses, combined_tokengraph

verbalunits = [vu for result in results for vu in result.verbalunits]
tokengraph = combined_tokengraph(results)

warnings = write_analyses(sentences, verbalunits, tokengraph, "analysis.txt")
for w in warnings:
    print(f"Warning: {w}")

tokengraph, verbalunits, sentences = read_analyses("analysis.txt")
```

`serialize_analyses(sentences, verbalunits, tokengraph)` builds the exact same text and returns it as a string (plus the same warnings) instead of writing it to a file -- `write_analyses()` is just a thin wrapper around it. Use this whenever you want the serialized format for something other than a standalone file: embedding it in a prompt, logging it, or handing it to some other file-writing code of your own.

The file has three labelled, pipe-delimited blocks (`#!sentences`, `#!verbal_units`, `#!tokens`), each with its own fixed header row -- see `serialization.py`'s module docstring for the exact format, why `sentences` is needed at all (it's the only place a citation is actually attached to a token id), and what `write_analyses()`'s warnings vs. `read_analyses()`'s errors each catch. Each of the three labels may appear more than once in the file; `read_analyses()` merges every instance of a label into that label's combined row list, in file order, so simply concatenating several `write_analyses()`/`serialize_analyses()` outputs together and reading the result back gives you one combined analysis. `read_analyses()` is otherwise deliberately strict: a malformed or internally inconsistent file raises `ValueError` naming the exact line and problem, rather than silently reconstructing something partial.

`read_analyses()` hands back flat, whole-file lists -- every sentence's `tokengraph`/`verbalunits` concatenated together, the same shape `combined_tokengraph()` produces. `split_analysis_by_sentence(tokengraph, verbalunits, sentences)` splits that back into one `(sentence_tokengraph, sentence_verbalunits)` slice per sentence, aligned with `sentences` itself:

```python
from diqduq import read_analyses, split_analysis_by_sentence

tokengraph, verbalunits, sentences = read_analyses("analysis.txt")
slices = split_analysis_by_sentence(tokengraph, verbalunits, sentences)

for sentence, (sentence_tokengraph, sentence_verbalunits) in zip(sentences, slices):
    ...  # render or inspect this one sentence's own analysis
```

An implied/elided token (see `models.py`'s `TokenAnalysis`) is included in whichever sentence's slice it's nested inside, but one sitting *after* a sentence's own last real token (rather than between two real tokens) falls just outside that sentence's slice -- see `split_analysis_by_sentence()`'s own docstring for why, and `read_analyses()`'s note on the same underlying [first, last] real-token-position convention.


## Reading passages from a delimited-text source file

`read_ctsdata()` (in `diqduq/ctsdata.py`) reads a list of citable passages -- each one a CTS URN paired with its own text -- out of a pipe-delimited file, the input-side counterpart to `write_analyses()`/`read_analyses()` above (which handle an analysis's *results*, not the passages you're about to analyze):

```python
from diqduq import read_ctsdata

rows = read_ctsdata("passages.txt")
for row in rows:
    citation = row.urnbase + row.citation  # reconstructs the full URN
    print(citation, "--", row.text)
```

The file has one or more `#!ctsdata` blocks, each with its own `urn|text` header row:

```
#!ctsdata
urn|text
urn:cts:compnov:bible.genesis.masoretic:1.1|בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃
```

Each row's `urn` column must be a 5-part, colon-separated CTS URN (e.g. `urn:cts:compnov:bible.genesis.masoretic:1.1`); `read_ctsdata()` splits it into `urnbase` (the first 4 parts, rejoined with `:`, plus a trailing `:` -- `urn:cts:compnov:bible.genesis.masoretic:` for that example) and `citation` (the 5th part, `1.1`). Pass `delimiter=...` if the file itself uses something other than `|`. Like `read_analyses()`, this is deliberately strict (a malformed row or a urn that doesn't split into exactly 5 parts raises `ValueError`, naming the line) and merges multiple `#!ctsdata` blocks in file order.


## Estimating and enforcing a `max_tokens` budget

`SyntaxAnalysis`'s output (a `reasoning` field plus JSON-serialized `verbalunits`/`tokengraph`) grows with how long and how syntactically complex a sentence is, not by a fixed amount, so a single hard-coded `max_tokens` value is eventually wrong: too small for a long or deeply subordinated sentence (truncation), too large for a short one (wasted budget). `diqduq/token_budget.py` addresses this with a calibrate-then-retry approach, and `pipeline.py`'s `analyze_sources()` already uses it -- both `analyze_sources()` and `analyze_passage()` get this for free, with nothing to change in your own calling code.

First, calibrate against your own configured model:

```bash
python3 calibrate_max_tokens.py
```

This is a live-LM script (real API cost, one call per `GOLD_EXAMPLES` entry) that measures how many completion tokens the real model actually uses for each gold example, fits `completion_tokens ~= intercept + slope * num_input_tokens` by least squares, and writes the result to `diqduq/token_budget_calibration.json`. Re-run it whenever the configured model, the `SyntaxAnalysis` prompt, or the `TokenAnalysis`/`VerbalExpression` schema changes substantially. Until you've run it at least once, `estimate_max_tokens()` falls back to an untuned, deliberately generous placeholder fit -- safe, but not a real measurement of your model.

```python
from diqduq import estimate_max_tokens

budget = estimate_max_tokens(num_tokens=25)  # -> an int max_tokens value
```

**A `dspy.clients.lm` warning that says `max_tokens=None` (or any other fixed number) is not reporting the real per-call budget, and setting `MAX_TOKENS` doesn't fix that.** dspy's own truncation-warning text (`_check_truncation()` in `dspy/clients/lm.py`) always reports the `dspy.LM` instance's *baseline* `max_tokens` -- whatever it was constructed with -- never the actual per-call value `analyze_with_retry()`/`segment_with_retry()` computed and sent for that specific attempt (the per-call `config={"max_tokens": ...}` override does correctly win for the real request; it just isn't what this one warning prints). An earlier version of this doc suggested setting `MAX_TOKENS` to make that warning "say something meaningful" -- don't: it only replaces one fixed, wrong-looking number (`None`) with a different fixed, equally-wrong-but-more-plausible-looking one (whatever you set `MAX_TOKENS` to), which is worse, not better, since a plausible number invites trusting it. There's no way to make this particular dspy warning line trustworthy from here; **ignore its number entirely** and look instead at this codebase's own `UserWarning`s (e.g. "retrying with a larger max_tokens=...", "still looks truncated after N retry(ies) (max_tokens=...)"), which always carry the real per-call number, or call `dspy.inspect_history()` for the actual request.

**Two different real symptoms, two different real fixes:**

- One of *this codebase's own* `UserWarning`s names an actual `max_tokens=<some number>` and the analysis/segmentation still succeeds after retrying -- this is `analyze_with_retry()`/`segment_with_retry()` working as designed. If it's landing on a larger budget than it feels like it should need, `calibrate_max_tokens.py` (above) is the fix for the analysis stage; segmentation has no calibration script yet (see its own note below) so widening its own fallback constants in `token_budget.py` is the equivalent move there.
- A hard *error* from your provider (rather than a warning) -- something like "tokens exceeded allowed length" as a rejection, not a truncated response -- means the *requested* `max_tokens` itself exceeded what your model actually allows, almost always because `token_budget.py`'s `DEFAULT_CEILING` (`8192`) is bigger than your configured `MODEL`'s real max-output-tokens limit. Neither `calibrate_max_tokens.py` nor the retry wrappers can fix that by themselves (they only ever ask for *less* than `ceiling`, never more); pass your model's real limit explicitly as `ceiling=`:

```python
from diqduq import analyze_with_retry

result = analyze_with_retry(passage, tokens, ceiling=4096)  # your model's real max output tokens
```

(`calibrate_max_tokens.py --calibration-ceiling` is the equivalent knob for the calibration run itself -- lower it if even calibration's own generous default of `8000` gets rejected outright.)

`estimate_max_tokens()` takes the calibrated (or fallback) fit, multiplies it by a `safety_margin` (default `1.4`, covering reasoning-length variance the fit alone doesn't), and clamps the result to `[floor, ceiling]`. Set `ceiling` to your actual model's real max-output-tokens limit -- the module's own `DEFAULT_CEILING` is only a placeholder stand-in, since that limit varies by provider/model and there's no single correct default.

For the retry half, `analyze_with_retry()` wraps `analyze()`:

```python
from diqduq import analyze_with_retry

result = analyze_with_retry(passage, tokens)
```

It starts from `estimate_max_tokens(len(tokens))` (or `initial_max_tokens`, if you pass one), and checks the result two ways: whether the returned `tokengraph` is missing any of `tokens`' own ids (the primary, provider-independent signal -- a real truncation, LM-JSON getting cut off mid-list, always shows up here), and, as a corroborating check, whether the LM's own `finish_reason` was `"length"`. If either signals truncation and a retry is still available (`max_retries`, default `1`) with budget left before `ceiling`, it multiplies the budget by `growth_factor` (default `2.0`) and calls again -- `max_tokens` is part of DSPy's own LM cache key, so the retry always reaches the LM again rather than replaying the same truncated cached response. If retries run out: a call that raised re-raises (nothing to fall back to); a call that returned an incomplete result is returned anyway, with a `UserWarning` naming the missing ids, rather than treated as fatal -- consistent with `validate()`'s own warn-don't-raise convention for imperfect LM output.

**Segmentation has its own budget too.** Unlike `arsgrammatica`, where the segmentation stage has no budget management at all, `diqduq`'s `segmentation_dspy.segment()` is wrapped the same way, by `segment_with_retry()`:

```python
from diqduq import segment_with_retry

sentences = segment_with_retry(sources)  # sources: List[CitedText]
```

This exists specifically because `segmentation_dspy.segment()` on its own passes no `config={"max_tokens": ...}` at all, so a `dspy.LM` configured without an explicit `max_tokens` falls straight through to the provider's own default, which can be small enough to truncate a longer passage's segmentation. `estimate_segmentation_max_tokens()` picks a budget from the combined input *character* count of `sources` (there's no per-token count to fit against before segmentation has run), using a deliberately generous, uncalibrated proxy rather than a real measured fit -- see `token_budget.py`'s "Segmentation budget" section for why. `pipeline.py`'s `analyze_sources()`/`analyze_passage()` already call `segment_with_retry()` rather than `segmentation_dspy.segment_sources()` directly, so this is automatic for ordinary use; call `segment_with_retry()` yourself only if you're driving the segmentation stage in isolation.

Because that proxy is a guess, not a fit, `segment_with_retry()` defaults `max_retries` to `3` (vs. `analyze_with_retry()`'s `1`) -- more retry headroom to compensate for a starting estimate with no real calibration behind it. In real testing, a short (~60-character) passage needed more than 1652 completion tokens (an 826 initial estimate, doubled once) before segmentation actually succeeded -- evidence that the `reasoning` field's own length dominates far more than input length for short passages. `_SEGMENTATION_FALLBACK_INTERCEPT` in `token_budget.py` was raised from `500.0` to `2000.0` in response to that observation; if segmentation still exhausts its retries on your own real passages, that constant (or `max_retries`/`growth_factor`, passed explicitly to `segment_with_retry()`) is the next thing to widen -- there's no calibration script for this stage yet to do it more precisely (see `token_budget.py`'s own comment on why: segmentation has no natural per-input-token count to fit against the way `SyntaxAnalysis` does).

`get_calibration()` reports which fit is currently active (a real one from a calibration script, or the untuned fallback) if you want to check before relying on an estimate.


## Harvesting gold examples from real analyses

`gold_example_from_analysis()`/`format_gold_example_source()` (in `tests/fixtures/harvest.py`) turn a real analysis's own `sentences`/`verbalunits`/`tokengraph` -- the same triple `write_analyses()`/`serialize_analyses()` take -- into a `GoldExample` (`tests/fixtures/gold_examples.py`), instead of hand-writing a `canned_answer` dict from scratch:

```python
from fixtures.harvest import gold_example_from_analysis, format_gold_example_source

sentences, results = analyze_passage("Some new passage you've reviewed by hand.")
result = results[0]  # one result per sentence; pick whichever one you're harvesting

example = gold_example_from_analysis(
    slug="some_new_construction_example",
    tags=["the construction this example is meant to cover"],
    sentences=sentences,
    verbalunits=result.verbalunits,
    tokengraph=result.tokengraph,
    reasoning=result.reasoning,  # dspy.ChainOfThought's own reasoning field
)
print(format_gold_example_source(example, "_SOME_NEW_CONSTRUCTION_ANSWER"))
```

`format_gold_example_source()`'s output is ready-to-paste Python: a `_SOME_NEW_CONSTRUCTION_ANSWER = {...}` dict literal followed by the `GoldExample(...)` entry that references it, in the same two-part shape every existing block in `gold_examples.py` already uses. `gold_example_from_analysis()` runs `validate()` against the given `sentences`/`verbalunits`/`tokengraph` before returning (pass `skip_validation=True` to bypass) -- catching a referentially-malformed analysis, but *not* judging whether the analysis is actually correct; that's still on you.


## `marimo` notebooks

`marimo/` holds `cts_text_menu.py` (a CTS file browser, predating this package build) and `hebrew_syntaxer_workflow.py`, modeled on `arsgrammatica`'s `latin_syntaxer_workflow.py`:

```bash
marimo edit marimo/hebrew_syntaxer_workflow.py   # interactive
marimo run marimo/hebrew_syntaxer_workflow.py    # read-only app view
```

Needs the same `.env` as `diqduq_main.py` (`API_BASE`/`MODEL`/`API_KEY`) -- it reuses `diqduq_main._configure_lm()` rather than duplicating its own LM setup. Enter a base URN (context), a passage reference, and the Hebrew text to analyze, then submit the form; the notebook runs `analyze_passage()` and displays the result several ways: a Mermaid diagram of the `tokengraph` (via `tokengraph_to_mermaid()`), plain reconstructed text (`tokengraph_to_text()`), an HTML reading view highlighted by verbal unit with cantillation marks dropped (`tokengraph_to_html(..., include_cantillation=False)`), and the same highlighting indented by depth of subordination (`tokengraph_to_depth_html()`). It also shows the dspy reasoning trace, optional token/cost/prompt inspection, and lets you download the analysis (as `.cex` or `.txt`, via `serialize_analyses()`) or the Mermaid diagram source.

Any further notebooks analogous to `arsgrammatica`'s `syntaxer.py` / `latin_syntaxer_ctsdata.py` / `latin_syntaxer_review.py` (see that project's own USAGE.md for what each of those does) remain unimplemented for now.


## Files

- `diqduq_main.py` — command-line entry point: loads `.env`, configures the LM, and runs an analysis for a passage given on the command line.
- `calibrate_max_tokens.py` — loads `.env`, configures the LM, and fits `diqduq/token_budget.py`'s `max_tokens` estimate against real completion-token usage over `GOLD_EXAMPLES` (see "Estimating and enforcing a `max_tokens` budget" above).
- `diqduq/` — the package with the actual analysis logic:
  - `models.py` — pydantic models for `CitedText`, `Token`, `Sentence`, `VerbalExpression`, and `TokenAnalysis`, matching the fields and relation labels from `syntax_model.md`.
  - `segmentation_dspy.py` — the DSPy signature (`SegmentPassage`) that segments citation-labeled source text into sentences and tokens, assigning stable ids (`t0`, `t1`, ...) and tracking which citation each token came from.
  - `hebrew_syntax_dspy.py` — the DSPy signature (`SyntaxAnalysis`) that takes a sentence's tokens and produces `verbalunits` + `tokengraph`, plus `validate()` and `print_analysis()`.
  - `pipeline.py` — ties the two stages together: `analyze_sources()` runs the full pipeline over citation-labeled input and analyzes every sentence it finds (via `token_budget.analyze_with_retry()`, not `analyze()` directly -- see above); `analyze_passage()` is the convenience wrapper for a single bare passage string; `combined_tokengraph()` concatenates results for diagramming.
  - `token_budget.py` — `estimate_max_tokens()` picks a `max_tokens` budget for a SyntaxAnalysis call from a passage's token count, using a fit `calibrate_max_tokens.py` writes to `token_budget_calibration.json` (or an untuned fallback before that's ever been run); `analyze_with_retry()` wraps `analyze()` with that estimate and retries with a larger budget if the result still comes back truncated. `estimate_segmentation_max_tokens()`/`segment_with_retry()` do the same for the segmentation stage, from an uncalibrated input-character-count proxy rather than a real fit (see "Estimating and enforcing a `max_tokens` budget" above).
  - `serialization.py` — `serialize_analyses()`/`write_analyses()`/`read_analyses()` save and reload `sentences`/`verbalunits`/`tokengraph` as one deterministic, pipe-delimited plain-text file, or as an equivalent in-memory string; `split_analysis_by_sentence()` splits `read_analyses()`'s flat, whole-file lists back into one `(tokengraph, verbalunits)` slice per sentence (see "Saving and loading analyses" above).
  - `mermaid.py` — turns a `tokengraph` into a Mermaid flowchart: one node per substantive token, one labelled edge per `relatedtoken1`/`relationship1` and `relatedtoken2`/`relationship2` pair, colored by verbal unit, with same-depth verbal-unit anchors chained together via Mermaid's invisible-link syntax so the layout also respects depth of subordination.
  - `verbal_units.py` — `assign_verbal_units()` partitions a `tokengraph` into the verbal units its own relations imply, purely from the existing graph structure (no extra LM call); `assign_verbal_unit_colors()` builds on that to assign each verbal unit a stable palette color, in the same first-appearance order `mermaid.py` uses for its node coloring -- the single shared source both `mermaid.py` and `rendering.py` draw on so their colorings always agree. `compute_subordination_depths()` computes each verbal expression's *depth of subordination* (0 for an independent clause, 1 for a clause it introduces, and so on); `max_subordination_depth()` reduces that to the single deepest depth reached anywhere in the passage. `find_unanchored_coordinated_verbs()` is a heuristic sanity check, separate from `validate()`: it flags a "coordinating conjunction" pair where one side anchors its own verbal unit and the other doesn't. Returns a list of warning strings (empty if nothing looks wrong).
  - `rendering.py` — `tokengraph_to_text()` reconstructs a continuous, readable plain-text string from a `tokengraph`, with correct spacing/joining around the article, inseparable prepositions, proclitic conjunctions, enclitic pronouns, and the maqaf hyphen (unlike a plain `" ".join(...)`, which would put a space before every token, splitting fused Hebrew words apart). `tokengraph_to_html()` does the same join, but as an HTML string with lexical tokens (and any coordinating conjunction) wrapped in verbal-unit-colored `<span>`s. `tokengraph_to_depth_html()` renders the same colored tokens grouped into per-verbal-unit blocks, each CSS-indented by its depth of subordination; its optional `depth` parameter caps rendering to blocks at or below that depth.
  - `ctsdata.py` — `read_ctsdata()` reads one or more `#!ctsdata`-blocked, pipe-delimited files of citable passages (see "Reading passages from a delimited-text source file" above).
  - `gepa_metric.py` — `syntax_metric()`, the scoring/feedback function `optimize_gepa.py` uses with `dspy.GEPA` (see OPTIMIZING.md).
  - `__init__.py` — re-exports the public names above, so callers do `from diqduq import ...` rather than reaching into submodules.
- `optimize_gepa.py` — GEPA optimization pipeline for `SyntaxAnalysis`'s prompt (see OPTIMIZING.md).
- `tests/` — a pytest suite covering models, segmentation, analysis, validation, and coverage of the scheme's relation/type vocabulary (see TESTING.md).
- `docs/build_api_docs.py` — regenerates `docs/diqduq-api-docs.html`, a single self-contained HTML page documenting every name in `diqduq.__all__`, built with `pdoc` straight from the package's own docstrings and type hints. Run `python docs/build_api_docs.py` after changing a public docstring or signature to refresh it; requires `pdoc` (`pip install pdoc --break-system-packages`).
- `marimo/` — interactive notebooks (see "`marimo` notebooks" above): the pre-existing `cts_text_menu.py`, plus `hebrew_syntaxer_workflow.py` for interactively analyzing a passage and viewing/downloading the result.
- `syntax_model.md` — the authoritative description of the analytic scheme itself: verbal-expression categories, tokenization rules, and syntactic relations.


## Extending the scheme

`syntax_model.md` says the current relation set is partial (its own "TBA" section names two constructions not yet covered: subordinating conjunctions and the relative pronoun אֲשֶׁר). To add a new relation:

1. Add the new label to `RelationLabel` in `diqduq/models.py`.
2. Describe when to use it in `SyntaxAnalysis`'s docstring in
   `diqduq/hebrew_syntax_dspy.py`, following the pattern of the existing relations
   (which token gets `relatedtoken1`/`relationship1`, which gets the
   corresponding value on the other end).
3. Add a gold example exercising it to `tests/fixtures/gold_examples.py` and
   re-run `pytest` to confirm the models still validate before trying it
   against the real LM.
