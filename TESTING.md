# Testing without network access

```bash
pytest
```

Runs the whole suite against `tests/`, using DSPy's `DummyLM` in place of a real LM call — useful for confirming the models/signatures/pipeline still fit together after you change something, without spending API calls. Tests that call the actual configured LM are marked `live` and skipped by default (they're the only way to check the LM itself gets a scenario right, not just that the code can represent a correct answer); run them explicitly with:

```bash
pytest -m live
```

(`live` tests need a working `.env`, same as `diqduq_main.py`; they skip gracefully if `API_KEY`/`MODEL` isn't set.)

Some standard `pytest` shorthands:

- `pytest` to run all.
- `pytest -v` for per-test names instead of dots.
- `pytest tests/test_gold_examples.py` to run just one file.
- `pytest -k coverage` to run only tests matching a substring.
- `pytest --collect-only` if you just want to see what it discovered without running anything.
