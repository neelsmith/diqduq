# diqduq

`diqduq` is a python package leveraging LLMs with [dspy](https://dspy.ai) to analyze the syntax of passages of Biblical Hebrew.

It offers an alternative analytic scheme, designed to describe Biblical Hebrew syntax in terms convenient for research and teaching, documented in [`syntax_model.md`](syntax_model.md). The scheme is modeled on -- and the codebase closely mirrors -- [`arsgrammatica`](https://github.com/neelsmith/arsgrammatica), the same author's analyzer for Latin syntax.

Released under the [GNU General Public License v3 or later](LICENSE).


## Installing

To use `diqduq` from another project, install it straight from this repository (no PyPI account or release process needed):

```sh
pip install git+https://github.com/neelsmith/diqduq.git
```

That installs whatever's currently on the `main` branch. Pin to a specific branch, tag, or commit by appending `@<ref>`, e.g. `pip install git+https://github.com/neelsmith/diqduq.git@wip` for a development branch, or `@v0.1.0` once a version is tagged. Either way, only `diqduq/` itself is installed as a package -- `dspy`, `pydantic`, and `python-dotenv` come along automatically as declared dependencies; the marimo notebooks, tests, and other repo scripts are not part of the installed package and aren't needed to use it.

Working on `diqduq` itself (this repo checked out locally) rather than depending on it from elsewhere: `pip install -e .` from the repo root installs it in editable mode, so source edits take effect immediately without reinstalling.


## Using `diqduq`

- [USAGE.md](USAGE.md)
- [TESTING.md](TESTING.md)
- [OPTIMIZING.md](OPTIMIZING.md)
- [DEVELOPMENT.md](DEVELOPMENT.md) -- how the above fit together into one development loop
- [API documentation](docs/diqduq-api-docs.html)

See the [project issue tracker](https://github.com/neelsmith/diqduq/issues) for known gaps and work in progress -- in particular, `syntax_model.md` itself notes several constructions ("TBA": the functions of prepositions beyond "object of preposition", subordinating conjunctions, and the relative pronoun אֲשֶׁר) that the scheme doesn't cover yet.


## Background

- [`arsgrammatica`](https://github.com/neelsmith/arsgrammatica), the same author's DSPy-based analyzer for Latin syntax, which this project mirrors closely in architecture.
- Some initial work from 2023 on [an alternative to universal dependencies](https://neelsmith.github.io/GreekAndLatinSyntax/).
