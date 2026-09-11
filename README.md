# diqduq

`diqduq` is a python package leveraging LLMs with [dspy](https://dspy.ai) to analyze the syntax of passages of Biblical Hebrew.

It offers an alternative analytic scheme, designed to describe Biblical Hebrew syntax in terms convenient for research and teaching, documented in [`syntax_model.md`](syntax_model.md). The scheme is modeled on -- and the codebase closely mirrors -- [`arsgrammatica`](https://github.com/neelsmith/arsgrammatica), the same author's analyzer for Latin syntax.

Released under the [GNU General Public License v3 or later](LICENSE).

## Status

Planning stages only. Beta release expected in late 2026.

## Work in progress

- Project [issue tracker](https://github.com/neelsmith/arsgrammatica/issues)
- [Documentation](https://neelsmith.github.io/diqduq)


## Related work

Parallel python packages for language-specific syntactic analysis:

- [asgrammatica](https://github.com/neelsmith/asgrammatica) for Latin
- [grammatike](https://github.com/neelsmith/grammatike) for Ancient Greek


Packages for working with universal syntax models:

- [udsyntax](https://github.com/neelsmith/udsyntax), a Python package to get dependency data from spaCy into a simple syntax graph format
- [aat](https://github.com/neelsmith/aat), a Python package implementing a reduced model of natural-language syntax, Agent-Action-Target