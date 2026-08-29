"""
Round-trip tests for serialization.py -- modeled on arsgrammatica's
test_serialization.py (considerably condensed, since the format itself is
language-agnostic and already exercised at length there).
"""

from conftest import run_gold_example
from diqduq.models import IMPLIED_TOKENTYPES, Sentence, Token
from diqduq.serialization import read_analyses, serialize_analyses, split_analysis_by_sentence, write_analyses
from fixtures.gold_examples import GOLD_EXAMPLES

_CITATION = "urn:cts:compnov:bible.genesis.masoretic:1.1"


def _sentence_for(tokengraph, citation=_CITATION):
    return Sentence(
        tokens=[
            Token(id=tok.id, text=tok.token, citation=citation)
            for tok in tokengraph
            if tok.tokentype not in IMPLIED_TOKENTYPES
        ]
    )


def test_write_and_read_round_trips_genesis_1_1(tmp_path):
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("genesis_1_1"))
    _tokens, result = run_gold_example(example)
    sentence = _sentence_for(result.tokengraph)

    path = tmp_path / "analysis.cex"
    warnings = write_analyses([sentence], result.verbalunits, result.tokengraph, str(path))
    assert warnings == []

    tokengraph2, verbalunits2, sentences2 = read_analyses(str(path))

    assert [t.id for t in tokengraph2] == [t.id for t in result.tokengraph]
    assert [t.token for t in tokengraph2] == [t.token for t in result.tokengraph]
    assert [v.id for v in verbalunits2] == [v.id for v in result.verbalunits]
    assert [t.id for t in sentences2[0].tokens] == [t.id for t in sentence.tokens]
    assert all(t.citation == _CITATION for t in sentences2[0].tokens)


def test_root_sentinel_survives_the_round_trip(tmp_path):
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("genesis_1_1"))
    _tokens, result = run_gold_example(example)
    sentence = _sentence_for(result.tokengraph)

    path = tmp_path / "analysis.cex"
    write_analyses([sentence], result.verbalunits, result.tokengraph, str(path))
    tokengraph2, _verbalunits2, _sentences2 = read_analyses(str(path))

    root_tok = next(t for t in tokengraph2 if t.id == "t2")
    assert root_tok.relatedtoken1 == "root"


def test_implied_token_round_trips_with_none_text(tmp_path):
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("implied_sum"))
    _tokens, result = run_gold_example(example)
    sentence = _sentence_for(result.tokengraph)

    path = tmp_path / "analysis.cex"
    write_analyses([sentence], result.verbalunits, result.tokengraph, str(path))
    tokengraph2, _verbalunits2, sentences2 = read_analyses(str(path))

    implied = next(t for t in tokengraph2 if t.tokentype == "implied sum")
    assert implied.token is None
    # The implied token was never part of the sentence's own `tokens` list.
    assert all(t.id != implied.id for t in sentences2[0].tokens)


def test_split_analysis_by_sentence_matches_write_analyses_input():
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("genesis_2_3"))
    _tokens, result = run_gold_example(example)
    sentence = _sentence_for(result.tokengraph, citation="urn:cts:compnov:bible.genesis.masoretic:2.3")

    slices = split_analysis_by_sentence(result.tokengraph, result.verbalunits, [sentence])
    assert len(slices) == 1
    sentence_tokengraph, sentence_verbalunits = slices[0]
    assert [t.id for t in sentence_tokengraph] == [t.id for t in result.tokengraph]
    assert [v.id for v in sentence_verbalunits] == [v.id for v in result.verbalunits]


def test_serialize_analyses_rejects_a_pipe_character(monkeypatch):
    example = next(ex for ex in GOLD_EXAMPLES if ex.slug.startswith("genesis_1_1"))
    _tokens, result = run_gold_example(example)
    sentence = _sentence_for(result.tokengraph)
    result.tokengraph[0].lemma = "bad|value"
    try:
        serialize_analyses([sentence], result.verbalunits, result.tokengraph)
    except ValueError as e:
        assert "|" in str(e)
    else:
        raise AssertionError("expected ValueError for a '|' in a field value")
