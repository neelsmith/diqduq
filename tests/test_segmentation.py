"""
Tests for segmentation_dspy.py, driven by DummyLM against
fixtures/segmentation_examples.py -- modeled on arsgrammatica's
test_segmentation.py.
"""

import pytest

from conftest import run_segmentation_example
from fixtures.segmentation_examples import SEGMENTATION_EXAMPLES


@pytest.mark.parametrize("example", SEGMENTATION_EXAMPLES, ids=lambda ex: ex.slug)
def test_segmentation_example_ids_are_sequential_and_unique(example):
    sentences = run_segmentation_example(example)
    all_ids = [tok.id for sentence in sentences for tok in sentence.tokens]
    assert len(all_ids) == len(set(all_ids))


@pytest.mark.parametrize("example", SEGMENTATION_EXAMPLES, ids=lambda ex: ex.slug)
def test_segmentation_example_tokens_match_canned_sentences(example):
    sentences = run_segmentation_example(example)
    expected_sentences = example.canned_sentences["sentences"]
    assert len(sentences) == len(expected_sentences)
    for sentence, expected in zip(sentences, expected_sentences):
        got = [(tok.id, tok.text, tok.citation) for tok in sentence.tokens]
        want = [(tok["id"], tok["text"], tok["citation"]) for tok in expected["tokens"]]
        assert got == want


def test_genesis_1_1_splits_article_and_preposition_from_their_noun():
    example = next(ex for ex in SEGMENTATION_EXAMPLES if ex.slug.startswith("genesis_1_1"))
    sentences = run_segmentation_example(example)
    texts = [tok.text for tok in sentences[0].tokens]
    assert texts[0] == "בְּ"  # preposition, split from רֵאשִׁ֖ית
    assert texts[5] == "הַ"  # article, split from שָּׁמַ֖יִם
    assert texts[7] == "וְ"  # proclitic conjunction, its own token
