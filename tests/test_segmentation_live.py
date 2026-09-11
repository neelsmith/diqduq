"""
A live-LM smoke test for segmentation_dspy.py -- skipped by default (see
notes/TESTING.md); run with `pytest -m live` once a real .env is configured.
Modeled on arsgrammatica's test_segmentation_live.py.
"""

import pytest

from diqduq.models import CitedText
from diqduq.segmentation_dspy import segment_sources


@pytest.mark.live
def test_segments_a_real_verse_into_at_least_one_sentence(real_lm):
    sources = [
        CitedText(
            citation="urn:cts:compnov:bible.genesis.masoretic:1.1",
            text="בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃",
        )
    ]
    sentences = segment_sources(sources)
    assert len(sentences) >= 1
    assert all(sentence.tokens for sentence in sentences)
    # Every token should carry the one citation given.
    for sentence in sentences:
        for tok in sentence.tokens:
            assert tok.citation == sources[0].citation
