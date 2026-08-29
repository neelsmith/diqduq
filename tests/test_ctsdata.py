"""
Tests for ctsdata.py's read_ctsdata() -- modeled on arsgrammatica's
test_ctsdata.py.
"""

import pytest

from diqduq.ctsdata import read_ctsdata


def _write(tmp_path, content):
    path = tmp_path / "passages.cex"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_reads_a_single_block(tmp_path):
    content = (
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:compnov:bible.genesis.masoretic:1.1|בְּרֵאשִׁית בָּרָא אֱלֹהִים\n"
        "urn:cts:compnov:bible.genesis.masoretic:1.2|וְהָאָרֶץ הָיְתָה תֹהוּ\n"
    )
    rows = read_ctsdata(_write(tmp_path, content))
    assert len(rows) == 2
    assert rows[0].urnbase == "urn:cts:compnov:bible.genesis.masoretic:"
    assert rows[0].citation == "1.1"
    assert rows[0].text == "בְּרֵאשִׁית בָּרָא אֱלֹהִים"


def test_concatenates_multiple_blocks_in_file_order(tmp_path):
    content = (
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:compnov:bible.genesis.masoretic:1.1|A\n"
        "\n"
        "#!ctsdata\n"
        "urn|text\n"
        "urn:cts:compnov:bible.genesis.masoretic:1.2|B\n"
    )
    rows = read_ctsdata(_write(tmp_path, content))
    assert [r.citation for r in rows] == ["1.1", "1.2"]


def test_missing_block_raises(tmp_path):
    with pytest.raises(ValueError, match="appears before any"):
        read_ctsdata(_write(tmp_path, "just some text\n"))


def test_truly_empty_file_raises_no_block_found(tmp_path):
    with pytest.raises(ValueError, match="no.*block"):
        read_ctsdata(_write(tmp_path, "\n\n"))


def test_malformed_urn_raises(tmp_path):
    content = "#!ctsdata\nurn|text\nnot-a-cts-urn|hello\n"
    with pytest.raises(ValueError, match="colon-separated"):
        read_ctsdata(_write(tmp_path, content))


def test_wrong_column_count_raises(tmp_path):
    content = "#!ctsdata\nurn|text\nurn:cts:compnov:bible.genesis.masoretic:1.1|too|many|columns\n"
    with pytest.raises(ValueError, match="column"):
        read_ctsdata(_write(tmp_path, content))


def test_empty_text_column_raises(tmp_path):
    content = "#!ctsdata\nurn|text\nurn:cts:compnov:bible.genesis.masoretic:1.1|\n"
    with pytest.raises(ValueError, match="empty text"):
        read_ctsdata(_write(tmp_path, content))


def test_custom_delimiter(tmp_path):
    content = "#!ctsdata\nurn\ttext\nurn:cts:compnov:bible.genesis.masoretic:1.1\tHello\n"
    rows = read_ctsdata(_write(tmp_path, content), delimiter="\t")
    assert rows[0].text == "Hello"
