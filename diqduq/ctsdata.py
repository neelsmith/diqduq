"""
Reads a delimited-text *source* file listing Biblical Hebrew passages to
analyze -- each one identified by a CTS URN and paired with its own text
content -- distinct from serialization.py's format (which reads/writes the
*results* of an analysis). Modeled directly on (and file-format-compatible
with) arsgrammatica's ctsdata.py. This is the input side of the same
workflow: pick a passage out of a file like this one, then hand its text
and citation to analyze_passage() exactly as if it had been typed in by
hand.

File shape: one or more blocks, each introduced by the label line
'#!ctsdata' alone on its own line, immediately followed by the header line
'urn|text' (using '|' as the column delimiter by default -- pass a
different `delimiter` to read_ctsdata() if the file itself uses another
character), then one data row per passage:

    #!ctsdata
    urn|text
    urn:cts:compnov:bible.genesis.masoretic:1.1|בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃

Each row's own urn column is a 5-part, colon-separated CTS URN (e.g.
'urn:cts:compnov:bible.genesis.masoretic:1.1'); read_ctsdata() splits it
into the first 4 parts -- rejoined with ':', plus a trailing ':' -- as
`urnbase`, and the 5th part as `citation`. For the example above that's
'urn:cts:compnov:bible.genesis.masoretic:' and '1.1'.

Multiple '#!ctsdata' blocks are allowed in one file -- every instance's
rows are concatenated, in file order, into the one list read_ctsdata()
returns.

read_ctsdata() is deliberately strict, matching serialization.py's own
read_analyses(): a missing block, a header line that doesn't match exactly,
a row that isn't exactly 2 columns, a blank urn or text column, or a urn
that doesn't split into exactly 5 colon-separated parts, all raise
ValueError immediately, naming the offending line, rather than silently
skipping or guessing.
"""

from dataclasses import dataclass
from typing import List

CTSDATA_LABEL = "#!ctsdata"


@dataclass
class CtsDataRow:
    """One passage from a `#!ctsdata` source file: `urnbase` (the first 4
    colon-separated parts of the row's own CTS URN, rejoined with ':', plus
    a trailing ':') and `citation` (the URN's 5th part) together
    reconstruct the full URN as `urnbase + citation`. `text` is the
    passage's own surface text, verbatim."""

    urnbase: str
    citation: str
    text: str


def read_ctsdata(path: str, delimiter: str = "|") -> List[CtsDataRow]:
    """Read every `#!ctsdata` block in `path` and return their rows,
    concatenated in file order, as a list of CtsDataRow -- see this
    module's docstring for the file shape and what counts as malformed.

    `delimiter` is the column separator used both for the header line
    ('urn' + delimiter + 'text') and for splitting each data row; '|' by
    default. Pass a different character if the source file's own text
    content might contain '|' (there is no escaping mechanism for whichever
    character is chosen as the delimiter).

    Raises ValueError, naming the offending line, for: a data line
    appearing before any '#!ctsdata' label; a label line with no header
    line before the next block or before the file ends; a header line that
    doesn't match `delimiter`-joined 'urn'/'text' exactly; a data row that
    isn't exactly 2 columns; a blank urn or text column; or a urn that
    doesn't split into exactly 5 colon-separated parts. Raises ValueError
    (not returning an empty list) if the file has no '#!ctsdata' block at
    all, so a caller can't mistake "wrong file" for "file with zero
    passages".
    """
    expected_header = delimiter.join(["urn", "text"])

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    rows: List[CtsDataRow] = []
    seen_block = False
    awaiting_header = False

    for line_no, line in enumerate(raw_lines, start=1):
        if line.strip() == "":
            continue

        if line == CTSDATA_LABEL:
            if awaiting_header:
                raise ValueError(
                    f"line {line_no}: a {CTSDATA_LABEL!r} block has a label "
                    "line but no header line before the next block starts"
                )
            seen_block = True
            awaiting_header = True
            continue

        if not seen_block:
            raise ValueError(
                f"line {line_no}: data line {line!r} appears before any "
                f"{CTSDATA_LABEL!r} block label"
            )

        if awaiting_header:
            if line != expected_header:
                raise ValueError(
                    f"line {line_no}: expected header {expected_header!r} "
                    f"for a {CTSDATA_LABEL!r} block, got {line!r}"
                )
            awaiting_header = False
            continue

        parts = line.split(delimiter)
        if len(parts) != 2:
            raise ValueError(
                f"line {line_no}: {CTSDATA_LABEL!r} row has {len(parts)} "
                f"column(s) (delimiter {delimiter!r}), expected 2: {line!r}"
            )
        urn, text = parts
        if urn == "":
            raise ValueError(f"line {line_no}: {CTSDATA_LABEL!r} row has an empty urn column")
        if text == "":
            raise ValueError(f"line {line_no}: {CTSDATA_LABEL!r} row has an empty text column")

        urn_parts = urn.split(":")
        if len(urn_parts) != 5:
            raise ValueError(
                f"line {line_no}: urn {urn!r} has {len(urn_parts)} "
                "colon-separated part(s), expected 5 (e.g. "
                "'urn:cts:compnov:bible.genesis.masoretic:1.1')"
            )
        citation = urn_parts[4]
        if citation == "":
            raise ValueError(
                f"line {line_no}: urn {urn!r} has an empty final (citation) part"
            )
        urnbase = ":".join(urn_parts[:4]) + ":"

        rows.append(CtsDataRow(urnbase=urnbase, citation=citation, text=text))

    if not seen_block:
        raise ValueError(f"file has no {CTSDATA_LABEL!r} block")
    if awaiting_header:
        raise ValueError(
            f"a {CTSDATA_LABEL!r} block has a label line but no header "
            "line (and no data) -- the file ends too early"
        )

    return rows
