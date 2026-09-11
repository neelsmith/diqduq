"""
DSPy program that segments a sequence of citation-labeled Biblical Hebrew
source units into sentences, and each sentence into tokens, per
syntax_model.md's "Tokenization" section -- while also tracking which
citation each token came from.

Input is `sources: List[CitedText]` rather than a single passage string
specifically so that a sentence spanning more than one citation unit (rare,
but possible if a verse-final sof pasuq doesn't align with a CitedText
boundary) is representable: sentence boundaries do not need to respect
CitedText boundaries, but every token still records the citation it came
from via Token.citation.

This is a separate stage from SyntaxAnalysis (hebrew_syntax_dspy.py) on
purpose, following arsgrammatica's own two-stage design: SegmentPassage's
output (List[Sentence]) feeds SyntaxAnalysis as `tokens: List[Token]` per
sentence. See pipeline.py for the module that ties the two stages
together, including its analyze_passage() convenience wrapper.

Run this file directly for a quick smoke test against the configured LM:
    python -m diqduq.segmentation_dspy
"""

from typing import List

import dspy

from .models import CitedText, Sentence


class SegmentPassage(dspy.Signature):
    """Segment a sequence of citation-labeled Biblical Hebrew source units
    into sentences, and each sentence into tokens, following
    syntax_model.md's tokenization scheme.

    `sources` is given in reading order; treat its units' text as one
    continuous passage for sentence-splitting purposes. Every token you
    produce must carry the `citation` of whichever `sources` unit its
    surface text came from, even for a sentence that spans more than one
    unit.

    - A "sentence" here corresponds to one verse-level unit of analysis:
      split primarily at a *sof pasuq* (the cantillation mark ׃ that ends a
      Masoretic verse), and also before a paragraph marker (פ/ס) that
      follows one. A verse may contain more than one independent verbal
      expression -- e.g. a chain of narrative wayyiqtol clauses joined by
      repeated וְ (see hebrew_syntax_dspy.SyntaxAnalysis's docstring on
      coordinating conjunctions), or a framing verb of speech together with
      its directly quoted content -- all of that still belongs to ONE
      sentence, analyzed together in a single SyntaxAnalysis call, exactly
      as syntax_model.md's own worked examples do (e.g. Genesis 1.3, where
      וַיֹּאמֶר and the direct quote יְהִי אֹור and the notice-of-fulfillment
      וַיְהִי־אֹור are all one unit). Do not split a sentence at a clause
      boundary just because a new independent verb or a quotation begins --
      only at a *sof pasuq* (or, if a verse genuinely contains no internal
      sof pasuq at all and none is expected before the passage ends, at the
      passage's own end).

    - Within each sentence, segment tokens as: *cantillation* (any of the
      te'amim -- e.g. sof pasuq ׃, silluq, atnach), *paragraph* (a
      standalone פ or ס marking a semantic division), *enclitic pronoun*
      (a pronoun bound as the object of a preposition or verb, or as a
      possessive with a noun), *proclitic conjunction* (specifically the
      conjunction וְ, including its vocalized forms such as וַ/וּ/וִ before a
      following consonant/vowel), *maqaf* (the joining hyphen ־), *lexical*
      (a continuous alphabetic sequence together with its own niqqud,
      dagesh, mappiq, and sin/shin dot -- but never cantillation marks,
      which are always their own separate token), or *editorial* (any
      other Unicode punctuation character or editorial mark, such as the
      masora circle).

    - A lexical word that is itself prefixed with the article הַ (or its
      vocalized variants, e.g. הָ before a guttural) or with an inseparable
      preposition (בְּ/כְּ/לְ) IS split from that prefix: the article or
      preposition becomes its own *lexical* token immediately before the
      noun/verb it attaches to, even though nothing separates them in the
      unpointed surface text. This is required so that relations such as
      "article" and "object of preposition" (see
      hebrew_syntax_dspy.SyntaxAnalysis's docstring) have a token of their
      own to attach to. For example הַשָּׁמַיִם is TWO lexical tokens, הַ
      then שָׁמַיִם, not one fused token; likewise בְּרֵאשִׁית is TWO
      lexical tokens, בְּ then רֵאשִׁית. (The proclitic conjunction וְ and
      the enclitic pronoun suffixes are split the same way, but are tagged
      with their own tokentypes -- *proclitic conjunction* and *enclitic
      pronoun* -- rather than *lexical*.)

    - Assign token ids sequentially across the WHOLE input, in reading
      order: t0, t1, t2, .... Do not restart numbering at each sentence or
      at each source unit. Every token, across every sentence and every
      source unit, has a unique id, and running this on the same `sources`
      again must produce the same ids for the same tokens.
    """

    sources: List[CitedText] = dspy.InputField(
        desc="Citation-labeled source units, in reading order, to segment as one continuous passage."
    )
    sentences: List[Sentence] = dspy.OutputField(
        desc="The sentences found across all of `sources`, in order. Token ids are global (see instructions); each token's `citation` names the source unit it came from."
    )


segment = dspy.ChainOfThought(SegmentPassage)


def segment_sources(sources: List[CitedText]) -> List[Sentence]:
    """Run the segmentation stage and return its sentences."""
    result = segment(sources=sources)
    return result.sentences
