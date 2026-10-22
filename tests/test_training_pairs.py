"""
Training pairs for the embedder.

The properties that matter: positives really contain the evidence, the hard
negative never does, and nothing is built from an unanswerable question.
"""

from docusense.evaluation.qasper_harness import normalize_text
from docusense.evaluation.qasper_loader import QASPERPaper, QASPERQuestion
from docusense.training.qasper_pairs import build_pairs, retrieval_eval_set

FILLER = (
    "This paragraph discusses unrelated background material about the field "
    "and its history in enough words to form a chunk of its own. "
) * 30


def _paper() -> QASPERPaper:
    evidence = (
        "We train the tagger on the Penn Treebank and report accuracy on "
        "section 23, where it reaches 97.3 percent."
    )
    sections = [
        {"section": "Introduction", "paragraphs": [FILLER]},
        {"section": "Related Work", "paragraphs": [FILLER.replace("field", "area")]},
        {"section": "Experiments", "paragraphs": [evidence]},
        {"section": "Analysis", "paragraphs": [FILLER.replace("history", "trends")]},
        {"section": "Conclusion", "paragraphs": [FILLER.replace("material", "notes")]},
    ]
    return QASPERPaper(
        paper_id="p1",
        title="A Tagger",
        abstract="We present a tagger.",
        full_text_sections=sections,
        questions=[
            QASPERQuestion(
                question="Which treebank is the tagger trained on?",
                answers=["Penn Treebank"],
                evidence=[evidence],
            ),
            QASPERQuestion(question="Is it open source?", is_unanswerable=True),
        ],
    )


def test_positive_contains_the_evidence():
    pairs, _, stats = build_pairs([_paper()])

    assert stats["unanswerable"] == 1
    assert pairs, "the answerable question should yield a pair"
    for pair in pairs:
        assert "penn treebank" in normalize_text(pair.positive)


def test_hard_negative_is_never_evidence():
    pairs, _, _ = build_pairs([_paper()])

    for pair in pairs:
        if pair.negative is not None:
            assert "penn treebank" not in normalize_text(pair.negative)
            assert pair.negative != pair.positive


def test_eval_set_points_at_real_chunks():
    paper = _paper()
    _, by_paper, _ = build_pairs([paper])
    queries, corpus, relevant = retrieval_eval_set([paper], by_paper)

    assert len(queries) == 1
    (qid,) = queries
    assert relevant[qid] <= set(corpus)


def test_every_candidate_negative_is_kept():
    pairs, _, _ = build_pairs([_paper()])
    for pair in pairs:
        if pair.negative is not None:
            assert pair.negative in pair.negatives
        for negative in pair.negatives:
            assert "penn treebank" not in normalize_text(negative)
