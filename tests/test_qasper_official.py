"""
The official QASPER metric, reproduced in docusense.evaluation.qasper_official.

These pin the behaviours a paraphrase would most likely get wrong: the max over
annotators, extractive spans joined before scoring, and how an empty
prediction scores against an unanswerable question.
"""

import pytest

from docusense.evaluation.qasper_official import (
    evaluate,
    get_answers_and_evidence,
    normalize_answer,
    paragraph_f1_score,
    token_f1_score,
)


def _raw():
    return {
        "p1": {"qas": [
            {"question_id": "q1", "answers": [
                {"answer": {"unanswerable": False, "extractive_spans": ["BERT", "ELMo"],
                            "free_form_answer": "", "yes_no": None,
                            "evidence": ["Para A", "FLOAT SELECTED: Table 1"]}},
                {"answer": {"unanswerable": False, "extractive_spans": [],
                            "free_form_answer": "they compare to BERT", "yes_no": None,
                            "evidence": ["Para B"]}},
            ]},
            {"question_id": "q2", "answers": [
                {"answer": {"unanswerable": True, "extractive_spans": [],
                            "free_form_answer": "", "yes_no": None, "evidence": []}},
            ]},
            {"question_id": "q3", "answers": [
                {"answer": {"unanswerable": False, "extractive_spans": [],
                            "free_form_answer": "", "yes_no": False, "evidence": ["Para C"]}},
            ]},
        ]},
        "p2": {"qas": [
            {"question_id": "q4", "answers": [
                {"answer": {"unanswerable": False, "extractive_spans": ["x"],
                            "free_form_answer": "", "yes_no": None, "evidence": []}},
            ]},
        ]},
    }


def test_normalization_matches_squad():
    assert normalize_answer("The  BERT, model!") == "bert model"


def test_token_f1():
    assert token_f1_score("BERT and ELMo", "BERT, ELMo") == pytest.approx(0.8)
    assert token_f1_score("nothing", "BERT") == 0.0


def test_paragraph_f1_empty_on_both_sides_is_perfect():
    assert paragraph_f1_score([], []) == 1.0
    assert paragraph_f1_score(["a"], []) == 0.0


def test_gold_parsing():
    gold = get_answers_and_evidence(_raw(), text_evidence_only=True)

    assert gold["q1"][0]["answer"] == "BERT, ELMo"
    assert gold["q1"][0]["evidence"] == ["Para A"]  # figure evidence dropped
    assert gold["q1"][1]["type"] == "abstractive"
    assert gold["q2"][0] == {"answer": "Unanswerable", "evidence": [], "type": "none"}
    assert gold["q3"][0]["answer"] == "No"  # yes_no False is an answer, not a gap


def test_paper_subset():
    gold = get_answers_and_evidence(_raw(), paper_ids=["p1"])
    assert set(gold) == {"q1", "q2", "q3"}


def test_best_annotator_wins():
    gold = get_answers_and_evidence(_raw(), paper_ids=["p1"])
    predicted = {
        "q1": {"answer": "they compare to BERT", "evidence": ["Para B"]},
        "q2": {"answer": "Unanswerable", "evidence": []},
        "q3": {"answer": "No", "evidence": ["Para C"]},
    }
    result = evaluate(gold, predicted)

    assert result["Answer F1"] == pytest.approx(1.0)
    assert result["Evidence F1"] == pytest.approx(1.0)
    assert result["Missing predictions"] == 0
    assert result["per_question"]["q1"]["type"] == "abstractive"


def test_missing_prediction_scores_zero():
    gold = get_answers_and_evidence(_raw(), paper_ids=["p1"])
    result = evaluate(gold, {"q2": {"answer": "Unanswerable", "evidence": []}})

    assert result["Missing predictions"] == 2
    assert result["Answer F1"] == pytest.approx(1 / 3)
