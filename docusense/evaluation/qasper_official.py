"""
QASPER's official Answer-F1 and Evidence-F1.

The scoring functions below are the ones in `qasper_evaluator.py`, released
by AllenAI with the dataset, reproduced so the metric can be
unit-tested and run without the dataset directory. The answer normalization
and token F1 are SQuAD v1.1's, by way of that script. Nothing about the
scoring is changed: `scripts/qasper_answers.py` also runs the official script
itself on the same predictions file when it is present, and refuses to report
if the two disagree.

WHAT THE METRIC IS
------------------
Every question has several independent annotations. A prediction is scored
against each and keeps the best:

    Answer-F1    token F1 against the reference answer -- the extractive
                 spans joined with ", ", the free-form answer, "Yes"/"No", or
                 "Unanswerable"
    Evidence-F1  F1 between the set of predicted paragraphs and the set the
                 annotator cited

Reference answers are short (a median of a few words), so a long prose answer
scores low even when it is right. That is a property of the benchmark, and why
the product's prose answers are not what is submitted here; see
scripts/qasper_answers.py.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def paragraph_f1_score(prediction: List[str], ground_truth: List[str]) -> float:
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def get_answers_and_evidence(
    data: Dict[str, Any],
    text_evidence_only: bool = True,
    paper_ids: Optional[Iterable[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Gold references per question id, from the raw released JSON.

    `paper_ids` restricts the gold set to a subset of papers. The official
    script scores every question in the file and counts a missing prediction
    as zero, so scoring an 80-paper sample against the whole test file would
    report a number about 336 papers nobody predicted on.
    """
    wanted = set(paper_ids) if paper_ids is not None else None
    answers_and_evidence: Dict[str, List[Dict[str, Any]]] = {}

    for paper_id, paper_data in data.items():
        if wanted is not None and paper_id not in wanted:
            continue
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                    continue
                if answer_info["extractive_spans"]:
                    answer = ", ".join(answer_info["extractive_spans"])
                    answer_type = "extractive"
                elif answer_info["free_form_answer"]:
                    answer = answer_info["free_form_answer"]
                    answer_type = "abstractive"
                elif answer_info["yes_no"]:
                    answer = "Yes"
                    answer_type = "boolean"
                elif answer_info["yes_no"] is not None:
                    answer = "No"
                    answer_type = "boolean"
                else:
                    raise RuntimeError(
                        f"Annotation {answer_info.get('annotation_id')} does not contain an answer"
                    )
                if text_evidence_only:
                    evidence = [t for t in answer_info["evidence"] if "FLOAT SELECTED" not in t]
                else:
                    evidence = answer_info["evidence"]
                references.append({"answer": answer, "evidence": evidence, "type": answer_type})
            answers_and_evidence[question_id] = references

    return answers_and_evidence


def evaluate(
    gold: Dict[str, List[Dict[str, Any]]],
    predicted: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score predictions ({question_id: {"answer", "evidence"}}) against gold.

    Returns the official script's output, plus the per-question scores so a
    paired comparison between two systems is possible.
    """
    max_answer_f1s: List[float] = []
    max_evidence_f1s: List[float] = []
    by_type: Dict[str, List[float]] = {
        "extractive": [], "abstractive": [], "boolean": [], "none": [],
    }
    per_question: Dict[str, Dict[str, float]] = {}
    num_missing = 0

    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing += 1
            max_answer_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            continue
        answer_f1s_and_types = [
            (token_f1_score(predicted[question_id]["answer"], ref["answer"]), ref["type"])
            for ref in references
        ]
        max_answer_f1, answer_type = sorted(
            answer_f1s_and_types, key=lambda x: x[0], reverse=True
        )[0]
        max_answer_f1s.append(max_answer_f1)
        by_type[answer_type].append(max_answer_f1)
        max_evidence_f1 = max(
            paragraph_f1_score(predicted[question_id]["evidence"], ref["evidence"])
            for ref in references
        )
        max_evidence_f1s.append(max_evidence_f1)
        per_question[question_id] = {
            "answer_f1": max_answer_f1,
            "evidence_f1": max_evidence_f1,
            "type": answer_type,
        }

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {k: mean(v) for k, v in by_type.items()},
        "Questions by type": {k: len(v) for k, v in by_type.items()},
        "Evidence F1": mean(max_evidence_f1s),
        "Missing predictions": num_missing,
        "per_question": per_question,
    }


__all__ = [
    "evaluate",
    "get_answers_and_evidence",
    "normalize_answer",
    "paragraph_f1_score",
    "token_f1_score",
]
