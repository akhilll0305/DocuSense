"""
Normalising a model's reply into QASPER's answer form.

Token F1 against a short reference punishes every extra word, so what is
stripped here moves the score; these pin that only wrapping is removed.
"""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "qasper_answers", Path(__file__).resolve().parent.parent / "scripts" / "qasper_answers.py"
)
qasper_answers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qasper_answers)
clean_answer = qasper_answers.clean_answer


def test_reasoning_block_is_dropped():
    assert clean_answer("<think>the excerpt says 3,044</think>3,044 sentences") == "3,044 sentences"


def test_answer_prefix_and_quotes_are_dropped():
    assert clean_answer('"Answer: BERT and ELMo"') == "BERT and ELMo"
    assert clean_answer("Final answer: Yes") == "Yes"


def test_only_the_first_line_is_kept():
    assert clean_answer("SQuAD 1.1\nThis is because the paper...") == "SQuAD 1.1"


def test_unanswerable_is_canonical():
    assert clean_answer("Unanswerable - the excerpts do not say.") == "Unanswerable"
    assert clean_answer("unanswerable") == "Unanswerable"


def test_empty_reply():
    assert clean_answer("") == ""
    assert clean_answer(None) == ""
