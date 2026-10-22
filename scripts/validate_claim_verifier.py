"""
How well does the claim verifier tell supported claims from unsupported ones?

    python scripts/validate_claim_verifier.py
    python scripts/validate_claim_verifier.py --model-file onnx/model.onnx   # fp32

A faithfulness number is only as good as the judge behind it, so the judge is
measured first, on QASPER *dev* (the test split is kept for the system).

THE LABELLED SET -- NO HAND LABELS, NO LLM JUDGE
-----------------------------------------------
QASPER's free-form answers are claims written by people who had the paper
open, about evidence paragraphs they marked. That gives labels by
construction:

    supported      an annotator's free-form answer, against the evidence
                   paragraphs that same annotator cited
    unsupported    (a) the same answer against the evidence for a *different*
                   question about the same paper -- same topic, same
                   vocabulary, wrong passage; the hard case
                   (b) the answer with one number changed, against its own
                   evidence -- the "right paper, wrong number" failure
                   citation checking cannot see

Only answers of six or more words are used; "BERT, ELMo" is an answer, not a
claim. Free-form answers are often abstractive ("They use a BiLSTM because
it handles long contexts"), so the supported set is harder than copying a
sentence out of the source, and recall on it is the honest number.

OUTPUT
------
ROC-AUC of the entailment score, precision / recall / F1 for "unsupported" at
each threshold, and the threshold that maximises balanced accuracy -- which is
what claim_support_threshold is set to.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

DEV = "data/benchmarks/qasper/qasper-dev-v0.3.json"
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")


def perturb_number(text: str, rng: random.Random):
    """Change one number in the text to a different, plausible one."""
    matches = list(_NUMBER.finditer(text))
    if not matches:
        return None
    m = rng.choice(matches)
    value = m.group(1)
    if "." in value:
        new = f"{float(value) + rng.choice([-1, 1]) * rng.uniform(1.0, 9.0):.{len(value.split('.')[1])}f}"
    else:
        n = int(value)
        new = str(n + rng.choice([-1, 1]) * max(1, int(abs(n) * rng.uniform(0.2, 0.6)) or 1))
    if new == value or new.startswith("-"):
        # Never negative, never unchanged: move up instead.
        new = str(round(float(value) + 7, 3)).rstrip("0").rstrip(".") if "." in value else str(int(value) + 7)
    return text[: m.start(1)] + new + text[m.end(1):]


def build_examples(data, seed: int, min_words: int = 6):
    rng = random.Random(seed)
    examples = []  # (claim, sources, label, kind)

    for paper in data.values():
        per_question = []
        for qa in paper["qas"]:
            for ann in qa["answers"]:
                a = ann["answer"]
                answer = (a.get("free_form_answer") or "").strip()
                evidence = [e for e in a.get("evidence", []) if "FLOAT SELECTED" not in e]
                if len(answer.split()) >= min_words and evidence:
                    per_question.append((qa["question_id"], answer, evidence))
                    break  # one annotation per question

        for i, (qid, answer, evidence) in enumerate(per_question):
            examples.append((answer, evidence, 1, "supported"))

            others = [p for p in per_question if p[0] != qid
                      and not set(p[2]) & set(evidence)]
            if others:
                _, _, other_evidence = rng.choice(others)
                examples.append((answer, other_evidence, 0, "other_question_evidence"))

            changed = perturb_number(answer, rng)
            if changed and changed != answer:
                examples.append((changed, evidence, 0, "number_changed"))

    return examples


def roc_auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEV)
    parser.add_argument("--repo", default=None, help="NLI model repo (default: settings)")
    parser.add_argument("--model-file", default=None, help="ONNX file inside the NLI repo")
    parser.add_argument("--limit", type=int, default=None, help="Cap on supported claims")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--out", default="data/benchmarks/claim_verifier_validation.json")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from docusense.generation.claim_verifier import ClaimVerifier, NLIModel

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    examples = build_examples(data, args.seed)
    if args.limit:
        keep_ids = set()
        supported_seen = 0
        for i, ex in enumerate(examples):
            if ex[2] == 1:
                supported_seen += 1
            if supported_seen <= args.limit:
                keep_ids.add(i)
        examples = [ex for i, ex in enumerate(examples) if i in keep_ids]

    counts = {}
    for _, _, _, kind in examples:
        counts[kind] = counts.get(kind, 0) + 1
    print(json.dumps(counts))

    model = NLIModel(repo=args.repo, model_file=args.model_file)
    verifier = ClaimVerifier(model=model, threshold=0.5)

    started = time.time()
    scores = []
    for n, (claim, sources, _, _) in enumerate(examples, 1):
        # The claim is passed whole: these are annotator answers, and the
        # question is whether the passage supports what they wrote.
        result = verifier.verify(claim, sources, claims=[claim])
        scores.append(result.claims[0].entailment)
        if n % 200 == 0:
            print(f"  {n}/{len(examples)} ({time.time() - started:.0f}s)")
    elapsed = time.time() - started

    labels = [ex[2] for ex in examples]
    kinds = [ex[3] for ex in examples]

    def at_threshold(t):
        # "Positive" here is the flag: the claim is unsupported.
        flagged = [s < t for s in scores]
        tp = sum(f and y == 0 for f, y in zip(flagged, labels))
        fp = sum(f and y == 1 for f, y in zip(flagged, labels))
        fn = sum((not f) and y == 0 for f, y in zip(flagged, labels))
        tn = sum((not f) and y == 1 for f, y in zip(flagged, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support_recall = tn / (tn + fp) if tn + fp else 0.0
        by_kind = {
            k: sum(f for f, kk in zip(flagged, kinds) if kk == k) / max(1, kinds.count(k))
            for k in sorted(set(kinds))
        }
        return {
            "threshold": t,
            "flag_precision": round(precision, 4),
            "flag_recall": round(recall, 4),
            "flag_f1": round(f1, 4),
            "supported_kept": round(support_recall, 4),
            "balanced_accuracy": round((recall + support_recall) / 2, 4),
            "flag_rate_by_kind": {k: round(v, 4) for k, v in by_kind.items()},
        }

    thresholds = [round(x * 0.05, 2) for x in range(1, 20)]
    table = [at_threshold(t) for t in thresholds]
    best = max(table, key=lambda r: r["balanced_accuracy"])

    report = {
        "dataset": args.dataset,
        "model": f"{model.repo}/{model.model_file}",
        "examples": counts,
        "roc_auc": round(roc_auc(scores, labels), 4),
        "roc_auc_by_negative_kind": {
            kind: round(roc_auc(
                [s for s, k in zip(scores, kinds) if k in ("supported", kind)],
                [y for y, k in zip(labels, kinds) if k in ("supported", kind)],
            ), 4)
            for kind in ("other_question_evidence", "number_changed")
        },
        "at_0.5": at_threshold(0.5),
        "best_balanced": best,
        "thresholds": table,
        "seconds": round(elapsed, 1),
        "ms_per_claim": round(1000 * elapsed / max(1, len(examples)), 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nROC-AUC {report['roc_auc']}  by kind {report['roc_auc_by_negative_kind']}")
    for row in (report["at_0.5"], best):
        print(json.dumps(row))
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
