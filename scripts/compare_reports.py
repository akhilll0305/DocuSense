"""
Compare one arm of two benchmark reports, question by question.

    python scripts/compare_reports.py \
        data/benchmarks/qasper_ablation_report.json hybrid_rerank \
        data/benchmarks/finetuned_ablation_report.json hybrid_rerank

The ablation compares arms inside one run. Changing the embedding model, the
chunk size or the reranker needs a second run over a re-ingested corpus, and
the two runs can only be compared pairwise if they asked the same questions in
the same order -- which this checks before computing anything, because a
paired test over misaligned lists is a confident number about nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docusense.evaluation.qasper_harness import paired_bootstrap  # noqa: E402


def arm(report: dict, name: str) -> dict:
    for entry in report["arms"]:
        if entry["arm"]["name"] == name:
            return entry
    raise SystemExit(f"arm '{name}' not in report; has {[a['arm']['name'] for a in report['arms']]}")


def main(argv) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    path_a, arm_a, path_b, arm_b = argv
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))

    qa, qb = a.get("queries"), b.get("queries")
    if qa is None or qb is None:
        print("ERROR: a report has no 'queries' list; re-run it with the current benchmark.py",
              file=sys.stderr)
        return 1
    if qa != qb:
        print(f"ERROR: the reports asked different questions ({len(qa)} vs {len(qb)})",
              file=sys.stderr)
        return 1

    ea, eb = arm(a, arm_a), arm(b, arm_b)
    label_a = f"{Path(path_a).stem}:{arm_a}"
    label_b = f"{Path(path_b).stem}:{arm_b}"

    print(f"{len(qa)} paired questions")
    print(f"{'metric':<10}{'A':>10}{'B':>10}{'delta':>10}{'95% CI':>22}{'p':>9}")
    out = []
    for metric, key in (("MRR", "reciprocal_rank"), ("NDCG@10", "ndcg_at_10")):
        c = paired_bootstrap(ea["per_query"][key], eb["per_query"][key], metric, label_a, label_b)
        out.append(c.to_dict())
        print(f"{metric:<10}{c.baseline_mean:>10.4f}{c.variant_mean:>10.4f}{c.delta:>+10.4f}"
              f"   [{c.ci_low:+.4f}, {c.ci_high:+.4f}]{c.p_value:>9.4f}")
    for m in ("P@1", "Recall@10", "MAP"):
        va, vb = ea["retrieval_metrics"][m], eb["retrieval_metrics"][m]
        print(f"{m:<10}{va:>10.4f}{vb:>10.4f}{vb - va:>+10.4f}")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
