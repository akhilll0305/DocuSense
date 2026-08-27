"""
Benchmark Runner - Run evaluations and generate reports.

Phase 6: Evaluation & Metrics (Step 5)

Orchestrates running benchmarks on sample or QASPER datasets,
generating evaluation reports with all retrieval + answer metrics.

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger

from docusense.evaluation.evaluator import (
    RAGEvaluator,
    EvaluationResult,
    EvaluationSample
)
from docusense.evaluation.qasper_loader import QASPERLoader


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str = "docusense_benchmark"
    dataset_path: Optional[str] = None  # Path to QASPER JSON
    use_sample_dataset: bool = True     # Use built-in sample if no dataset
    max_samples: Optional[int] = None
    skip_unanswerable: bool = True
    evaluate_retrieval: bool = True
    evaluate_answers: bool = True
    output_dir: str = "data/benchmarks"


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    config: BenchmarkConfig
    result: EvaluationResult
    num_samples: int = 0
    benchmark_time: float = 0.0
    timestamp: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.config.name,
            "num_samples": self.num_samples,
            "benchmark_time": round(self.benchmark_time, 2),
            "timestamp": self.timestamp,
            "results": self.result.to_dict() if self.result else {},
            "summary": self.summary,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        lines = [
            f"═══ Benchmark Report: {self.config.name} ═══",
            f"  Samples: {self.num_samples}",
            f"  Time: {self.benchmark_time:.2f}s",
        ]
        if self.result:
            lines.append(str(self.result))
        if self.summary:
            lines.append(f"  Summary: {json.dumps(self.summary, indent=2)}")
        return "\n".join(lines)


class BenchmarkRunner:
    """
    Run evaluation benchmarks and generate reports.

    Usage:
        runner = BenchmarkRunner()

        # Quick benchmark with sample data
        report = runner.run_sample_benchmark()
        print(report)

        # QASPER benchmark
        report = runner.run(BenchmarkConfig(
            dataset_path="qasper-test-v0.3.json"
        ))

        # Custom data
        samples = [EvaluationSample(...)]
        report = runner.run_with_samples(samples)
    """

    def __init__(self):
        self.evaluator = RAGEvaluator()
        self.loader = QASPERLoader()
        logger.info("🏃 BenchmarkRunner initialized")

    def run(self, config: BenchmarkConfig) -> BenchmarkReport:
        """
        Run a benchmark with the given configuration.

        Args:
            config: BenchmarkConfig

        Returns:
            BenchmarkReport
        """
        start = time.time()
        logger.info(f"🏃 Running benchmark: {config.name}")

        # Load samples
        samples = self._load_samples(config)
        if not samples:
            logger.warning("⚠️ No samples loaded, returning empty report")
            return BenchmarkReport(config=config, result=EvaluationResult())

        warnings = self._check_samples(samples, config)

        # Run evaluation
        result = self.evaluator.evaluate(
            samples,
            evaluate_retrieval=config.evaluate_retrieval,
            evaluate_answers=config.evaluate_answers
        )

        elapsed = time.time() - start

        # Build summary
        summary = self._build_summary(result)

        report = BenchmarkReport(
            config=config,
            result=result,
            num_samples=len(samples),
            benchmark_time=elapsed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            summary=summary,
            warnings=warnings,
        )

        # Save report
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{config.name}_report.json"
        self._save_report(report, report_path)

        logger.success(f"✅ Benchmark complete: {report_path}")
        return report

    def run_sample_benchmark(self) -> BenchmarkReport:
        """
        Run a quick benchmark with built-in sample data.

        No external dataset needed. This is a smoke test of the evaluation
        plumbing, not a measurement: the built-in samples carry no retrieved
        ids and no generated answers, so the report comes back with warnings
        and no metrics. Real numbers come from `scripts/benchmark.py`, which
        ingests QASPER papers and runs the retrieval pipeline over them.
        """
        config = BenchmarkConfig(
            name="sample_benchmark",
            use_sample_dataset=True,
        )
        return self.run(config)

    def run_with_samples(
        self,
        samples: List[EvaluationSample],
        name: str = "custom_benchmark"
    ) -> BenchmarkReport:
        """
        Run benchmark with pre-built samples.

        Args:
            samples: Pre-built evaluation samples
            name: Benchmark name

        Returns:
            BenchmarkReport
        """
        start = time.time()

        result = self.evaluator.evaluate(samples)
        elapsed = time.time() - start

        return BenchmarkReport(
            config=BenchmarkConfig(name=name),
            result=result,
            num_samples=len(samples),
            benchmark_time=elapsed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            summary=self._build_summary(result)
        )

    @staticmethod
    def _check_samples(
        samples: List[EvaluationSample],
        config: BenchmarkConfig
    ) -> List[str]:
        """
        Say plainly when the samples cannot produce the requested metrics.

        `RAGEvaluator` skips any sample without `retrieved_ids`/`relevant_ids`
        (retrieval) or without `generated_answer` (answers). A config that asks
        for metrics nothing can supply used to yield a report with an empty
        `results` block and no explanation — indistinguishable from a genuine
        score of zero. These warnings are logged and travel with the report.

        Loading a dataset alone never fills those fields: running the pipeline
        does. See `QASPERHarness`.
        """
        warnings: List[str] = []

        if config.evaluate_retrieval:
            scorable = sum(1 for s in samples if s.retrieved_ids and s.relevant_ids)
            if scorable == 0:
                warnings.append(
                    "No retrieval metrics: no sample has both retrieved_ids and "
                    "relevant_ids. Run the retrieval pipeline over an ingested "
                    "corpus first (see QASPERHarness)."
                )
            elif scorable < len(samples):
                warnings.append(
                    f"Retrieval metrics cover {scorable}/{len(samples)} samples; "
                    "the rest lack retrieved_ids or relevant_ids."
                )

        if config.evaluate_answers:
            scorable = sum(1 for s in samples if s.generated_answer)
            if scorable == 0:
                warnings.append(
                    "No answer metrics: no sample has a generated_answer. "
                    "Generate answers before evaluating them."
                )
            elif scorable < len(samples):
                warnings.append(
                    f"Answer metrics cover {scorable}/{len(samples)} samples; "
                    "the rest have no generated_answer."
                )

        for warning in warnings:
            logger.warning(warning)

        return warnings

    def _load_samples(self, config: BenchmarkConfig) -> List[EvaluationSample]:
        """Load evaluation samples based on config."""
        if config.dataset_path:
            papers = self.loader.load(config.dataset_path)
            return self.loader.to_evaluation_samples(
                papers,
                skip_unanswerable=config.skip_unanswerable,
                max_samples=config.max_samples
            )

        if config.use_sample_dataset:
            entries = QASPERLoader.create_sample_dataset()
            return QASPERLoader.from_custom_dataset(entries)

        return []

    @staticmethod
    def _build_summary(result: EvaluationResult) -> Dict[str, Any]:
        """Build a human-readable summary from results."""
        summary = {}

        if result.retrieval:
            r = result.retrieval
            if r.mrr >= 0.7:
                retrieval_grade = "Excellent"
            elif r.mrr >= 0.5:
                retrieval_grade = "Good"
            elif r.mrr >= 0.3:
                retrieval_grade = "Fair"
            else:
                retrieval_grade = "Needs Improvement"

            summary["retrieval_grade"] = retrieval_grade
            summary["retrieval_highlights"] = {
                "MRR": round(r.mrr, 4),
                "NDCG@5": round(r.ndcg_at_5, 4),
                "P@5": round(r.precision_at_5, 4),
            }

        if result.answer:
            a = result.answer
            if a.citation_f1 >= 0.7:
                citation_grade = "Excellent"
            elif a.citation_f1 >= 0.5:
                citation_grade = "Good"
            elif a.citation_f1 >= 0.3:
                citation_grade = "Fair"
            else:
                citation_grade = "Needs Improvement"

            summary["answer_grade"] = citation_grade
            summary["answer_highlights"] = {
                "token_overlap": round(a.token_overlap, 4),
                "citation_f1": round(a.citation_f1, 4),
                "completeness": round(a.completeness, 4),
            }

        return summary

    @staticmethod
    def _save_report(report: BenchmarkReport, path: Path) -> None:
        """Save benchmark report to JSON."""
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
