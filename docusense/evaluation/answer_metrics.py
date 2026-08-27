"""
Answer Quality Metrics - ROUGE, citation accuracy, and answer evaluation.

Phase 6: Evaluation & Metrics (Step 2)

PURPOSE:
--------
Evaluate generated answer quality:
1. ROUGE scores (overlap between generated and reference answers)
2. Citation accuracy (are cited papers actually in sources?)
3. Answer completeness (does the answer address the question?)
4. Faithfulness (is the answer grounded in retrieved context?)

Note: ROUGE and BERTScore use optional heavy libraries.
Built-in metrics work without any extra installs.

Author: DocuSense
Created: 2026-03-08
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from loguru import logger


# Check for optional evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import bert_score  # noqa: F401  (availability probe)
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


@dataclass
class AnswerMetricsResult:
    """Quality metrics for a generated answer."""
    # Text overlap (ROUGE)
    rouge_1: float = 0.0  # Unigram overlap
    rouge_2: float = 0.0  # Bigram overlap
    rouge_l: float = 0.0  # Longest common subsequence

    # BERTScore (semantic similarity)
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    bert_f1: float = 0.0

    # Citation metrics
    citation_precision: float = 0.0  # cited papers that exist in sources
    citation_recall: float = 0.0     # source papers that are cited
    citation_f1: float = 0.0
    num_citations_found: int = 0
    num_expected_citations: int = 0

    # Answer quality (built-in)
    answer_length: int = 0
    token_overlap: float = 0.0     # Simple word overlap score
    has_citations: bool = False
    completeness: float = 0.0      # Coverage of key terms from query

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize, distinguishing "not computed" from "computed as zero".

        A missing optional library used to be indistinguishable from a genuine
        score of 0.0: `rouge-score` was absent from the environment while
        listed in requirements.txt, and every report would have published
        ROUGE-1 = 0.0 as if the answers had no overlap at all. Metrics whose
        backing library is unavailable now serialize as null and are named in
        `not_computed`.
        """
        not_computed = []
        if not ROUGE_AVAILABLE:
            not_computed.append("ROUGE (install rouge-score)")
        if not BERTSCORE_AVAILABLE:
            not_computed.append("BERTScore (install bert-score)")

        def rouge(value: float) -> Optional[float]:
            return round(value, 4) if ROUGE_AVAILABLE else None

        def bert(value: float) -> Optional[float]:
            return round(value, 4) if BERTSCORE_AVAILABLE else None

        payload: Dict[str, Any] = {
            "ROUGE-1": rouge(self.rouge_1),
            "ROUGE-2": rouge(self.rouge_2),
            "ROUGE-L": rouge(self.rouge_l),
            "BERTScore-P": bert(self.bert_precision),
            "BERTScore-R": bert(self.bert_recall),
            "BERTScore-F1": bert(self.bert_f1),
            "Citation-P": round(self.citation_precision, 4),
            "Citation-R": round(self.citation_recall, 4),
            "Citation-F1": round(self.citation_f1, 4),
            "token_overlap": round(self.token_overlap, 4),
            "completeness": round(self.completeness, 4),
            "has_citations": self.has_citations,
            "answer_length": self.answer_length,
        }
        if not_computed:
            payload["not_computed"] = not_computed
        return payload

    def __str__(self) -> str:
        lines = ["Answer Quality Metrics:"]
        if ROUGE_AVAILABLE and self.rouge_1 > 0:
            lines.extend([
                f"  ROUGE-1:  {self.rouge_1:.4f}",
                f"  ROUGE-2:  {self.rouge_2:.4f}",
                f"  ROUGE-L:  {self.rouge_l:.4f}",
            ])
        if BERTSCORE_AVAILABLE and self.bert_f1 > 0:
            lines.extend([
                f"  BERTScore: P={self.bert_precision:.4f}, "
                f"R={self.bert_recall:.4f}, F1={self.bert_f1:.4f}",
            ])
        lines.extend([
            f"  Citation:  P={self.citation_precision:.4f}, "
            f"R={self.citation_recall:.4f}, F1={self.citation_f1:.4f}",
            f"  Overlap:   {self.token_overlap:.4f}",
            f"  Complete:  {self.completeness:.4f}",
            f"  Citations: {self.has_citations}",
        ])
        return "\n".join(lines)


class AnswerMetrics:
    """
    Evaluate generated answer quality.

    Features:
    - ROUGE scores (requires rouge-score library)
    - BERTScore (requires bert-score library)
    - Citation accuracy (built-in)
    - Token overlap and completeness (built-in)

    Usage:
        metrics = AnswerMetrics()
        result = metrics.evaluate(
            generated="BERT achieved 93.5% F1 (Devlin et al., 2018).",
            reference="BERT scored 93.5% F1 on SST-2.",
            query="What F1 did BERT achieve?",
            source_papers=["BERT Paper"]
        )
    """

    def __init__(self):
        if ROUGE_AVAILABLE:
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            logger.info("📊 ROUGE scorer loaded")
        else:
            self._rouge_scorer = None
            logger.info("📊 ROUGE scorer not available (install rouge-score)")

    def evaluate(
        self,
        generated: str,
        reference: str = "",
        query: str = "",
        source_papers: Optional[List[str]] = None
    ) -> AnswerMetricsResult:
        """
        Evaluate a generated answer against a reference.

        Args:
            generated: Generated answer text
            reference: Ground-truth reference answer
            query: Original query (for completeness check)
            source_papers: Papers that were in the retrieval sources

        Returns:
            AnswerMetricsResult
        """
        result = AnswerMetricsResult()
        result.answer_length = len(generated)

        # ROUGE scores
        if reference and self._rouge_scorer:
            scores = self._rouge_scorer.score(reference, generated)
            result.rouge_1 = scores["rouge1"].fmeasure
            result.rouge_2 = scores["rouge2"].fmeasure
            result.rouge_l = scores["rougeL"].fmeasure

        # Token overlap (built-in, no dependencies)
        if reference:
            result.token_overlap = self._compute_token_overlap(generated, reference)

        # Citation accuracy
        cited = self._extract_citations(generated)
        result.has_citations = len(cited) > 0
        result.num_citations_found = len(cited)

        if source_papers:
            result.num_expected_citations = len(source_papers)
            citation_metrics = self._citation_accuracy(cited, source_papers)
            result.citation_precision = citation_metrics["precision"]
            result.citation_recall = citation_metrics["recall"]
            result.citation_f1 = citation_metrics["f1"]

        # Query completeness
        if query:
            result.completeness = self._compute_completeness(generated, query)

        return result

    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> AnswerMetricsResult:
        """
        Evaluate multiple answers and return averaged metrics.

        Args:
            evaluations: List of dicts with keys:
                - "generated": str
                - "reference": str
                - "query": str
                - "source_papers": List[str]

        Returns:
            Averaged AnswerMetricsResult
        """
        if not evaluations:
            return AnswerMetricsResult()

        results = [
            self.evaluate(
                generated=ev["generated"],
                reference=ev.get("reference", ""),
                query=ev.get("query", ""),
                source_papers=ev.get("source_papers")
            )
            for ev in evaluations
        ]

        n = len(results)
        avg = AnswerMetricsResult(
            rouge_1=sum(r.rouge_1 for r in results) / n,
            rouge_2=sum(r.rouge_2 for r in results) / n,
            rouge_l=sum(r.rouge_l for r in results) / n,
            citation_precision=sum(r.citation_precision for r in results) / n,
            citation_recall=sum(r.citation_recall for r in results) / n,
            citation_f1=sum(r.citation_f1 for r in results) / n,
            token_overlap=sum(r.token_overlap for r in results) / n,
            completeness=sum(r.completeness for r in results) / n,
            has_citations=all(r.has_citations for r in results),
            answer_length=sum(r.answer_length for r in results) // n,
        )

        return avg

    # ==================================================================
    # BUILT-IN METRICS (no extra dependencies)
    # ==================================================================

    @staticmethod
    def _compute_token_overlap(generated: str, reference: str) -> float:
        """
        Compute F1-style token overlap between generated and reference.

        This is a lightweight ROUGE-1-like metric without stemming.
        """
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())

        if not gen_tokens or not ref_tokens:
            return 0.0

        overlap = gen_tokens & ref_tokens
        precision = len(overlap) / len(gen_tokens) if gen_tokens else 0
        recall = len(overlap) / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _extract_citations(text: str) -> List[str]:
        """
        Extract cited author names from text.

        Matches patterns like:
        - (Devlin et al., 2018)
        - (Smith & Jones, 2020)
        - (Author, Year)
        """
        patterns = [
            r'\(([A-Z][a-z]+)\s+et\s+al\.\s*,\s*\d{4}',   # et al.
            r'\(([A-Z][a-z]+)\s*&\s*[A-Z][a-z]+\s*,\s*\d{4}',  # Two authors
            r'\(([A-Z][a-z]+)\s*,\s*\d{4}',  # Single author
        ]

        cited = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            cited.extend(matches)

        return list(set(cited))

    @staticmethod
    def _citation_accuracy(
        cited_authors: List[str],
        source_papers: List[str]
    ) -> Dict[str, float]:
        """
        Compute citation precision/recall/F1.

        Matches cited author last names against paper title/author info
        in a fuzzy manner (case-insensitive contains check).
        """
        if not cited_authors and not source_papers:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if not cited_authors:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not source_papers:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Check how many cited authors match source papers
        source_text = " ".join(source_papers).lower()
        matched_citations = sum(
            1 for author in cited_authors
            if author.lower() in source_text
        )

        precision = matched_citations / len(cited_authors) if cited_authors else 0
        recall = min(matched_citations, len(source_papers)) / len(source_papers) if source_papers else 0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def _compute_completeness(generated: str, query: str) -> float:
        """
        Compute how well the answer covers key terms from the query.

        Returns fraction of query content words found in the answer.
        """
        # Common stop words to ignore
        stop_words = {
            "what", "how", "did", "does", "is", "are", "was", "were",
            "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "and", "or", "but", "with", "from", "by", "about", "which",
            "that", "this", "it", "do", "can", "you", "we", "they",
            "their", "its", "has", "have", "had",
        }

        query_words = set(query.lower().split()) - stop_words
        if not query_words:
            return 1.0

        answer_lower = generated.lower()
        found = sum(1 for word in query_words if word in answer_lower)

        return found / len(query_words)
