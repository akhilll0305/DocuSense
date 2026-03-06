"""
Generation Module - Answer generation with academic citations.

Phase 4: Answer Generation with Citations

Components:
-----------
1. AnswerGenerator: Core answer generation with academic prompts
2. CitationFormatter: Academic citation formatting (APA, BibTeX)
3. GenerationPipeline: End-to-end query → answer orchestration
"""

from .answer_generator import (
    AnswerGenerator,
    GeneratedAnswer,
    generate_answer
)

from .citation_formatter import (
    CitationFormatter,
    CitationStyle,
    FormattedCitation,
    format_citation
)

from .generation_pipeline import (
    GenerationPipeline,
    PipelineResponse
)

__all__ = [
    # Answer Generation
    "AnswerGenerator",
    "GeneratedAnswer",
    "generate_answer",
    # Citation Formatting
    "CitationFormatter",
    "CitationStyle",
    "FormattedCitation",
    "format_citation",
    # Pipeline
    "GenerationPipeline",
    "PipelineResponse",
]
