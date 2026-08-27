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

# Grouped by module rather than sorted alphabetically: the comment headings
# below are the point of the list. RUF022 wants a flat isort-style sort,
# which would scatter each group across the others.
__all__ = [  # noqa: RUF022
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
