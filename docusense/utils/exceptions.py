"""
Custom exceptions for DocuSense.
"""


class DocuSenseError(Exception):
    """Base exception for DocuSense."""
    pass


class ConfigurationError(DocuSenseError):
    """Raised when configuration is invalid or missing."""
    pass


class DocumentProcessingError(DocuSenseError):
    """Raised when document processing fails."""
    pass


class EmbeddingError(DocuSenseError):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(DocuSenseError):
    """Raised when retrieval fails."""
    pass


class LLMError(DocuSenseError):
    """Raised when LLM API call fails."""
    pass


class ValidationError(DocuSenseError):
    """Raised when data validation fails."""
    pass
