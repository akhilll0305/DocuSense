"""
DocuSense Test Suite

Run with: pytest tests/
"""


def test_import():
    """Test that docusense package can be imported."""
    import docusense
    assert docusense.__version__ == "0.1.0"


def test_settings():
    """Test that settings load correctly."""
    from docusense.config import settings
    
    assert settings.project_name == "DocuSense"
    # chunk_size was the legacy knob; the chunker works in token targets.
    assert settings.target_chunk_tokens > 0
    assert settings.top_k_results > 0
