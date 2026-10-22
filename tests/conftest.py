"""
Suite-wide defaults.

Claim verification loads an 87MB NLI model from the Hugging Face Hub the first
time an answer is built. Tests that build answers are testing everything else,
so it is off unless a test turns it on (tests/test_claim_verifier.py uses a
fake model instead).
"""

import pytest

from docusense.config.settings import settings


@pytest.fixture(autouse=True)
def _no_claim_verification(monkeypatch):
    monkeypatch.setattr(settings, "enable_claim_verification", False)
