"""
Claim-level verification.

The NLI model itself is replaced by a fake that entails a claim exactly when
the premise contains it, so these tests pin the plumbing -- claim extraction,
windowing, best-window selection, the threshold -- without a download.
"""

import numpy as np

from docusense.generation.claim_verifier import (
    ClaimVerifier,
    extract_claims,
    source_windows,
    split_sentences,
)


class ContainsNLI:
    """Entailment 0.9 when the premise contains the claim, else neutral."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        out = np.zeros((len(pairs), 2))  # (entailment, contradiction)
        for i, (premise, hypothesis) in enumerate(pairs):
            if hypothesis.rstrip(".").lower() in premise.lower():
                out[i] = [0.9, 0.05]
            else:
                out[i] = [0.1, 0.1]
        return out


def test_split_keeps_abbreviations_whole():
    text = "Results follow Devlin et al. 2019 closely. See Fig. 2 for details. Done."
    assert split_sentences(text) == [
        "Results follow Devlin et al. 2019 closely.",
        "See Fig. 2 for details.",
        "Done.",
    ]


def test_claims_drop_citations_headings_and_bibliography():
    answer = (
        "## Summary\n"
        "The model reaches 88.5 F1 [1]. It was trained on SQuAD (Devlin et al., 2018).\n"
        "- Uses a learning rate of 3e-5 [Source 2]\n"
        "Short one.\n\n"
        "Sources:\n"
        "[1] Devlin, J. et al. BERT. 2018."
    )
    assert extract_claims(answer) == [
        "The model reaches 88.5 F1.",
        "It was trained on SQuAD.",
        "Uses a learning rate of 3e-5",
    ]


def test_windows_overlap_and_cover_the_end():
    text = " ".join(f"Sentence number {i}." for i in range(7))
    windows = source_windows(text, size=4, stride=2)
    assert windows[0].startswith("Sentence number 0.")
    assert windows[-1].endswith("Sentence number 6.")
    assert len(windows) == 3


def test_short_source_is_one_window():
    assert source_windows("One. Two.") == ["One. Two."]


def test_verify_flags_the_unsupported_claim():
    verifier = ClaimVerifier(model=ContainsNLI(), threshold=0.5)
    sources = ["We train on SQuAD for three epochs. The model reaches 88.5 F1."]
    answer = "The model reaches 88.5 F1. Training used eight TPUs on the cluster."

    result = verifier.verify(answer, sources)

    assert [c.supported for c in result.claims] == [True, False]
    assert result.faithfulness == 0.5
    assert result.unsupported[0].claim.startswith("Training used")
    assert result.claims[0].source_index == 0


def test_best_source_is_reported():
    verifier = ClaimVerifier(model=ContainsNLI(), threshold=0.5)
    sources = ["Nothing relevant here at all.", "The corpus has 3044 sentences in total."]

    result = verifier.verify("The corpus has 3044 sentences in total.", sources)

    assert result.claims[0].supported
    assert result.claims[0].source_index == 1


def test_no_claims_means_no_score():
    verifier = ClaimVerifier(model=ContainsNLI())
    result = verifier.verify("Yes.", ["anything"])
    assert result.faithfulness is None
    assert result.to_dict()["num_claims"] == 0


def test_no_sources_supports_nothing():
    verifier = ClaimVerifier(model=ContainsNLI())
    result = verifier.verify("The model reaches 88.5 F1 on dev.", [])
    assert result.faithfulness == 0.0


def test_one_model_call_per_answer():
    """Every (window, claim) pair goes in one predict call, batched inside."""
    model = ContainsNLI()
    ClaimVerifier(model=model).verify(
        "First claim is here today. Second claim is here too.",
        ["Source one text. More text.", "Source two text."],
    )
    assert model.calls == 1


def test_prefilter_scores_only_the_best_windows():
    """Windows sharing no words with the claim never reach the model."""
    seen = []

    class Recording(ContainsNLI):
        def predict(self, pairs):
            seen.extend(pairs)
            return super().predict(pairs)

    sources = [f"Unrelated sentence about gardening number {i}." for i in range(10)]
    sources.append("The corpus has 3044 sentences in total.")
    verifier = ClaimVerifier(model=Recording(), max_windows_per_claim=2)

    result = verifier.verify("The corpus has 3044 sentences in total.", sources)

    assert len(seen) == 2
    assert result.claims[0].supported
    assert result.claims[0].source_index == 10


def test_structure_is_not_checked_as_claims():
    """Bold labels and list lead-ins assert nothing; seen live from a hosted model."""
    answer = (
        "The two papers report the following results for average delay:\n\n"
        "**Rahmani et al. (2021)**\n"
        "*   **ARTNet Performance:** The model reduced average delay by 23.4% against fixed-time.\n"
        "*   **Jinan benchmark**: the gap to MaxPressure was 1.5% on the irregular grid.\n"
    )
    assert extract_claims(answer) == [
        "The model reduced average delay by 23.4% against fixed-time.",
        "the gap to MaxPressure was 1.5% on the irregular grid.",
    ]
