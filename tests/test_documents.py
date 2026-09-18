"""Every number in the rendered documents must be present in results.json."""
from pathlib import Path

import pytest

from src.config import ROOT
from src.reporting.check_numbers import CHECKED, check_documents, check_text, tokens

RESULTS = {"a": 0.12345, "b": {"c": 1234.5, "d": [0.95, 1000]}, "e": "text"}


def test_fabricated_number_is_caught():
    assert check_text("Net benefit was $9,999 per day.", RESULTS) == ["$9,999"]


def test_numbers_from_results_pass_in_several_formats():
    text = "Rate 12.3% and 12.35%, amount $1,234 or 1,234.50, CI 95%, 1,000 resamples, 0.123."
    assert check_text(text, RESULTS) == []


def test_identifiers_code_and_links_are_not_numbers():
    text = "Columns V339, id_30, card1, H1 and P90; see `x = 42` and [fig](outputs/fig_7.png)."
    assert tokens(text) == []


def test_negative_values_match_their_magnitude():
    assert check_text("a loss of -$1,234", RESULTS) == []


@pytest.mark.skipif(not (ROOT / "outputs" / "results.json").exists()
                    or not all((ROOT / d).exists() for d in CHECKED),
                    reason="documents not rendered yet (run `make all`)")
def test_committed_documents_only_contain_numbers_from_results():
    problems = check_documents()
    assert problems == [], "\n".join(problems)
