"""Pytest fixtures."""
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_paragraph_unit():
    return {
        "granularity": "paragraph",
        "article_id": 1,
        "paragraph_id": "test_para_001",
        "paragraph_text": "Axel Foley is a Detroit cop who goes to Beverly Hills.",
        "links": [
            {"plain_text_rel_char_start": 0, "plain_text_rel_char_end": 10},
            {"plain_text_rel_char_start": 29, "plain_text_rel_char_end": 41},
        ],
    }


@pytest.fixture
def sample_sentence_unit():
    return {
        "granularity": "sentence",
        "sentence_id": "test_sent_001",
        "sentence_text": "Eddie Murphy stars in the film.",
        "links": [{"plain_text_rel_char_start": 0, "plain_text_rel_char_end": 13}],
    }
