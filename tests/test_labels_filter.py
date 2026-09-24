"""Labels and child-content filter — no GPU, no weights."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.filter import banned_images_from_queries, query_blocked
from src.labels import blend_target, expert_majority, expert_to_unit


def test_expert_majority_needs_two_votes():
    assert expert_majority([1, 1, 4]) == 1
    assert expert_majority([1, 2, 3]) is None
    assert expert_majority([4, 4, 4]) == 4


def test_blend_prefers_expert_when_crowd_missing():
    assert blend_target(4, None) == pytest.approx(1.0)
    assert blend_target(1, None) == pytest.approx(0.0)
    # 0.6 * 1.0 + 0.4 * 0.0
    assert blend_target(4, 0.0) == pytest.approx(0.6)


def test_expert_unit_interval():
    assert expert_to_unit(1) == 0.0
    assert expert_to_unit(4) == 1.0


def test_query_blocked():
    assert query_blocked("A baby is playing in the park with toys.")
    assert query_blocked("Two blonde boys one in a camouflage shirt")
    assert not query_blocked("A brown dog sits in long grass.")


def test_banned_images_from_query_id_prefix():
    banned = banned_images_from_queries(
        ["2549968784_39bfbe44f9.jpg#2", "661749711_6f408dad62.jpg#0"],
        [
            "A young child is wearing blue goggles",
            "A brown dog sits in long grass.",
        ],
    )
    assert banned == {"2549968784_39bfbe44f9.jpg"}
