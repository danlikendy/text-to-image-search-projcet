"""Turn 1–4 expert votes + crowd share into a [0, 1] target."""

from collections import Counter
from typing import Optional, Sequence

import pandas as pd

from .config import EXPERT_MIN_VOTES, TARGET_BLEND


def expert_majority(votes: Sequence[float], min_votes: int = EXPERT_MIN_VOTES) -> Optional[float]:
    """2-of-3 majority. Full disagreement → None."""
    clean = [int(v) for v in votes if pd.notna(v)]
    if not clean:
        return None
    rating, n = Counter(clean).most_common(1)[0]
    return float(rating) if n >= min_votes else None


def expert_to_unit(score: float) -> float:
    """Map 1–4 onto [0, 1]."""
    return (float(score) - 1.0) / 3.0


def blend_target(expert_score: float, crowd_share: Optional[float] = None) -> float:
    """0.6 × expert + 0.4 × crowd when crowd exists; else expert only."""
    e = expert_to_unit(expert_score)
    if crowd_share is None or (isinstance(crowd_share, float) and pd.isna(crowd_share)):
        return e
    w_e, w_c = TARGET_BLEND
    return w_e * e + w_c * float(crowd_share)
