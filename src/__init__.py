"""Text → image scoring: ResNet50 + MiniLM + MLP."""

from .config import DISCLAIMER, TARGET_BLEND
from .filter import CHILD_PATTERN, banned_images_from_queries, query_blocked
from .labels import expert_majority, blend_target

__version__ = "1.0.0"
__all__ = [
    "DISCLAIMER",
    "TARGET_BLEND",
    "CHILD_PATTERN",
    "banned_images_from_queries",
    "query_blocked",
    "expert_majority",
    "blend_target",
]
