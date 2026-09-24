"""Refuse queries / drop catalog images that mention children."""

import re
from typing import Iterable, Set

CHILD_PATTERN = re.compile(
    r"\b(?:child(?:ren)?|kid(?:s)?|bab(?:y|ies)|toddler|infant|"
    r"young (?:boy|girl|child)|little (?:boy|girl|blond)|teen(?:ager)?|"
    r"tyke|son|daughter|boy(?:s)?|girl(?:s)?)\b",
    re.I,
)


def query_blocked(text: str) -> bool:
    return bool(text) and bool(CHILD_PATTERN.search(text))


def image_from_query_id(query_id: str) -> str:
    """query_id is `{filename}#{caption_index}`."""
    return str(query_id).split("#", 1)[0]


def banned_images_from_queries(query_ids: Iterable[str], query_texts: Iterable[str]) -> Set[str]:
    banned: Set[str] = set()
    for qid, text in zip(query_ids, query_texts):
        if query_blocked(text):
            banned.add(image_from_query_id(qid))
    return banned
