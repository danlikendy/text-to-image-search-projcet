"""Paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / "cache"

RANDOM_STATE = 42
EXPERT_MIN_VOTES = 2
TARGET_BLEND = (0.6, 0.4)  # expert, crowd
IMAGE_EMBED_DIM = 2048
TEXT_EMBED_DIM = 384

DISCLAIMER = "This image is unavailable in your country in compliance with local laws."

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
