"""Score a query against a catalog. Needs trained artifacts on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np

from .config import ARTIFACTS_DIR, DISCLAIMER
from .filter import query_blocked


@dataclass
class SearchHit:
    image: Optional[str]
    score: Optional[float]
    blocked: bool
    message: Optional[str] = None


class SearchIndex:
    """MLP on L2-normalized [image 2048 | text 384]."""

    def __init__(self, artifacts_dir: Optional[Path] = None) -> None:
        self._dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self._model = None
        self._img_norm = None
        self._text_norm = None
        self._text_encode = None

    def load(self) -> "SearchIndex":
        self._model = joblib.load(self._dir / "mlp.joblib")
        self._img_norm = joblib.load(self._dir / "img_normalizer.joblib")
        self._text_norm = joblib.load(self._dir / "text_normalizer.joblib")
        return self

    def encode_text(self, text: str) -> np.ndarray:
        if self._text_encode is None:
            from sentence_transformers import SentenceTransformer
            from .config import TEXT_MODEL_NAME

            self._text_encode = SentenceTransformer(TEXT_MODEL_NAME)
        return np.asarray(self._text_encode.encode([text], convert_to_numpy=True)[0])

    def score(self, query: str, image_vec: np.ndarray) -> float:
        if self._model is None:
            self.load()
        img = self._img_norm.transform(image_vec.reshape(1, -1))
        txt = self._text_norm.transform(self.encode_text(query).reshape(1, -1))
        x = np.hstack([img, txt])
        return float(np.clip(self._model.predict(x)[0], 0, 1))

    def search(
        self,
        query: str,
        catalog: Dict[str, np.ndarray],
    ) -> Tuple[SearchHit, Dict[str, float]]:
        if query_blocked(query):
            return SearchHit(None, None, True, DISCLAIMER), {}
        scores = {name: self.score(query, vec) for name, vec in catalog.items()}
        best = max(scores, key=scores.get)
        return SearchHit(best, scores[best], False), scores
