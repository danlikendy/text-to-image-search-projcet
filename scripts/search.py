"""Score a query against cached test-image embeddings (if present)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import ARTIFACTS_DIR, CACHE_DIR
from src.search import SearchIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Text → image search")
    parser.add_argument("query", help="English scene description")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CACHE_DIR / "test_image_embeddings.npz",
        help="npz with arrays `names` and `vectors`",
    )
    args = parser.parse_args()

    if not (ARTIFACTS_DIR / "mlp.joblib").exists():
        print(
            "No artifacts/. Train from the notebook (or dump mlp + normalizers to artifacts/).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.catalog.exists():
        print(f"No catalog cache at {args.catalog}", file=sys.stderr)
        sys.exit(1)

    blob = np.load(args.catalog, allow_pickle=True)
    catalog = {str(n): blob["vectors"][i] for i, n in enumerate(blob["names"])}
    hit, _ = SearchIndex().search(args.query, catalog)
    if hit.blocked:
        print(hit.message)
        return
    print(f"{hit.score:.3f}  {hit.image}")


if __name__ == "__main__":
    main()
