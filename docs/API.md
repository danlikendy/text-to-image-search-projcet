# API

## Target

`expert_majority(votes)` — 2-of-3 on `{1,2,3,4}`. Tie-all-different → drop the pair.

`blend_target(expert, crowd_share)` — `(score-1)/3`, then `0.6 × expert + 0.4 × crowd` when crowd is present.

## Filter

`query_blocked(text)` — regex on child-related English.

`banned_images_from_queries(ids, texts)` — if the caption under a `query_id` matches, the image prefix of that id is out of train.

## `SearchIndex`

`load()` from `artifacts/`. `search(query, catalog)` → `SearchHit` + per-image scores. Catalog is `{filename: (2048,) ndarray}`.

Blocked queries never touch the MLP.

## CLI

```bash
python scripts/download_dataset.py [--data-dir data] [--force]
python scripts/search.py "…" [--catalog cache/test_image_embeddings.npz]
```
