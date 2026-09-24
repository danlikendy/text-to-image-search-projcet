# Rank photos by a sentence

A catalog query: English caption in, a relevance score in `[0, 1]` out, top-1 image from the gallery. Image tower is frozen **ResNet50** (ImageNet). Text tower is **all-MiniLM-L6-v2**. A small **MLP** reads the concatenated L2-normalized pair.

Live write-up: **[danlikendy.github.io/text-to-image-search-projcet](https://danlikendy.github.io/text-to-image-search-projcet/)**

This is **not** CLIP. The two towers never saw each other during pretraining. The MLP is what has to learn the joint space. Retrieval top-1 on a 100-image hold-out is weak; ranking MAE is where the work actually shows.

Queries that mention children are refused (legal constraint on the catalog).

---

## Problem

**5 822** image–caption pairs on **1 000** photos. Three experts score match on a 1–4 scale; crowd gives a yes-share. Experts disagree on a slice of pairs — I drop those (need 2-of-3). Target is `0.6 × expert + 0.4 × crowd` when crowd exists, else expert only, mapped to `[0, 1]`. Mean target sits around **0.17**: most captions are a poor match. A dummy predictor is already “pretty good” on MAE. You have to beat that.

**288** images leave the train set because their `query_id` captions talk about children. That is prefix-of-`query_id`, not a vision detector.

## What I shipped

| Piece | Choice |
|---|---|
| Image | ResNet50, 2048-d, frozen |
| Text | MiniLM-L6-v2, 384-d (not TF-IDF — tried, lost) |
| Features | L2 per tower, concat **2432** |
| Split | `GroupShuffleSplit` on `image_id`, 70/30 — no photo in both sides |
| Metric | **MAE** (primary), ROC-AUC on a high/low cut |
| Model | Dummy → Ridge → MLP. Winner: **MLP (128)** |

| Model | MAE (val) | ROC-AUC |
|---|---:|---:|
| DummyRegressor | 0.214 | 0.500 |
| Ridge (α=10) | 0.197 | 0.723 |
| **MLP (128)** | **0.163** | **0.836** |

Ten random test queries, top-1 vs the labeled photo: **0/10**. Expected with unaligned towers. I leave the number here so nobody reads 0.836 AUC as “the search works.”

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py   # if data/ is empty
pytest tests/ -q
```

Notebook (EDA + encode + train + demo search): `notebooks/eda_and_training.ipynb`. First ResNet pass on CPU is ~15–20 min; embeddings cache under `cache/`.

After you dump `artifacts/mlp.joblib` + normalizers and `cache/test_image_embeddings.npz`:

```bash
python scripts/search.py "A brown dog sits in long grass."
```

Child-mention query prints the legal refusal string instead of a filename.

More: [docs/RUN.md](docs/RUN.md) · [docs/API.md](docs/API.md)

## Layout

```
src/           labels, child filter, SearchIndex
scripts/       download dataset, CLI search
notebooks/     full training path
tests/         labels + filter (no GPU)
data/          pairs, expert/crowd TSV, jpg catalogs
```

`cache/` and `artifacts/` are gitignored.

---

Artem Tsygantsov · [tsygantsov.ru](https://tsygantsov.ru) · MIT
