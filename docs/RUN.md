# Run

Python 3.10+. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Markup CSVs/TSVs are in `data/`. Images live in `data/train_images/` and `data/test_images/`. If those folders are empty:

```bash
python scripts/download_dataset.py
```

## Tests that don’t need a GPU

```bash
pytest tests/ -q
```

Expert majority, target blend, child-query gate.

## Train / encode

That’s the notebook: `notebooks/eda_and_training.ipynb`. It writes embedding cache under `cache/` and the fitted MLP. Copy the sklearn objects into `artifacts/` if you want the CLI:

- `artifacts/mlp.joblib`
- `artifacts/img_normalizer.joblib`
- `artifacts/text_normalizer.joblib`
- `cache/test_image_embeddings.npz` with `names`, `vectors`

## Search CLI

```bash
python scripts/search.py "A hiker poses for a picture in front of stunning mountains."
```

Blocked query → disclaimer text, exit 0.

Classes: [API.md](API.md).
