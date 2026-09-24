# Data

Paired captions, expert/crowd scores, and two image folders.

| Path | What |
|---|---|
| `train_dataset.csv` | image, query_id, query_text (5 822 rows) |
| `ExpertAnnotations.tsv` | three 1–4 scores per pair |
| `CrowdAnnotations.tsv` | yes-share per pair |
| `test_queries.csv` | hold-out captions |
| `test_images.csv` | 100 test filenames |
| `train_images/` | 1 000 jpg |
| `test_images/` | 100 jpg |

```bash
python scripts/download_dataset.py
```

Archive: [dsplus_integrated_project_4.zip](https://code.s3.yandex.net/datasets/dsplus_integrated_project_4.zip)
