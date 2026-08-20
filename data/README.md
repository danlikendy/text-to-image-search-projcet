# Данные проекта

Файлы разметки и изображения для PoC поиска фото по тексту.

## Структура

| Файл / папка | Описание |
|---|---|
| `train_dataset.csv` | пары image — query_id — query_text для обучения |
| `CrowdAnnotations.tsv` | краудсорсинговые оценки соответствия |
| `ExpertAnnotations.tsv` | экспертные оценки (1–4) |
| `test_queries.csv` | тестовые запросы и эталонные изображения |
| `test_images.csv` | список тестовых изображений |
| `train_images/` | 1000 обучающих изображений |
| `test_images/` | 100 тестовых изображений |

## Скачать заново

```bash
python scripts/download_dataset.py
```

Источник: [dsplus_integrated_project_4.zip](https://code.s3.yandex.net/datasets/dsplus_integrated_project_4.zip)
