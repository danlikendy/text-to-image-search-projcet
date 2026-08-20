# Text-to-Image Search PoC

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP%20%7C%20Ridge-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PoC поиска фотографий по текстовому описанию для фотохостинга **With Sense**.

Пользователь вводит описание сцены — модель возвращает число от 0 до 1 (степень соответствия) и находит наиболее релевантное изображение в каталоге.

> Учебный проект Yandex Practicum · Data Science · multimodal retrieval

## Demo

| Вход | Выход |
|---|---|
| Текстовый запрос на английском | Top-1 изображение из `test_images/` |
| Запрос с контентом о детях | Дисклеймер вместо результата |

## Стек

| Компонент | Технология |
|---|---|
| Эмбеддинги изображений | ResNet50 (PyTorch, ImageNet), **2048** dim |
| Эмбеддинги текста | TF-IDF (uni/bi-grams), **500** dim |
| Модель сходства | LinearRegression, Ridge, **MLP (512→256→128)** |
| Метрика | MAE (основная), ROC-AUC |
| Split | GroupShuffleSplit по `image_id` (70/30) |

## Результаты

| Модель | MAE (val) | ROC-AUC |
|---|:---:|:---:|
| LinearRegression | 0.217 | 0.634 |
| Ridge (α=10) | 0.216 | 0.636 |
| **MLP (512,256,128)** | **0.191** | **0.704** |

Top-1 на 10 случайных тестовых запросах: **0/10** — ожидаемо для PoC без CLIP/BERT.

## Структура

```
.
├── data/                         # датасет (разметка + изображения)
│   ├── train_dataset.csv
│   ├── ExpertAnnotations.tsv
│   ├── CrowdAnnotations.tsv
│   ├── test_queries.csv
│   ├── train_images/             # 1000 jpg
│   └── test_images/              # 100 jpg
├── notebooks/
│   └── image_search_poc.ipynb    # основной ноутбук
├── scripts/
│   └── download_dataset.py
├── cache/                        # кэш эмбеддингов (gitignore)
├── requirements.txt
└── LICENSE
```

## Быстрый старт

```bash
git clone https://github.com/danlikendy/text-to-image-search-projcet.git
cd text-to-image-search-projcet

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# если data/ пустой:
python scripts/download_dataset.py

jupyter notebook notebooks/image_search_poc.ipynb
```

Первый прогон ResNet50 на CPU: ~15–20 мин. Повторные запуски — секунды (кэш в `cache/`).

## Пайплайн

1. **EDA** — голосование экспертов 2/3, target = 0.6×expert + 0.4×crowd
2. **Фильтрация** — исключение изображений с детьми по ключевым словам
3. **Векторизация** — ResNet50 + TF-IDF → concat **2548** признаков
4. **Обучение** — сравнение линейных моделей и MLP, выбор лучшей по MAE
5. **Поиск** — `search_image()` + юридический дисклеймер

## Данные

| Файл | Описание |
|---|---|
| `train_dataset.csv` | 5822 пары image + текст |
| `ExpertAnnotations.tsv` | оценки 3 экспертов (1–4) |
| `CrowdAnnotations.tsv` | краудсорсинговые оценки |
| `test_queries.csv` | 499 тестовых запросов |

Источник: [dsplus_integrated_project_4.zip](https://code.s3.yandex.net/datasets/dsplus_integrated_project_4.zip) · подробнее в [data/README.md](data/README.md)

## Ограничения

- TF-IDF + ResNet50 не выровнены в общем embedding-space (в отличие от CLIP)
- PoC, не production-ready
- Датасет — только для обучения

## Автор

[danlikendy](https://github.com/danlikendy)

## Лицензия

MIT — см. [LICENSE](LICENSE).
