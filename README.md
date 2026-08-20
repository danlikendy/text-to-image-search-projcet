# With Sense — Image Search PoC

PoC сервиса поиска фотографий по текстовому описанию для фотохостинга **With Sense**.

Модель оценивает соответствие пары «текст + изображение» числом от 0 до 1. На основе лучшей модели реализован демо-поиск по каталогу тестовых изображений.

## Задача

Пользователь вводит описание сцены — система возвращает наиболее подходящую фотографию. Учитываются юридические ограничения: контент с детьми фильтруется, при запросе с «вредными» словами показывается дисклеймер.

## Стек

- **Python 3.10+**
- **PyTorch** + ResNet50 (эмбеддинги изображений, 2048 dim)
- **TF-IDF** (эмбеддинги текста, 500 dim)
- **scikit-learn**: LinearRegression, Ridge, MLPRegressor
- **Jupyter Notebook**

## Структура репозитория

```
.
├── data/                    # датасет (разметка + изображения)
├── notebooks/
│   └── image_search_poc.ipynb
├── scripts/
│   └── download_dataset.py
├── cache/                   # кэш эмбеддингов (не в git)
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
git clone <repo-url>
cd <repo-name>

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# если data/ пустой:
python scripts/download_dataset.py

jupyter notebook notebooks/image_search_poc.ipynb
```

Запускайте ячейки сверху вниз. Первый прогон ResNet50 на CPU занимает ~15–20 мин; повторные — секунды (кэш в `cache/`).

## Данные

| Источник | Содержимое |
|---|---|
| `train_dataset.csv` | 5822 пары image + текст |
| `ExpertAnnotations.tsv` | оценки 3 экспертов (шкала 1–4) |
| `CrowdAnnotations.tsv` | краудсорсинговые оценки |
| `test_queries.csv` | 499 тестовых запросов |
| `train_images/` | 1000 jpg |
| `test_images/` | 100 jpg |

Подробнее: [data/README.md](data/README.md)

## Метод

1. **EDA** — агрегация экспертных оценок (голосование 2/3), смешивание с краудом (0.6 / 0.4)
2. **Фильтрация** — исключение изображений с детьми по ключевым словам в описаниях
3. **Векторизация** — ResNet50 + TF-IDF → конкатенация 2548 признаков
4. **Обучение** — GroupShuffleSplit по image_id, сравнение линейных моделей и MLP
5. **Поиск** — функция `search_image()` с дисклеймером для запрещённых запросов

## Результаты

| Модель | MAE (val) | ROC-AUC |
|---|---:|---:|
| LinearRegression | 0.217 | 0.634 |
| Ridge (α=10) | 0.216 | 0.636 |
| **MLP (512,256,128)** | **0.191** | **0.704** |

Top-1 accuracy на 10 случайных тестовых запросах: 0/10 — типично для PoC без CLIP/BERT.

## Ограничения PoC

- TF-IDF не улавливает семантику так же хорошо, как BERT/CLIP
- ResNet50 обучен на ImageNet, не на text-image alignment
- Для продакшена: multimodal embeddings, ANN-индекс, мониторинг bias

## Лицензия

MIT — см. [LICENSE](LICENSE).

Датасет: учебный проект Yandex Practicum, используйте только в образовательных целях.
