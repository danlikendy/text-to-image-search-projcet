#!/usr/bin/env python3
"""Скачивание и распаковка датасета Yandex Practicum (project 4)."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://code.s3.yandex.net/datasets/dsplus_integrated_project_4.zip"

FILES = [
    "train_dataset.csv",
    "CrowdAnnotations.tsv",
    "ExpertAnnotations.tsv",
    "test_queries.csv",
    "test_images.csv",
]
DIRS = ["train_images", "test_images"]


def download_dataset(target_dir: Path, force: bool = False) -> Path:
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not force and (target_dir / "train_dataset.csv").exists():
        print(f"Данные уже есть: {target_dir}")
        return target_dir

    zip_path = target_dir / "dsplus_integrated_project_4.zip"
    print("Скачивание...")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    print("Распаковка...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)

    zip_path.unlink(missing_ok=True)

    # архив может распаковаться во вложенную папку
    for path in target_dir.rglob("train_dataset.csv"):
        root = path.parent
        if root == target_dir:
            return target_dir
        for name in FILES + DIRS:
            src = root / name
            dst = target_dir / name
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
        return target_dir

    raise FileNotFoundError("train_dataset.csv не найден после распаковки")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download image-search dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Каталог для данных (по умолчанию: ../data)",
    )
    parser.add_argument("--force", action="store_true", help="Перекачать даже если файлы есть")
    args = parser.parse_args()
    path = download_dataset(args.data_dir, force=args.force)
    print(f"Готово: {path}")


if __name__ == "__main__":
    main()
