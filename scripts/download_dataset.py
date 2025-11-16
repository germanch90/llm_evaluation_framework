#!/usr/bin/env python3
"""
Automated downloader for Docugami KG-RAG datasets.

Downloads the upstream GitHub repository archive, extracts the requested
dataset folders (PDFs, questions, etc.), and stages them under
data/kg-rag-dataset/.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List

import requests
from tqdm import tqdm

ZIP_URL = "https://codeload.github.com/docugami/KG-RAG-datasets/zip/refs/heads/main"
DEFAULT_DATASETS: List[str] | None = None
DATA_DIR = Path("data") / "kg-rag-dataset"


def download_zip(destination: Path) -> None:
    """Download the KG-RAG repository archive."""
    response = requests.get(ZIP_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(destination, "wb") as handle, tqdm(
        total=total_size or None,
        unit="B",
        unit_scale=True,
        desc="kg-rag-datasets.zip",
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)
            progress.update(len(chunk))


def _discover_dataset_dirs(root_folder: Path) -> List[str]:
    """Return a list of dataset directory names in the archive."""
    return [item.name for item in root_folder.iterdir() if item.is_dir()]


def extract_datasets(
    zip_path: Path, datasets: Iterable[str] | None, overwrite: bool
) -> List[str]:
    """
    Extract requested datasets into DATA_DIR.

    Returns list of dataset names that were successfully extracted.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extracted: List[str] = []

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_dir)
        root_folder = next(temp_dir.iterdir())  # KG-RAG-datasets-main
        available = _discover_dataset_dirs(root_folder)
        lookup = {name.lower(): name for name in available}
        if datasets is None:
            selected = list(available)
        else:
            selected = []
            missing = []
            for requested in datasets:
                match = lookup.get(requested.lower())
                if match:
                    selected.append(match)
                else:
                    missing.append(requested)
            if missing:
                print("⚠️  Some requested datasets were not found: " + ", ".join(missing))
                print("   Available options:")
                for name in available:
                    print(f"     - {name}")

        if not selected:
            return []

        for dataset in selected:
            source = root_folder / dataset
            if not source.exists():
                print(f"⚠️  Dataset '{dataset}' not found in archive; skipping.")
                continue

            destination = DATA_DIR / dataset
            if destination.exists():
                if not overwrite:
                    print(f"↷ Dataset '{dataset}' already exists. Use --overwrite to replace.")
                    continue
                shutil.rmtree(destination)

            shutil.copytree(source, destination)
            extracted.append(dataset)
            print(f"✓ Extracted {dataset} → {destination}")

    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Docugami KG-RAG datasets into data/kg-rag-dataset/"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset folders to extract (default: all available)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset directories if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading KG-RAG repository archive...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "kg-rag-datasets.zip"
        download_zip(zip_path)
        extracted = extract_datasets(
            zip_path,
            args.datasets if args.datasets is not None else DEFAULT_DATASETS,
            args.overwrite,
        )

    if not extracted:
        print("No datasets were extracted. Verify names or use --overwrite.")
    else:
        print("\nDatasets available in data/kg-rag-dataset/:")
        for dataset in extracted:
            print(f"  - {dataset}")


if __name__ == "__main__":
    main()
