#!/usr/bin/env python3
"""
Download Docugami KG-RAG dataset
"""
import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url: str, destination: Path) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=destination.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def main():
    """Download and extract KG-RAG dataset."""
    # GitHub repository info
    repo_url = "https://github.com/docugami/KG-RAG-datasets"
    
    print("=" * 60)
    print("Docugami KG-RAG Dataset Downloader")
    print("=" * 60)
    print(f"\nDataset repository: {repo_url}")
    print("\nPlease follow these steps:")
    print("1. Visit the repository URL above")
    print("2. Clone or download the datasets you need")
    print("3. Place them in: data/kg-rag-dataset/")
    print("\nRecommended datasets:")
    print("  - Agriculture")
    print("  - Automotive") 
    print("  - Biomedical")
    print("  - Finance")
    print("  - Legal")
    print("\nEach dataset contains:")
    print("  - Documents (PDFs)")
    print("  - Questions with ground truth answers")
    print("  - Difficulty levels (single-chunk to multi-doc)")
    print("=" * 60)
    
    # Create directory
    dataset_dir = Path("data/kg-rag-dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Created directory: {dataset_dir}")
    print("\nManual setup required - see instructions above.")

if __name__ == "__main__":
    main()
