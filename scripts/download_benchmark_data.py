"""
Download benchmark datasets (not committed to git).

Usage: python scripts/download_benchmark_data.py
"""

import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  exists: {path}")
        return
    print(f"  downloading: {url}")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  saved: {path}")
    except Exception as e:
        print(f"  FAILED: {url} -> {e}")


def main():
    tasks = {}

    # LongBench v1 (351MB)
    lb1 = os.path.join(ROOT, "research", "longbench_v1", "data")
    if not os.path.exists(os.path.join(lb1, "data", "passage_retrieval_en.jsonl")):
        tasks["longbench_v1"] = (
            "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip",
            os.path.join(lb1, "data.zip"),
            lb1,
        )

    # Loong (114MB)
    loong = os.path.join(ROOT, "research", "loong", "data")
    if not os.path.exists(os.path.join(loong, "doc", "legal", "legal.json")):
        tasks["loong"] = (
            "http://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip",
            os.path.join(loong, "doc.zip"),
            loong,
        )
        # loong.jsonl from repo
        import shutil
        if not os.path.exists(os.path.join(loong, "loong.jsonl")):
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/MozerWang/Loong.git",
                            os.path.join("/tmp", "Loong_data")], check=False)
            src = os.path.join("/tmp", "Loong_data", "data", "loong.jsonl")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(loong, "loong.jsonl"))

    # Zero-SCROLLS
    zs = os.path.join(ROOT, "research", "zero_scrolls", "data")
    for task in ["qasper", "quality", "gov_report", "qmsum"]:
        dest = os.path.join(zs, task, "test.jsonl")
        if not os.path.exists(dest):
            tasks[f"zero_scrolls_{task}"] = (
                f"https://huggingface.co/datasets/tau/zero_scrolls/resolve/main/{task}.zip",
                os.path.join(zs, f"{task}.zip"),
                os.path.join(zs),
            )

    # LongBench v2 (from HF datasets, needs manual)
    print("\n=== Download tasks ===")
    for name, (url, zip_path, extract_to) in tasks.items():
        download(url, zip_path)
        if zip_path.endswith(".zip"):
            print(f"  extracting to {extract_to}")
            subprocess.run(["unzip", "-q", "-o", zip_path, "-d", extract_to])
            os.remove(zip_path)

    print("\nDone. Some datasets (LongBench v2) require manual download via datasets library.")


if __name__ == "__main__":
    main()

def download_books():
    """Download Project Gutenberg books for long_context benchmarks."""
    base = os.path.join(ROOT, "research", "long_context", "data")
    books = {
        "pride_and_prejudice.txt": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "moby_dick.txt": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "war_and_peace.txt": "https://www.gutenberg.org/files/2600/2600-0.txt",
        "alice_in_wonderland.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    }
    print("\n=== Download books ===")
    for name, url in books.items():
        path = os.path.join(base, name)
        if not os.path.exists(path):
            download(url, path)
