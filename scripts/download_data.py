"""Download the ATLAS Higgs dataset from CERN Open Data into data/.

    python scripts/download_data.py
"""

from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

URL = "https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GZ_PATH = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv.gz"
CSV_PATH = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv"


def _report(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        sys.stdout.write(f"\r  downloading... {pct:5.1f}%")
        sys.stdout.flush()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print(f"Already present: {CSV_PATH}")
        return

    print(f"Fetching {URL}")
    urllib.request.urlretrieve(URL, GZ_PATH, _report)
    print("\nDecompressing...")
    with gzip.open(GZ_PATH, "rb") as fin, open(CSV_PATH, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    GZ_PATH.unlink(missing_ok=True)
    print(f"Done -> {CSV_PATH}")


if __name__ == "__main__":
    main()
