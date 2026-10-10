"""Download the original OULAD zip from the Open University and unpack the CSVs."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oulad.config import RAW_DIR  # noqa: E402
from oulad.load import ensure_extracted  # noqa: E402

# The OU dataset page still links the first URL, but that CDN path currently 404s.
# UCI hosts the same Kuzilek et al. release. Row counts are checked after load.
ZIP_URLS = (
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
    "https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip",
    "http://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip",
)
CHECKSUM_URL = "https://analyse.kmi.open.ac.uk/open-dataset/downloadCheckSum"
USER_AGENT = "oulad-early-warning/0.1 (research reproduction; CC BY 4.0 dataset)"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    checksum_path = RAW_DIR / "checksum.txt"
    if not checksum_path.exists():
        try:
            _download(CHECKSUM_URL, checksum_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not fetch the checksum page ({exc}). Continuing.")
    zip_path = RAW_DIR / "oulad_uci.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        last_error = None
        for url in ZIP_URLS:
            try:
                print(f"Downloading {url}")
                _download(url, zip_path)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"  failed: {exc}")
        else:
            raise SystemExit(f"Could not download OULAD: {last_error}")
    digest = hashlib.md5()
    with zip_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    print(f"MD5 {digest.hexdigest()}")
    if checksum_path.exists():
        text = checksum_path.read_text(errors="replace")
        if digest.hexdigest() not in text.replace(" ", "").lower() and digest.hexdigest() not in text.lower():
            print("The published checksum page did not contain this MD5. Inspect data/raw/checksum.txt.")
        else:
            print("MD5 matches the OU checksum page.")
    ensure_extracted(RAW_DIR)
    print(f"CSVs are in {RAW_DIR}")


if __name__ == "__main__":
    main()
