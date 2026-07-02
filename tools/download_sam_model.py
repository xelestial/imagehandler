from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEFAULT_OUTPUT = Path("models") / "sam_vit_b_01ec64.pth"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, output: Path, force: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"exists: {output}")
        print(f"sha256: {sha256_file(output)}")
        return

    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"downloading: {url}")
    print(f"output: {output}")
    with urlopen(url, timeout=60) as response, tmp.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else 0
        received = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            received += len(chunk)
            if total_bytes:
                pct = received * 100.0 / total_bytes
                print(f"\r{received / (1024 * 1024):.1f} MiB / {total_bytes / (1024 * 1024):.1f} MiB ({pct:.1f}%)", end="")
        if total_bytes:
            print()

    os.replace(tmp, output)
    print(f"saved: {output}")
    print(f"size: {output.stat().st_size} bytes")
    print(f"sha256: {sha256_file(output)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SAM ViT-B checkpoint for ImageHandler item proposals.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=SAM_VIT_B_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.url, args.output, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
