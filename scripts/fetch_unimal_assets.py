#!/usr/bin/env python3
"""Download the 100 UNIMAL body files used in the paper and check their hashes.

The 100 bodies are the training set of the UNIMAL-100 benchmark released
with MetaMorph (Gupta et al., ICLR 2022), which MetaMorph chose from the
UNIMAL bodies that DERL evolved (Gupta et al., Nature Communications 2021).
The upstream repositories (github.com/agrimgupta92/derl,
github.com/agrimgupta92/metamorph) declare no license, so this repository
does not redistribute the files. This script
fetches the upstream archive, extracts the 100 training bodies
(unimals_100/train/xml/*.xml) and checks every file against the SHA-256
manifest in assets/unimals_100/SHA256SUMS, which was computed from the exact
files used to train the reported controllers.

Upstream source (MetaMorph README, "Benchmark"):
    gdown 1LyKYTCevnqWrDle1LTBMlBF58RmCjSzM    # unimals_100.zip

Usage:
    python scripts/fetch_unimal_assets.py                 # download, verify, extract
    python scripts/fetch_unimal_assets.py --zip unimals_100.zip   # use a local copy
    python scripts/fetch_unimal_assets.py --check         # verify an existing directory

Exit status is non-zero if any file is missing or differs from the manifest.
Nothing is written to the destination unless all 100 files verify. Only the
Python standard library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "assets" / "unimals_100" / "SHA256SUMS"
DEFAULT_DEST = REPO / "assets" / "unimals_100" / "xml"

GDRIVE_ID = "1LyKYTCevnqWrDle1LTBMlBF58RmCjSzM"
DOWNLOAD_URL = f"https://drive.usercontent.google.com/download?id={GDRIVE_ID}&export=download&confirm=t"
ARCHIVE_MEMBER_DIR = "unimals_100/train/xml/"
# SHA-256 of unimals_100.zip as downloaded on 2026-10-07 (7,359,913 bytes).
# A different archive hash is reported but is not an error by itself: the
# per-file hashes below decide.
ARCHIVE_SHA256 = "cac50822f41246d23d0c0fdacc07be3bf52cd97cf7dc82611643f52817a69c77"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    if len(out) != 100:
        raise SystemExit(f"ERROR: {MANIFEST} lists {len(out)} files, expected 100.")
    return out


def download(url: str) -> bytes:
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_unimal_assets/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if not data.startswith(b"PK"):
        raise SystemExit(
            "ERROR: the download is not a zip archive (Google Drive may have returned a "
            "confirmation or quota page). Download it manually, for example with\n"
            f"    pip install gdown && gdown {GDRIVE_ID}\n"
            "and rerun with --zip unimals_100.zip."
        )
    return data


def verify_dir(dest: Path, manifest: dict[str, str]) -> int:
    bad = 0
    for name, digest in sorted(manifest.items()):
        p = dest / name
        if not p.is_file():
            print(f"  MISSING  {name}")
            bad += 1
        elif sha256(p.read_bytes()) != digest:
            print(f"  MISMATCH {name}")
            bad += 1
    print(f"{len(manifest) - bad} of {len(manifest)} files verified in {dest}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"output directory (default: {DEFAULT_DEST.relative_to(REPO)})")
    ap.add_argument("--zip", type=Path, help="use a local copy of unimals_100.zip instead of downloading")
    ap.add_argument("--url", default=DOWNLOAD_URL, help="override the download URL (for a mirror)")
    ap.add_argument("--check", action="store_true", help="only verify the files already in --dest")
    args = ap.parse_args()

    manifest = read_manifest()
    if args.check:
        return verify_dir(args.dest, manifest)

    data = args.zip.read_bytes() if args.zip else download(args.url)
    digest = sha256(data)
    if digest == ARCHIVE_SHA256:
        print("Archive hash matches the archive recorded on 2026-10-07.")
    else:
        print(f"NOTE: archive hash {digest} differs from the recorded {ARCHIVE_SHA256}; checking files one by one.")

    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in manifest:
            member = ARCHIVE_MEMBER_DIR + name
            try:
                files[name] = zf.read(member)
            except KeyError:
                pass
    missing = sorted(set(manifest) - set(files))
    mismatched = sorted(n for n, b in files.items() if sha256(b) != manifest[n])
    for n in missing:
        print(f"  MISSING  {n}")
    for n in mismatched:
        print(f"  MISMATCH {n}")
    if missing or mismatched:
        print(f"ERROR: {len(missing)} missing and {len(mismatched)} mismatched files; nothing was written.")
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.dest.parent) as tmp:
        for name, blob in files.items():
            (Path(tmp) / name).write_bytes(blob)
        for name in files:
            shutil.move(str(Path(tmp) / name), args.dest / name)
    print(f"Wrote {len(files)} verified UNIMAL XMLs to {args.dest}")
    return verify_dir(args.dest, manifest)


if __name__ == "__main__":
    sys.exit(main())
