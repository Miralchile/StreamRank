from __future__ import annotations

import hashlib
import json
import os
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

KUAI_RAND_PURE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
KUAI_RAND_PURE_MD5 = "0820331067a3784d9691136f772b35a7"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"archive path traversal detected: {member.name}")
        archive.extractall(destination)


def download_kuairand_pure(destination: str | Path) -> dict[str, object]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "KuaiRand-Pure.tar.gz"
    extracted = destination / "KuaiRand-Pure/data"
    if not archive_path.is_file() or file_md5(archive_path) != KUAI_RAND_PURE_MD5:
        temporary = archive_path.with_suffix(".tar.gz.part")
        try:
            with urllib.request.urlopen(KUAI_RAND_PURE_URL, timeout=60) as response:
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if file_md5(temporary) != KUAI_RAND_PURE_MD5:
                raise ValueError("downloaded KuaiRand-Pure archive failed MD5 verification")
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    digest = file_md5(archive_path)
    if digest != KUAI_RAND_PURE_MD5:
        raise ValueError("KuaiRand-Pure archive failed MD5 verification")
    if not extracted.is_dir():
        safe_extract_tar(archive_path, destination)
    report = {
        "dataset": "KuaiRand-Pure",
        "official_source": KUAI_RAND_PURE_URL,
        "license": "CC-BY-SA-4.0",
        "archive": str(archive_path),
        "md5": digest,
        "data_dir": str(extracted),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
