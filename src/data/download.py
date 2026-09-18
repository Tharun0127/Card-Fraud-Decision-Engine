"""Download the IEEE-CIS labelled files through the Kaggle API.

Two sources, chosen by ``kaggle.source`` in ``config/pipeline.yaml``:

- ``competition``: the official competition files. Requires accepting the
  competition rules on kaggle.com; otherwise the API refuses the download.
- ``mirror``: a public Kaggle dataset that re-uploads the same files. Used in
  this project because the official download was refused (FINDINGS.md,
  deviation D-1).

Whatever the source, every file must match the byte size published on the
official competition listing, and its SHA-256 is recorded for provenance.
Credentials: ``~/.kaggle/access_token``, ``~/.kaggle/kaggle.json``, or the
KAGGLE_* environment variables.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import paths, pipeline_config


def _kaggle_executable() -> str:
    exe = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    if exe.exists():
        return str(exe)
    found = shutil.which("kaggle")
    if not found:
        raise RuntimeError("kaggle CLI not found; run `make setup` first")
    return found


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_files(raw: Path, expected_bytes: dict[str, int]) -> dict[str, dict]:
    """Raise if any file is missing or its size differs from the official listing."""
    report, problems = {}, []
    for name, size in expected_bytes.items():
        f = raw / name
        if not f.exists():
            problems.append(f"{name}: missing")
            continue
        actual = f.stat().st_size
        if actual != size:
            problems.append(f"{name}: {actual:,} bytes, expected {size:,}")
        report[name] = {"bytes": actual, "expected_bytes": size, "sha256": sha256(f)}
    if problems:
        raise RuntimeError("Downloaded data does not match the official file listing:\n" + "\n".join(problems))
    return report


def _command(cfg_k: dict, name: str, raw: Path) -> list[str]:
    exe = _kaggle_executable()
    if cfg_k["source"] == "competition":
        return [exe, "competitions", "download", "-c", cfg_k["competition"], "-f", name, "-p", str(raw)]
    if cfg_k["source"] == "mirror":
        return [exe, "datasets", "download", cfg_k["mirror_dataset"], "-f", name, "-p", str(raw)]
    raise ValueError(f"unknown kaggle.source: {cfg_k['source']}")


def download(force: bool = False) -> dict[str, dict]:
    cfg = pipeline_config()
    raw = paths(cfg).ensure().raw
    k = cfg["kaggle"]
    files = k["files"]
    if force or not all((raw / f).exists() for f in files):
        for f in files:
            cmd = _command(k, f, raw)
            print("[data] " + " ".join(cmd), flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Kaggle download failed for {f} (source: {k['source']}).\n"
                    f"stdout: {res.stdout}\nstderr: {res.stderr}\n"
                    "For source=competition, accept the rules at "
                    "https://www.kaggle.com/c/ieee-fraud-detection/rules"
                )
        for z in raw.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw)
            z.unlink()
    else:
        print(f"[data] raw files already present in {raw}", flush=True)
    report = verify_files(raw, k["expected_bytes"])
    print("[data] file sizes match the official competition listing", flush=True)
    return report


if __name__ == "__main__":
    download()
