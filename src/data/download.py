"""Download the IEEE-CIS labelled files through the Kaggle API.

Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or KAGGLE_USERNAME /
KAGGLE_KEY) and acceptance of the competition rules on kaggle.com; without the
latter the API returns HTTP 403.
"""
from __future__ import annotations

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


def raw_files_present(raw_dir: Path, files: list[str]) -> bool:
    return all((raw_dir / f).exists() and (raw_dir / f).stat().st_size > 0 for f in files)


def download(force: bool = False) -> Path:
    cfg = pipeline_config()
    raw = paths(cfg).ensure().raw
    files = cfg["kaggle"]["files"]
    if raw_files_present(raw, files) and not force:
        print(f"[data] raw files already present in {raw}")
        return raw
    comp = cfg["kaggle"]["competition"]
    for f in files:
        cmd = [_kaggle_executable(), "competitions", "download", "-c", comp, "-f", f, "-p", str(raw)]
        print("[data] " + " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed for {f}.\nstdout: {res.stdout}\nstderr: {res.stderr}\n"
                "Check ~/.kaggle/kaggle.json and that the competition rules are accepted at "
                "https://www.kaggle.com/c/ieee-fraud-detection/rules"
            )
    for z in raw.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw)
        z.unlink()
    missing = [f for f in files if not (raw / f).exists()]
    if missing:
        raise RuntimeError(f"Download finished but files are missing: {missing}")
    return raw


if __name__ == "__main__":
    download()
