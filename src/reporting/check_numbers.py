"""Fail if a number in the rendered documents is absent from results.json.

A numeric token in a document is accepted when some numeric value in
results.json, rounded to the token's number of decimals, equals it either
directly or as a percentage (value x 100). Tokens that are part of an
identifier (``V339``, ``id_30``, ``H1``, ``P90``) are not numbers and are
skipped, as are fenced code blocks, inline code, and link / image targets.

Run: ``python -m src.reporting.check_numbers`` (exit code 1 on failure).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.config import ROOT

CHECKED = ("README.md", "REPORT.md", "FINDINGS.md", "MONITORING_PLAN.md")
TOKEN = re.compile(r"(?<![\w.\-/])-?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?%?(?![\w/])")


def leaves(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            out += leaves(v)
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        try:
            out.append(float(obj))
        except ValueError:
            pass
    return out


def strip_non_prose(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)       # fenced code
    text = re.sub(r"`[^`\n]*`", " ", text)                    # inline code
    text = re.sub(r"\]\([^)]*\)", "]", text)                  # link / image targets
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)       # comments
    text = re.sub(r"^\s*\d+\.\s", " ", text, flags=re.M)      # ordered-list markers
    return text


def tokens(text: str) -> list[str]:
    return [m.group(0) for m in TOKEN.finditer(strip_non_prose(text))]


def _parse(tok: str) -> tuple[float, int]:
    core = tok.replace("$", "").replace("%", "").replace(",", "")
    dec = len(core.split(".")[1]) if "." in core else 0
    return abs(float(core)), dec


def matches(tok: str, values: list[float]) -> bool:
    v, d = _parse(tok)
    tol = 0.5 * 10 ** (-d) + 1e-9
    for x in values:
        for cand in (abs(x), abs(x) * 100):
            if abs(round(cand, d) - v) <= 1e-9 or (d > 0 and abs(cand - v) < tol and round(cand, d) == v):
                return True
    return False


def check_text(text: str, results: dict) -> list[str]:
    vals = leaves(results)
    return [t for t in tokens(text) if not matches(t, vals)]


def check_documents(root: Path = ROOT, results_path: Path | None = None) -> list[str]:
    results_path = results_path or root / "outputs" / "results.json"
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    problems = []
    for doc in CHECKED:
        p = Path(root) / doc
        if not p.exists():
            problems.append(f"{doc}: missing")
            continue
        for t in check_text(p.read_text(encoding="utf-8"), results):
            problems.append(f"{doc}: '{t}' not found in results.json")
    return problems


def main() -> None:
    problems = check_documents()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("All numbers in the documents are sourced from outputs/results.json")


if __name__ == "__main__":
    main()
