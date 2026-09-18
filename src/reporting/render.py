"""Render README.md, REPORT.md and FINDINGS.md from templates and results.json.

Templates live in ``docs_templates/``. Every number in a rendered document
comes from ``results.json`` through a formatting filter; no metric is typed
into a template. ``src/reporting/check_numbers.py`` enforces this.
"""
from __future__ import annotations

from pathlib import Path

import jinja2

from src.config import ROOT

TEMPLATES = ROOT / "docs_templates"
DOCUMENTS = ("README.md", "REPORT.md", "FINDINGS.md", "MONITORING_PLAN.md")


def _num(v):
    if v is None:
        raise ValueError("value missing from results.json")
    return float(v)


def usd(v, d: int = 0) -> str:
    v = _num(v)
    s = f"${abs(v):,.{d}f}"
    return f"-{s}" if v < 0 and round(abs(v), d) != 0 else s


def pct(v, d: int = 1) -> str:
    return f"{_num(v) * 100:.{d}f}%"


def num(v, d: int = 3) -> str:
    return f"{_num(v):,.{d}f}"


def integer(v) -> str:
    return f"{int(round(_num(v))):,}"


def _defined(c: dict) -> bool:
    return c.get("point") is not None and c.get("ci_low") is not None


def ci_usd(c: dict, d: int = 0) -> str:
    if not _defined(c):
        return "n/a"
    return f"{usd(c['point'], d)} (95% CI {usd(c['ci_low'], d)} to {usd(c['ci_high'], d)})"


def ci_num(c: dict, d: int = 4) -> str:
    if not _defined(c):
        return "n/a"
    return f"{num(c['point'], d)} (95% CI {num(c['ci_low'], d)} to {num(c['ci_high'], d)})"


def ci_pp(c: dict, d: int = 1) -> str:
    """Difference of two rates, in percentage points."""
    if not _defined(c):
        return "n/a"
    f = lambda v: f"{_num(v) * 100:.{d}f}"  # noqa: E731
    return f"{f(c['point'])} pp (95% CI {f(c['ci_low'])} to {f(c['ci_high'])} pp)"


def verdict(c: dict) -> str:
    """Plain statement of whether an interval excludes zero."""
    if c.get("n_boot_valid", 1) == 0 or c.get("ci_low") is None:
        return "undefined: the statistic cannot be computed (for example, a policy that flags nothing)"
    if c["excludes_zero"]:
        return "interval excludes zero" if c["point"] > 0 else "interval excludes zero, in the unfavourable direction"
    return "interval includes zero: not distinguishable from no difference"


def passfail(b) -> str:
    return "PASS" if b else "FAIL"


def environment() -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)), undefined=jinja2.StrictUndefined,
                             keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
    env.filters.update(usd=usd, pct=pct, num=num, int=integer, ci_usd=ci_usd, ci_num=ci_num, ci_pp=ci_pp,
                       verdict=verdict, passfail=passfail)
    return env


def render_all(results: dict, out_dir: Path = ROOT) -> list[Path]:
    env = environment()
    written = []
    for doc in DOCUMENTS:
        text = env.get_template(doc + ".j2").render(r=results)
        out = Path(out_dir) / doc
        out.write_text(text, encoding="utf-8")
        written.append(out)
    return written
