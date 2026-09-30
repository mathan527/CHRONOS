"""Export a JSONL trace as ONE standalone HTML file (data embedded, works offline, no server).

    python -m chronos.trace.export traces/<session>.jsonl              # -> traces/<session>.html
    python -m chronos.trace.export traces/<session>.jsonl -o demo.html
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("visualizer.html")
MARKER = "/*CHRONOS_TRACE_DATA*/null"  # appears exactly once in the template
DEFAULT_TITLE = "<title>CHRONOS Trace Timeline</title>"


def read_trace(path: Path) -> tuple[list[dict[str, Any]], int]:
    """-> (rows, number_of_skipped_lines). Blank lines are ignored; junk lines are counted."""
    rows: list[dict[str, Any]] = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except ValueError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            bad += 1
    return rows, bad


def render_html(rows: list[dict[str, Any]], title: str | None = None) -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    if page.count(MARKER) != 1:
        raise RuntimeError("visualizer.html must contain the data marker exactly once")
    data = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    # The JSON sits inside a <script>. Escaping every "<" (valid in JSON and in JS) means no
    # "</script>", "<!--" or "<script" sequence can appear in the data at all; U+2028/2029 are
    # legal JSON but were line terminators in JS string literals, so escape those too.
    data = (data.replace("<", "\\u003c").replace("\u2028", "\\u2028")
                .replace("\u2029", "\\u2029"))
    page = page.replace(MARKER, data)
    if title:
        page = page.replace(DEFAULT_TITLE, f"<title>{htmllib.escape(title)}</title>")
    return page


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chronos.trace.export",
                                 description="Export a CHRONOS trace as a standalone HTML timeline.")
    ap.add_argument("trace", type=Path, help="path to traces/<session>.jsonl")
    ap.add_argument("-o", "--output", type=Path, help="output file (default: next to the trace)")
    args = ap.parse_args(argv)

    if not args.trace.is_file():
        print(f"error: no such trace file: {args.trace}", file=sys.stderr)
        return 2
    rows, bad = read_trace(args.trace)
    if bad:
        print(f"warning: skipped {bad} line(s) that were not JSON objects", file=sys.stderr)
    if not rows:
        print("error: the trace contains no events", file=sys.stderr)
        return 1
    session = rows[0].get("session_id")
    out = args.output or args.trace.with_suffix(".html")
    out.write_text(render_html(rows, f"CHRONOS trace {session}" if session else None),
                   encoding="utf-8")
    print(f"wrote {out} ({len(rows)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
