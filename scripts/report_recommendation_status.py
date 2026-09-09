#!/usr/bin/env python3
"""Print local R1 status; optionally create one new report file.

Exit 0: waiting/pending or validated delivery. Exit 1: attention required.
Exit 2: invalid arguments or output destination. No live operation is invoked.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

# A status query must not create bytecode files in the working project.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ops.recommendation_status import (  # noqa: E402
    KST,
    build_recommendation_status,
    format_recommendation_status,
)
from signals.recommend_snapshot import DEFAULT_SNAPSHOT_ROOT  # noqa: E402
from notifier.delivery_receipt import DEFAULT_RECEIPT_ROOT  # noqa: E402


def _write_new_output(path: Path, content: str, *, protected_roots: list[Path]) -> None:
    destination = path.absolute()
    resolved = destination.resolve()
    if any(resolved.is_relative_to(root.resolve()) for root in protected_roots):
        raise ValueError("report output must not be inside a canonical evidence tree")
    parent_stat = destination.parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("output parent must be an existing non-symlink directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(destination.parent, flags)
    try:
        opened = os.fstat(parent_fd)
        if (opened.st_dev, opened.st_ino) != (parent_stat.st_dev, parent_stat.st_ino):
            raise ValueError("output parent changed while opening")
        descriptor = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", help="Canonical KST decision date; defaults to now's KST date")
    parser.add_argument("--now", help="Aware ISO timestamp used only for this report")
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--output", type=Path, help="Create a new file under an existing parent; never overwrite")
    args = parser.parse_args(argv)
    try:
        now = datetime.fromisoformat(args.now) if args.now else datetime.now(KST)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("--now must include a timezone")
        asof = args.asof if args.asof is not None else now.astimezone(KST).date().isoformat()
        report = build_recommendation_status(
            asof, now=now, snapshot_root=args.snapshot_root, receipt_root=args.receipt_root,
        )
        content = (
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            if args.format == "json" else format_recommendation_status(report)
        )
        if args.output is not None:
            roots = [DEFAULT_SNAPSHOT_ROOT, DEFAULT_RECEIPT_ROOT,
                     ROOT / "output/recommend_score_labels"]
            roots.extend(root for root in (args.snapshot_root, args.receipt_root) if root is not None)
            _write_new_output(args.output, content, protected_roots=roots)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(content)
    return 1 if report["attention_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
