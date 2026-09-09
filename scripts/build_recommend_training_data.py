"""Inspect frozen R1 training-data readiness, with optional new-file export."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ops.recommendation_evidence import EvidenceError  # noqa: E402
from signals.recommend_training_data import (  # noqa: E402
    build_training_readiness,
    readiness_summary,
)


def write_new_report(
    path: str | Path, report: dict, *, root: Path = _ROOT,
    protected_roots: tuple[Path, ...] = (),
) -> None:
    """Atomically publish a new JSON file inside root; never replace any target.

    Parent directories must already exist.  Directory descriptors prevent
    symlink traversal and keep a concurrently replaced parent from redirecting
    publication.  Temporary files are created in that same directory.
    """
    root = root.resolve()
    destination = Path(os.path.abspath(path)) if Path(path).is_absolute() else root / path
    relative = destination.relative_to(root)
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("output must be a new file inside the project")
    canonical_roots = tuple(root / "output" / name for name in (
        "recommend_snapshots", "recommend_score_labels", "recommend_receipts",
    ))
    resolved = destination.resolve()
    if any(resolved.is_relative_to(item.resolve()) for item in (*canonical_roots, *protected_roots)):
        raise ValueError("report output must be outside canonical input roots")
    content = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".{relative.name}.{uuid.uuid4().hex}.tmp"
    created = False
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        created = True
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, relative.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False)
        os.fsync(directory_fd)
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory_fd)
        os.close(directory_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--slot", choices=("both", "open", "preopen"), default="both")
    parser.add_argument("--now", help="timezone-aware evidence cutoff; default=current UTC")
    parser.add_argument("--snapshot-root", type=Path, default=_ROOT / "output/recommend_snapshots")
    parser.add_argument("--label-root", type=Path, default=_ROOT / "output/recommend_score_labels")
    parser.add_argument("--receipt-root", type=Path, default=_ROOT / "output/recommend_receipts")
    parser.add_argument("--min-train-dates", type=int, default=10,
                        help="initial diagnostic value, not a promotion gate (default=10)")
    parser.add_argument("--validation-dates", type=int, default=5,
                        help="initial diagnostic block size (default=5)")
    parser.add_argument("--output", type=Path, help="optional NEW JSON inside the project; never overwritten")
    args = parser.parse_args(argv)
    try:
        now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
        report = build_training_readiness(
            snapshot_root=args.snapshot_root, label_root=args.label_root,
            receipt_root=args.receipt_root, start_date=args.start_date, end_date=args.end_date,
            now=now, slots=("open", "preopen") if args.slot == "both" else (args.slot,),
            min_train_dates=args.min_train_dates, validation_dates=args.validation_dates,
        )
        if args.output is not None:
            write_new_report(args.output, report, protected_roots=(
                args.snapshot_root, args.label_root, args.receipt_root,
            ))
        print(json.dumps(readiness_summary(report), ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "model_fitted": False,
                          "deployable": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
