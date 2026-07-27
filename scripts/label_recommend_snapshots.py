"""recommend score snapshot 전 유니버스의 24h 사후 라벨 CLI.

단일 파일:
    python scripts/label_recommend_snapshots.py \
      --snapshot output/recommend_snapshots/2026-07-24/open_r1.json

날짜 폴더 전체(open/preopen, R1/R2/A1):
    python scripts/label_recommend_snapshots.py --date 2026-07-24

어제까지 미완결 artifact 재시도:
    python scripts/label_recommend_snapshots.py --through-date 2026-07-24

active ranking·알림·ledger에는 손대지 않고 별도 atomic artifact만 만든다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date as calendar_date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.recommend_score_labels import (  # noqa: E402
    DEFAULT_LABEL_ROOT,
    M15_DB_PATH,
    label_recommend_snapshot,
)
from signals.recommend_snapshot import DEFAULT_SNAPSHOT_ROOT  # noqa: E402


def _iso_date(value: str) -> str:
    try:
        return calendar_date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "날짜는 YYYY-MM-DD 형식이어야 합니다"
        ) from exc


def _resolve_inputs(
    *,
    snapshot: str | None,
    date: str | None,
    through_date: str | None,
    snapshot_root: str | Path,
) -> list[Path]:
    if snapshot:
        return [Path(snapshot)]
    root = Path(snapshot_root)
    if date:
        calendar_date.fromisoformat(date)
        candidates = (root / date).glob("*.json")
    else:
        if through_date is None:
            raise ValueError("snapshot, date, through_date 중 하나가 필요합니다")
        cutoff = calendar_date.fromisoformat(str(through_date))

        def within_cutoff(path: Path) -> bool:
            try:
                return calendar_date.fromisoformat(path.parent.name) <= cutoff
            except ValueError:
                return False

        candidates = (
            p
            for p in root.glob("*/*.json")
            if within_cutoff(p)
        )
    return sorted(
        p for p in candidates
        if (
            p.is_file()
            and not p.name.startswith(".")
            and ".limit" not in p.stem
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="recommend snapshot 전 유니버스 24h path 사후 라벨"
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", help="단일 snapshot JSON")
    source.add_argument(
        "--date",
        type=_iso_date,
        help="YYYY-MM-DD snapshot 폴더 전체",
    )
    source.add_argument(
        "--through-date",
        type=_iso_date,
        help="YYYY-MM-DD까지 모든 snapshot 처리(과거 partial 자동 재시도)",
    )
    ap.add_argument(
        "--snapshot-root", default=str(DEFAULT_SNAPSHOT_ROOT),
        help="--date 조회 root",
    )
    ap.add_argument(
        "--output-root", default=str(DEFAULT_LABEL_ROOT),
        help="라벨 artifact root",
    )
    ap.add_argument("--db", default=str(M15_DB_PATH), help="Upbit 15m SQLite DB")
    ap.add_argument(
        "--now",
        default=None,
        help="성숙도 판정 KST 시각(테스트/재현용, default=현재)",
    )
    args = ap.parse_args()

    inputs = _resolve_inputs(
        snapshot=args.snapshot,
        date=args.date,
        through_date=args.through_date,
        snapshot_root=args.snapshot_root,
    )
    if not inputs:
        print("label skip: matching snapshot JSON 없음", file=sys.stderr)
        return 3

    summaries = []
    had_partial = False
    had_error = False
    for path in inputs:
        try:
            result = label_recommend_snapshot(
                path,
                output_root=args.output_root,
                db_path=args.db,
                now=args.now,
            )
            summaries.append({
                "snapshot": str(path),
                "status": result["artifact_status"],
                "artifact": result.get("artifact_path"),
                "written": result.get("written", False),
                "reused": result.get("artifact_reused", False),
                "summary": result.get("summary"),
                "reason": result.get("reason"),
            })
            if result["artifact_status"] == "partial":
                had_partial = True
        except Exception as exc:  # CLI batch에서 나머지 snapshot은 계속 처리한다.
            summaries.append({
                "snapshot": str(path),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            had_error = True

    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
    if had_error:
        return 1
    if had_partial:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
