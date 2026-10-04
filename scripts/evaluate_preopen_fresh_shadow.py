#!/usr/bin/env python3
"""08:50 그림자 기록 평가 — 그림자 Top3 vs 같은 날 09:05 R1 Top3 (수동, 스케줄 없음).

scripts/record_preopen_fresh_shadow.py 가 4~8주 쌓인 뒤 1회 수동 실행한다.

비교 (같은 날짜 D 끼리):
  1) 겹침 — 그림자 Top3 와 output/recommend_snapshots/D/open_r1.json Top3 의
     겹치는 종목 수(/3), 3개 모두 일치 비율, 후보 순위 Spearman, 그림자 Top3 의
     09:05 순위. 참고로 지금 08:50 (preopen_r1.json) 도 같은 방식으로 본다.
  2) 결과 — data/upbit_15m.db (read-only) 로 D 09:00 15분봉 시가에 진입했다고
     보고 96봉(24시간) 경로에서 +10%·+5% 고가 도달, −5% 저가 도달 비율.
     arm 별로 날짜 평균을 낸 뒤 날짜 단위 paired bootstrap 으로 차이 CI95.
★ 비용 미반영(gross) 진단치 — 왕복 거래비용을 차감하지 않은 고가·저가 도달 여부만
  본다. 사이징·청산 규칙도 없다. net 성과가 아니며 운영 판단 lever 아님.
  (같은 09:00 진입 기준 arm 간 paired 비교라 차이 추정에는 비용 영향이 작다.)
★ 그림자 기록은 status=ok 이고 rank_basis 가 'R1_riskreward' 로 시작하며
  freshness.cron_log.health_gate == 'passed' 인 것만 쓴다 (degraded·stale_input·
  failed, R1 preopen D1 health gate 실패·미확인 제외 — 그날 라이브 08:50 은 gate
  실패면 발송되지 않는다). 제외된 날짜는 사유별로 shadow_excluded 에 보고한다.
★ 읽기 전용 — DB 는 mode=ro, 출력은 stdout (또는 --out 지정 파일).

    python scripts/evaluate_preopen_fresh_shadow.py [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHADOW_ROOT = ROOT / "output" / "preopen_fresh_shadow"
DEFAULT_SNAPSHOT_ROOT = ROOT / "output" / "recommend_snapshots"
DEFAULT_M15_DB = ROOT / "data" / "upbit_15m.db"

# 초기값(placeholder) — 평가 정의. 변경 시 결과 비교 불가하므로 기록과 함께 바꾼다.
ENTRY_HHMM = (9, 0)
PATH_BARS = 96            # 15분봉 96개 = 24시간
BAR = pd.Timedelta(minutes=15)
UP_THRESHOLDS = {"hit10": 0.10, "hit5": 0.05}
DOWN_THRESHOLDS = {"dn5": -0.05}
METRICS = ["hit10", "hit5", "dn5"]
N_BOOT = 4000
SEED = 7

ARMS = ("shadow", "open0905", "preopen0850")
R1_RANK_BASIS_PREFIX = "R1_riskreward"
COST_BASIS_NOTE = ("비용 미반영(gross) 진단치 — 왕복 거래비용 미차감, net 아님. "
                   "arm 간 paired 차이 진단용이며 운영 판단 lever 아님.")


# --------------------------------------------------------------------------
# 1. 입력
# --------------------------------------------------------------------------
def _shadow_exclusion(rec: dict) -> str | None:
    """평가에서 뺄 사유 (None = 사용)."""
    result = rec.get("result")
    status = rec.get("status")
    if status != "ok" or not result:
        return f"status={status}" if status != "ok" else "status=ok without result"
    if not str(result.get("rank_basis") or "").startswith(R1_RANK_BASIS_PREFIX):
        return "non-R1 rank_basis"
    cron = (rec.get("freshness") or {}).get("cron_log") or {}
    gate = cron.get("health_gate")
    if gate != "passed":
        # health_gate 도입 전 기록(필드 없음)도 확인 불가로 뺀다.
        return f"health_gate={gate or 'missing'}"
    return None


def load_shadow_with_exclusions(shadow_root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """({date: result} 평가 대상, {date 또는 파일명: 제외 사유})."""
    out: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    for path in sorted(shadow_root.glob("????-??-??.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            excluded[path.stem] = f"unreadable: {type(exc).__name__}"
            continue
        reason = _shadow_exclusion(rec)
        day = str(rec.get("asof") or path.stem)
        if reason is not None:
            excluded[day] = reason
            continue
        out[day] = rec["result"]
    return out, excluded


def load_shadow(shadow_root: Path) -> dict[str, dict]:
    """status=ok·R1 rank_basis·health_gate=passed 인 그림자 기록만 {date: result}.

    degraded(대체 정렬)·stale_input·failed 는 제외. rank_basis 도 다시 확인한다
    (degraded 도입 전 기록이 섞여도 fallback Top3 가 평가에 들어가지 않게).
    R1 preopen D1 health gate 가 failed·unknown(또는 필드 없음)인 날도 제외.
    """
    return load_shadow_with_exclusions(shadow_root)[0]


def exclusion_summary(excluded: dict[str, str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for reason in excluded.values():
        counts[reason] = counts.get(reason, 0) + 1
    return {"n_days": len(excluded), "by_reason": dict(sorted(counts.items())),
            "days": dict(sorted(excluded.items()))}


def load_snapshot(snapshot_root: Path, day: str, name: str) -> dict | None:
    path = snapshot_root / day / name
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {
        "top3": [u["coin"] for u in doc.get("top3") or []],
        "ranks": {u["coin"]: u["rank"] for u in doc.get("universe") or []},
    }


def shadow_arm(result: dict) -> dict:
    return {
        "top3": [u["coin"] for u in result.get("top3") or []],
        "ranks": {c[0]: c[1] for c in result.get("candidates") or []},
    }


# --------------------------------------------------------------------------
# 2. 겹침
# --------------------------------------------------------------------------
def overlap(a: list[str], b: list[str]) -> int:
    return len(set(a) & set(b))


def rank_spearman(ra: dict[str, int], rb: dict[str, int], min_common: int = 10) -> float:
    common = [c for c in ra if c in rb]
    if len(common) < min_common:
        return float("nan")
    x = pd.Series([ra[c] for c in common], dtype=float)
    y = pd.Series([rb[c] for c in common], dtype=float)
    return float(x.rank().corr(y.rank()))


def overlap_summary(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
    if not pairs:
        return {"n_days": 0}
    ov = [overlap(a["top3"], b["top3"]) for a, b in pairs]
    rho = [rank_spearman(a["ranks"], b["ranks"]) for a, b in pairs]
    ranks_in_b = [b["ranks"].get(c) for a, b in pairs for c in a["top3"]]
    known = [r for r in ranks_in_b if r is not None]
    return {
        "n_days": len(pairs),
        "mean_overlap_of_3": round(float(np.mean(ov)), 4),
        "share_3of3": round(float(np.mean([v == 3 for v in ov])), 4),
        "share_0of3": round(float(np.mean([v == 0 for v in ov])), 4),
        "overlap_dist": {str(k): int(sum(v == k for v in ov)) for k in range(4)},
        "spearman_median": (round(float(np.nanmedian(rho)), 4)
                            if not all(np.isnan(rho)) else None),
        "a_top3_rank_in_b_median": float(np.median(known)) if known else None,
        "a_top3_in_b_top10": (round(float(np.mean([r <= 10 for r in known])), 4)
                              if known else None),
        "a_top3_missing_in_b": int(sum(r is None for r in ranks_in_b)),
    }


# --------------------------------------------------------------------------
# 3. 경로 결과 (15m)
# --------------------------------------------------------------------------
def path_outcome(bars: pd.DataFrame, entry_ts: pd.Timestamp) -> dict[str, float] | None:
    """entry_ts 시가 진입 → 96봉 창의 도달 여부. 진입봉이 없으면 None.

    bars: timestamp(오름차순)·open·high·low 열. 창 = [entry_ts, entry_ts + 96봉).
    """
    if bars.empty:
        return None
    end = entry_ts + PATH_BARS * BAR
    w = bars[(bars["timestamp"] >= entry_ts) & (bars["timestamp"] < end)]
    if w.empty or w["timestamp"].iloc[0] != entry_ts:
        return None
    entry = float(w["open"].iloc[0])
    if not np.isfinite(entry) or entry <= 0:
        return None
    hi = float(w["high"].max()) / entry - 1.0
    lo = float(w["low"].min()) / entry - 1.0
    out = {k: float(hi >= thr) for k, thr in UP_THRESHOLDS.items()}
    out.update({k: float(lo <= thr) for k, thr in DOWN_THRESHOLDS.items()})
    out["max_up"] = hi
    out["max_dn"] = lo
    out["n_bars"] = float(len(w))
    return out


def load_m15(db: Path, markets: list[str], start: pd.Timestamp,
             end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    if not markets:
        return {}
    q = ("SELECT market, timestamp, open, high, low FROM candles "  # noqa: S608
         "WHERE timestamp >= ? AND timestamp < ? AND market IN ({})".format(
             ",".join("?" * len(markets))))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(q, con, params=[str(start), str(end), *markets],
                               parse_dates=["timestamp"])
    finally:
        con.close()
    df = df.sort_values(["market", "timestamp"])
    return {m: g.reset_index(drop=True) for m, g in df.groupby("market")}


def m15_last_ts(db: Path) -> pd.Timestamp | None:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT MAX(timestamp) FROM candles").fetchone()
    finally:
        con.close()
    return pd.Timestamp(row[0]) if row and row[0] else None


def day_means(picks: list[str], bars: dict[str, pd.DataFrame],
              entry_ts: pd.Timestamp) -> dict[str, float] | None:
    """arm 의 Top3 결과를 그날 평균으로. 결과가 하나도 없으면 None."""
    rows = []
    for coin in picks:
        r = path_outcome(bars.get(coin, pd.DataFrame()), entry_ts)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS} | {"n_picks": len(rows)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict[str, float]:
    """날짜 단위 paired bootstrap — mean(a-b) 와 CI95."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        return {"n_days": 0, "mean_diff": float("nan"), "ci95": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_days": int(d.size), "mean_diff": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


# --------------------------------------------------------------------------
# 4. 평가
# --------------------------------------------------------------------------
def evaluate(shadow_root: Path, snapshot_root: Path, m15_db: Path | None) -> dict[str, Any]:
    shadow, excluded = load_shadow_with_exclusions(shadow_root)
    days = sorted(shadow)
    arms_by_day: dict[str, dict[str, dict]] = {}
    for day in days:
        arms = {"shadow": shadow_arm(shadow[day])}
        op = load_snapshot(snapshot_root, day, "open_r1.json")
        if op:
            arms["open0905"] = op
        pr = load_snapshot(snapshot_root, day, "preopen_r1.json")
        if pr:
            arms["preopen0850"] = pr
        arms_by_day[day] = arms

    out: dict[str, Any] = {
        "cost_basis": "gross",
        "note": COST_BASIS_NOTE,
        "shadow_days_ok": len(days),
        "shadow_excluded": exclusion_summary(excluded),
        "definition": {
            "entry": f"D {ENTRY_HHMM[0]:02d}:{ENTRY_HHMM[1]:02d} 15m open",
            "path_bars": PATH_BARS, "metrics": METRICS,
            "costs": "none (gross — 비용 미반영 진단치)",
            "bootstrap": {"unit": "date", "n_boot": N_BOOT, "seed": SEED},
        },
        "overlap": {
            "shadow_vs_open0905": overlap_summary(
                [(a["shadow"], a["open0905"]) for a in arms_by_day.values() if "open0905" in a]),
            "preopen0850_vs_open0905": overlap_summary(
                [(a["preopen0850"], a["open0905"]) for a in arms_by_day.values()
                 if "open0905" in a and "preopen0850" in a]),
        },
        "outcomes": None,
    }
    if m15_db is None or not m15_db.exists():
        out["outcomes"] = {"skipped": "15m DB unavailable"}
        return out

    last_ts = m15_last_ts(m15_db)
    per_day: dict[str, dict[str, dict]] = {}
    incomplete: list[str] = []
    for day, arms in arms_by_day.items():
        entry_ts = pd.Timestamp(day) + pd.Timedelta(hours=ENTRY_HHMM[0], minutes=ENTRY_HHMM[1])
        end = entry_ts + PATH_BARS * BAR
        if last_ts is None or last_ts < end - BAR:
            incomplete.append(day)
            continue
        coins = sorted({c for a in arms.values() for c in a["top3"]})
        bars = load_m15(m15_db, coins, entry_ts, end)
        per_day[day] = {}
        for name, arm in arms.items():
            r = day_means(arm["top3"], bars, entry_ts)
            if r is not None:
                per_day[day][name] = r

    summary: dict[str, Any] = {"n_days_complete": len(per_day),
                               "incomplete_days": incomplete, "arms": {}, "diff": {}}
    for name in ARMS:
        vals = [d[name] for d in per_day.values() if name in d]
        if vals:
            summary["arms"][name] = {"n_days": len(vals)} | {
                m: round(float(np.mean([v[m] for v in vals])), 4) for m in METRICS}
    for a, b in (("shadow", "open0905"), ("shadow", "preopen0850"),
                 ("open0905", "preopen0850")):
        common = [d for d in per_day.values() if a in d and b in d]
        summary["diff"][f"{a}-{b}"] = {
            m: paired_bootstrap(np.array([d[a][m] for d in common]),
                                np.array([d[b][m] for d in common]))
            for m in METRICS}
    out["outcomes"] = summary
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="08:50 fresh 그림자 vs 09:05 R1 평가 (읽기 전용)")
    ap.add_argument("--shadow-root", type=Path, default=DEFAULT_SHADOW_ROOT)
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    ap.add_argument("--m15-db", type=Path, default=DEFAULT_M15_DB)
    ap.add_argument("--out", type=Path, default=None, help="결과 JSON 파일 (기본 stdout)")
    args = ap.parse_args()
    result = evaluate(args.shadow_root, args.snapshot_root, args.m15_db)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    sys.exit(0)


if __name__ == "__main__":
    main()
