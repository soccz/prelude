"""Dashboard JSON builder — paper_ledger 두 개 → soccz.github.io/projects/prelude/dashboard/data/*.json

산출물 (3개):
  summary.json   — 채널별 KPI (총 alert, hit rate, 가상 누적 PnL)
  history.json   — 전체 알림 행 (예측 + 실제 OHLC + 결과). 날짜 내림차순.
  accuracy.json  — rolling 30일 hit rate 시계열

가상 PnL 룰 (텔레그램 가이드와 동일):
  - 알림 1건 = 1단위 자본 (equal weight)
  - +5% 도달 시 익절, 아니면 EOD close (자동 손절 X)
  - 거래비용 ROUND_TRIP_COST_PCT 차감
  - cum_pnl_pct = sum of per-alert net_return (compounding 안 함, 단순합)

운영:
  매일 close cron 끝에 호출. JSON 만 갱신, html/JS 는 그대로 둔다.

사용:
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --out-dir <github.io path>
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.config import ROUND_TRIP_COST_PCT


DEFAULT_OUT_DIR = "/home/soccz/22tb/soccz.github.io/projects/prelude/dashboard/data"
ROLLING_WINDOW_DAYS = 30
TP_PCT = 0.05  # 사용자 가이드: "5% 오르면 즉시 매도"
DEFAULT_PIN = "9963"
PBKDF2_ITERATIONS = 250000


# ============================================================================
# Per-channel metrics
# ============================================================================
def _virtual_pnl_per_alert(max_ret_pct: float, close_ret_pct: float) -> float:
    """텔레그램 가이드 룰: TP 5% 도달 시 5% 익절, 아니면 EOD close. 비용 차감.

    Returns: net return as decimal (e.g. 0.034 == +3.4%).
    """
    if pd.isna(max_ret_pct):
        return np.nan
    max_d = max_ret_pct / 100.0
    close_d = (close_ret_pct or 0.0) / 100.0
    gross = TP_PCT if max_d >= TP_PCT else close_d
    return gross - ROUND_TRIP_COST_PCT


def _per_alert_pnl_series(closed: pd.DataFrame, max_col: str, close_col: str) -> pd.Series:
    """알림별 net PnL (TP rule 적용, 비용 차감) — 단일 시리즈."""
    pnl = closed.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    ).dropna()
    return pnl


def compute_quant_metrics(closed: pd.DataFrame, max_col: str, close_col: str) -> dict:
    """헤드라인 metric — Sharpe/Sortino/Calmar/Profit Factor/Expectancy.

    daily aggregation 으로 Sharpe (그날 net 의 평균 / std × √252).
    Calmar = annualized return / |MDD|.
    Profit Factor = sum(positive) / |sum(negative)|.
    Expectancy = avg per trade.
    """
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    if len(pnl) == 0:
        return {"n_trades": 0}

    closed = closed.copy()
    closed["pnl"] = closed.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    )
    closed = closed.dropna(subset=["pnl"])
    closed["date_dt"] = pd.to_datetime(closed["date"])

    # daily aggregate (같은 날 알림 N 개면 그날 net = mean)
    daily = closed.groupby(closed["date_dt"].dt.date)["pnl"].sum().sort_index()
    n_days = len(daily)
    avg_daily = float(daily.mean())
    std_daily = float(daily.std(ddof=1)) if n_days > 1 else float("nan")
    downside_daily = daily[daily < 0]
    downside_std = (
        float(downside_daily.std(ddof=1)) if len(downside_daily) > 1 else float("nan")
    )

    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily and std_daily > 0 else float("nan")
    sortino = (avg_daily / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else float("nan")

    cum = daily.cumsum()
    mdd = float((cum - cum.cummax()).min())
    annualized_return = avg_daily * 252
    calmar = annualized_return / abs(mdd) if mdd < 0 else float("nan")

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("nan")
    )
    expectancy = float(pnl.mean())
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")
    win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss and not np.isnan(avg_loss) and avg_loss < 0 else float("nan")

    return {
        "n_trades": int(len(pnl)),
        "n_days": int(n_days),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "sharpe_ann": _maybe_float(sharpe),
        "sortino_ann": _maybe_float(sortino),
        "calmar": _maybe_float(calmar),
        "profit_factor": _maybe_float(profit_factor),
        "expectancy_pct": _maybe_float(expectancy * 100),
        "avg_win_pct": _maybe_float(avg_win * 100) if len(wins) else None,
        "avg_loss_pct": _maybe_float(avg_loss * 100) if len(losses) else None,
        "win_loss_ratio": _maybe_float(win_loss_ratio),
        "annualized_return_pct": _maybe_float(annualized_return * 100),
    }


def _maybe_float(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 4)


def compute_bootstrap_ci(closed: pd.DataFrame, max_col: str, close_col: str,
                         n_iter: int = 1000, seed: int = 42) -> dict:
    """Trade-level bootstrap 95% CI for cum PnL, mean PnL, Sharpe.

    표본이 28 정도라 CI 는 매우 넓을 것 — 그게 "정직한" 신호.
    """
    pnl = _per_alert_pnl_series(closed, max_col, close_col).reset_index(drop=True)
    n = len(pnl)
    if n < 5:
        return {"n_trades": int(n), "note": "n<5: bootstrap skipped"}
    rng = np.random.default_rng(seed)
    cum_samples = np.empty(n_iter)
    mean_samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sample = pnl.iloc[idx].values
        cum_samples[i] = sample.sum()
        mean_samples[i] = sample.mean()
    return {
        "n_trades": int(n),
        "n_iter": int(n_iter),
        "cum_pnl_pct_ci95": [
            _maybe_float(np.quantile(cum_samples, 0.025) * 100),
            _maybe_float(np.quantile(cum_samples, 0.975) * 100),
        ],
        "mean_pnl_pct_ci95": [
            _maybe_float(np.quantile(mean_samples, 0.025) * 100),
            _maybe_float(np.quantile(mean_samples, 0.975) * 100),
        ],
    }


def compute_distribution_summary(df: pd.DataFrame) -> dict:
    """distribution paper_ledger → KPI dict."""
    closed = df[df["status"].astype(str) == "closed"].copy()
    out = {
        "n_alerts_total": int(len(df)),
        "n_closed": int(len(closed)),
        "n_pending": int(len(df) - len(closed)),
    }
    if len(df) > 0:
        out["first_alert_date"] = str(pd.to_datetime(df["date"]).min().date())
        out["last_alert_date"] = str(pd.to_datetime(df["date"]).max().date())
    if len(closed) == 0:
        return out

    # Hit rate (head 정의 그대로)
    for k in ("hit_h2", "hit_h6", "hit_h5"):
        if k in closed.columns:
            out[f"{k}_pct"] = float(closed[k].dropna().mean() * 100)

    out["avg_max_return_pct"] = float(closed["next_max_return_pct"].dropna().mean())
    out["avg_min_return_pct"] = float(closed["next_min_return_pct"].dropna().mean())
    out["avg_close_return_pct"] = float(closed["next_close_return_pct"].dropna().mean())
    out["median_max_return_pct"] = float(closed["next_max_return_pct"].dropna().median())
    out["win_rate_pct"] = float(
        (closed["next_close_return_pct"].dropna() > 0).mean() * 100
    )

    # 가상 누적 PnL (TP5 룰)
    pnl = closed.apply(
        lambda r: _virtual_pnl_per_alert(
            r["next_max_return_pct"], r["next_close_return_pct"]
        ),
        axis=1,
    ).dropna()
    if len(pnl) > 0:
        cum = pnl.cumsum()
        out["virtual"] = {
            "rule": "5% TP / EOD close, equal weight, cost 0.15% 차감",
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": float(cum.iloc[-1] * 100),
            "max_dd_pct": float((cum - cum.cummax()).min() * 100),
            "avg_pnl_per_trade_pct": float(pnl.mean() * 100),
            "tp_hit_rate_pct": float(
                (closed["next_max_return_pct"].dropna() / 100.0 >= TP_PCT).mean() * 100
            ),
        }
        out["quant"] = compute_quant_metrics(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["bootstrap"] = compute_bootstrap_ci(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
    return out


def compute_preopen_summary(df: pd.DataFrame) -> dict:
    """preopen paper_ledger → KPI dict (1h horizon)."""
    closed = df[df["status"].astype(str) == "closed"].copy()
    out = {
        "n_alerts_total": int(len(df)),
        "n_closed": int(len(closed)),
        "n_pending": int(len(df) - len(closed)),
    }
    if len(df) > 0:
        out["first_alert_date"] = str(pd.to_datetime(df["date"]).min().date())
        out["last_alert_date"] = str(pd.to_datetime(df["date"]).max().date())
    if len(closed) == 0:
        return out

    for k in (
        "hit_first15_3pct",
        "hit_first15_5pct",
        "hit_first1h_3pct",
        "hit_first1h_5pct",
    ):
        if k in closed.columns:
            out[f"{k}_pct"] = float(closed[k].dropna().mean() * 100)

    if "first_1h_max_return_pct" in closed.columns:
        out["avg_max_return_pct"] = float(closed["first_1h_max_return_pct"].dropna().mean())
        out["avg_min_return_pct"] = float(closed["first_1h_min_return_pct"].dropna().mean())
        out["avg_close_return_pct"] = float(
            closed["first_1h_close_return_pct"].dropna().mean()
        )
        # 가상 누적 PnL (1h 안 +5% 못 가면 1h close)
        pnl = closed.apply(
            lambda r: _virtual_pnl_per_alert(
                r["first_1h_max_return_pct"], r["first_1h_close_return_pct"]
            ),
            axis=1,
        ).dropna()
        if len(pnl) > 0:
            cum = pnl.cumsum()
            out["virtual"] = {
                "rule": "1h horizon: 5% TP / 1h close, equal weight, cost 0.15% 차감",
                "n_trades": int(len(pnl)),
                "cum_pnl_pct": float(cum.iloc[-1] * 100),
                "max_dd_pct": float((cum - cum.cummax()).min() * 100),
                "avg_pnl_per_trade_pct": float(pnl.mean() * 100),
                "tp_hit_rate_pct": float(
                    (closed["first_1h_max_return_pct"].dropna() / 100.0 >= TP_PCT).mean() * 100
                ),
            }
            out["quant"] = compute_quant_metrics(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["bootstrap"] = compute_bootstrap_ci(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
    return out


# ============================================================================
# History (combined rows)
# ============================================================================
def history_rows(df_dist: pd.DataFrame, df_preopen: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df_dist.iterrows():
        max_pct = r.get("next_max_return_pct")
        close_pct = r.get("next_close_return_pct")
        rows.append({
            "date": str(r["date"]),
            "channel": "distribution",
            "coin": str(r["coin"]).replace("KRW-", ""),
            "setups": str(r.get("setup_ids", "") or ""),
            "btc_regime": str(r.get("btc_regime", "")),
            "entry_price": _safe_float(r.get("next_open")),
            "p_h2": _safe_float(r.get("p_h2_3pct_4h")),
            "p_h6": _safe_float(r.get("p_h6_5pct_24h")),
            "p_h5": _safe_float(r.get("p_h5_20pct_tail")),
            "composite": _safe_float(r.get("composite_score")),
            "next_max_pct": _safe_float(max_pct),
            "next_min_pct": _safe_float(r.get("next_min_return_pct")),
            "next_close_pct": _safe_float(close_pct),
            "hit_h2": _safe_int(r.get("hit_h2")),
            "hit_h6": _safe_int(r.get("hit_h6")),
            "hit_h5": _safe_int(r.get("hit_h5")),
            "virtual_pnl_pct": _maybe_pct(_virtual_pnl_per_alert(max_pct, close_pct)),
            "status": str(r.get("status", "")),
        })

    for _, r in df_preopen.iterrows():
        max_pct = r.get("first_1h_max_return_pct")
        close_pct = r.get("first_1h_close_return_pct")
        # preopen 알림 시점 가격 = entry_price_proxy (alert 시점 close).
        # 09:00 첫 봉 시가 (first_open) 와 약간 다를 수 있음.
        entry = r.get("entry_price_proxy")
        if pd.isna(entry):
            entry = r.get("first_open")
        rows.append({
            "date": str(r["date"]),
            "channel": "preopen",
            "coin": str(r["coin"]).replace("KRW-", ""),
            "setups": "",
            "btc_regime": str(r.get("btc_regime", "")),
            "entry_price": _safe_float(entry),
            "p_first15_3": _safe_float(r.get("p_first15_3pct")),
            "p_first15_5": _safe_float(r.get("p_first15_5pct")),
            "p_first1h_3": _safe_float(r.get("p_first1h_3pct")),
            "composite": _safe_float(r.get("composite_score")),
            "next_max_pct": _safe_float(max_pct),
            "next_min_pct": _safe_float(r.get("first_1h_min_return_pct")),
            "next_close_pct": _safe_float(close_pct),
            "hit_h2": _safe_int(r.get("hit_first1h_3pct")),
            "hit_h6": _safe_int(r.get("hit_first1h_5pct")),
            "hit_h5": None,
            "virtual_pnl_pct": _maybe_pct(_virtual_pnl_per_alert(max_pct, close_pct)),
            "status": str(r.get("status", "")),
        })

    rows.sort(key=lambda x: (x["date"], x["channel"], x["coin"]), reverse=True)
    return rows


def _safe_float(v):
    try:
        f = float(v)
        if not np.isfinite(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_pct(v):
    if v is None or pd.isna(v):
        return None
    return round(v * 100, 3)


# ============================================================================
# Accuracy time series (rolling)
# ============================================================================
def rolling_accuracy(df: pd.DataFrame, hit_cols: list[str],
                      window_days: int = ROLLING_WINDOW_DAYS) -> list[dict]:
    """alert 시점 기준 rolling window 의 hit rate 시계열.

    각 date 마다 [date - window + 1 days, date] 의 alert 들을 모아 hit rate.
    """
    if len(df) == 0:
        return []
    closed = df[df["status"].astype(str) == "closed"].copy()
    if len(closed) == 0:
        return []
    closed["date"] = pd.to_datetime(closed["date"])
    dates = sorted(closed["date"].dt.normalize().unique())
    results = []
    for d in dates:
        start = d - pd.Timedelta(days=window_days - 1)
        window = closed[(closed["date"] >= start) & (closed["date"] <= d)]
        if len(window) == 0:
            continue
        row = {"date": str(d.date()), "n": int(len(window))}
        for c in hit_cols:
            if c in window.columns:
                vals = window[c].dropna()
                row[f"{c}_pct"] = float(vals.mean() * 100) if len(vals) else None
        # cum virtual pnl up to d
        pnl_col = "virtual_pnl_dec"
        if pnl_col not in window.columns:
            pass
        results.append(row)
    return results


def cumulative_pnl_series(df: pd.DataFrame,
                           max_col: str, close_col: str) -> list[dict]:
    """일자별 누적 가상 PnL 시계열.

    같은 날 알림 N 개면 그날 net = sum(per-alert net). cum = 일별 net 누적.
    """
    closed = df[df["status"].astype(str) == "closed"].copy()
    if len(closed) == 0:
        return []
    closed["date"] = pd.to_datetime(closed["date"])
    closed["pnl"] = closed.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    )
    closed = closed.dropna(subset=["pnl"])
    if len(closed) == 0:
        return []
    daily = closed.groupby(closed["date"].dt.date)["pnl"].sum().sort_index()
    cum = daily.cumsum()
    return [
        {
            "date": str(d),
            "daily_pnl_pct": float(daily.loc[d] * 100),
            "cum_pnl_pct": float(cum.loc[d] * 100),
        }
        for d in cum.index
    ]


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--paper-ledger-preopen", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--asof", type=str, help="기준 시점 (default=now)")
    parser.add_argument(
        "--pin",
        default=os.environ.get("PRELUDE_DASHBOARD_PIN", DEFAULT_PIN),
        help="암호화 PIN (default 9963 또는 PRELUDE_DASHBOARD_PIN env). "
             "빈 값 ('') 으로 주면 평문 출력 (테스트용).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dashboard")

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pin = args.pin or None
    log.info(f"encryption: {'PIN ' + ('*' * len(pin)) if pin else 'OFF (plaintext)'}")

    df_dist = _load_or_empty(args.paper_ledger)
    df_pre = _load_or_empty(args.paper_ledger_preopen)
    log.info(f"loaded: distribution {len(df_dist)} rows, preopen {len(df_pre)} rows")

    # 1) summary.json
    summary = {
        "asof": asof.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "channels": {
            "distribution": compute_distribution_summary(df_dist),
            "preopen": compute_preopen_summary(df_pre),
        },
    }
    _write_json(out_dir / "summary.json", summary, passphrase=pin)
    log.info(f"saved summary.json")

    # 2) history.json
    history = {
        "asof": asof.isoformat(),
        "rows": history_rows(df_dist, df_pre),
    }
    _write_json(out_dir / "history.json", history, passphrase=pin)
    log.info(f"saved history.json ({len(history['rows'])} rows)")

    # 3) accuracy.json (rolling + cum pnl 둘 다)
    accuracy = {
        "asof": asof.isoformat(),
        "window_days": ROLLING_WINDOW_DAYS,
        "distribution": {
            "rolling": rolling_accuracy(
                df_dist, ["hit_h2", "hit_h6", "hit_h5"]
            ),
            "cum_pnl": cumulative_pnl_series(
                df_dist, "next_max_return_pct", "next_close_return_pct"
            ),
        },
        "preopen": {
            "rolling": rolling_accuracy(
                df_pre,
                ["hit_first15_3pct", "hit_first15_5pct",
                 "hit_first1h_3pct", "hit_first1h_5pct"],
            ),
            "cum_pnl": cumulative_pnl_series(
                df_pre, "first_1h_max_return_pct", "first_1h_close_return_pct"
            ),
        },
    }
    _write_json(out_dir / "accuracy.json", accuracy, passphrase=pin)
    log.info(f"saved accuracy.json")

    # quick stdout summary
    print()
    print("=== Dashboard summary ===")
    for ch, s in summary["channels"].items():
        v = s.get("virtual", {})
        print(
            f"  {ch:<13} alerts={s.get('n_alerts_total',0):>4} "
            f"closed={s.get('n_closed',0):>4} "
            f"cum_pnl={v.get('cum_pnl_pct', float('nan')):+.2f}% "
            f"avg_max={s.get('avg_max_return_pct', float('nan')):+.2f}% "
            f"avg_min={s.get('avg_min_return_pct', float('nan')):+.2f}%"
        )
    print(f"\nout_dir: {out_dir}")


def _load_or_empty(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "status" in df.columns:
        df["status"] = df["status"].fillna("").astype(str)
    return df


def _write_json(path: Path, payload: dict, passphrase: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if passphrase:
        payload = _encrypt_payload(payload, passphrase)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _encrypt_payload(payload: dict, passphrase: str) -> dict:
    """PBKDF2-HMAC-SHA256 + AES-256-CBC + HMAC-SHA256(salt||iv||ct).

    Viewer 의 decryptPayload (papers index.html 동일) 와 호환. PIN brute-force
    완전 차단은 불가능 (4-digit + client-side) — 외부 일반인 차단 수준.
    """
    from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    plaintext = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    salt = os.urandom(16)
    iv = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=64,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    keymat = kdf.derive(passphrase.encode("utf-8"))
    aes_key, mac_key = keymat[:32], keymat[32:]

    pad_len = 16 - (len(plaintext) % 16)
    padded = plaintext + bytes([pad_len]) * pad_len
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    ct = cipher.encryptor().update(padded) + cipher.encryptor().finalize()

    h = crypto_hmac.HMAC(mac_key, hashes.SHA256())
    h.update(salt + iv + ct)
    mac = h.finalize()

    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    return {
        "encrypted": True,
        "version": 1,
        "kdf": "PBKDF2-HMAC-SHA256",
        "cipher": "AES-256-CBC-HMAC-SHA256",
        "iterations": PBKDF2_ITERATIONS,
        "salt": b64(salt),
        "iv": b64(iv),
        "ct": b64(ct),
        "mac": b64(mac),
    }


if __name__ == "__main__":
    main()
