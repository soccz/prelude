"""Adversarial recalculation for binance_leadlag_v1 audit (quant-evaluator).

Independent re-read of output/binance_leadlag_v1_oof_picks.csv (the OOF pick rows
the researcher emitted) + raw DBs. Does NOT trust the run.log numbers — recomputes.

Covers suspicion points:
  1 leak / boundary lag (independent BTC corr lag0 vs lag±1)
  2 per-fold net consistency (the pool +1.24% — is it one fold?)
  3 regime split net (bear_quiet / bear_volatile / bull) — live is bear_quiet
  4 time concentration (monthly, coin-day dedup)
  5 selection deflate (DSR with trials)
  6 net sim honesty — the killer. tpsl model optimism quantified + conservative redo
  7 tradeability / freshness
  8 bn_only vs baseline cross-check

Writes nothing to existing files. Prints a structured report.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OOF = ROOT / "output" / "binance_leadlag_v1_oof_picks.csv"
LIFT = ROOT / "output" / "binance_leadlag_v1_lift.csv"
UPBIT_DB = ROOT / "data" / "upbit_d1.db"
BINANCE_DB = ROOT / "data" / "binance_d1.db"
SOURCE = ROOT / "scripts" / "binance_leadlag_v1.py"

COST = 0.0015
COST_CONS = 0.005
TP, SL = 0.05, -0.03
SEED = 42
DRAWS = 2_000
ARTIFACT_FEATURE_CUTOFF = pd.Timestamp("2026-05-03 09:00:00")
now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
current_completed_feature_cutoff = pd.Timestamp(
    (now_kst - timedelta(hours=9)).date() - timedelta(days=1)
) + pd.Timedelta(hours=9)
if ARTIFACT_FEATURE_CUTOFF > current_completed_feature_cutoff:
    raise ValueError("sealed artifact feature cutoff is in the future")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path: Path) -> dict:
    stat = path.stat()
    wal = Path(f"{path}-wal")
    wal_stat = wal.stat() if wal.exists() else None
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "wal_bytes": wal_stat.st_size if wal_stat else 0,
        "wal_mtime_ns": wal_stat.st_mtime_ns if wal_stat else None,
    }


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def require_finite(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    for column in sorted(columns):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: {column} contains missing/non-finite values")


required_paths = (OOF, LIFT, UPBIT_DB, BINANCE_DB, SOURCE)
missing_paths = [str(path) for path in required_paths if not path.is_file()]
if missing_paths:
    raise FileNotFoundError(f"missing audit inputs: {missing_paths}")
database_snapshots_before = {
    UPBIT_DB.name: snapshot(UPBIT_DB),
    BINANCE_DB.name: snapshot(BINANCE_DB),
}


# ---------------------------------------------------------------------------
# Load OOF picks (these are the BASELINE fire rows; binance subsets via columns)
# ---------------------------------------------------------------------------
oof = pd.read_csv(OOF, parse_dates=["timestamp"])
required_columns = {
    "market", "feature_date", "timestamp", "roc_7d_rank", "u_roc_7d",
    "u_atr_pct_14", "b_ret_1d", "b_ret_7d", "b_vol_surge",
    "bn_minus_upbit_ret_1d", "pump_max_return", "pump20_bin",
    "entry_open_D", "high_D", "close_D",
}
missing_columns = sorted(required_columns.difference(oof.columns))
if missing_columns or oof.empty:
    raise ValueError(f"OOF artifact missing/empty: {missing_columns}")
oof["feature_date"] = pd.to_datetime(oof["feature_date"], errors="raise")
if oof["timestamp"].isna().any() or oof["feature_date"].isna().any():
    raise ValueError("OOF artifact contains invalid/null dates")
if oof.duplicated(["market", "timestamp"]).any():
    raise ValueError("OOF artifact contains duplicate market/timestamp rows")
if not (
    oof["feature_date"].dt.date == oof["timestamp"].dt.date
).all():
    raise ValueError("OOF feature_date and timestamp disagree")
if oof["timestamp"].max() > ARTIFACT_FEATURE_CUTOFF:
    raise ValueError("OOF artifact extends past its sealed Binance cutoff")
require_finite(
    oof,
    {
        "roc_7d_rank", "u_roc_7d",
        "pump_max_return", "pump20_bin", "entry_open_D", "high_D", "close_D",
    },
    "OOF artifact",
)
unused_atr_missing = int(
    pd.to_numeric(oof["u_atr_pct_14"], errors="coerce").isna().sum()
)
if not oof["pump20_bin"].isin({0, 1}).all():
    raise ValueError("OOF pump20_bin is not binary")
expected_pump20 = oof["pump_max_return"].ge(0.20).astype(int)
if not oof["pump20_bin"].eq(expected_pump20).all():
    raise ValueError("OOF pump20 label mismatch")
oof["label_date"] = oof["timestamp"] + pd.Timedelta(days=1)  # day D
oof["ym"] = oof["timestamp"].dt.to_period("M").astype(str)

# Recompute fold assignment is NOT in the picks file? check
hr("0. OOF picks sanity")
print(f"rows={len(oof)}  cols={list(oof.columns)}")
print(f"timestamp(feature_date) range {oof['timestamp'].min().date()} ~ {oof['timestamp'].max().date()}")
print(f"unique (market,feature_date) = {oof.groupby(['market','feature_date']).ngroups}")
print(f"pump20_bin base rate among baseline fires = {oof['pump20_bin'].mean():.4f}")

# entry/exit returns from raw bar fields
op = oof["entry_open_D"].astype(float)
hi = oof["high_D"].astype(float)
cl = oof["close_D"].astype(float)
oof["ret_eod"] = cl / op - 1.0
oof["ret_high"] = hi / op - 1.0
oof["tp_hit"] = oof["ret_high"] >= TP


# ---------------------------------------------------------------------------
# Need low_D too for honest path bound — pull from raw DB (not in picks file!)
# This is the crux: net_sim never used low. Pull it.
# ---------------------------------------------------------------------------
hr("1b. Pull low_D from raw DB (picks file omitted it — needed for honest SL)")
ucon = readonly(UPBIT_DB)
raw = pd.read_sql_query(
    "SELECT market, timestamp, open, high, low, close FROM candles", ucon,
    parse_dates=["timestamp"])
ucon.close()
raw["label_date"] = pd.to_datetime(raw["timestamp"])
if raw.empty or raw["label_date"].isna().any():
    raise ValueError("D1 database is empty or has invalid timestamps")
if raw.duplicated(["market", "label_date"]).any():
    raise ValueError("D1 database contains duplicate market/timestamp rows")
require_finite(raw, {"open", "high", "low", "close"}, "D1 database")
if (raw[["open", "high", "low", "close"]] <= 0).any().any():
    raise ValueError("D1 database contains non-positive OHLC")

# Independently reconstruct the persisted Binance features from raw candles.
bcon = readonly(BINANCE_DB)
binance_raw = pd.read_sql_query(
    """
    SELECT market, timestamp, close, quote_volume
    FROM candles
    ORDER BY market, timestamp
    """,
    bcon,
    parse_dates=["timestamp"],
)
bcon.close()
if binance_raw.empty or binance_raw["timestamp"].isna().any():
    raise ValueError("Binance D1 database is empty or has invalid timestamps")
if binance_raw.duplicated(["market", "timestamp"]).any():
    raise ValueError("Binance D1 database contains duplicate market/timestamp rows")
require_finite(binance_raw, {"close", "quote_volume"}, "Binance D1 database")
if (binance_raw["close"] <= 0).any() or (binance_raw["quote_volume"] < 0).any():
    raise ValueError("Binance D1 database contains invalid price/volume")
binance_raw = binance_raw.sort_values(["market", "timestamp"]).copy()
binance_raw["feature_date"] = binance_raw["timestamp"].dt.normalize()
binance_raw["recalc_b_ret_1d"] = np.log(
    binance_raw["close"]
    / binance_raw.groupby("market", sort=False)["close"].shift(1)
)
binance_raw["recalc_b_ret_7d"] = (
    binance_raw.groupby("market", sort=False)["close"].pct_change(7)
)
binance_raw["recalc_b_vol_surge"] = (
    binance_raw["quote_volume"]
    / binance_raw.groupby("market", sort=False)["quote_volume"].transform(
        lambda values: values.rolling(20, min_periods=10).mean()
    )
)
oof["bn_market_recalc"] = (
    "BINANCE-" + oof["market"].str.split("-", n=1).str[1] + "USDT"
)
rebuild = binance_raw[
    [
        "market",
        "feature_date",
        "recalc_b_ret_1d",
        "recalc_b_ret_7d",
        "recalc_b_vol_surge",
    ]
].rename(columns={"market": "bn_market_recalc"})
oof = oof.merge(
    rebuild,
    on=["bn_market_recalc", "feature_date"],
    how="left",
    validate="many_to_one",
)
feature_rebuild_errors = {}
for persisted, recalculated in (
    ("b_ret_1d", "recalc_b_ret_1d"),
    ("b_ret_7d", "recalc_b_ret_7d"),
    ("b_vol_surge", "recalc_b_vol_surge"),
):
    persisted_numeric = pd.to_numeric(oof[persisted], errors="coerce")
    present = oof[persisted].notna()
    if not np.isfinite(persisted_numeric[present].to_numpy(float)).all():
        raise ValueError(f"OOF artifact has non-finite persisted {persisted}")
    if oof.loc[present, recalculated].isna().any():
        raise ValueError(f"Binance feature rebuild is missing {persisted} rows")
    error = float(
        np.max(
            np.abs(
                persisted_numeric[present].to_numpy(float)
                - oof.loc[present, recalculated].to_numpy(float)
            )
        )
    )
    feature_rebuild_errors[persisted] = error
    if error > 1e-12:
        raise ValueError(f"Binance feature mismatch for {persisted}: {error}")

upbit_feature = raw[["market", "label_date", "close"]].rename(
    columns={"label_date": "feature_timestamp"}
).sort_values(["market", "feature_timestamp"])
upbit_feature["feature_date"] = upbit_feature["feature_timestamp"].dt.normalize()
upbit_feature["recalc_u_log_return_1d"] = np.log(
    upbit_feature["close"]
    / upbit_feature.groupby("market", sort=False)["close"].shift(1)
)
oof = oof.merge(
    upbit_feature[["market", "feature_date", "recalc_u_log_return_1d"]],
    on=["market", "feature_date"],
    how="left",
    validate="many_to_one",
)
catchup_present = oof["bn_minus_upbit_ret_1d"].notna()
if not np.isfinite(
    pd.to_numeric(
        oof.loc[catchup_present, "bn_minus_upbit_ret_1d"],
        errors="coerce",
    ).to_numpy(float)
).all():
    raise ValueError("OOF artifact has non-finite persisted catch-up values")
catchup_expected = (
    oof.loc[catchup_present, "b_ret_1d"]
    - oof.loc[catchup_present, "recalc_u_log_return_1d"]
)
catchup_error = float(
    np.max(
        np.abs(
            oof.loc[catchup_present, "bn_minus_upbit_ret_1d"].to_numpy(float)
            - catchup_expected.to_numpy(float)
        )
    )
)
if catchup_error > 1e-12:
    raise ValueError(f"catch-up feature rebuild mismatch: {catchup_error}")
feature_rebuild_frame = oof[
    [
        "market",
        "feature_date",
        "recalc_b_ret_1d",
        "recalc_b_ret_7d",
        "recalc_b_vol_surge",
        "recalc_u_log_return_1d",
    ]
]
feature_rebuild_payload = (
    feature_rebuild_frame.sort_values(["feature_date", "market"])
    .to_json(orient="records", date_format="iso", double_precision=15)
    .encode()
)
feature_rebuild_sha256 = hashlib.sha256(feature_rebuild_payload).hexdigest()
# join low_D by (market, label_date == day D)
oof = oof.merge(
    raw[["market", "label_date", "low", "open", "high", "close"]].rename(
        columns={"low": "low_D_raw", "open": "open_chk", "high": "high_chk", "close": "close_chk"}),
    on=["market", "label_date"], how="left", validate="one_to_one")
miss = oof["low_D_raw"].isna().sum()
print(f"low_D matched: {len(oof)-miss}/{len(oof)} (miss {miss})")
if miss:
    raise ValueError(f"D1 database is missing {miss} OOF label rows")
# integrity: does picks entry_open_D == raw open on day D?
chk = oof.dropna(subset=["open_chk"])
dop = (chk["entry_open_D"] - chk["open_chk"]).abs()
dhi = (chk["high_D"] - chk["high_chk"]).abs()
print(f"entry_open_D vs raw open(dayD): max abs diff = {dop.max():.6g}  (should be ~0 => label join correct)")
print(f"high_D vs raw high(dayD):       max abs diff = {dhi.max():.6g}")
if dop.max() > 1e-12 or dhi.max() > 1e-12:
    raise ValueError("OOF entry/high values disagree with D1 database")
oof["ret_low"] = oof["low_D_raw"].astype(float) / op - 1.0
oof["sl_hit"] = oof["ret_low"] <= SL


# ---------------------------------------------------------------------------
# 6. NET SIM HONESTY — the optimistic bias, quantified, then conservative redo
# ---------------------------------------------------------------------------
hr("6. NET SIM: researcher 'tpsl' optimism vs conservative path models")


def net_models(df):
    """Return dict of per-trade net arrays under different path assumptions."""
    if df.empty:
        raise ValueError("net model received an empty cohort")
    o = df["entry_open_D"].astype(float).values
    h = df["high_D"].astype(float).values
    low = df["low_D_raw"].astype(float).values
    c = df["close_D"].astype(float).values
    if (
        not np.isfinite(np.column_stack([o, h, low, c])).all()
        or (o <= 0).any()
        or (h < np.maximum(o, c)).any()
        or (low > np.minimum(o, c)).any()
        or (h < low).any()
    ):
        raise ValueError("net model received invalid OHLC values")
    eod = c / o - 1.0
    rh = h / o - 1.0
    rl = low / o - 1.0
    tp_hit = rh >= TP
    sl_hit = rl <= SL
    # (A) researcher tpsl: TP if high>=TP else max(eod,SL). NEVER penalizes intraday low.
    g_research = np.where(tp_hit, TP, np.maximum(eod, SL))
    # (B) conservative: if BOTH tp & sl touched in a daily bar => assume SL first (pessimistic).
    #     tp only -> +TP ; sl only -> SL ; neither -> eod
    g_cons = np.where(sl_hit & tp_hit, SL,
             np.where(tp_hit, TP,
             np.where(sl_hit, SL, eod)))
    # (C) plain EOD close (no bracket) — honest single-number
    g_eod = eod
    # (D) optimistic floor only on eod but ALSO apply real SL when low hit (mid)
    #     tp if high>=TP (ignore order) else if low<=SL -> SL else eod
    g_mid = np.where(tp_hit, TP, np.where(sl_hit, SL, eod))
    return {"research_tpsl": g_research, "cons_slfirst": g_cons,
            "eod_close": g_eod, "mid_realSL": g_mid,
            "both_touch_frac": float((tp_hit & sl_hit).mean())}


def summ(g, cost):
    if len(g) == 0 or not np.isfinite(g).all():
        raise ValueError("summary received empty/non-finite returns")
    net = g - cost
    return dict(n=len(g), gross=float(np.nanmean(g)),
                net_mean=float(np.nanmean(net)),
                net_median=float(np.nanmedian(net)),
                winrate=float(np.mean(net > 0)),
                net_std=float(np.nanstd(net)),
                sharpe_pertrade=float(np.nanmean(net) / (np.nanstd(net) + 1e-12)))


for rule_name, mask in [
    ("baseline", pd.Series(True, index=oof.index)),
    ("base_AND_bn_volsurge(>1.5)", oof["b_vol_surge"] > 1.5),
    ("base_AND_bn_up(>0)", oof["b_ret_1d"] > 0),
]:
    sub = oof[mask.fillna(False)]
    m = net_models(sub)
    print(f"\n--- {rule_name}  (n={len(sub)}, both TP&SL touched in same bar = {m['both_touch_frac']*100:.1f}%) ---")
    for model in ["research_tpsl", "cons_slfirst", "mid_realSL", "eod_close"]:
        s15 = summ(m[model], COST)
        s50 = summ(m[model], COST_CONS)
        print(f"  {model:14s}  net@0.15%={s15['net_mean']*100:+.3f}%  win={s15['winrate']*100:.1f}%  "
              f"sharpe/trade={s15['sharpe_pertrade']:+.3f} | net@0.50%={s50['net_mean']*100:+.3f}%")


# ---------------------------------------------------------------------------
# 2. PER-FOLD net consistency. Reconstruct folds from PurgedWalkForward.
# ---------------------------------------------------------------------------
hr("2. PER-FOLD net (is the pooled +1.24% one bull fold?)")
sys.path.insert(0, str(ROOT))
from signals.validate import PurgedWalkForward  # noqa

# Rebuild the exact date axis used by build_upbit_panel.  Starting at the first
# emitted fire row is wrong because fire rows are only a sparse subset and shifts
# every PurgedWalkForward boundary.  The source panel requires seven prior rows
# for u_roc_7d and an exact next-day label row.
axis_rows = raw[["market", "label_date"]].rename(
    columns={"label_date": "timestamp"}
).sort_values(["market", "timestamp"])
axis_rows["_prior_rows"] = axis_rows.groupby("market").cumcount()
axis_rows["_next_timestamp"] = axis_rows.groupby("market")["timestamp"].shift(-1)
eligible_axis_rows = axis_rows[
    axis_rows["_prior_rows"].ge(7)
    & axis_rows["_next_timestamp"].eq(
        axis_rows["timestamp"] + pd.Timedelta(days=1)
    )
    & axis_rows["timestamp"].le(ARTIFACT_FEATURE_CUTOFF)
]
axis = eligible_axis_rows["timestamp"].drop_duplicates().sort_values()
if axis.empty:
    raise ValueError("could not reconstruct the source panel date axis")
spl = PurgedWalkForward(n_folds=5, embargo_days=10, holdout_days=180)
fold_windows = []
for fi, (tr, va) in enumerate(spl.split(axis), 1):
    fold_windows.append((fi, pd.to_datetime(pd.Series(va)).min(), pd.to_datetime(pd.Series(va)).max()))
    print(f"  fold {fi} val window: {pd.to_datetime(pd.Series(va)).min().date()} ~ {pd.to_datetime(pd.Series(va)).max().date()}")


def assign_fold(ts):
    for fi, lo, hi_ in fold_windows:
        if lo <= ts <= hi_:
            return fi
    return -1  # holdout or gap


oof["fold"] = oof["timestamp"].map(assign_fold)
print(f"\npicks assigned to folds: {(oof['fold']>0).sum()}/{len(oof)} "
      f"(fold=-1 holdout/embargo dropped: {(oof['fold']<0).sum()})")
if oof["fold"].lt(1).any():
    raise ValueError("OOF artifact contains rows outside reconstructed folds")
lift = pd.read_csv(LIFT)
required_lift = {"rule", "fold", "n_fire"}
if not required_lift.issubset(lift.columns):
    raise ValueError("lift artifact is missing fold/fire columns")
baseline_fire = (
    lift[lift["rule"].eq("baseline_roc7")]
    .set_index("fold")["n_fire"]
    .astype(int)
    .sort_index()
)
reconstructed_fire = oof.groupby("fold").size().astype(int).sort_index()
if not reconstructed_fire.equals(baseline_fire):
    raise ValueError(
        "reconstructed folds disagree with persisted baseline fire counts: "
        f"{reconstructed_fire.to_dict()} vs {baseline_fire.to_dict()}"
    )

for rule_name, mask in [("baseline", pd.Series(True, index=oof.index)),
                        ("volsurge>1.5", oof["b_vol_surge"] > 1.5)]:
    print(f"\n--- per-fold net ({rule_name}) — conservative (cons_slfirst) & research_tpsl ---")
    sub = oof[mask.fillna(False) & (oof["fold"] > 0)]
    for fi in sorted(sub["fold"].unique()):
        f = sub[sub["fold"] == fi]
        m = net_models(f)
        s_res = summ(m["research_tpsl"], COST)
        s_con = summ(m["cons_slfirst"], COST)
        s_eod = summ(m["eod_close"], COST)
        print(f"  fold{fi} n={len(f):4d}  research_net={s_res['net_mean']*100:+.3f}%  "
              f"cons_net={s_con['net_mean']*100:+.3f}%  eod_net={s_eod['net_mean']*100:+.3f}%  "
              f"win(res)={s_res['winrate']*100:.0f}%")


# ---------------------------------------------------------------------------
# 4. TIME CONCENTRATION
# ---------------------------------------------------------------------------
hr("4. TIME CONCENTRATION (volsurge>1.5 picks, monthly)")
vs = oof[(oof["b_vol_surge"] > 1.5).fillna(False)]
by_m = vs.groupby("ym").agg(n=("pump20_bin", "size"),
                            hits=("pump20_bin", "sum"),
                            net_res=("ret_eod", "size"))
# monthly net research model
mr = []
for ym, g in vs.groupby("ym"):
    m = net_models(g)
    mr.append((ym, len(g), int(g["pump20_bin"].sum()),
               float(np.mean(m["research_tpsl"]) - COST) * 100,
               float(np.mean(m["cons_slfirst"]) - COST) * 100))
mrdf = pd.DataFrame(mr, columns=["ym", "n", "hits", "res_net%", "cons_net%"])
print(mrdf.to_string(index=False))
print(f"\nfire concentration: top-3 months hold "
      f"{mrdf.nlargest(3,'n')['n'].sum()}/{mrdf['n'].sum()} = "
      f"{mrdf.nlargest(3,'n')['n'].sum()/mrdf['n'].sum()*100:.0f}% of fires")
# how many months have POSITIVE conservative net
print(f"months with positive cons_net: {(mrdf['cons_net%']>0).sum()}/{len(mrdf)}")
print(f"months with positive research_net: {(mrdf['res_net%']>0).sum()}/{len(mrdf)}")


# ---------------------------------------------------------------------------
# 3. REGIME SPLIT — classify each pick's feature_date BTC regime (leak-free, D-1)
# ---------------------------------------------------------------------------
hr("3. REGIME SPLIT net (live regime = bear_quiet)")
ucon = readonly(UPBIT_DB)
btc = pd.read_sql_query("SELECT timestamp, close FROM candles WHERE market='KRW-BTC' ORDER BY timestamp",
                        ucon, parse_dates=["timestamp"])
ucon.close()
btc = btc.sort_values("timestamp").reset_index(drop=True)
EPS = 1e-12
lr = np.log(btc["close"] / btc["close"].shift(1) + EPS)
ma = btc["close"].rolling(200).mean()
btc["ma_dist"] = (btc["close"] - ma) / (ma + EPS)
btc["rv30"] = lr.rolling(30).std()
btc["intensity"] = btc["rv30"].rolling(252).rank(pct=True)


def reg(b, v):
    if pd.isna(b) or pd.isna(v):
        return "unknown"
    if b > 0 and v <= 0.5:
        return "bull_quiet"
    if b > 0 and v > 0.5:
        return "bull_volatile"
    if b <= 0 and v <= 0.5:
        return "bear_quiet"
    return "bear_volatile"


btc["regime"] = [reg(b, v) for b, v in zip(btc["ma_dist"], btc["intensity"])]
btc["feature_date"] = btc["timestamp"].dt.normalize()
# regime at feature_date (D-1) — strictly the info available before day D
rmap = dict(zip(btc["feature_date"], btc["regime"]))
oof["btc_regime"] = oof["feature_date"].dt.normalize().map(rmap)

for rule_name, mask in [("volsurge>1.5", oof["b_vol_surge"] > 1.5),
                        ("baseline", pd.Series(True, index=oof.index))]:
    print(f"\n--- regime split net ({rule_name}) ---")
    sub = oof[mask.fillna(False)]
    for rg in [
        "bull_quiet",
        "bull_volatile",
        "bear_quiet",
        "bear_volatile",
        "unknown",
    ]:
        g = sub[sub["btc_regime"] == rg]
        if len(g) < 20:
            print(f"  {rg:14s} n={len(g):4d}  (too few)")
            continue
        m = net_models(g)
        s_res = summ(m["research_tpsl"], COST)
        s_con = summ(m["cons_slfirst"], COST)
        s_eod = summ(m["eod_close"], COST)
        hit = g["pump20_bin"].mean()
        print(f"  {rg:14s} n={len(g):4d}  hit={hit*100:.1f}%  "
              f"research_net={s_res['net_mean']*100:+.3f}%  cons_net={s_con['net_mean']*100:+.3f}%  "
              f"eod_net={s_eod['net_mean']*100:+.3f}%")


# ---------------------------------------------------------------------------
# 5. SELECTION DEFLATE — DSR with trials. Bootstrap CI on net.
# ---------------------------------------------------------------------------
hr("5. SELECTION DEFLATE + bootstrap CI (volsurge>1.5, research vs cons model)")
sub = oof[(oof["b_vol_surge"] > 1.5).fillna(False)]
m = net_models(sub)
for model in ["research_tpsl", "cons_slfirst", "eod_close"]:
    trade_net = m[model] - COST
    if not np.isfinite(trade_net).all():
        raise ValueError(f"{model}: non-finite net returns")
    daily = (
        pd.DataFrame(
            {"feature_date": sub["feature_date"].to_numpy(), "net": trade_net}
        )
        .groupby("feature_date", sort=True)["net"]
        .mean()
    )
    g = daily.to_numpy(float)
    mu, sd = g.mean(), g.std(ddof=1)
    n = len(g)
    sharpe = mu / sd if sd > 0 else np.nan
    # Date-cluster bootstrap: same-day fires aren't independent observations.
    rng = np.random.default_rng(SEED + list(m).index(model))
    indices = rng.integers(0, n, size=(DRAWS, n))
    boot = g[indices].mean(axis=1)
    lo, hi_ = np.percentile(boot, [2.5, 97.5])
    print(f"  {model:14s}  daily_net_mean={mu*100:+.4f}%  date-CI95=[{lo*100:+.4f}%, {hi_*100:+.4f}%]  "
          f"daily_sharpe={sharpe:+.4f}  {'>>POSITIVE' if lo>0 else 'CI includes 0' if hi_>0 else 'NEGATIVE'}")
# This script doesn't claim a DSR because the full trial-return matrix wasn't
# persisted.  The post-hoc trial count remains a mandatory evidence limitation.
print("\n  (selection note: at least 5 rules were compared post-hoc; no DSR is")
print("   claimed without the complete common-cohort trial return matrix.)")


# ---------------------------------------------------------------------------
# 7. TRADEABILITY / FRESHNESS + annualized framing
# ---------------------------------------------------------------------------
hr("7. TRADEABILITY: fires/day, freshness, annualized Sharpe framing")
vs = oof[(oof["b_vol_surge"] > 1.5).fillna(False)].copy()
fires_per_day = vs.groupby(vs["feature_date"].dt.date).size()
print(f"volsurge fires: total={len(vs)} over {fires_per_day.shape[0]} active days  "
      f"=> mean {fires_per_day.mean():.2f}/day, p95 {fires_per_day.quantile(.95):.0f}/day")
print(
    "Sealed artifact Binance feature cutoff = "
    f"{ARTIFACT_FEATURE_CUTOFF.date()} (later DB appends are outside this audit)"
)
# Annualize the daily portfolio series, not correlated individual trades.
for model in ["research_tpsl", "cons_slfirst", "eod_close"]:
    trade_net = net_models(vs)[model] - COST
    daily = (
        pd.DataFrame(
            {"feature_date": vs["feature_date"].to_numpy(), "net": trade_net}
        )
        .groupby("feature_date", sort=True)["net"]
        .mean()
    )
    daily_std = float(daily.std())
    annualized = (
        float(daily.mean() / daily_std * np.sqrt(365))
        if daily_std > 0
        else np.nan
    )
    daily_sharpe = float(daily.mean() / daily_std) if daily_std > 0 else np.nan
    print(
        f"  {model:14s} daily Sharpe={daily_sharpe:+.4f} "
        f"=> sqrt(365) annualized={annualized:+.2f}"
    )


# ---------------------------------------------------------------------------
# 8. bn_only vs baseline cross-check (cold-start consistency)
# ---------------------------------------------------------------------------
hr("8. catchup reverse-causality + bn_only roc_7d_rank distribution")
print("roc_7d_rank distribution (baseline fires already > 0.85 by construction):")
print(f"  baseline: mean roc_7d_rank = {oof['roc_7d_rank'].mean():.3f}")
print(f"  volsurge>1.5 subset: mean roc_7d_rank = {oof[oof['b_vol_surge']>1.5]['roc_7d_rank'].mean():.3f}")
print(f"  catchup subset: mean roc_7d_rank = {oof[oof['bn_minus_upbit_ret_1d']>0]['roc_7d_rank'].mean():.3f}")
# Note: bn_only is NOT in oof picks (oof = baseline fires only). So bn_only's 4.92x
# lift cannot be net-verified from this file — flag it.
print("\nNOTE: bn_only_surge+mom picks are NOT in oof_picks.csv (file = baseline fires only).")
print("      => bn_only 4.92x lift has NO net verification possible from emitted artifacts.")

database_snapshots_after = {
    UPBIT_DB.name: snapshot(UPBIT_DB),
    BINANCE_DB.name: snapshot(BINANCE_DB),
}
if database_snapshots_after != database_snapshots_before:
    raise RuntimeError("a source database changed during recalculation")
cache_columns = [
    "market",
    "feature_date",
    "label_date",
    "fold",
    "entry_open_D",
    "high_D",
    "low_D_raw",
    "close_D",
]
cache_payload = (
    oof[cache_columns]
    .sort_values(["feature_date", "market"])
    .to_json(orient="records", date_format="iso", double_precision=15)
    .encode()
)
provenance = {
    "input_sha256": {
        "oof": sha(OOF),
        "lift": sha(LIFT),
        "source": sha(SOURCE),
    },
    "source_database_snapshots": database_snapshots_before,
    "artifact_feature_cutoff": str(ARTIFACT_FEATURE_CUTOFF),
    "reconstructed_axis": {
        "dates": int(axis.nunique()),
        "start": str(axis.min()),
        "end": str(axis.max()),
        "fold_fire_counts_match_lift": True,
    },
    "joined_path_cache_sha256": hashlib.sha256(cache_payload).hexdigest(),
    "feature_rebuild": {
        "max_abs_errors": {
            **feature_rebuild_errors,
            "bn_minus_upbit_ret_1d": catchup_error,
        },
        "recalculated_values_sha256": feature_rebuild_sha256,
    },
    "costs": {
        "round_trip_default": COST,
        "round_trip_conservative": COST_CONS,
    },
    "unused_input_diagnostics": {
        "u_atr_pct_14_missing_rows": unused_atr_missing,
        "note": "column is not consumed by any recalculated rule or return",
    },
    "bootstrap": {
        "unit": "trading date",
        "seed_base": SEED,
        "draws": DRAWS,
    },
    "population_scope": {
        "audited_rows": (
            "every persisted baseline-fire OOF row across all five folds"
        ),
        "raw_rebuild": "all D1 rows needed to reproduce Upbit/Binance features",
        "evidence_boundary": (
            "the emitted OOF artifact omits non-baseline and bn_only candidate "
            "rows, so those populations cannot be net-recalculated here"
        ),
    },
    "output_contract": (
        "read-only and stdout-only; partial stdout is invalid unless the "
        "process exits zero and emits the final [DONE] sentinel"
    ),
}
print("\n=== PROVENANCE / CONTRACTS ===")
print(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False))
print("\n[DONE]")
