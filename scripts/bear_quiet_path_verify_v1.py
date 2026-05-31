"""Independent re-verification of bear_quiet 15m path-exit (quant-evaluator).

Recomputes from raw paths with HONEST dedup (coin-day, no fold-overlap triple count),
day-level (cluster-robust) Sharpe, block-bootstrap CI95, and DSR with true trials.
Does NOT trust researcher CSV — rebuilds entries + simulates paths.
"""
from __future__ import annotations
import sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.database import list_markets, load_candles
from signals.features import compute_btc_features
from scripts.univariate_precursor_lift_v1 import build_market_features, add_cross_sectional

D1_DB="data/upbit_d1.db"; M15_DB="data/upbit_15m.db"
COST=0.0015; N_FOLDS=5; EMBARGO=5; Q=0.90
CANDS=["f_qv_surge_7d","f_qv_surge_30d","f_bounce_off_7d_low","f_atr_xs_decile"]
BPH=4

def build_panel():
    frames=[]
    for m in list_markets(D1_DB):
        df=load_candles(D1_DB,m)
        if df is None or len(df)<70: continue
        df=df.copy(); df["market"]=m
        frames.append(build_market_features(df))
    p=pd.concat(frames,ignore_index=True)
    p["timestamp"]=pd.to_datetime(p["timestamp"]); p["date"]=p["timestamp"].dt.date
    return p.sort_values(["date","market"]).reset_index(drop=True)

def attach_bq(panel):
    btc=load_candles(D1_DB,"KRW-BTC")
    bf=compute_btc_features(btc)[["timestamp","btc_regime"]].copy()
    bf["timestamp"]=pd.to_datetime(bf["timestamp"]); bf=bf.sort_values("timestamp")
    bf["regime_dm1"]=bf["btc_regime"].shift(1)
    p=panel.merge(bf[["timestamp","regime_dm1"]],on="timestamp",how="left")
    p["is_bq"]=(p["regime_dm1"]=="bear_quiet").astype(float)
    return p

def collect_entries(panel,feat):
    """Same WF as researcher BUT dedup coin-day at the end (keep first fold occurrence)."""
    d=panel[panel["is_bq"]==1.0][["market","date",feat,"timestamp"]].dropna(subset=[feat])
    all_dates=np.sort(panel["date"].unique())
    fs=len(all_dates)//(N_FOLDS+1); sel=[]
    for k in range(1,N_FOLDS+1):
        tr_end=fs*k; te_start=tr_end+EMBARGO; te_end=min(fs*(k+1)+tr_end,len(all_dates))
        if te_start>=len(all_dates): break
        trd=set(all_dates[:tr_end]); ted=set(all_dates[te_start:te_end])
        tr=d[d["date"].isin(trd)]; te=d[d["date"].isin(ted)]
        if len(tr)<50 or len(te)<10: continue
        cut=tr[feat].quantile(Q); s=te[te[feat]>=cut].copy(); s["fold"]=k; sel.append(s)
    if not sel: return pd.DataFrame(columns=["market","date",feat,"fold"])
    e=pd.concat(sel,ignore_index=True)
    e_dedup=e.drop_duplicates(subset=["market","date"]).reset_index(drop=True)
    return e, e_dedup

def load_paths(entries):
    conn=sqlite3.connect(M15_DB); paths={}
    for _,r in entries[["market","date"]].drop_duplicates().iterrows():
        m,dt=r["market"],pd.Timestamp(r["date"])
        s=dt.strftime("%Y-%m-%d 09:00:00"); e=(dt+pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
        rows=conn.execute("SELECT open,high,low,close FROM candles WHERE market=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",(m,s,e)).fetchall()
        if rows: paths[(m,dt.date())]=rows
    conn.close(); return paths

def sim(bars,hard_sl,tp,ts_h,trail):
    if not bars: return np.nan,"nodata",np.nan
    entry=bars[0][0]
    if not np.isfinite(entry) or entry<=0: return np.nan,"nodata",np.nan
    sl_px=entry*(1-hard_sl) if hard_sl is not None else None
    tp_px=entry*(1+tp) if tp is not None else None
    ts_bars=int(round(ts_h*BPH)) if ts_h is not None else None
    rh=entry
    for i,(o,h,l,c) in enumerate(bars):
        if trail is not None and h>rh: rh=h
        trail_px=rh*(1-trail) if trail is not None else None
        if sl_px is not None and l<=sl_px: return -hard_sl,"sl",(i+1)/BPH
        if trail_px is not None and l<=trail_px: return trail_px/entry-1,"trail",(i+1)/BPH
        if tp_px is not None and h>=tp_px: return tp,"tp",(i+1)/BPH
        if ts_bars is not None and (i+1)>=ts_bars: return c/entry-1,"timestop",(i+1)/BPH
    return bars[-1][3]/entry-1,"eod",len(bars)/BPH

def day_metrics(td):
    """Day-equal-weight (cluster-robust) net metrics. td: date,net,outcome."""
    if len(td)==0: return {}
    daily=td.groupby("date")["net"].mean().sort_index()
    mu,sig=daily.mean(),daily.std(ddof=1)
    sharpe=float(mu/sig*np.sqrt(365)) if sig and sig>0 else np.nan
    eq=(1+daily).cumprod(); peak=eq.cummax(); mdd=float(((eq-peak)/peak).min())
    downside=daily[daily<0]
    dsig=downside.std(ddof=1) if len(downside)>1 else np.nan
    sortino=float(mu/dsig*np.sqrt(365)) if dsig and dsig>0 else np.nan
    calmar=float((daily.mean()*365)/abs(mdd)) if mdd<0 else np.nan
    return dict(day_sharpe=sharpe, day_sortino=sortino, calmar=calmar,
                day_mu=float(mu), cum=float(eq.iloc[-1]-1), mdd=mdd,
                n_days=int(len(daily)), n_trades=int(len(td)),
                trade_mean=float(td["net"].mean()),
                hit_trade=float((td["net"]>0).mean()),
                hit_day=float((daily>0).mean()),
                tp_rate=float((td["outcome"]=="tp").mean()),
                sl_rate=float((td["outcome"]=="sl").mean()),
                eod_rate=float((td["outcome"]=="eod").mean()),
                daily=daily)

def block_bootstrap_sharpe(daily, n_boot=5000, block=5, seed=42):
    """Stationary-block bootstrap on daily returns → Sharpe CI95."""
    rng=np.random.default_rng(seed); x=daily.values; n=len(x)
    if n<5: return (np.nan,np.nan,np.nan)
    sh=[]
    nblocks=int(np.ceil(n/block))
    for _ in range(n_boot):
        idx=[]
        for _b in range(nblocks):
            st=rng.integers(0,n)
            idx.extend([(st+j)%n for j in range(block)])
        s=x[idx[:n]]
        sd=s.std(ddof=1)
        sh.append(s.mean()/sd*np.sqrt(365) if sd>0 else 0.0)
    sh=np.array(sh)
    return float(np.percentile(sh,2.5)), float(np.median(sh)), float(np.percentile(sh,97.5))

def block_bootstrap_cum(daily, n_boot=5000, block=5, seed=42):
    rng=np.random.default_rng(seed); x=daily.values; n=len(x)
    if n<5: return (np.nan,np.nan,np.nan)
    cums=[]; nblocks=int(np.ceil(n/block))
    for _ in range(n_boot):
        idx=[]
        for _b in range(nblocks):
            st=rng.integers(0,n); idx.extend([(st+j)%n for j in range(block)])
        s=x[idx[:n]]; cums.append(np.prod(1+s)-1)
    cums=np.array(cums)
    return float(np.percentile(cums,2.5)), float(np.median(cums)), float(np.percentile(cums,97.5))

def psr_dsr(sharpe_ann, daily, n_trials):
    """PSR vs 0 and DSR. sharpe_ann is annualized; convert to per-period (daily)."""
    n=len(daily)
    if n<5 or not np.isfinite(sharpe_ann): return (np.nan,np.nan,np.nan)
    sr=sharpe_ann/np.sqrt(365)  # per-day SR
    r=daily.values
    g3=stats.skew(r); g4=stats.kurtosis(r,fisher=False)
    # PSR vs benchmark 0
    denom=np.sqrt(1 - g3*sr + (g4-1)/4*sr**2)
    psr=stats.norm.cdf((sr*np.sqrt(n-1))/denom) if denom>0 else np.nan
    # DSR: expected max SR from N trials (variance of trial SRs ~ approximated)
    e=np.euler_gamma
    z1=stats.norm.ppf(1-1.0/n_trials); z2=stats.norm.ppf(1-1.0/(n_trials*np.e))
    sr0=np.sqrt(1.0/(n))*((1-e)*z1+e*z2)  # using 1/n as var-of-trials proxy (V[SR]~1/n)
    dsr=stats.norm.cdf((sr-sr0)*np.sqrt(n-1)/denom) if denom>0 else np.nan
    # MinTRL: track record length needed for PSR=0.95 at this SR
    if sr!=0:
        mintrl=1+(1-g3*sr+(g4-1)/4*sr**2)*(stats.norm.ppf(0.95)/sr)**2
    else: mintrl=np.nan
    return float(psr),float(dsr),float(mintrl)

def main():
    panel=add_cross_sectional(build_panel()); panel=attach_bq(panel)
    conn=sqlite3.connect(M15_DB)
    mn,mx=conn.execute("SELECT MIN(timestamp),MAX(timestamp) FROM candles").fetchone(); conn.close()
    m15_start=pd.Timestamp(mn).date()
    bq_all=panel[panel["is_bq"]==1.0]["date"].nunique()
    bq_in15=panel[(panel["is_bq"]==1.0)&(panel["date"]>=m15_start)]["date"].nunique()
    print(f"15m window {mn}..{mx}  bear_quiet days: full={bq_all} in15m={bq_in15}")

    # Key policies to evaluate (researcher 'best' + hardSL replicate + baselines)
    POLICIES={
        "EOD_noSLTP":      (None,None,None,None),
        "TP10_noSL_EOD":   (None,0.10,None,None),
        "TP5_noSL_EOD":    (None,0.05,None,None),
        "TP8_noSL_EOD":    (None,0.08,None,None),
        "hardSL5_TP10":    (0.05,0.10,None,None),
        "hardSL5_only":    (0.05,None,None,None),
        "timestop4h_TP10": (None,0.10,4.0,None),
        "trail10_TP10":    (None,0.10,None,0.10),
    }
    full_grid_trials = 4*5*4*4  # HARD_SL x TP x TS x TRAIL = 320 per pattern

    out=[]
    for feat in CANDS:
        e_full,e=collect_entries(panel,feat)
        e=e[e["date"]>=m15_start].reset_index(drop=True)
        bars_map=load_paths(e)
        e=e[e.apply(lambda r:(r["market"],r["date"]) in bars_map,axis=1)].reset_index(drop=True)
        n_full=len(e_full[e_full["date"]>=m15_start])
        print(f"\n### {feat}: fold-overlap n={n_full}  DEDUP coin-day n={len(e)}  days={e['date'].nunique()}  infl={n_full/max(len(e),1):.2f}x")
        # what-hit-first (TP10/SL5)
        both=tpf=slf=0; sl_touch=tp_touch=0
        for _,r in e.iterrows():
            bars=bars_map[(r["market"],r["date"])]; entry=bars[0][0]
            tpx,spx=entry*1.10,entry*0.95
            ht=any(h>=tpx for o,h,l,c in bars); hs=any(l<=spx for o,h,l,c in bars)
            if ht: tp_touch+=1
            if hs: sl_touch+=1
            if ht and hs:
                both+=1
                for o,h,l,c in bars:
                    if l<=spx and h>=tpx: slf+=1; break
                    if l<=spx: slf+=1; break
                    if h>=tpx: tpf+=1; break
        print(f"  what-hit-first(TP10/SL5 dedup): both={both} tp_first={tpf} sl_first={slf} sl_share={slf/both:.2f}" if both else "  no ambiguous")
        print(f"  anytime touch: SL5={sl_touch} ({100*sl_touch/len(e):.1f}%)  TP10={tp_touch} ({100*tp_touch/len(e):.1f}%)")
        for pname,(sl,tp,ts,tr) in POLICIES.items():
            recs=[]
            for _,r in e.iterrows():
                g,oc,hold=sim(bars_map[(r["market"],r["date"])],sl,tp,ts,tr)
                if np.isfinite(g): recs.append((r["date"],g-COST,oc,hold))
            td=pd.DataFrame(recs,columns=["date","net","outcome","hold"])
            m=day_metrics(td)
            daily=m.pop("daily")
            row=dict(feature=feat,policy=pname,**m)
            out.append((row,daily))
            # print headline
            print(f"  {pname:18s} dSh={m['day_sharpe']:+.2f} dSort={m.get('day_sortino',float('nan')):+.2f} cum={m['cum']:+.3f} mdd={m['mdd']:+.3f} hitD={m['hit_day']:.2f} trMean={m['trade_mean']:+.4f} tp/sl/eod={m['tp_rate']:.2f}/{m['sl_rate']:.2f}/{m['eod_rate']:.2f} nD={m['n_days']} nT={m['n_trades']}")

    # CI + DSR for the leading candidate combos
    print("\n===== BOOTSTRAP CI95 + DSR (day-level, block=5, n_boot=5000) =====")
    leaders=[("f_qv_surge_7d","TP10_noSL_EOD"),("f_qv_surge_30d","TP10_noSL_EOD"),
             ("f_bounce_off_7d_low","TP5_noSL_EOD"),("f_atr_xs_decile","TP5_noSL_EOD"),
             ("f_qv_surge_7d","hardSL5_TP10"),("f_qv_surge_7d","EOD_noSLTP")]
    dmap={(r["feature"],r["policy"]):d for r,d in out}
    rmap={(r["feature"],r["policy"]):r for r,d in out}
    for key in leaders:
        d=dmap.get(key); r=rmap.get(key)
        if d is None or len(d)<5: print(key,"insufficient"); continue
        slo,smed,shi=block_bootstrap_sharpe(d)
        clo,cmed,chi=block_bootstrap_cum(d)
        psr,dsr,mintrl=psr_dsr(r["day_sharpe"],d,full_grid_trials)
        print(f"{key[0]:20s}/{key[1]:14s} dSharpe={r['day_sharpe']:+.2f} CI95[{slo:+.2f},{shi:+.2f}] | cum={r['cum']:+.3f} CI95[{clo:+.3f},{chi:+.3f}] | PSR={psr:.2f} DSR(trials={full_grid_trials})={dsr:.2f} MinTRL={mintrl:.0f}d (have {len(d)}d)")

if __name__=="__main__": main()
