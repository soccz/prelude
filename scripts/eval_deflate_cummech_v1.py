"""quant-evaluator — (1) A1 selection deflate (DSR/PSR, trials=8) 사후표기,
(2) A2a bear_quiet cum(+0.040) vs net_mean(-0.0008) 불일치 기전 규명.
self-contained.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "output"


# ---------- (1) DSR/PSR on A1 daily net (day-eq), trials = 8 ----------
def psr(sr_hat, n, skew, kurt, sr_star=0.0):
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado).
    sr_hat, sr_star: per-observation Sharpe (not annualized)."""
    num = (sr_hat - sr_star) * np.sqrt(n - 1)
    den = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2)
    return float(stats.norm.cdf(num / den))


def dsr(sr_hat, n, skew, kurt, n_trials, var_sr_trials):
    """Deflated Sharpe Ratio: sr_star = expected max Sharpe over N trials."""
    from scipy.stats import norm
    e_max = np.sqrt(var_sr_trials) * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
        + np.euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return psr(sr_hat, n, skew, kurt, sr_star=e_max), float(e_max)


def daily_net(path, col, val):
    d = pd.read_csv(OUT / path)
    d = d[d[col] == val].dropna(subset=["net"]).copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d.groupby("date")["net"].mean().sort_index()


print("=== (1) A1 selection deflate (day-eq daily net, trials=8) ===")
# A1 8조합 daily-net Sharpe 분포로 var_sr 추정 (deflate 용)
comp = pd.read_csv(OUT / "ch_sustainability_compare_v1.csv")
a1rows = comp[comp.policy == "A1_sustain"]
# 8조합의 (per-observation) Sharpe = sharpe_annual / sqrt(365)
sr_trials = (a1rows["sharpe"].values / np.sqrt(365))
var_sr = float(np.var(sr_trials, ddof=1))
print(f"  8조합 annual Sharpe: {np.round(a1rows['sharpe'].values,3)}")
print(f"  per-obs Sharpe var across trials = {var_sr:.3e}")

best = daily_net("ch_sustainability_picks_v1.csv", "policy", "A1_sustain")
n = len(best)
mu, sd = best.mean(), best.std(ddof=1)
sr = mu / sd
sk = float(stats.skew(best)); ku = float(stats.kurtosis(best, fisher=False))
print(f"  best A1(dump_B q0.6) daily net: n={n} mean={mu:+.5f} sd={sd:.5f} "
      f"per-obs SR={sr:+.4f} (annual {sr*np.sqrt(365):+.3f}) skew={sk:+.2f} kurt={ku:.2f}")
print(f"  PSR(SR>0) = {psr(sr,n,sk,ku):.4f}")
d, emax = dsr(sr, n, sk, ku, 8, var_sr)
print(f"  DSR (trials=8, E[max SR]={emax:+.4f}/obs) = {d:.4f}")
print(f"  → net Sharpe 음수({sr*np.sqrt(365):+.2f} annual) 이므로 PSR/DSR 모두 낮음(예상). "
      f"net 은 deflate 전에도 음수 → 학술지표 사후표기일 뿐.")


# ---------- (2) bear_quiet cum vs net_mean 기전 ----------
print("\n=== (2) A2a bear_quiet: cum(day-eq) vs net_mean(trade-eq) 불일치 기전 ===")
df = pd.read_csv(OUT / "ch_regime_split_picks_v1.csv")
df["date"] = pd.to_datetime(df["date"]).dt.date
bq = df[(df.policy == "A2a") & (df.regime == "bear_quiet")].dropna(subset=["net"]).copy()
print(f"  n_trades={len(bq)} n_days={bq.date.nunique()} picks/day={len(bq)/bq.date.nunique():.2f}")
print(f"  trade-eq net_mean = {bq.net.mean():+.6f} (음수)")
daily = bq.groupby("date")["net"].mean().sort_index()
print(f"  day-eq daily-mean: mean={daily.mean():+.6f} (이것도 음수면 cum>0 은 순전히 복리/순서효과)")
cum = (1 + daily).cumprod().iloc[-1] - 1
print(f"  cum(day-eq, 복리) = {cum:+.5f}")
print(f"  sum(daily) = {daily.sum():+.5f}  (단리 합)")
# 기여 분해: 큰 양수 날 몇 개가 끌어올리나
top = daily.sort_values(ascending=False).head(5)
print(f"  상위 5일 일평균 net: {np.round(top.values,4)}  (이 며칠이 cum 견인 여부)")
print(f"  daily>0 비율 = {(daily>0).mean():.3f}, daily 중앙값 = {daily.median():+.5f}")
# fold5 단일 의존 확인
print("\n  fold 별 bear_quiet net (단일 fold 의존 여부):")
for f, g in bq.groupby("fold"):
    print(f"    fold{f}: n={len(g):3d} net_mean={g.net.mean():+.6f} days={g.date.nunique()}")
