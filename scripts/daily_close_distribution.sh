#!/usr/bin/env bash
# Distribution paper/shadow ledger close-out entry.
#
# Runs after KST 09:00 daily candle boundary so yesterday's target day has
# final 4h bars. This keeps distribution paper ledger and shadow ledger aligned
# before the dashboard/idea-validation publish step.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_close_$TODAY.log"

echo "=== prelude distribution close KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

echo "[1/3] data update — d1 + 4h all 2 days" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"
python -m data.collector_4h --all --days 2 >> "$LOG" 2>&1 || echo "  4h update warn" >> "$LOG"

# ★ set -e 가드: close_paper_ledger 가 실패해도 아래 close_recommend_ledger(R1 SHADOW
#   실현)와 champion_selector(재선정)는 반드시 돌아야 한다 — 다음날 08:50/09:05 챔피언
#   결정에 직결. 가드가 없으면 set -e 가 여기서 스크립트를 죽여 R1 행 미실현 + 챔피언
#   stale 가 되고, close 스크립트엔 실패 알림이 없어 조용히 방치된다.
echo "[2/3] close_paper_ledger" >> "$LOG"
if python scripts/close_paper_ledger.py >> "$LOG" 2>&1; then
    EXIT=0
else
    EXIT=$?
    echo "  paper close warn (exit=$EXIT) — close_recommend/champion_selector 는 계속" >> "$LOG"
fi

# R1 SHADOW recommend ledger 실현 (전일 open 행을 -3%SL/+5%TP 15m 경로로 청산 + pump_hit).
# forward 표본 평가가능하려면 매일 청산 필수 — 별도 timer 없이 기존 close(09:30)에 fold.
echo "[2b/3] close_recommend_ledger (R1 SHADOW 실현)" >> "$LOG"
python scripts/close_recommend_ledger.py >> "$LOG" 2>&1 || echo "  recommend close warn" >> "$LOG"

# R2 challenger ledger 실현 (동일 -3%SL/+5%TP 15m 경로). champion_selector 가 R1 vs R2 를
# forward CLOSED 로 비교하려면 R2 도 매일 청산돼야 함. 실패해도 R1/champion 무관(가드).
python scripts/close_recommend_ledger.py --ledger output/shadow_ledger_recommend_r2.csv >> "$LOG" 2>&1 || echo "  R2 recommend close warn" >> "$LOG"

# A1 sustainability challenger ledger 실현 (동일 -3%SL/+5%TP 15m 경로). champion_selector 가
# R1 vs A1 을 forward CLOSED 로 비교하려면 A1 도 매일 청산돼야 함. 실패해도 R1/champion 무관(가드).
python scripts/close_recommend_ledger.py --ledger output/shadow_ledger_recommend_sustain.csv >> "$LOG" 2>&1 || echo "  A1 recommend close warn" >> "$LOG"

# PUMP hunter rule detector ledger 실현 (동일 -3%SL/+5%TP 15m 경로 + pump20_hit).
# policy_competition 이 pump20 recall / net / downside 를 기존 모델들과 비교하려면
# 별도 shadow ledger 도 매일 CLOSED 로 전환돼야 한다. 실패해도 R1/champion 무관(가드).
python scripts/close_recommend_ledger.py --ledger output/shadow_ledger_pump_hunter.csv >> "$LOG" 2>&1 || echo "  PUMP hunter close warn" >> "$LOG"

# PUMP hunter v2 (Binance volsurge radar) ledger 실현 — 동일 경로 + exit lab 7 잣대.
python scripts/close_recommend_ledger.py --ledger output/shadow_ledger_pump_hunter_v2.csv >> "$LOG" 2>&1 || echo "  PUMP v2 close warn" >> "$LOG"

# champion/challenger 재선정 (unattended). 위 close 들이 forward CLOSED 행을 갱신한 *뒤*
# 돌려야 rolling 윈도가 최신 → 다음날 발송(08:50/09:05)이 새 champion 을 쓴다. 별도 timer 불필요.
echo "[2c/3] champion_selector (forward 갱신 후 재선정 — 다음날 발송용)" >> "$LOG"
python -m ops.champion_selector >> "$LOG" 2>&1 || echo "  champion_selector warn" >> "$LOG"

echo "[2d/3] policy_competition (model + send-policy forward audit)" >> "$LOG"
python -m ops.policy_competition >> "$LOG" 2>&1 || echo "  policy_competition warn" >> "$LOG"

echo "[3/4] train_recommendation_meta (shadow-gated)" >> "$LOG"
python scripts/train_recommendation_meta.py >> "$LOG" 2>&1 || echo "  recommendation meta train warn" >> "$LOG"

echo "[4/4] idea_validation_report" >> "$LOG"
python scripts/idea_validation_report.py >> "$LOG" 2>&1 || echo "  idea validation warn" >> "$LOG"
python scripts/build_idea_validation_html.py >> "$LOG" 2>&1 || echo "  idea validation html warn" >> "$LOG"

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
