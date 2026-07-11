#!/usr/bin/env bash
# Distribution beta cron entry — KST 09:05.
# Telegram sends every day: ACTIVE recommendation or concise silence/status.
# 사용: '5 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_run_distribution.sh'

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# .env (NOT used for telegram in Stage 1, but harmless to load)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

LOG_DIR="output"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)
LOG="$LOG_DIR/cron_dist_$TODAY.log"

echo "=== prelude distribution beta KST $(date +%Y-%m-%d\ %H:%M:%S) ===" >> "$LOG"

# 1) Data update (incremental). 실패해도 health gate 가 잡음
echo "[1/3] data update — d1 + 4h all (close-out 가 4h 필요)" >> "$LOG"
python -m data.collector_d1 --update >> "$LOG" 2>&1 || echo "  d1 update warn" >> "$LOG"
# 4h ALL coins (close-out 이 모든 alert coin 의 어제 4h 봉 필요).
# --days 2 로 incremental (1-2 페이지/코인, 252 × ~0.5s ≈ 2분).
python -m data.collector_4h --all --days 2 >> "$LOG" 2>&1 || echo "  4h update warn" >> "$LOG"

# 2) Health gate — stale DB 로 paper entry 만들지 않음
echo "[2/3] health_check gate (distribution: d1 + 4h)" >> "$LOG"
python scripts/health_check.py --channel distribution --no-telegram >> "$LOG" 2>&1

# 3) Distribution beta — RECORD ONLY (telegram OFF; paper_ledger/대시보드 유지). R1 레이더가 발송.
#    send 플래그 제거 → dry_run, paper_ledger append 는 window-gate 라 그대로 기록됨.
echo "[3/4] predict_today_distribution (record only — no send flags)" >> "$LOG"
python scripts/predict_today_distribution.py \
    --universe top100 \
    --top-k 10 \
    >> "$LOG" 2>&1 || echo "  dist predict warn (record only)" >> "$LOG"

# 4) R1 risk-reward 레이더 — 최소관심 모드로 record-only 강등 (2026-07-11 집행,
#    사전등록 블록 PHASES.md "recommend record-only 강등" 조항 · DECISIONS #2 비준 2026-07-08).
#    --dry-run = 텔레그램 발송 X. R1 ledger 기록(recommend_today)은 그대로 유지 — 판정 재료 연속성.
echo "[4/5] recommend_send (R1 radar, open slot — record-only/최소관심) + recommend_today (R1 ledger)" >> "$LOG"
python scripts/recommend_send.py --slot open --dry-run >> "$LOG" 2>&1 || echo "  R1 send warn" >> "$LOG"
python scripts/recommend_today.py >> "$LOG" 2>&1
EXIT=$?

# 5) R2 challenger (downside-penalized) — SHADOW · record-only (텔레그램 X). forward 표본
#    적립용으로 자기 ledger(shadow_ledger_recommend_r2.csv)에만 기록. challenger_only=True 라
#    champion_selector 가 영구 차단(절대 발송 안 됨). 30거래일 후 R1 과 하방-우선 비교.
#    실패해도 R1 운영과 무관(가드).
echo "[5/6] recommend_today --ranking R2 (R2 challenger ledger, record-only)" >> "$LOG"
python scripts/recommend_today.py --ranking R2 >> "$LOG" 2>&1 || echo "  R2 record warn" >> "$LOG"

# 6) A1 sustainability challenger — SHADOW · record-only (텔레그램 X). R1 top-3 위에
#    dump head 로 dump-prone 픽 강등→교체. 자기 ledger(shadow_ledger_recommend_sustain.csv)
#    에만 기록. challenger_only=True 라 champion_selector 가 영구 차단(절대 발송 안 됨).
#    30거래일 후 R1 과 하방-우선 비교. 실패해도 R1 운영과 무관(가드).
echo "[6/7] recommend_today --ranking A1 (A1 sustainability ledger, record-only)" >> "$LOG"
python scripts/recommend_today.py --ranking A1 >> "$LOG" 2>&1 || echo "  A1 record warn" >> "$LOG"

# 7) PUMP hunter rule detector — SHADOW · record-only (텔레그램 X). pump_rule_discovery_v1
#    에서 채굴한 D-1 roc_7d/ATR/log_return 룰을 매일 별도 ledger 에 기록한다.
#    challenger_only=True 라 champion_selector 가 발송 승격하지 않음. policy_competition 이
#    CLOSED forward rows 로 기존 모델들과 pump20 recall/net/downside 를 비교.
echo "[7/9] pump_detector_today (PUMP hunter ledger, record-only)" >> "$LOG"
python scripts/pump_detector_today.py >> "$LOG" 2>&1 || echo "  PUMP hunter record warn" >> "$LOG"

# 8) Binance d1 incremental refresh — v2 detector 의 b_vol_surge 용. Binance D-1 일봉은
#    00:00 UTC = KST 09:00 마감이라 이 시점 (09:10+) 에 fresh 하게 받는다.
#    메인 알림 (위 1-4) 이 모두 끝난 뒤라 실패/지연해도 기존 운영 무영향 (가드).
echo "[8/9] collector_binance_d1 --days 3 (v2 feature incremental)" >> "$LOG"
python -m data.collector_binance_d1 --all --days 3 >> "$LOG" 2>&1 || echo "  binance refresh warn" >> "$LOG"

# 9) PUMP hunter v2 — Binance volsurge 융합 radar. 사용자 컨펌 (2026-06-11) 으로
#    🎯 텔레그램 발사 (후보 있을 때만 / binance stale 시 경고 1줄 / 후보 0 + 정상 = 무소음).
#    shadow ledger 기록. champion 승격은 challenger_only 차단 — 별도 radar 채널.
echo "[9/9] pump_detector_v2_today (🎯 radar telegram + shadow ledger)" >> "$LOG"
python scripts/pump_detector_v2_today.py --send-telegram >> "$LOG" 2>&1 || echo "  PUMP v2 warn" >> "$LOG"

# 1h/15m incremental update 는 별도 research cron 으로 분리 (Phase B/C 용, 운영 critical X)

echo "[done] $(date +%H:%M:%S) exit=$EXIT" >> "$LOG"
echo "" >> "$LOG"
exit $EXIT
