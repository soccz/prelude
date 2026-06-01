#!/usr/bin/env bash
# prelude systemd install — systemd timers are more robust than cron.
# xsec_alpha 패턴 차용. sudo 필요.
#
# 설치:
#   sudo bash deploy/install_systemd.sh
#
# 결과:
#   - prelude-preopen.service        + .timer  (KST 08:50, ACTIVE-only telegram)
#   - prelude-distribution.service   + .timer  (KST 09:05, ACTIVE-only telegram)
#   - prelude-close.service          + .timer  (KST 09:30, distribution close)
#   - prelude-preopen-close.service  + .timer  (KST 10:05, preopen close)
#   - prelude-publish-dashboard      + .timer  (KST 10:10)
#   - signal timers 는 늦은 catch-up 이 paper ledger 를 오염시켜 Persistent=false

set -euo pipefail

REPO="/home/soccz/22tb/prelude"
UNIT_DIR="/etc/systemd/system"

echo "Installing prelude systemd units..."

cp "$REPO/deploy/prelude-distribution.service" "$UNIT_DIR/prelude-distribution.service"
cp "$REPO/deploy/prelude-distribution.timer"    "$UNIT_DIR/prelude-distribution.timer"
cp "$REPO/deploy/prelude-close.service"          "$UNIT_DIR/prelude-close.service"
cp "$REPO/deploy/prelude-close.timer"            "$UNIT_DIR/prelude-close.timer"
# Pre-open trigger (08:55) + close (09:30 same as distribution close, separate ledger)
cp "$REPO/deploy/prelude-preopen.service"        "$UNIT_DIR/prelude-preopen.service"
cp "$REPO/deploy/prelude-preopen.timer"          "$UNIT_DIR/prelude-preopen.timer"
cp "$REPO/deploy/prelude-preopen-close.service"  "$UNIT_DIR/prelude-preopen-close.service"
cp "$REPO/deploy/prelude-preopen-close.timer"    "$UNIT_DIR/prelude-preopen-close.timer"
# Dashboard publish (10:10 KST — close cron 둘 다 끝난 후)
cp "$REPO/deploy/prelude-publish-dashboard.service" "$UNIT_DIR/prelude-publish-dashboard.service"
cp "$REPO/deploy/prelude-publish-dashboard.timer"   "$UNIT_DIR/prelude-publish-dashboard.timer"
# Phase X+3 운영 안전 — DB 백업 (04:00) + heartbeat (10:30)
cp "$REPO/deploy/prelude-backup.service"            "$UNIT_DIR/prelude-backup.service"
cp "$REPO/deploy/prelude-backup.timer"              "$UNIT_DIR/prelude-backup.timer"
cp "$REPO/deploy/prelude-heartbeat.service"         "$UNIT_DIR/prelude-heartbeat.service"
cp "$REPO/deploy/prelude-heartbeat.timer"           "$UNIT_DIR/prelude-heartbeat.timer"

# -----------------------------------------------------------------------------
# [SUPERSEDED — 설치하지 말 것] v-next 추천 레이더 timer 3종.
#   ★ R1 레이더 발송/기록/청산은 이제 기존 timer 스크립트에 통합됨:
#     - 08:50 prelude-preopen → daily_run_preopen.sh: preopen 발송 OFF + recommend_send(preopen)
#     - 09:05 prelude-distribution → daily_run_distribution.sh: distribution 발송 OFF + recommend_send(open) + recommend_today
#     - 09:30 prelude-close → daily_close_distribution.sh: + close_recommend_ledger (R1 실현)
#   따라서 아래 recommend-am/open/close timer 는 불필요하며, 설치하면 같은 시각 중복 발송됨.
#   기존 7 timer 만 활성(아래 enable). 별도 sudo 재실행 불필요 — 스크립트 내용만 바뀜.
# cp "$REPO/deploy/prelude-recommend-am.service"      "$UNIT_DIR/prelude-recommend-am.service"
# cp "$REPO/deploy/prelude-recommend-am.timer"        "$UNIT_DIR/prelude-recommend-am.timer"
# cp "$REPO/deploy/prelude-recommend-open.service"    "$UNIT_DIR/prelude-recommend-open.service"
# cp "$REPO/deploy/prelude-recommend-open.timer"      "$UNIT_DIR/prelude-recommend-open.timer"
# cp "$REPO/deploy/prelude-recommend-close.service"   "$UNIT_DIR/prelude-recommend-close.service"
# cp "$REPO/deploy/prelude-recommend-close.timer"     "$UNIT_DIR/prelude-recommend-close.timer"
# -----------------------------------------------------------------------------

systemctl daemon-reload
systemctl enable --now prelude-distribution.timer
systemctl enable --now prelude-close.timer
systemctl enable --now prelude-preopen.timer
systemctl enable --now prelude-preopen-close.timer
systemctl enable --now prelude-publish-dashboard.timer
systemctl enable --now prelude-backup.timer
systemctl enable --now prelude-heartbeat.timer
# [OPTIONAL] 추천 레이더 (SHADOW) — 위 cp 블록과 함께 주석 해제 시 활성화.
# systemctl enable --now prelude-recommend-am.timer
# systemctl enable --now prelude-recommend-open.timer
# systemctl enable --now prelude-recommend-close.timer

echo ""
echo "=== Installed timers ==="
systemctl list-timers | grep prelude || true

echo ""
echo "=== Test ==="
echo "  sudo systemctl start prelude-distribution.service  # 수동 1회 실행"
echo "  systemctl status prelude-distribution.timer        # 다음 실행 시각 확인"
echo "  journalctl -u prelude-distribution.service -n 50   # 로그 확인"

echo ""
echo "✅ install complete. cron 항목은 별도 제거 필요 (중복 방지)."
