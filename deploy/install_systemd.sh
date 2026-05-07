#!/usr/bin/env bash
# prelude systemd install — systemd timers are more robust than cron.
# xsec_alpha 패턴 차용. sudo 필요.
#
# 설치:
#   sudo bash deploy/install_systemd.sh
#
# 결과:
#   - prelude-distribution.service  + .timer  (KST 09:05)
#   - prelude-close.service          + .timer  (KST 09:30)
#   - 둘 다 enable + start
#   - close-out timer 만 부팅 후 누락 실행 catch up
#     (distribution alert 는 늦은 catch-up 이 paper ledger 를 오염시켜 Persistent=false)

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

systemctl daemon-reload
systemctl enable --now prelude-distribution.timer
systemctl enable --now prelude-close.timer
systemctl enable --now prelude-preopen.timer
systemctl enable --now prelude-preopen-close.timer
systemctl enable --now prelude-publish-dashboard.timer

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
