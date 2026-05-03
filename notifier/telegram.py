"""텔레그램 봇 — 메시지 전송.

설정 (.env 또는 환경변수):
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

사용:
    from notifier.telegram import send_telegram
    send_telegram("hello")
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

# .env 자동 로드 (PROJECT_ROOT/.env)
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger("telegram")


# ============================================================================
# 설정
# ============================================================================
def get_token() -> Optional[str]:
    return os.environ.get("TELEGRAM_BOT_TOKEN")


def get_chat_id() -> Optional[str]:
    return os.environ.get("TELEGRAM_CHAT_ID")


# ============================================================================
# 전송
# ============================================================================
def send_telegram(
    message: str,
    chat_id: Optional[str] = None,
    token: Optional[str] = None,
    parse_mode: Optional[str] = None,
    retries: int = 3,
    backoff: float = 2.0,
    dry_run: bool = False,
) -> bool:
    """
    텔레그램 메시지 전송.

    parse_mode: 'Markdown' / 'HTML' / None (plain)
    retries: 실패 시 재시도

    return: 성공 여부
    """
    token = token or get_token()
    chat_id = chat_id or get_chat_id()

    if dry_run or not token or not chat_id:
        if not dry_run:
            logger.warning("TELEGRAM_BOT_TOKEN / CHAT_ID 미설정 — dry-run")
        print("=" * 60)
        print(f"[telegram dry-run] chat_id={chat_id}")
        print("-" * 60)
        print(message)
        print("=" * 60)
        return True

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # 텔레그램 4096 자 제한
    chunks = []
    for i in range(0, len(message), 4000):
        chunks.append(message[i:i + 4000])

    for chunk in chunks:
        payload = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        for attempt in range(1, retries + 1):
            try:
                res = requests.post(url, data=payload, timeout=10)
                res.raise_for_status()
                break
            except Exception as e:
                logger.warning(f"telegram attempt {attempt} fail: {e}")
                if attempt < retries:
                    time.sleep(backoff ** attempt)
                else:
                    logger.error("telegram send FAILED after retries")
                    return False

    return True


# ============================================================================
# 직접 실행 — 봇 셋업 / 테스트
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="prelude test message")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = get_token()
    chat_id = get_chat_id()
    if not args.dry_run and (not token or not chat_id):
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수 미설정")
        print("설정: export TELEGRAM_BOT_TOKEN=...")
        print("       export TELEGRAM_CHAT_ID=...")
        print("       또는 .env 파일")
        import sys; sys.exit(1)

    ok = send_telegram(args.message, dry_run=args.dry_run)
    print("OK" if ok else "FAIL")
