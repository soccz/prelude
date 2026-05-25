from __future__ import annotations

import pandas as pd

from scripts.predict_preopen_trigger import should_send_telegram as should_send_preopen_telegram
from scripts.predict_today_distribution import should_send_telegram as should_send_distribution_telegram


def test_distribution_telegram_sends_only_active_by_default():
    active = pd.DataFrame([{"market": "KRW-AAA"}])
    empty = pd.DataFrame()

    assert should_send_distribution_telegram(dry_run=False, alerts=active, send_silence=False)
    assert not should_send_distribution_telegram(dry_run=False, alerts=empty, send_silence=False)
    assert should_send_distribution_telegram(dry_run=False, alerts=empty, send_silence=True)
    assert not should_send_distribution_telegram(dry_run=True, alerts=active, send_silence=True)


def test_preopen_telegram_sends_only_active_by_default():
    active = pd.DataFrame([{"market": "KRW-AAA"}])
    empty = pd.DataFrame()

    assert should_send_preopen_telegram(send_tg=True, alerts=active, send_silence=False)
    assert not should_send_preopen_telegram(send_tg=True, alerts=empty, send_silence=False)
    assert should_send_preopen_telegram(send_tg=True, alerts=empty, send_silence=True)
    assert not should_send_preopen_telegram(send_tg=False, alerts=active, send_silence=True)
