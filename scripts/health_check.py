"""Daily health check — cron 정상 작동 + DB 신선도 + risk state 점검.

매일 KST 09:30 또는 별도 cron.
문제 발견 시 텔레그램 alert.
"""
from __future__ import annotations

import argparse
import math
import re
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.collector_d1 import get_krw_markets
from data.database import connect_readonly, market_timestamp_ranges_readonly
from data.market_universe import (
    signal_eligible_markets,
    signal_market_exclusions,
)
from notifier.telegram import send_telegram
from ops.artifact_provenance import (
    ArtifactValidationError,
    strict_json_object,
)


# 이전 CLI와의 호환용 진단 임계. PIT Top100에 들어온 candidate-eligible market의
# exact required row 누락은 이 값으로 완화할 수 없고 항상 실패한다.
DEFAULT_MIN_UNIVERSE_COVERAGE = 1.0
# 초기값: NTP/API 경계의 작은 clock skew만 허용한다. candle timestamp가 이보다
# 미래면 손상/시간대 오류로 보고 fail closed 한다.
DEFAULT_MAX_FUTURE_HOURS = 5.0 / 60.0
SIGNAL_UNIVERSE_TOP_N = 100
SIGNAL_MIN_HISTORY = 70
KST = timezone(timedelta(hours=9))
_KRW_MARKET_RE = re.compile(r"KRW-[A-Z0-9]+")
_RISK_STATE_KEYS = frozenset(
    {
        "is_active",
        "silenced_until",
        "trigger_reason",
        "last_daily_pnl_pct",
        "current_mdd_pct",
    }
)
_DRIFT_STATE_KEYS = frozenset(
    {"state", "triggers", "last_check", "details"}
)
_DRIFT_STATES = frozenset({"OK", "WARN", "FREEZE"})
_DRIFT_TRIGGERS = frozenset(
    {"SIGN_FLIP", "HALF_DROP", "HIT_RATE_DROP", "DIST_SHIFT"}
)
_RISK_TRIGGER_METRICS = {
    "DAILY_LOSS_LIMIT": "last_daily_pnl_pct",
    "MDD_LIMIT": "current_mdd_pct",
}
_DRIFT_DETAIL_RANGES = {
    "ic_24h": (-1.0, 1.0),
    "ic_7d_ma": (-1.0, 1.0),
    "hit_7d": (0.0, 1.0),
    "hit_30d": (0.0, 1.0),
    "dist_ks_pvalue": (0.0, 1.0),
}
_DRIFT_DETAIL_PAIRS = (
    ("ic_24h", "ic_7d_ma"),
    ("hit_7d", "hit_30d"),
)
_DRIFT_REQUIRED_DETAILS = {
    "SIGN_FLIP": frozenset({"ic_24h", "ic_7d_ma"}),
    "HALF_DROP": frozenset({"ic_24h", "ic_7d_ma"}),
    "HIT_RATE_DROP": frozenset({"hit_7d", "hit_30d"}),
    "DIST_SHIFT": frozenset({"dist_ks_pvalue"}),
}
_DRIFT_FREEZE_TRIGGERS = frozenset({"SIGN_FLIP", "HALF_DROP"})


def _kst_now_naive() -> datetime:
    """Return the KST wall clock used by timezone-naive exchange timestamps."""
    return datetime.now(KST).replace(tzinfo=None)


@dataclass(frozen=True)
class _FreshnessScope:
    channel: str
    rank_at: datetime
    d1_required_at: datetime
    d1_boundary_name: str


def _wall_clock_datetime(value) -> datetime:
    """DB의 timezone-naive KST timestamp와 같은 wall-clock으로 정규화."""
    dt = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if not isinstance(dt, datetime):
        raise TypeError(f"not a datetime: {value!r}")
    if dt.tzinfo is not None:
        dt = dt.astimezone(KST)
    return dt.replace(tzinfo=None)


def _lag_hours(latest, now: datetime) -> float:
    return (
        _wall_clock_datetime(now) - _wall_clock_datetime(latest)
    ).total_seconds() / 3600.0


def check_db_freshness(
    db_path: str,
    market: str = "KRW-BTC",
    max_lag_hours: float = 30,
    *,
    max_future_hours: float = DEFAULT_MAX_FUTURE_HOURS,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """단일 market freshness를 DB 생성 없이 검사한다."""
    if max_lag_hours <= 0:
        raise ValueError("max_lag_hours must be positive")
    if max_future_hours < 0:
        raise ValueError("max_future_hours must be non-negative")
    try:
        latest = market_timestamp_ranges_readonly(db_path).get(market, (None, None))[1]
    except Exception as exc:
        return False, f"DB {db_path}: read-only freshness check failed: {exc}"
    if latest is None:
        return False, f"DB {db_path}: no data for {market}"
    lag_hours = _lag_hours(latest, now or _kst_now_naive())
    if lag_hours < -max_future_hours:
        return (
            False,
            f"DB {db_path}: future timestamp for {market} "
            f"(latest={latest}, ahead={-lag_hours:.1f}h, "
            f"tolerance={max_future_hours:.2f}h)",
        )
    if lag_hours > max_lag_hours:
        return (
            False,
            f"DB {db_path}: stale {market} "
            f"(latest={latest}, lag={lag_hours:.1f}h, max={max_lag_hours:.1f}h)",
        )
    return True, f"DB {db_path}: latest {latest} (lag {lag_hours:.1f}h)"


def db_checks_for_channel(channel: str) -> list[tuple[str, int]]:
    """Return required DB freshness checks for the operating channel."""
    if channel == "recommend":
        return [
            ("data/upbit_d1.db", 30),
        ]
    if channel == "recommend-preopen":
        return [
            ("data/upbit_d1.db", 48),
        ]
    if channel == "preopen":
        return [
            ("data/upbit_d1.db", 48),
            ("data/upbit_15m.db", 2),
        ]
    if channel == "distribution":
        return [
            ("data/upbit_d1.db", 30),
            ("data/upbit_4h.db", 8),
        ]
    if channel == "all":
        return [
            ("data/upbit_d1.db", 30),
            ("data/upbit_4h.db", 8),
            ("data/upbit_15m.db", 2),
        ]
    raise ValueError(f"unknown channel: {channel}")


def log_names_for_channel(channel: str, today: str) -> list[str]:
    if channel == "recommend":
        return [f"output/cron_dist_{today}.log"]
    if channel == "recommend-preopen":
        return [f"output/cron_preopen_{today}.log"]
    if channel == "preopen":
        return [f"output/cron_preopen_{today}.log"]
    if channel == "distribution":
        return [f"output/cron_dist_{today}.log"]
    if channel == "all":
        return [
            f"output/cron_preopen_{today}.log",
            f"output/cron_dist_{today}.log",
        ]
    raise ValueError(f"unknown channel: {channel}")


def _timestamp_text(value: datetime) -> str:
    return _wall_clock_datetime(value).strftime("%Y-%m-%d %H:%M:%S")


def _d1_rank_inputs_readonly(
    db_path: str,
    rank_at: datetime,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Read PIT rank values and prior bar counts without creating a DB."""
    timestamp = _timestamp_text(rank_at)
    with connect_readonly(db_path) as conn:
        quote_rows = conn.execute(
            """
            SELECT market, quote_volume
            FROM candles
            WHERE timestamp = ?
            """,
            (timestamp,),
        ).fetchall()
        count_rows = conn.execute(
            """
            SELECT market, COUNT(*)
            FROM candles
            WHERE timestamp <= ?
            GROUP BY market
            """,
            (timestamp,),
        ).fetchall()
    return (
        {str(market): quote_volume for market, quote_volume in quote_rows},
        {str(market): int(count) for market, count in count_rows},
    )


def _markets_at_timestamp_readonly(
    db_path: str,
    required_at: datetime,
) -> set[str]:
    """Return market identities having the exact required candle boundary."""
    timestamp = _timestamp_text(required_at)
    with connect_readonly(db_path) as conn:
        rows = conn.execute(
            "SELECT market FROM candles WHERE timestamp = ?",
            (timestamp,),
        ).fetchall()
    return {str(row[0]) for row in rows}


def _current_intraday_start(now: datetime, interval_minutes: int) -> datetime:
    """Current Upbit minute-candle start (boundaries are aligned in UTC)."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    wall = _wall_clock_datetime(now)
    utc_wall = wall - timedelta(hours=9)
    epoch = datetime(1970, 1, 1)
    interval_seconds = interval_minutes * 60
    elapsed_seconds = int((utc_wall - epoch).total_seconds())
    floored = elapsed_seconds - elapsed_seconds % interval_seconds
    return epoch + timedelta(seconds=floored, hours=9)


def _last_closed_intraday_start(
    now: datetime,
    interval_minutes: int,
) -> datetime:
    return _current_intraday_start(now, interval_minutes) - timedelta(
        minutes=interval_minutes
    )


def _freshness_scope(channel: str, now: datetime) -> _FreshnessScope:
    wall = _wall_clock_datetime(now)
    today = wall.replace(hour=9, minute=0, second=0, microsecond=0)
    if channel in {"preopen", "recommend-preopen"}:
        return _FreshnessScope(
            channel=channel,
            rank_at=today - timedelta(days=2),
            d1_required_at=today - timedelta(days=1),
            d1_boundary_name="last_closed_start",
        )
    if channel in {"recommend", "distribution", "all"}:
        return _FreshnessScope(
            channel=channel,
            rank_at=today - timedelta(days=1),
            d1_required_at=today,
            d1_boundary_name="current_start",
        )
    raise ValueError(f"unknown channel: {channel}")


def _infer_channel(db_checks: Sequence[tuple[str, float]]) -> str:
    names = {Path(path).name for path, _ in db_checks}
    if "upbit_15m.db" in names:
        return "preopen"
    if "upbit_4h.db" in names:
        return "distribution"
    return "recommend"


def _rank_signal_universe(
    live_markets: set[str],
    quote_values: Mapping[str, Any],
    history_counts: Mapping[str, int],
    *,
    top_n: int,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Replicate ``rank(method='min') <= top_n`` on PIT D1 quote volume."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    ranked_values: dict[str, float] = {}
    invalid_qv: set[str] = set()
    insufficient_history: set[str] = set()
    for market in live_markets:
        if history_counts.get(market, 0) < SIGNAL_MIN_HISTORY:
            insufficient_history.add(market)
            continue
        try:
            value = float(quote_values[market])
        except (KeyError, TypeError, ValueError):
            invalid_qv.add(market)
            continue
        if not math.isfinite(value) or value < 0:
            invalid_qv.add(market)
            continue
        ranked_values[market] = value

    ordered = sorted(ranked_values.items(), key=lambda item: (-item[1], item[0]))
    expected: set[str] = set()
    lower_ranked: set[str] = set()
    prior_value: float | None = None
    rank = 0
    for position, (market, value) in enumerate(ordered, start=1):
        if prior_value is None or value != prior_value:
            rank = position
            prior_value = value
        if rank <= top_n:
            expected.add(market)
        else:
            lower_ranked.add(market)
    return expected, lower_ranked, insufficient_history, invalid_qv


def _identity_summary(markets: Iterable[str], *, limit: int = 8) -> str:
    identities = sorted(set(markets))
    shown = ",".join(identities[:limit]) or "none"
    if len(identities) > limit:
        shown += f",+{len(identities) - limit}"
    return f"{len(identities)}[{shown}]"


def _strict_optional_state(path: Path) -> dict[str, Any] | None:
    """Read one optional state without treating a dangling symlink as absent."""
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ArtifactValidationError(
            f"cannot inspect state artifact: {path}"
        ) from exc
    return strict_json_object(path)


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    artifact: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ArtifactValidationError(
            f"{artifact} schema keys mismatch: "
            f"missing={missing}, unknown={unknown}"
        )


def _optional_finite_number(
    value: Any,
    *,
    field: str,
) -> float | None:
    if value is None:
        return None
    # bool is an int subclass in Python, but is not a JSON number in this
    # contract.
    if type(value) not in (int, float):
        raise ArtifactValidationError(
            f"{field} must be a finite JSON number or null"
        )
    try:
        number = float(value)
    except OverflowError as exc:
        raise ArtifactValidationError(
            f"{field} must be a finite JSON number or null"
        ) from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(
            f"{field} must be a finite JSON number or null"
        )
    return number


def _require_canonical_date(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise ArtifactValidationError(f"{field} must be YYYY-MM-DD")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ArtifactValidationError(f"{field} must be YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise ArtifactValidationError(f"{field} must be YYYY-MM-DD")
    return value


def _require_canonical_iso_datetime(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise ArtifactValidationError(
            f"{field} must be a canonical ISO-8601 datetime"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{field} must be a canonical ISO-8601 datetime"
        ) from exc
    if parsed.isoformat() != value:
        raise ArtifactValidationError(
            f"{field} must be a canonical ISO-8601 datetime"
        )
    return value


def _validate_risk_silence(
    silenced_until: Any,
    trigger_reason: Any,
    metrics: Mapping[str, float | None],
) -> None:
    if silenced_until is None:
        raise ArtifactValidationError(
            "SILENCED risk_state requires silenced_until"
        )
    if type(trigger_reason) is not str:
        raise ArtifactValidationError(
            "SILENCED risk_state requires trigger_reason"
        )

    trigger_name, separator, _ = trigger_reason.partition(" ")
    metric_name = _RISK_TRIGGER_METRICS.get(trigger_name)
    if separator != " " or metric_name is None:
        raise ArtifactValidationError(
            "risk_state.trigger_reason has an unknown trigger enum"
        )
    trigger_value = metrics[metric_name]
    if trigger_value is None or trigger_value >= 0:
        raise ArtifactValidationError(
            f"{trigger_name} requires its matching negative metric"
        )
    expected_reason = (
        f"{trigger_name} (-{abs(trigger_value) * 100:.1f}%)"
    )
    if trigger_reason != expected_reason:
        raise ArtifactValidationError(
            "risk_state.trigger_reason does not match its metric"
        )


def _validate_risk_state(payload: Mapping[str, Any]) -> bool:
    _require_exact_keys(payload, _RISK_STATE_KEYS, artifact="risk_state")

    is_active = payload["is_active"]
    if type(is_active) is not bool:
        raise ArtifactValidationError("risk_state.is_active must be boolean")

    silenced_until = payload["silenced_until"]
    if silenced_until is not None:
        _require_canonical_date(
            silenced_until,
            field="risk_state.silenced_until",
        )

    trigger_reason = payload["trigger_reason"]
    if trigger_reason is not None and type(trigger_reason) is not str:
        raise ArtifactValidationError(
            "risk_state.trigger_reason must be string or null"
        )

    metrics = {
        field: _optional_finite_number(
            payload[field],
            field=f"risk_state.{field}",
        )
        for field in _RISK_TRIGGER_METRICS.values()
    }

    if is_active:
        if silenced_until is not None or trigger_reason is not None:
            raise ArtifactValidationError(
                "ACTIVE risk_state cannot retain silence metadata"
            )
        return True

    _validate_risk_silence(silenced_until, trigger_reason, metrics)
    return False


def _validate_drift_triggers(value: Any) -> list[str]:
    if type(value) is not list:
        raise ArtifactValidationError(
            "drift_state.triggers must be a JSON array"
        )
    for trigger in value:
        if type(trigger) is not str or trigger not in _DRIFT_TRIGGERS:
            raise ArtifactValidationError(
                "drift_state.triggers contains an unknown trigger enum"
            )
    if len(value) != len(set(value)):
        raise ArtifactValidationError(
            "drift_state.triggers contains duplicate values"
        )
    return value


def _validate_drift_details(value: Any) -> dict[str, float]:
    if type(value) is not dict:
        raise ArtifactValidationError(
            "drift_state.details must be a JSON object"
        )
    unknown = sorted(set(value) - set(_DRIFT_DETAIL_RANGES))
    if unknown:
        raise ArtifactValidationError(
            f"drift_state.details contains unknown keys: {unknown}"
        )

    parsed_details: dict[str, float] = {}
    for key, raw_value in value.items():
        parsed = _optional_finite_number(
            raw_value,
            field=f"drift_state.details.{key}",
        )
        if parsed is None:
            raise ArtifactValidationError(
                "drift_state.details values cannot be null"
            )
        lower, upper = _DRIFT_DETAIL_RANGES[key]
        if not lower <= parsed <= upper:
            raise ArtifactValidationError(
                f"drift_state.details.{key} must be in [{lower:g}, {upper:g}]"
            )
        parsed_details[key] = parsed

    for left, right in _DRIFT_DETAIL_PAIRS:
        if (left in parsed_details) != (right in parsed_details):
            raise ArtifactValidationError(
                f"drift_state.details requires {left} and {right} together"
            )
    return parsed_details


def _expected_drift_state(triggers: set[str]) -> str:
    if not triggers:
        return "OK"
    if triggers & _DRIFT_FREEZE_TRIGGERS:
        return "FREEZE"
    return "WARN"


def _validate_drift_state(
    payload: Mapping[str, Any],
) -> tuple[str, list[str]]:
    _require_exact_keys(payload, _DRIFT_STATE_KEYS, artifact="drift_state")

    state = payload["state"]
    if type(state) is not str or state not in _DRIFT_STATES:
        raise ArtifactValidationError(
            "drift_state.state must be one of OK, WARN, FREEZE"
        )

    triggers = _validate_drift_triggers(payload["triggers"])

    _require_canonical_iso_datetime(
        payload["last_check"],
        field="drift_state.last_check",
    )

    parsed_details = _validate_drift_details(payload["details"])

    trigger_set = set(triggers)
    for trigger in triggers:
        missing = _DRIFT_REQUIRED_DETAILS[trigger] - set(parsed_details)
        if missing:
            raise ArtifactValidationError(
                f"drift trigger {trigger} is missing details: "
                f"{sorted(missing)}"
            )

    expected_state = _expected_drift_state(trigger_set)
    if state != expected_state:
        raise ArtifactValidationError(
            "drift_state.state is inconsistent with triggers"
        )
    return state, triggers


def check_log_age(log_path: str, max_age_hours: int = 26) -> tuple[bool, str]:
    p = Path(log_path)
    try:
        parent_before = p.parent.lstat()
        before = p.lstat()
    except FileNotFoundError:
        return False, f"{log_path}: missing"
    except OSError as exc:
        return False, f"{log_path}: cannot inspect ({exc})"
    if not stat.S_ISDIR(parent_before.st_mode):
        return False, f"{log_path}: parent must be a real directory"
    if not stat.S_ISREG(before.st_mode):
        return False, f"{log_path}: must be a regular non-symlink file"
    mtime = datetime.fromtimestamp(before.st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    try:
        parent_after = p.parent.lstat()
        after = p.lstat()
    except OSError as exc:
        return False, f"{log_path}: changed during inspection ({exc})"
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        parent_before.st_dev,
        parent_before.st_ino,
        parent_before.st_mode,
        parent_before.st_mtime_ns,
        parent_before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        parent_after.st_dev,
        parent_after.st_ino,
        parent_after.st_mode,
        parent_after.st_mtime_ns,
        parent_after.st_ctime_ns,
    ):
        return False, f"{log_path}: changed during inspection"
    if age.total_seconds() < -DEFAULT_MAX_FUTURE_HOURS * 3600:
        return False, f"{log_path}: future-dated by {-age}"
    if age.total_seconds() / 3600 > max_age_hours:
        return False, f"{log_path}: stale by {age}"
    return True, f"{log_path}: {age.total_seconds()/3600:.1f}h ago"


def check_universe_coverage(
    db_checks: Sequence[tuple[str, float]],
    *,
    min_coverage_ratio: float = DEFAULT_MIN_UNIVERSE_COVERAGE,
    live_markets: Iterable[str] | None = None,
    now: datetime | None = None,
    channel: str | None = None,
    top_n: int = SIGNAL_UNIVERSE_TOP_N,
) -> tuple[bool, str]:
    """Validate exact PIT candidate inputs, not blanket all-live freshness.

    Open channels rank the prior completed D1 candle and require today's 09:00
    row for that candidate-eligible Top100.  Pre-open ranks D-2 and requires
    yesterday 09:00.  Lower-ranked/new raw ticker identities cannot enter the
    scorer and are reported as explicit exclusions instead of false failures.
    """
    if not 0.0 < min_coverage_ratio <= 1.0:
        raise ValueError("min_coverage_ratio must be in (0, 1]")
    if not db_checks:
        raise ValueError("db_checks must not be empty")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    for db_path, max_lag_hours in db_checks:
        if not db_path:
            raise ValueError("db path must not be empty")
        if max_lag_hours <= 0:
            raise ValueError("max_lag_hours must be positive")

    try:
        raw_live = list(
            get_krw_markets(include_stablecoins_for_audit=True)
            if live_markets is None
            else live_markets
        )
    except Exception as exc:
        return False, f"universe coverage: live KRW ticker fetch failed: {exc}"
    if (
        any(
            not isinstance(market, str)
            or _KRW_MARKET_RE.fullmatch(market) is None
            for market in raw_live
        )
        or len(set(raw_live)) != len(raw_live)
    ):
        return False, "universe coverage: invalid live KRW ticker identities"
    signal_excluded = signal_market_exclusions(raw_live)
    live = set(signal_eligible_markets(raw_live))
    if not live:
        return False, "universe coverage: live KRW ticker set is empty"

    check_now = now or _kst_now_naive()
    resolved_channel = channel or _infer_channel(db_checks)
    scope = _freshness_scope(resolved_channel, check_now)
    d1_path = db_checks[0][0]
    if Path(d1_path).name != "upbit_d1.db":
        raise ValueError("first db check must be upbit_d1.db")
    try:
        quote_values, history_counts = _d1_rank_inputs_readonly(
            d1_path,
            scope.rank_at,
        )
    except Exception as exc:
        return False, f"universe coverage: D1 PIT rank read failed: {exc}"
    expected, lower_ranked, insufficient_history, invalid_qv = (
        _rank_signal_universe(
            live,
            quote_values,
            history_counts,
            top_n=top_n,
        )
    )
    if not expected:
        return (
            False,
            "universe coverage: candidate-eligible PIT universe is empty "
            f"(rank_at={_timestamp_text(scope.rank_at)})",
        )

    db_exact: dict[str, set[str]] = {}
    details = [
        f"scope={resolved_channel} live_signal={len(live)} "
        f"rank_at={_timestamp_text(scope.rank_at)} "
        f"pit_top={len(expected)}/{top_n} threshold={min_coverage_ratio:.2%}",
        f"signal_market_excluded={_identity_summary(signal_excluded)}",
        f"lower_ranked_excluded={_identity_summary(lower_ranked)}",
        f"insufficient_history_excluded={_identity_summary(insufficient_history)}",
        f"invalid_rank_qv_excluded={_identity_summary(invalid_qv)}",
    ]
    all_ok = True

    d1_exact: set[str] = set()
    for db_path, _max_lag_hours in db_checks:
        db_name = Path(db_path).name
        if db_name == "upbit_d1.db":
            required_at = scope.d1_required_at
            boundary_name = scope.d1_boundary_name
            max_allowed_start = scope.d1_required_at
        elif db_name == "upbit_15m.db":
            required_at = _last_closed_intraday_start(check_now, 15)
            boundary_name = "last_closed_start"
            max_allowed_start = _current_intraday_start(check_now, 15)
        elif db_name == "upbit_4h.db":
            required_at = _last_closed_intraday_start(check_now, 240)
            boundary_name = "last_closed_start"
            max_allowed_start = _current_intraday_start(check_now, 240)
        else:
            raise ValueError(f"unsupported Upbit DB freshness target: {db_path}")
        try:
            ranges = market_timestamp_ranges_readonly(db_path)
            exact = _markets_at_timestamp_readonly(db_path, required_at)
        except Exception as exc:
            db_exact[db_path] = set()
            all_ok = False
            details.append(f"{Path(db_path).name}=read_error[{exc}]")
            continue

        stored = {m for m in ranges if m.startswith("KRW-")}
        db_exact[db_path] = exact
        if db_name == "upbit_d1.db":
            d1_exact = exact
        missing = sorted(expected - exact)
        future: list[str] = []
        exact_required = sorted(expected & exact)
        for market in sorted(expected & stored):
            latest = ranges[market][1]
            latest_wall = _wall_clock_datetime(latest)
            if latest_wall > max_allowed_start:
                future.append(market)

        ratio = len(exact_required) / len(expected)
        inactive_retained = len(stored - expected)
        db_ok = (
            ratio + 1e-12 >= min_coverage_ratio
            and not missing
            and not future
        )
        all_ok = all_ok and db_ok
        details.append(
            f"{db_name}=exact {len(exact_required)}/{len(expected)} "
            f"({ratio:.2%}) expected_{boundary_name}={_timestamp_text(required_at)} "
            f"missing={_identity_summary(missing)} "
            f"future={_identity_summary(future)} "
            f"inactive_retained={inactive_retained}"
        )

    raw_no_current = (live - expected) - d1_exact
    details.append(
        "raw_confirmed_no_current_excluded="
        f"{_identity_summary(raw_no_current)}"
    )

    # D1 과 채널별 intraday DB 의 candidate-universe 불일치를 함께 노출한다.
    d1_markets = db_exact.get(d1_path, set()) & expected
    for ref_path, _ in db_checks[1:]:
        ref_markets = db_exact.get(ref_path, set()) & expected
        d1_only = sorted(d1_markets - ref_markets)
        ref_only = sorted(ref_markets - d1_markets)
        details.append(
            f"{Path(d1_path).stem}<->{Path(ref_path).stem} "
            f"d1_only={len(d1_only)} ref_only={len(ref_only)}"
        )

    return all_ok, "universe coverage: " + "; ".join(details)


def check_risk_state() -> tuple[bool, str]:
    p = Path("output/risk_state.json")
    try:
        s = _strict_optional_state(p)
    except ArtifactValidationError as exc:
        return False, f"risk_state: invalid artifact ({exc})"
    if s is None:
        # ledger/risk.py is intentionally not wired to the record-only runtime.
        # Missing remains an explicit bootstrap state; once a file exists it
        # must satisfy the full persisted RiskState contract below.
        return True, "risk_state: BOOTSTRAP_UNINITIALIZED (evaluator unwired)"
    try:
        is_active = _validate_risk_state(s)
    except ArtifactValidationError as exc:
        return False, f"risk_state: invalid artifact ({exc})"
    if is_active:
        return True, "risk: ACTIVE"
    return (
        False,
        "risk: SILENCED until "
        f"{s['silenced_until']} ({s['trigger_reason']})",
    )


def check_drift_state() -> tuple[bool, str]:
    p = Path("output/drift_state.json")
    try:
        s = _strict_optional_state(p)
    except ArtifactValidationError as exc:
        return False, f"drift_state: invalid artifact ({exc})"
    if s is None:
        # No production caller currently invokes evaluate_drift. Treat absence
        # as explicit bootstrap, never as an implicitly parsed "OK" document.
        return True, "drift_state: BOOTSTRAP_UNINITIALIZED (evaluator unwired)"
    try:
        state, triggers = _validate_drift_state(s)
    except ArtifactValidationError as exc:
        return False, f"drift_state: invalid artifact ({exc})"
    if state in ("WARN", "FREEZE"):
        return False, f"drift: {state} ({triggers})"
    return True, f"drift: {state}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="distribution",
                        choices=[
                            "recommend",
                            "recommend-preopen",
                            "distribution",
                            "preopen",
                            "all",
                        ],
                        help="Operating channel to validate")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Do not send Telegram on failure; exit nonzero only")
    parser.add_argument(
        "--min-universe-coverage",
        type=float,
        default=DEFAULT_MIN_UNIVERSE_COVERAGE,
        help=(
            "diagnostic minimum PIT coverage (required candidate row missing "
            "still fails closed; default: 1.0)"
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.min_universe_coverage <= 1.0:
        parser.error("--min-universe-coverage must be in (0, 1]")

    now_kst = datetime.now(KST)
    print(f"=== prelude health {now_kst.isoformat()} channel={args.channel} ===")
    issues = []

    # Live KRW universe coverage + market별 freshness. DB별 grouped query 1회로
    # BTC 포함 모든 live market을 함께 검사하므로 별도 BTC query를 반복하지 않는다.
    db_checks = db_checks_for_channel(args.channel)
    ok, msg = check_universe_coverage(
        db_checks,
        min_coverage_ratio=args.min_universe_coverage,
        channel=args.channel,
    )
    print(f"  {'OK' if ok else 'FAIL'}: {msg}")
    if not ok:
        issues.append(msg)

    # Log age
    today = now_kst.strftime("%Y%m%d")
    for log_name in log_names_for_channel(args.channel, today):
        ok, msg = check_log_age(log_name)
        print(f"  {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            issues.append(msg)

    # Risk
    ok, msg = check_risk_state()
    print(f"  {'OK' if ok else 'WARN'}: {msg}")
    if not ok:
        issues.append(msg)

    # Drift
    ok, msg = check_drift_state()
    print(f"  {'OK' if ok else 'WARN'}: {msg}")
    if not ok:
        issues.append(msg)

    if issues:
        msg = "⚠️ prelude health issues:\n" + "\n".join(f"  • {i}" for i in issues)
        if not args.no_telegram:
            send_telegram(msg)
        sys.exit(1)
    else:
        print("\n✅ ALL OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
