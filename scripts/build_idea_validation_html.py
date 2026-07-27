"""Build a local HTML idea validation report.

The hosted dashboard can consume idea_validation.json, but this repo-local HTML
keeps the same evidence readable without editing the external site repository.
"""
from __future__ import annotations

import argparse
import html
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.portfolio_metrics import normalize_kst_date
from ops.policy_competition import (
    POLICY_DB,
    PolicyArtifactError,
    load_policy_artifact,
)
from scripts.idea_validation_report import (
    load_idea_validation_artifact,
    validate_idea_validation_payload,
)


def _fmt(value, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _finite_number(value, default: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _state_class(state: str) -> str:
    s = str(state)
    if "PROMOTE" in s:
        return "good"
    if "DEMOTE" in s:
        return "bad"
    return "watch"


def _gate_cards(payload: dict) -> str:
    rows = payload.get("policy_gate", {}).get("rows", [])
    cards = []
    for row in rows:
        reasons = "".join(f"<li>{html.escape(str(r))}</li>" for r in row.get("reasons", []))
        cards.append(
            f"""
            <section class="card {_state_class(row.get('state'))}">
              <div class="eyebrow">{html.escape(str(row.get('channel', '')))}</div>
              <h2>{html.escape(str(row.get('state', '')))}</h2>
              <p class="action">{html.escape(str(row.get('action', '')))}</p>
              <dl>
                <div><dt>confidence</dt><dd>{html.escape(str(row.get('confidence', '')))}</dd></div>
                <div><dt>replay closed</dt><dd>{_fmt(row.get('replay_n_closed'), digits=0)}</dd></div>
                <div><dt>replay net</dt><dd>{_fmt(row.get('replay_net_pnl_sum_pct'), '%p')}</dd></div>
                <div><dt>delta</dt><dd>{_fmt(row.get('delta_net_pnl_sum_pct'), '%p')}</dd></div>
                <div><dt>CI low</dt><dd>{_fmt(row.get('replay_bootstrap_avg_ci95_low_pct'), '%p')}</dd></div>
                <div><dt>late avg</dt><dd>{_fmt(row.get('replay_late_avg_net_pnl_pct'), '%p')}</dd></div>
              </dl>
              <ul>{reasons}</ul>
            </section>
            """
        )
    return "\n".join(cards)


def _replay_table(payload: dict) -> str:
    rows = payload.get("policy_replay", {}).get("rows", [])
    body = []
    for row in rows:
        obs = row.get("observed_active", {})
        rep = row.get("replay_active", {})
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('channel')))}</td>"
            f"<td>{_fmt(obs.get('n_closed'), digits=0)}</td>"
            f"<td>{_fmt(obs.get('net_pnl_sum_pct'), '%p')}</td>"
            f"<td>{_fmt(rep.get('n_closed'), digits=0)}</td>"
            f"<td>{_fmt(rep.get('net_pnl_sum_pct'), '%p')}</td>"
            f"<td>{_fmt(row.get('delta_net_pnl_sum_pct'), '%p')}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _idea_table(payload: dict) -> str:
    rows = [
        r for r in payload.get("tables", {}).get("summary", [])
        if r.get("dimension") == "idea"
    ]
    rows.sort(key=lambda r: (str(r.get("channel")), -(r.get("net_pnl_sum_pct") or -999999)))
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('channel')))}</td>"
            f"<td>{html.escape(str(row.get('idea_id')))}</td>"
            f"<td>{html.escape(str(row.get('setup_quality')))}</td>"
            f"<td>{html.escape(str(row.get('btc_regime')))}</td>"
            f"<td>{_fmt(row.get('n_closed'), digits=0)}</td>"
            f"<td>{_fmt(row.get('tp5_hit_rate_pct'), '%')}</td>"
            f"<td>{_fmt(row.get('net_pnl_sum_pct'), '%p')}</td>"
            f"<td>{_fmt(row.get('avg_net_pnl_pct'), '%p')}</td>"
            f"<td>{html.escape(str(row.get('evidence_tier')))}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _model_card(payload: dict) -> str:
    card = payload.get("model_card", {})
    if not card:
        return ""
    layers = "".join(f"<li>{html.escape(str(x))}</li>" for x in card.get("decision_layers", []))
    controls = "".join(f"<li>{html.escape(str(x))}</li>" for x in card.get("risk_controls", []))
    refs = "".join(
        f"<li><a href=\"{html.escape(str(r.get('url', '')))}\">{html.escape(str(r.get('label', '')))}</a></li>"
        for r in card.get("methodology_references", [])
    )
    ev = card.get("current_evidence", {})
    trained = card.get("trained_meta_model", {})
    artifact_status = str(trained.get("artifact_status", "")).strip().upper()
    if artifact_status:
        trained_status = artifact_status
    elif trained.get("deployable"):
        trained_status = "DECLARED_DEPLOYABLE"
    else:
        trained_status = "INACTIVE"
    trained_text = (
        f"{html.escape(str(trained.get('model_id', 'not trained')))} "
        f"({trained_status}) · n={_fmt(trained.get('n_samples'), digits=0)} · "
        f"reason: {html.escape(str(trained.get('reason', '')))}"
    ) if trained.get("available") else html.escape(str(trained.get("reason", "not trained")))
    return f"""
    <section class="block">
      <h2>Model Card</h2>
      <div class="two">
        <div>
          <p><strong>{html.escape(str(card.get('name', '')))}</strong></p>
          <p class="muted">{html.escape(str(card.get('intended_use', '')))}</p>
          <dl>
            <div><dt>version</dt><dd>{html.escape(str(card.get('version', '')))}</dd></div>
            <div><dt>closed evidence</dt><dd>{_fmt(ev.get('n_closed'), digits=0)}</dd></div>
            <div><dt>net</dt><dd>{_fmt(ev.get('net_pnl_sum_pct'), '%p')}</dd></div>
            <div><dt>TP5 hit</dt><dd>{_fmt(ev.get('tp5_hit_rate_pct'), '%')}</dd></div>
          </dl>
          <p class="muted"><strong>Trained meta model:</strong> {trained_text}</p>
        </div>
        <div>
          <h3>Decision Layers</h3>
          <ul>{layers}</ul>
          <h3>Risk Controls</h3>
          <ul>{controls}</ul>
          <h3>References</h3>
          <ul>{refs}</ul>
        </div>
      </div>
    </section>
    """


def _quality_table(payload: dict) -> str:
    q = payload.get("recommendation_quality", {})
    rows = q.get("rows", [])
    if not rows:
        note = html.escape(str(q.get("note", "No recommendation-quality rows yet.")))
        return f"<tr><td colspan=\"7\" class=\"muted\">{note}</td></tr>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('channel')))}</td>"
            f"<td>{html.escape(str(row.get('confidence_tier')))}</td>"
            f"<td>{_fmt(row.get('n_candidates'), digits=0)}</td>"
            f"<td>{_fmt(row.get('avg_confidence_score'), digits=1)}</td>"
            f"<td>{_fmt(row.get('n_closed'), digits=0)}</td>"
            f"<td>{_fmt(row.get('net_pnl_sum_pct'), '%p')}</td>"
            f"<td>{_fmt(row.get('tp5_hit_rate_pct'), '%')}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _metric_table(metrics: dict, threshold: dict) -> str:
    rows = [
        ("holdout n", metrics.get("n"), ""),
        ("positive rate", metrics.get("positive_rate_pct"), "%"),
        ("AUC", metrics.get("auc"), ""),
        ("average precision", metrics.get("average_precision"), ""),
        ("Brier", metrics.get("brier"), ""),
        ("all holdout net", metrics.get("net_pnl_sum_pct"), "%p"),
        ("selected n", threshold.get("n_selected"), ""),
        ("selected precision", threshold.get("precision_pct"), "%"),
        ("selected net", threshold.get("net_pnl_sum_pct"), "%p"),
        ("selected avg", threshold.get("avg_net_pnl_pct"), "%p"),
    ]
    body = []
    for label, value, suffix in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{_fmt(value, suffix)}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _coef_table(rows: list[dict]) -> str:
    if not rows:
        return "<tr><td colspan=\"2\" class=\"muted\">No coefficients available.</td></tr>"
    body = []
    for row in rows[:10]:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('feature')))}</td>"
            f"<td>{_fmt(row.get('coef'), digits=4)}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _accuracy_section(payload: dict) -> str:
    trained = payload.get("model_card", {}).get("trained_meta_model", {})
    if not trained.get("available"):
        reason = html.escape(str(trained.get("reason", "trained meta model not available")))
        return f"""
        <section class="block">
          <h2>Recommendation Accuracy</h2>
          <p class="muted">{reason}</p>
        </section>
        """
    artifact_status = str(trained.get("artifact_status", "")).strip().upper()
    if artifact_status:
        status = artifact_status
    elif trained.get("deployable"):
        status = "DECLARED_DEPLOYABLE"
    else:
        status = "INACTIVE"
    status_class = (
        "good"
        if status == "DEPLOYED" and trained.get("deployable")
        else "watch"
    )
    metrics = trained.get("holdout_metrics", {}) or {}
    threshold = trained.get("holdout_threshold_stats", {}) or {}
    return f"""
    <section class="block">
      <h2>Recommendation Accuracy</h2>
      <div class="statusline">
        <span class="pill {status_class}">{status}</span>
        <span class="muted">{html.escape(str(trained.get('reason', '')))}</span>
      </div>
      <div class="two">
        <div>
          <h3>Holdout Gate</h3>
          <table>
            <tbody>{_metric_table(metrics, threshold)}</tbody>
          </table>
        </div>
        <div>
          <h3>Top Coefficients</h3>
          <table>
            <thead><tr><th>feature</th><th>coef</th></tr></thead>
            <tbody>{_coef_table(trained.get('top_coefficients', []))}</tbody>
          </table>
        </div>
      </div>
    </section>
    """


def _policy_competition_table(payload: dict) -> str:
    pc = payload.get("policy_competition") or {}
    rows = [
        r for r in pc.get("rows", [])
        if isinstance(r, dict) and _finite_number(r.get("n_closed"), 0) > 0
    ]
    if not rows:
        return "<tr><td colspan=\"9\" class=\"muted\">No closed policy competition rows yet.</td></tr>"
    rows.sort(
        key=lambda r: (
            _finite_number(r.get("pump20_recall_pct"), -1),
            _finite_number(r.get("net_mean_pct"), -999),
            _finite_number(r.get("n_closed"), 0),
        ),
        reverse=True,
    )
    body = []
    for row in rows[:16]:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('participant_id')))}</td>"
            f"<td>{html.escape(str(row.get('objective')))}</td>"
            f"<td>{_fmt(row.get('n_closed'), digits=0)}</td>"
            f"<td>{_fmt(row.get('n_days'), digits=0)}</td>"
            f"<td>{_fmt(row.get('net_mean_pct'), '%p')}</td>"
            f"<td>{_fmt(row.get('deep_loss_freq_pct'), '%')}</td>"
            f"<td>{_fmt(row.get('pump20_precision_pct'), '%')}</td>"
            f"<td>{_fmt(row.get('pump20_recall_pct'), '%')}</td>"
            f"<td>{_fmt(row.get('pump20_captured'), digits=0)} / {_fmt(row.get('pump20_actual'), digits=0)}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _policy_competition_section(payload: dict) -> str:
    pc = payload.get("policy_competition") or {}
    if not pc:
        return """
        <section class="block">
          <h2>Policy Competition</h2>
          <p class="muted">No policy competition artifact yet.</p>
        </section>
        """
    cfg = pc.get("config", {})
    return f"""
    <section class="block">
      <h2>Policy Competition</h2>
      <p class="muted">
        Model + send-policy audit · asof {html.escape(str(pc.get('asof', '')))}
        · P(+20%) threshold {_fmt(cfg.get('pump20_threshold'), digits=2)}
        · record-only, no Telegram promotion.
      </p>
      <table>
        <thead>
          <tr>
            <th>participant</th><th>objective</th><th>closed</th><th>days</th>
            <th>net avg</th><th>deep loss</th><th>pump precision</th>
            <th>pump recall</th><th>captured</th>
          </tr>
        </thead>
        <tbody>{_policy_competition_table(payload)}</tbody>
      </table>
    </section>
    """


def _validated_payload(payload: dict, *, asof=None) -> tuple[dict, str]:
    if not isinstance(payload, dict):
        raise ValueError("idea validation artifact must be a JSON object")
    cutoff = normalize_kst_date(asof)
    validate_idea_validation_payload(
        payload,
        asof=cutoff,
        require_current=True,
    )
    artifact_asof = normalize_kst_date(payload["asof"])

    safe = dict(payload)
    policy = safe.get("policy_competition")
    if policy is not None:
        safe["policy_competition"] = _safe_policy_competition(
            policy,
            cutoff=artifact_asof,
        )
    return safe, str(cutoff.date())


def _safe_policy_competition(policy, *, cutoff) -> dict | None:
    if not isinstance(policy, dict) or not policy.get("asof"):
        return None
    try:
        policy_asof = normalize_kst_date(policy["asof"])
    except ValueError:
        return None
    cutoff_date = normalize_kst_date(cutoff)
    if policy_asof > cutoff_date:
        return None
    rows = []
    for row in policy.get("rows", []):
        if not isinstance(row, dict) or not row.get("asof"):
            continue
        try:
            row_asof = normalize_kst_date(row["asof"])
        except ValueError:
            continue
        if row_asof == policy_asof and row_asof <= cutoff_date:
            rows.append(row)
    safe = dict(policy)
    safe["asof"] = str(policy_asof.date())
    safe["rows"] = rows
    return safe


def render_html(
    payload: dict,
    *,
    asof=None,
    policy_competition: dict | None = None,
) -> str:
    payload, report_cutoff = _validated_payload(payload, asof=asof)
    if payload.get("policy_competition") is None and policy_competition is not None:
        payload["policy_competition"] = _safe_policy_competition(
            policy_competition,
            cutoff=payload["asof"],
        )
    generated = html.escape(str(payload.get("generated_at", "")))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rendered = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prelude Idea Validation</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d9dee8;
      --good: #0f8a5f;
      --bad: #b42318;
      --watch: #9a6700;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 6px; font-size: 20px; letter-spacing: 0; }}
    main {{ padding: 24px 32px 40px; max-width: 1440px; margin: 0 auto; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 8px;
      padding: 18px;
    }}
    .card.good {{ border-left-color: var(--good); }}
    .card.bad {{ border-left-color: var(--bad); }}
    .card.watch {{ border-left-color: var(--watch); }}
    .eyebrow {{ color: var(--muted); text-transform: uppercase; font-size: 12px; font-weight: 700; }}
    .action {{ margin: 0 0 14px; font-weight: 700; }}
    h3 {{ margin: 12px 0 4px; font-size: 14px; letter-spacing: 0; }}
    dl {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 14px; margin: 0 0 12px; }}
    dl div {{ min-width: 0; }}
    dt {{ color: var(--muted); font-size: 12px; }}
    dd {{ margin: 2px 0 0; font-weight: 700; }}
    section.block {{ margin-top: 24px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    ul {{ margin: 10px 0 0; padding-left: 18px; color: var(--muted); }}
    a {{ color: inherit; }}
    .two {{ display: grid; grid-template-columns: minmax(280px, 0.85fr) minmax(320px, 1.15fr); gap: 22px; }}
    .statusline {{ display: flex; align-items: center; gap: 10px; margin: 10px 0 16px; flex-wrap: wrap; }}
    .pill {{ display: inline-flex; align-items: center; min-height: 24px; padding: 0 10px; border-radius: 999px; font-size: 12px; font-weight: 800; }}
    .pill.good {{ color: var(--good); background: #e8f7ef; }}
    .pill.watch {{ color: var(--watch); background: #fff5d6; }}
    @media (max-width: 820px) {{ .two {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Prelude Idea Validation</h1>
    <div class="muted">asof: {html.escape(str(payload.get('asof')))} KST | report cutoff: {html.escape(report_cutoff)} KST | source generated: {generated} | html generated UTC: {html.escape(now)} | cost: {_fmt(payload.get('round_trip_cost_pct'), '%')}</div>
  </header>
  <main>
    {_model_card(payload)}
    {_accuracy_section(payload)}
    {_policy_competition_section(payload)}
    <div class="grid">
      {_gate_cards(payload)}
    </div>
    <section class="block">
      <h2>Recommendation Quality</h2>
      <table>
        <thead><tr><th>channel</th><th>tier</th><th>candidates</th><th>avg conf</th><th>closed</th><th>net</th><th>TP5 hit</th></tr></thead>
        <tbody>{_quality_table(payload)}</tbody>
      </table>
    </section>
    <section class="block">
      <h2>Policy Replay</h2>
      <table>
        <thead><tr><th>channel</th><th>observed n</th><th>observed net</th><th>replay n</th><th>replay net</th><th>delta</th></tr></thead>
        <tbody>{_replay_table(payload)}</tbody>
      </table>
    </section>
    <section class="block">
      <h2>Idea Attribution</h2>
      <table>
        <thead><tr><th>channel</th><th>idea</th><th>setup</th><th>regime</th><th>n</th><th>TP5 hit</th><th>net</th><th>avg</th><th>tier</th></tr></thead>
        <tbody>{_idea_table(payload)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _invalid_html(reason: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Prelude Idea Validation — INVALID</title></head><body>"
        "<h1>INVALID AS-OF ARTIFACT</h1>"
        f"<p>{html.escape(reason)}</p>"
        "<p>No metrics were rendered.</p></body></html>"
    )


def build_html(input_json: str | Path, out_html: str | Path,
               policy_competition_json: str | Path | None = None,
               *,
               asof=None) -> None:
    out = Path(out_html)
    try:
        payload = load_idea_validation_artifact(
            input_json,
            asof=asof,
            require_current=True,
        )
        policy_override = None
        if (
            policy_competition_json is not None
            and isinstance(payload, dict)
            and payload.get("policy_competition") is None
        ):
            try:
                pc_path = Path(policy_competition_json)
                policy_override = load_policy_artifact(
                    pc_path,
                    csv_path=pc_path.with_suffix(".csv"),
                    db_path=POLICY_DB,
                    asof=payload.get("asof"),
                    require_current=True,
                )
            except PolicyArtifactError:
                policy_override = None
        rendered = render_html(
            payload,
            asof=asof,
            policy_competition=policy_override,
        )
    except (OSError, ValueError) as exc:
        _atomic_write_text(out, _invalid_html(str(exc)))
        raise ValueError(f"cannot render idea validation report: {exc}") from exc
    _atomic_write_text(out, rendered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", default="output/idea_validation_summary.json")
    parser.add_argument("--out-html", default="output/idea_validation_report.html")
    parser.add_argument("--policy-competition-json", default="output/policy_competition_summary.json")
    parser.add_argument("--asof", help="inclusive KST cutoff (default=today KST)")
    args = parser.parse_args()
    build_html(
        args.input_json,
        args.out_html,
        args.policy_competition_json,
        asof=args.asof,
    )
    print(f"saved {args.out_html}")


if __name__ == "__main__":
    main()
