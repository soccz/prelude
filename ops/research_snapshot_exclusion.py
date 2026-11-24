"""Only a validated, explicitly delivery-only R1 can be omitted from research."""

from pathlib import Path

from signals import recommend_live_snapshot as delivery

# Explicitly reviewed issuer of the original delivery-only format. New issuer
# versions require review; a re-sealed document with an arbitrary 64-hex value
# is not grounds to suppress a research validation error.
REVIEWED_ISSUERS = {"f075f436c6b0fc68d456d197a4fe2b8956464ba4772b0401158179a88ba997b9"}


def marked_exclusion(path: Path, error: Exception, **identity) -> bool:
    if not isinstance(
        error, delivery.SnapshotError
    ) or "probability nesting violated in top-k:" not in str(error):
        return False
    if path.name not in {"open_r1.json", "preopen_r1.json"}:
        return False
    slot = path.name.split("_")[0]
    expected = {
        "asof": path.parent.name,
        "ranking": "R1",
        "slot": slot,
        "model_id": f"recommend_r1_{slot}",
        **identity,
    }
    # A bad checksum, wrong identity, missing/forged marker or unrelated
    # invariant must still fail the caller; the exception text alone is not proof.
    document = delivery.load_snapshot(path, **expected)
    return (
        "probability_quality" in document
        and document.get("delivery_policy_source_sha256") in REVIEWED_ISSUERS
    )
