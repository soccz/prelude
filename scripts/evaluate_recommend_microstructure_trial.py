"""Read-only cumulative evaluation of the single record-only microstructure trial."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signals.recommend_microstructure_trial import (  # noqa: E402
    DEFAULT_RECEIPT_ROOT, DEFAULT_TRIAL_ROOT, evaluate_microstructure_trials, write_new_trial_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", default=str(DEFAULT_TRIAL_ROOT))
    parser.add_argument("--label-root", default=str(ROOT / "output/recommend_score_labels"))
    parser.add_argument("--receipt-root", default=str(DEFAULT_RECEIPT_ROOT))
    parser.add_argument("--now", help="aware ISO evaluation time; default actual UTC now")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="optional NEW report path; never overwrites")
    args = parser.parse_args(argv)
    try:
        report = evaluate_microstructure_trials(
            args.trial_root, label_root=args.label_root, receipt_root=args.receipt_root,
            now=args.now, n_boot=args.n_boot, seed=args.seed,
        )
        if args.output:
            write_new_trial_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "blocked", "deployable": False,
                          "reason": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
