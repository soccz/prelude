"""봉인 파일 무결성 — 2026-10-04 알림 문구·기록 전용 변경이 점수/forward 소스를 건드리지 않았음을 고정.

봉인 대상 36개 항목(35개 고유 파일, data/database.py 는 양쪽 목록에 중복):
  - R1 점수 소스 9개: output/recommend_snapshots/2026-10-03/open_r1.json 의 code.score_source_files
  - L1 forward 소스 27개: output/recommend_book_forward/design.json 의 sources
기대 sha256 은 2026-10-04 06시 KST 라이브 작업트리에서 실측한 값이다.

★ 이 봉인 파일들을 *의도적으로* 바꾸는 후속 작업은 (승인 후) 이 표를 함께 갱신해야
  한다 — 그렇지 않으면 07:30 selftest 가 실패한다 (조용한 소스 변화 감지가 목적).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import signals.recommend_snapshot as recommend_snapshot

ROOT = Path(__file__).resolve().parents[1]

SCORE_SOURCE_FILES = ('signals/recommend.py', 'signals/features.py', 'signals/model_registry.py', 'data/database.py', 'data/market_universe.py', 'scripts/downside_head_riskreward_v1.py', 'scripts/recommendation_scorer_v1.py', 'scripts/univariate_precursor_lift_v1.py', 'scripts/regime_split_precursor_v1.py')
# 2026-10-03 open_r1.json code.score_source_sha256 (manifest 집계 해시)
SCORE_SOURCE_AGGREGATE_SHA256 = (
    "4ce4f0205e7ff87f7cfd2bce1951c55bece873f4b00de28de0dec01d3ece949b"
)
SEALED_SHA256 = {
    "signals/recommend.py":
        "32ce9f6280d2ed9bd926da591f2dd7c9bf911773692325f9de022de16d40b59d",
    "signals/features.py":
        "3c24c9fbe9e5b34c27fc51fa18c31a8694b20cc59b7af0ce5d2244a37b97a560",
    "signals/model_registry.py":
        "6510aea1a6cc9bc3eb57f4f71a3163b0885845d56805baf103ba74de6a5f56bb",
    "data/database.py":
        "40e5d7e9c1e1ecfd5e91769216dd297fc41018cd3a3d355144071ee8cbdcdddf",
    "data/market_universe.py":
        "15b5e2c1c19650d5e3c13bfa0e40eb5025552f4d56e33ac7ef10bb967b16f067",
    "scripts/downside_head_riskreward_v1.py":
        "fac5b48f8b7760b27cd84d5dc6d2dfb81619e1485613c02d986554f831ca018d",
    "scripts/recommendation_scorer_v1.py":
        "48f7ac0107c2c2d68db8fa44a578909832ba84590034c25124fcced912f8d076",
    "scripts/univariate_precursor_lift_v1.py":
        "cd5baf1ae80214f4ea309e47ef74eea81dc4c9e97c45e967589819cde61d44c3",
    "scripts/regime_split_precursor_v1.py":
        "bd6cc80b7f7e26439306bed7b4d325908daafc621a266cba7859d1ca792c4dd6",
    "data/upbit_microstructure.py":
        "582a64fa785cb02adcf4112f8f2fde4f22c2048c204214990608bab3730b9075",
    "ledger/config.py":
        "71ab3229c1cd3c0aa8573079f2584e73b26923270856e8edfd8c1ba31c321b3b",
    "ledger/path_quality.py":
        "dff77efd2df015d912ac29c0530f6fd08a59d6d79944837377400b368cc03028",
    "ledger/portfolio_metrics.py":
        "2bfeb38efabecd7d091a17a33af149b08b188ba75f68d5af0f4adc9266133ef7",
    "notifier/delivery_receipt.py":
        "a920d02f9c5c4d58d67e4858ae420874673e847c008979589da20d9653ffe6e4",
    "notifier/telegram.py":
        "e9cc6e13fee12847648cf804e95e012d054b99c35b69ed575b0f8d51831ceefc",
    "ops/artifact_provenance.py":
        "ef33127caf4320c35d46231458ab47887289863a8b5820d5b6a1cb8786a037bc",
    "ops/file_lock.py":
        "b39fc69278748f7c9d903a052c76bab1d996a99f462883e14b77eed3a073f452",
    "ops/recommend_book_execution.py":
        "39aa64f82a8e86853fd7992004b7a55cb72daa2198ea9b89766b134d6dc75dfc",
    "ops/recommend_book_forward.py":
        "6c5f301a880a19a451bdacdf1fcb27a60f5b0785e13a396f11078431bc1ed578",
    "ops/recommend_book_validation.py":
        "56fb35cbdcd9f4e92beee91e2356bc744d2d05b310800b5089dc23af81766e6c",
    "ops/recommendation_evidence.py":
        "d55a8d484a37199361c83da5cbe40be34d504b6125248ccd8ae301bf7c5b4a69",
    "scripts/daily_run_distribution.sh":
        "2c043bd032b290de8a26425c1a5c087368c38b8743fedeedc021feb43554dea5",
    "scripts/evaluate_recommend_entry_delay.py":
        "6be051dc7eb5a5249e3ab80702d99eb764576109a1ab77feba654a91204f2d1c",
    "signals/recommend_book_pressure.py":
        "d5ec80dd6f6293489e4bdbfb05fa5fda74a4e4371bacac171480b4e3e64f6e5f",
    "signals/recommend_book_readiness.py":
        "86d291415f4a1343bdab9509dd090a65948cc518c4ac013263066704cb26bb6a",
    "signals/recommend_book_top3.py":
        "880b7a046c035364015744c48f30caadc20a2e5c756071883b8b9d076ba61157",
    "signals/recommend_book_validation.py":
        "ce0f095fd1adf351f1d8e04c38c1002e937cb490e85d19f94f5fbbf38ebf51c5",
    "signals/recommend_experiment_eval.py":
        "30c6d9d7404f218248ace3ca35767c7bedc95cbc0966fdf6d8a021eae70354ee",
    "signals/recommend_microstructure.py":
        "6136114913e592797d557b57a9d2b605a1bcdc2926aa6331fc3f1355f3071e09",
    "signals/recommend_microstructure_trial.py":
        "65b523de5dc228cd7bc952ff5b3135e73700d4ad5e4a813417aa21d9e96a4dff",
    "signals/recommend_regime_replay.py":
        "f7199c7921d3b7e01ad66e9cadec752e430cfa1bfcf7f5b1a8f71aa65475791d",
    "signals/recommend_score_labels.py":
        "d7a81288f6fb2770acbbc72fc8a00395a8595ea3b2e51cd0d0721e2e317e5f23",
    "signals/recommend_snapshot.py":
        "48db8dc1cbf9330360e40002b85960fc9ce59cd1d52cfb1c6e7970dccba93a5b",
    "signals/recommend_spread_diagnostics.py":
        "cd75ca8af4577aae5b623ee61244da136c48fb7caa36e7d71f94c2ff028ac98d",
    "signals/recommend_trade_shortlist_trial.py":
        "1fc620d497758e42ebe7c489ce6292de25294be13d0ce5fbfa380afe790ac099",
}


def test_sealed_inventory_is_36_entries_35_unique_files():
    assert len(SEALED_SHA256) == 35
    assert set(SCORE_SOURCE_FILES) <= set(SEALED_SHA256)
    assert len(SCORE_SOURCE_FILES) + 27 == 36


def test_sealed_score_and_forward_sources_are_byte_identical():
    changed = {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel in SEALED_SHA256
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != SEALED_SHA256[rel]
    }
    assert changed == {}


def test_score_source_aggregate_matches_live_snapshot_identity():
    assert recommend_snapshot._SCORE_SOURCE_FILES == SCORE_SOURCE_FILES
    assert recommend_snapshot._source_sha256() == SCORE_SOURCE_AGGREGATE_SHA256
