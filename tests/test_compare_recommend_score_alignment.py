import json
from pathlib import Path

import pytest

import scripts.compare_recommend_score_alignment as cli
from ops.artifact_provenance import file_set_identity
from signals.recommend_experiment_eval import evaluate_comparison
from test_recommend_experiment_eval import _predictions


def save(path, report):
    path.write_text(json.dumps(report, allow_nan=False))


def seal(report):
    report.pop('report_payload_sha256', None)
    report['report_payload_sha256'] = cli.sha256_bytes(cli.canonical_json_bytes(report))
    return report


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'INPUTS', {'earlier': 'early.json', 'later': 'late.json'})
    monkeypatch.setattr(cli, 'SOURCES', ('alignment.py',))
    monkeypatch.setattr(cli, 'CONTRACTS', {
        name: {**value, 'start': '2026-09-01', 'end': '2026-09-06'}
        for name, value in cli.CONTRACTS.items()
    })
    source = tmp_path / 'alignment.py'
    source.write_text('# fixed alignment\n')
    kernel = tmp_path / 'kernel.py'
    kernel.write_text('# old kernel\n')
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{"fixed":true}')
    frame = _predictions(days=6, slots=('open', 'preopen'))
    frame['B_status'], frame['C_status'] = 'ok', 'ok'
    evaluation = evaluate_comparison(frame, n_boot=10)
    reports = {}
    for name, path in cli.INPUTS.items():
        report = {
            'schema': cli.CONTRACTS[name]['schema'], 'design_id': cli.CONTRACTS[name]['design_id'],
            'generator_files': file_set_identity({'kernel': kernel}, root=tmp_path),
            'data_provenance': {'files': file_set_identity({'evidence': evidence}, root=tmp_path)},
            'predictions': frame.to_dict(orient='records'), 'evaluation': evaluation,
        }
        save(tmp_path/path, seal(report))
        reports[name] = tmp_path/path
    design = tmp_path / 'design.json'
    save(design, cli.prepare_design())
    monkeypatch.setattr('requests.sessions.Session.request', lambda *a, **k: pytest.fail('network forbidden'))
    return {**reports, 'design': design, 'source': source, 'kernel': kernel, 'evidence': evidence}


def test_prepare_design_never_transforms_or_evaluates(case, monkeypatch):
    monkeypatch.setattr(cli, 'transform_predictions', lambda *a: pytest.fail('design must not transform'))
    monkeypatch.setattr(cli, 'evaluate_alignment', lambda *a, **k: pytest.fail('design must not evaluate'))
    design = cli.prepare_design()
    assert design['specification']['new_candidates'] == 1
    assert design['specification']['model_fitted'] is False


def test_run_is_offline_no_default_writes_with_both_segments_separate(case, tmp_path):
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = cli.run_comparison(case['design'])
    assert set(result['segments']) == {'earlier', 'later'}
    for segment in result['segments'].values():
        assert segment['coverage']['expected_date_slots'] == 12
        assert segment['coverage']['predicted_date_slots'] == 12
        assert segment['coverage']['missing_predictions'] == []
        assert segment['original_selection_parity'] == {'status':'passed','date_slots':12}
    for key in ('deployable','live_model_changed','model_fitted','model_artifact_saved',
                'automatic_adoption','is_untouched_holdout'):
        assert result[key] is False
    assert result['promotion_status'] == 'NOT_EVALUATED'
    digest = result.pop('report_payload_sha256')
    assert digest == cli.sha256_bytes(cli.canonical_json_bytes(result))
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('target', ['earlier','later','design','source','kernel','evidence'])
def test_input_changes_cannot_be_used_for_a_frozen_design(case, monkeypatch, target):
    case[target].write_text('changed')
    monkeypatch.setattr(cli, 'transform_predictions', lambda *a: pytest.fail('must stop before transform'))
    with pytest.raises(ValueError):
        cli.run_comparison(case['design'])


@pytest.mark.parametrize('field', ['configuration','contracts','specification','frozen_bundle'])
def test_configuration_and_dates_cannot_be_swept(case, field):
    design = json.loads(case['design'].read_text())
    design[field] = 'changed'
    save(case['design'], design)
    with pytest.raises(ValueError, match='frozen score alignment'):
        cli.run_comparison(case['design'])


@pytest.mark.parametrize('mutation', ['payload','contract'])
def test_source_report_validation_cannot_be_skipped_by_refreezing(case, mutation):
    source = json.loads(case['later'].read_text())
    source['schema'] = 'wrong'
    if mutation == 'contract':
        seal(source)
    save(case['later'], source)
    with pytest.raises(ValueError):
        cli.prepare_design()


@pytest.mark.parametrize('target', ['earlier','later','design','source','kernel','evidence'])
def test_changes_during_comparison_block_publication(case, monkeypatch, target):
    original = cli.evaluate_alignment
    def evaluate(*args, **kwargs):
        case[target].write_text('changed during evaluation')
        return original(*args, **kwargs)
    monkeypatch.setattr(cli, 'evaluate_alignment', evaluate)
    with pytest.raises(ValueError, match='changed during comparison'):
        cli.run_comparison(case['design'])


def test_original_raw_selection_is_reproduced_not_overwritten(case, monkeypatch):
    original = cli.transform_predictions
    def transform(frame):
        result, audit = original(frame)
        result['p_up10_B'] = 1. - result.p_up10_B
        return result, audit
    monkeypatch.setattr(cli, 'transform_predictions', transform)
    with pytest.raises(ValueError, match='original C selection changed'):
        cli.run_comparison(case['design'])


def test_missing_dates_are_visible_in_fixed_calendar_denominator(case):
    report = json.loads(case['later'].read_text())
    report['predictions'] = [r for r in report['predictions'] if r['date'] != '2026-09-02']
    save(case['later'], seal(report))
    save(case['design'], cli.prepare_design())
    result = cli.run_comparison(case['design'])
    coverage = result['segments']['later']['coverage']
    assert coverage['expected_date_slots'] == 12
    assert coverage['predicted_date_slots'] == 10
    assert coverage['missing_predictions'] == [{'date':'2026-09-02','slot':s} for s in ('open','preopen')]


def test_outside_dates_rejected_without_export(case, monkeypatch):
    original = cli.transform_predictions
    def transform(frame):
        result, audit = original(frame)
        result.loc[result.date.eq('2026-09-01'), 'date'] = '2026-09-30'
        return result, audit
    monkeypatch.setattr(cli, 'transform_predictions', transform)
    # Coverage uses the source frame; raw parity independently binds every date/slot.
    with pytest.raises(KeyError):
        cli.run_comparison(case['design'])


def test_cli_no_overwrite_and_no_output_on_failure(case, tmp_path, capsys):
    design = tmp_path / 'new-design.json'
    result = tmp_path / 'result.json'
    assert cli.main(['--prepare-design','--output',str(design)]) == 0
    assert cli.main(['--design',str(design),'--output',str(result)]) == 0
    before = result.read_bytes()
    assert cli.main(['--design',str(design),'--output',str(result)]) == 2
    assert result.read_bytes() == before
    assert cli.main(['--prepare-design']) == 2
    assert 'blocked' in capsys.readouterr().err


def test_export_protects_all_original_evidence(case, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'write_new_report', lambda *a, **k: calls.append((a,k)))
    assert cli.main(['--design',str(case['design']),'--output','_workspace/new.json']) == 0
    protected = calls[0][1]['protected_roots']
    assert all(path in protected for path in case.values())
    assert calls[0][0][0] == Path('_workspace/new.json')
