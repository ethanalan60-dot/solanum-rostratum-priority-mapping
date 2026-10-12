"""Focused v2 orchestration checks: CPU only, mocked selector, no real solves."""
from pathlib import Path
import argparse, datetime, json, math
import numpy as np
from scipy.optimize import OptimizeResult
import evaluate_common_management_v2 as v


def main(protocol_sha, maps_dir=None):
    out = v.PA / ('common_rm_v2_checks_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False); tests = []
    protocol, watched = v.load_protocol(protocol_sha)
    tests.append('reviewed_protocol_and_all_frozen_sources_match')
    try: v.load_protocol('0' * 64)
    except AssertionError: tests.append('wrong_protocol_sha_rejected')
    else: raise AssertionError('Wrong protocol accepted')
    source = out / 'source_guard.txt'; source.write_text('original', encoding='utf-8')
    source_watch = {str(source): v.ev.sha(source)}
    source.write_text('changed', encoding='utf-8')
    try: v.verify_hashes(source_watch)
    except AssertionError: tests.append('source_drift_hard_rejected')
    else: raise AssertionError('Changed source accepted')
    path = out / 'keys' / 'raw' / 'case.json'; path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf-8')
    assert v.relative_path(out, 'keys/raw/case.json') == v.relative_path(out, 'keys\\raw\\case.json') == path.resolve()
    assert list(v.file_hashes(out, [path])) == ['keys/raw/case.json']
    for key in ('../escape', '/absolute', 'C:/outside', 'a//b'):
        try: v.relative_path(out, key)
        except AssertionError: pass
        else: raise AssertionError('Unsafe path accepted: ' + key)
    tests.append('windows_separators_canonical_keys_and_traversal_rejection')
    for run in v.KNOWN_COMPLETED:
        try: v.reject_existing(run, out)
        except RuntimeError: pass
        else: raise AssertionError('Completed original model accepted')
    tests.append('all_three_completed_models_rejected_without_resolving')
    duplicate_root = out / 'duplicates'; duplicate_root.mkdir()
    for state in ('running', 'failed', 'completed'):
        folder = duplicate_root / ('cmv2_' + state); folder.mkdir()
        v.ev.write_json(folder / 'status.json', {'model_run': 'fixture_' + state, 'state': state})
        try: v.reject_existing('fixture_' + state, duplicate_root)
        except RuntimeError: pass
        else: raise AssertionError('Prior output accepted')
    claims = duplicate_root / 'common_rm_v2_claims'; claims.mkdir()
    v.ev.write_json(claims / 'fixture_claim.json', {'state': 'incomplete_creation'})
    try: v.reject_existing('fixture_claim', duplicate_root)
    except RuntimeError: tests.append('running_failed_completed_and_partial_claim_rejected')
    else: raise AssertionError('Prior claim accepted')
    raw_dir = out / 'raw_before_reject'; raw_dir.mkdir()
    (raw_dir / 'raw').mkdir(); (raw_dir / 'choices').mkdir()
    marker = {'candidate': 'uninterpreted evidence'}
    def reject_after_save(*args):
        assert v.ev.read(raw_dir / 'raw' / v.recovery.case_name(v.CASES[0])) == marker
        raise RuntimeError('intentional provenance rejection')
    try: v.save_then_recover(raw_dir, v.CASES[0], ['a'], np.ones(1), np.ones(1), marker, recover_fn=reject_after_save)
    except RuntimeError as error: assert 'intentional' in str(error)
    else: raise AssertionError('Expected rejection absent')
    assert len(list((raw_dir / 'raw').glob('*.json'))) == 1 and not list((raw_dir / 'choices').iterdir())
    tests.append('unmodified_raw_saved_before_recovery_rejection')
    intercepted = out / 'selector_internal_assert'; intercepted.mkdir(); (intercepted / 'solver_raw').mkdir()
    backend = v.policy.milp; backend_calls = []
    def inversion_backend(*args, **kwargs):
        backend_calls.append((args, kwargs))
        assert kwargs['options'] == v.policy.OPTIONS
        return OptimizeResult(x=np.array([1.]), status=0, fun=float('nan'), mip_dual_bound=0.,
                              mip_gap=float('inf'), mip_node_count=1, message='synthetic contradictory bound')
    v.policy.milp = inversion_backend
    try:
        try: v.capture_selector_call(intercepted, (1, 1, 1.), ['a'], np.ones(1), np.array([.8]), 1.)
        except AssertionError as error: assert 'upper bound below feasible primal' in str(error)
        else: raise AssertionError('Frozen internal bound assertion did not fire')
        assert v.policy.milp is inversion_backend and len(backend_calls) == 1
    finally: v.policy.milp = backend
    captured = v.ev.read(intercepted / 'solver_raw' / 'M1_R1_B100.json')
    assert captured['result']['x'] == [1.] and captured['solver_options_passed'] == v.policy.OPTIONS
    assert captured['result']['fun'] == {'nonfinite_float': 'NaN'}
    assert captured['result']['mip_gap'] == {'nonfinite_float': '+Infinity'}
    assert (intercepted / 'solver_raw' / 'M1_R1_B100_exception.json').exists()
    assert v.policy.milp is backend
    tests.append('actual_frozen_selector_internal_assert_keeps_solver_return_options_nonfinite_and_restores_backend_no_retry')
    pipeline = out / 'synthetic_pipeline'; pipeline.mkdir()
    dest = pipeline / 'predicted_area_policy'; dest.mkdir()
    rows = []
    for m in (1, 2, 4):
        for i, p, target in zip(('a', 'b', 'c'), (.8, .6, .2), (1, 0, 0)):
            rows.append({'M': m, 'id': f'm{m}_{i}', 'area': '1.0',
                         'predicted_R1': str(p), 'predicted_R2': str(p), 'predicted_R4': str(p), 'target': str(target)})
    v.ev.write_csv(pipeline / 'unit_predictions.csv', rows)
    allowed = v.project_selection_inputs(pipeline / 'unit_predictions.csv')
    assert all(set(row) == set(v.FIELDS) for row in allowed)
    tampered_labels = [dict(row, target='NOT_A_NUMBER', q_true='DO_NOT_READ') for row in rows]
    v.ev.write_csv(pipeline / 'unreadable_labels.csv', tampered_labels)
    assert v.project_selection_inputs(pipeline / 'unreadable_labels.csv') == allowed
    tests.append('selection_projection_discards_unreadable_labels')
    calls = []
    def mock_selector(ids, area, pred, budget):
        assert not (dest / 'selection_seal.json').exists()
        chosen = v.policy.prefix_ids(ids, area, pred, budget)
        v.policy.milp(options=v.policy.OPTIONS.copy())
        calls.append((list(ids), budget))
        if len(calls) == 1:
            return v.recovery.toy_raw(ids, area, pred, budget, [], candidate=list(ids), feasible=False, upper=math.fsum(pred))
        return v.recovery.toy_raw(ids, area, pred, budget, chosen, upper=math.fsum(pred))
    v.policy.milp = lambda **kwargs: OptimizeResult(x=None, status=1, fun=None, mip_dual_bound=None, mip_gap=None,
                                                   mip_node_count=0, message='mock backend; no real solve')
    try: seal = v.choose_all(dest, allowed, 3., {'synthetic_only': True}, selector=mock_selector)
    finally: v.policy.milp = backend
    assert len(calls) == 45 and seal['target_join_started'] is False
    assert len(seal['raw_sha256']) == len(seal['choices_sha256']) == 45
    assert v.verify_selection_seal(dest) == seal
    tests.append('all45_label_free_calls_raw_choices_and_seal_saved')
    ref_rows = []
    data = v.prepare_data(allowed, 3.)
    for m, r, b in v.CASES:
        ids, area, pp = data[m]
        q = v.policy.evaluate_selected(ids, v.policy.prefix_ids(ids, area, pp[r], b), area, pp[r], np.array([1., 0., 0.]))
        ref_rows.append({'M': m, 'R': r, 'budget': b, 'method': 'model_prefix', 'k': q['selected_count'],
                         'actual_area': q['actual_area_m2'], 'captured_target': q['captured_target_m2'],
                         'capture_fraction': q['capture_fraction'], 'zero_annotation_area': q['zero_annotation_area_m2']})
    v.ev.write_csv(pipeline / 'fixed_budgets.csv', ref_rows)
    selected = dest / 'choices' / v.recovery.case_name(v.CASES[0]); prior_bytes = selected.read_bytes()
    selected.write_text('{}', encoding='utf-8')
    read_csv = v.policy.read_csv
    def forbid_label_read(*args): raise RuntimeError('Labels must not be read before valid seal')
    v.policy.read_csv = forbid_label_read
    try:
        try: v.evaluate_choices(pipeline, dest, {'area_m2': 3., 'target_m2': 1.})
        except AssertionError: pass
        else: raise AssertionError('Tampered seal accepted')
    finally: v.policy.read_csv = read_csv; selected.write_bytes(prior_bytes)
    tests.append('tampered_choice_seal_rejected_before_any_csv_label_read')
    answer = v.evaluate_choices(pipeline, dest, {'area_m2': 3., 'target_m2': 1.})
    assert answer['combinations'] == 45 and answer['fallback_count'] == 1
    assert len(v.policy.read_csv(dest / 'all_45_deltas.csv')) == 45
    assert len(v.policy.read_csv(dest / 'all_45_prefix_reference.csv')) == 45
    assert v.verify_selection_seal(dest) == seal
    tests.append('full45_output_join_after_seal_and_strict_fallback_not_proven')
    actual = None
    if maps_dir:
        _, checked, _ = v.validate(maps_dir, protocol_sha)
        actual = {'maps_directory': str(Path(maps_dir).resolve()), 'run_id': checked[2]['run_id'],
                  'checkpoint_epoch': checked[2]['checkpoint_epoch'], 'accepted': True}
        tests.append('one_completed_remaining_export_passes_unchanged_validator')
    v.verify_hashes(watched)
    report = {'passed': True, 'created_at': v.ev.stamp(), 'checks': tests, 'check_count': len(tests),
              'new_real_solver_calls': 0, 'synthetic_mock_selector_calls': len(calls), 'model_calls': 0,
              'new_geometry_or_production_integration': False, 'test_or_R6_accessed': False,
              'actual_readonly_entry_check': actual, 'protocol_sha256': protocol_sha,
              'pipeline_source_sha256': v.ev.sha(v.__file__), 'test_source_sha256': v.ev.sha(__file__),
              'reused_checks': {'entry_rejection': 29, 'recovery': 13, 'rerun': False}}
    v.ev.write_json(out / 'focused_checks.json', report)
    print(json.dumps({'output': str(out), 'report': report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha256', required=True); parser.add_argument('--maps-dir')
    args = parser.parse_args(); main(args.protocol_sha256, args.maps_dir)
