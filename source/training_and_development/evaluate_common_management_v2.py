"""Frozen common-model development evaluation with audited numerical recovery.

No training/inference/test/R6. Parent label-reference MILP uses two threads;
the separate prediction-policy process uses the unchanged one-thread selector.
Pass the reviewed pipeline protocol SHA explicitly. Existing attempts are refused.
"""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, csv, datetime, json, math, shutil, subprocess, sys, time, traceback
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
import evaluate_common_management as original
import evaluate_management as ev
import diagnose_predicted_knapsack as policy
import recover_common_policy as recovery

ROOT = Path(__file__).resolve().parent
PA = ROOT / 'precision_agriculture'
PIPELINE = PA / 'RM_common_pipeline_v2_protocol.json'
RECOVERY_SHA = '97b12d1178dcba1357860c4b2517a041ca3aaf456eb8a575c2c931db5579cfdd'
EXTENSION_SHA = 'e73597bbfd233f82ecbd1fc04fbefda00143eb642fafa4b1d45039bf48a79e2d'
FROZEN = {
    'evaluate_common_management.py': '7682bbd5dba05606a62abad43d4e2178b4b7cfd8f2beae6fd06f0351db78f05e',
    'evaluate_management.py': '29d7b8a8bd7e6762cdae18bde2d491ecbe9cd459bf401ef084a9f827e990ff8d',
    'geo_management.py': '16c6fb38e1c2c1c9c5ea9f5f25db317d0fa021505631ddef9f07dea528549d8f',
    'diagnose_predicted_knapsack.py': 'f45f3210bbef82aa3e0dcfcec58a2b06d2704164c50417819d982660c6674750',
    'recover_common_policy.py': '7fdc3fac309b8b6e98d7c53934ad676fc305670643fd933f5d762443ca550225',
}
KNOWN_COMPLETED = (
    'dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17',
    'dev_common_v2_lr1e4_resnet18_soft_seg_L256_s17',
    'dev_common_v2_lr1e4_convnextv2_tiny_hard_seg_L256_s17',
)
CASES = tuple(recovery.CASES)
FIELDS = tuple(recovery.FIELDS)


def relative_path(root, key):
    """Read either legacy separator; emit only Path.relative_to(...).as_posix()."""
    parts = str(key).replace('\\', '/').split('/')
    assert all(x not in ('', '.', '..') and ':' not in x for x in parts), key
    result = Path(root).joinpath(*parts).resolve()
    assert result.is_relative_to(Path(root).resolve())
    return result


def file_hashes(root, paths):
    return {Path(p).relative_to(root).as_posix(): ev.sha(p) for p in paths}


def verify_hashes(mapping):
    for name, digest in mapping.items():
        assert ev.sha(name) == digest, 'Changed source/input: ' + str(name)


def load_protocol(expected_sha):
    assert len(expected_sha) == 64 and ev.sha(PIPELINE) == expected_sha
    protocol = ev.read(PIPELINE)
    assert protocol['schema'] == 'common_management_pipeline_v2'
    assert protocol['extension_protocol_sha256'] == EXTENSION_SHA == ev.sha(original.PROTOCOL)
    assert protocol['recovery_protocol_sha256'] == RECOVERY_SHA == ev.sha(recovery.PROTOCOL)
    assert protocol['frozen_source_sha256'] == FROZEN
    assert protocol['pipeline_source_sha256'] == ev.sha(__file__)
    assert protocol['strict_area_tolerance_m2'] == 0.0
    assert protocol['case_order'] == [list(x) for x in CASES]
    assert protocol['solver_options'] == policy.OPTIONS
    assert protocol['known_completed_rejected'] == list(KNOWN_COMPLETED)
    declared = {x['run_id'] for x in ev.read(original.PROTOCOL)['declared_runs']}
    assert set(protocol['remaining_run_ids']) == declared - set(KNOWN_COMPLETED)
    assert len(protocol['remaining_run_ids']) == 5
    watched = {str(ROOT / k): h for k, h in FROZEN.items()}
    watched.update({str(PIPELINE): expected_sha, str(original.PROTOCOL): EXTENSION_SHA,
                    str(recovery.PROTOCOL): RECOVERY_SHA, str(Path(__file__)): ev.sha(__file__)})
    for entry in protocol['reused_checks_and_evidence']:
        path = relative_path(ROOT, entry['path'])
        watched[str(path)] = entry['sha256']
    verify_hashes(watched)
    checks = {x['role']: ev.read(relative_path(ROOT, x['path']))
              for x in protocol['reused_checks_and_evidence'] if x['role'] in ('entry_29', 'recovery_13', 'geometry_metric')}
    assert checks['entry_29']['passed'] and checks['entry_29']['rejected_invalid_cases'] == 29
    assert checks['entry_29']['wrapper_sha256'] == FROZEN['evaluate_common_management.py']
    assert checks['recovery_13']['passed'] and len(checks['recovery_13']['checks']) == 13
    assert checks['geometry_metric']['passed']
    return protocol, watched


def reject_existing(run_id, pa=PA):
    if run_id in KNOWN_COMPLETED:
        raise RuntimeError('Already completed under audited original/recovery outputs; do not re-solve: ' + run_id)
    claim = pa / 'common_rm_v2_claims' / (run_id + '.json')
    if claim.exists():
        raise RuntimeError('Prior v2 attempt claim exists; preserve completed/failed/running output: ' + str(claim))
    for folder in pa.glob('cmv2_*'):
        state_path = folder / 'status.json'
        if state_path.exists() and ev.read(state_path).get('model_run') == run_id:
            raise RuntimeError('Prior v2 output exists; do not overwrite or repeat: ' + str(folder))


def validate(maps_dir, expected_sha):
    protocol, watched = load_protocol(expected_sha)
    checked = original.validate(Path(maps_dir).resolve())
    maps = checked[2]
    reject_existing(maps['run_id'])
    assert maps['run_id'] in protocol['remaining_run_ids']
    watched.update(checked[-1])
    # Keep model/checkpoint selection in the unchanged, audited original validator.
    verify_hashes(watched)
    return protocol, checked, watched


def archive_sources(out, protocol, watched):
    source = out / 'source'; source.mkdir()
    paths = [Path(__file__)] + [ROOT / x for x in FROZEN]
    paths += [PIPELINE, original.PROTOCOL, recovery.PROTOCOL,
              PA / 'RM_development_protocol_v1.json', PA / 'RM_policy_diagnostic_protocol_v2.json']
    paths += [relative_path(ROOT, x['path']) for x in protocol['reused_checks_and_evidence']]
    records = []
    for i, path in enumerate(paths):
        target = source / (f'{i:02d}_' + path.name)
        shutil.copy2(path, target)
        digest = ev.sha(path); assert digest == ev.sha(target)
        watched[str(path)] = digest
        records.append({'original': str(path), 'copy': target.relative_to(out).as_posix(), 'sha256': digest})
    ledger = {'created_at': ev.stamp(), 'records': records, 'watched_sha256': watched,
              'project_sources_archived_before_selection': True,
              'versions': {'python': sys.version, 'numpy': np.__version__, 'scipy': ev.scipy.__version__},
              'threadpool_info': threadpool_info(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
    ev.write_json(out / 'source_ledger.json', ledger)
    ev.write_json(out / 'toy_checks.json', {
        'passed': True, 'reused_without_rerunning': protocol['reused_checks_and_evidence'],
        'note': 'Frozen geometry/metric,29 entry-rejection and13 recovery checks; new orchestration checked separately.'})
    return ledger


def project_selection_inputs(path):
    # Target columns are parsed as uninterpreted CSV strings, then discarded.
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = [{key: row[key] for key in FIELDS} for row in csv.DictReader(stream)]
    assert rows and all(set(row) == set(FIELDS) for row in rows)
    return rows


def prepare_data(allowed, domain_area):
    assert all(set(row) == set(FIELDS) for row in allowed)
    assert {int(row['M']) for row in allowed} == {1, 2, 4}
    data = {}
    for m in (1, 2, 4):
        rows = sorted((x for x in allowed if int(x['M']) == m), key=lambda x: x['id'])
        ids = [x['id'] for x in rows]; area = np.array([float(x['area']) for x in rows])
        predictions = {r: np.array([float(x[f'predicted_R{r}']) for x in rows]) for r in (1, 2, 4)}
        for r in (1, 2, 4): recovery.validate_arrays(ids, area, predictions[r], .1)
        recovery.close(math.fsum(area), domain_area)
        data[m] = (ids, area, predictions)
    return data


def save_then_recover(dest, case, ids, area, pred, raw, *, recover_fn=recovery.recover_choice):
    """Persist the unmodified selector return before ANY recovery/strict audit."""
    name = recovery.case_name(case); raw_path = dest / 'raw' / name
    assert not raw_path.exists()
    ev.write_json(raw_path, raw)
    raw_sha = ev.sha(raw_path)
    # Bad provenance/unknown IDs/general errors hard-fail, leaving raw evidence.
    choice = recover_fn(ids, area, pred, case[2], ev.read(raw_path))
    assert ev.sha(raw_path) == raw_sha
    choice.update(M=case[0], R=case[1], budget=case[2], raw_origin='new_predeclared_case_solve',
                  raw_file=raw_path.relative_to(dest).as_posix(), raw_sha256=raw_sha,
                  recovery_source_sha256=FROZEN['recover_common_policy.py'],
                  requested_relative_gap=policy.OPTIONS['mip_rel_gap'])
    ev.write_json(dest / 'choices' / name, choice)
    return choice


def encode_solver_value(value):
    """Preserve IEEE nonfinite values explicitly in standards-compliant JSON."""
    if isinstance(value, np.ndarray): return [encode_solver_value(x) for x in value.tolist()]
    if isinstance(value, np.generic): return encode_solver_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')}
    if isinstance(value, dict): return {str(k): encode_solver_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [encode_solver_value(x) for x in value]
    assert value is None or isinstance(value, (str, int, float, bool)), type(value)
    return value


def capture_selector_call(dest, case, ids, area, pred, budget, *, selector=policy.select_predicted_area):
    """Intercept one actual scipy.milp return before frozen selector assertions.

    The same arguments/options and result object flow through unchanged. This
    records evidence only; selector errors hard-fail without retry/fallback.
    """
    backend = policy.milp
    calls = 0
    raw_path = dest / 'solver_raw' / recovery.case_name(case)
    assert not raw_path.exists()
    def captured_milp(*args, **kwargs):
        nonlocal calls
        assert calls == 0, 'More than one solver call for a predeclared case'
        calls += 1
        result = backend(*args, **kwargs)
        record = {'case': list(case), 'backend': 'scipy.optimize.milp', 'scipy_version': ev.scipy.__version__,
                  'solver_options_passed': encode_solver_value(kwargs.get('options')),
                  'actual_solver_calls_for_case': calls, 'persisted_before_selector_audit': True,
                  'selection_input_sha256': policy.payload_sha({'ids': ids, 'area': area.tolist(), 'predicted': pred.tolist(), 'budget': budget}),
                  'result': encode_solver_value(dict(result)),
                  'standard_fields_present': {k: k in result for k in ('x', 'status', 'fun', 'mip_dual_bound', 'mip_gap', 'mip_node_count', 'message')}}
        ev.write_json(raw_path, record)
        return result
    policy.milp = captured_milp
    try:
        answer = selector(ids, area, pred, budget)
        assert calls == 1 and raw_path.exists(), 'Expected exactly one actual solver return'
        return answer
    except BaseException as error:
        ev.write_json(dest / 'solver_raw' / (Path(raw_path).stem + '_exception.json'),
                      {'case': list(case), 'actual_solver_calls': calls, 'solver_return_persisted': raw_path.exists(),
                       'error': repr(error), 'traceback': traceback.format_exc(), 'automatically_retried': False})
        raise
    finally:
        policy.milp = backend


def choose_all(dest, allowed, domain_area, context, *, selector=policy.select_predicted_area):
    """Only label-free arrays enter this function; no reference/target read."""
    (dest / 'raw').mkdir(); (dest / 'choices').mkdir(); (dest / 'solver_raw').mkdir()
    ev.write_csv(dest / 'selection_inputs_without_annotations.csv', allowed)
    data = prepare_data(allowed, domain_area)
    for index, case in enumerate(CASES):
        m, r, b = case; ids, area, predictions = data[m]
        raw = capture_selector_call(dest, case, ids, area, predictions[r], b, selector=selector)
        save_then_recover(dest, case, ids, area, predictions[r], raw)
        ev.write_json(dest / 'progress.json', {'stage': 'label_free_selection', 'completed_choices': index + 1,
                      'solver_calls': index + 1, 'target_join_started': False})
    seal = {'created_at': ev.stamp(), 'all_45_final_choices_saved': True, 'target_join_started': False,
            'case_order': [list(x) for x in CASES], 'solver_calls': 45,
            'raw_sha256': file_hashes(dest, sorted((dest / 'raw').glob('*.json'))),
            'solver_raw_sha256': file_hashes(dest, sorted((dest / 'solver_raw').glob('*.json'))),
            'choices_sha256': file_hashes(dest, sorted((dest / 'choices').glob('*.json'))),
            'allowed_inputs_sha256': ev.sha(dest / 'selection_inputs_without_annotations.csv'),
            'implementation_sha256': ev.sha(__file__), 'recovery_source_sha256': FROZEN['recover_common_policy.py'],
            'context': context}
    assert len(seal['raw_sha256']) == len(seal['choices_sha256']) == len(seal['solver_raw_sha256']) == 45
    ev.write_json(dest / 'selection_seal.json', seal)
    return seal


def verify_selection_seal(dest):
    seal = ev.read(dest / 'selection_seal.json')
    assert seal['all_45_final_choices_saved'] and seal['target_join_started'] is False
    assert seal['case_order'] == [list(x) for x in CASES] and seal['solver_calls'] == 45
    for category, key in (('raw', 'raw_sha256'), ('choices', 'choices_sha256'), ('solver_raw', 'solver_raw_sha256')):
        expected = {f'{category}/{recovery.case_name(case)}' for case in CASES}
        assert set(seal[key]) == expected
        for name, digest in seal[key].items(): assert ev.sha(relative_path(dest, name)) == digest
    assert ev.sha(dest / 'selection_inputs_without_annotations.csv') == seal['allowed_inputs_sha256']
    assert seal['implementation_sha256'] == ev.sha(__file__)
    assert seal['recovery_source_sha256'] == FROZEN['recover_common_policy.py']
    return seal


def evaluate_choices(out, dest, domain):
    seal = verify_selection_seal(dest)  # mandatory gate BEFORE reading target/reference
    allowed = policy.read_csv(dest / 'selection_inputs_without_annotations.csv')
    data = prepare_data(allowed, domain['area_m2'])
    annotated = policy.read_csv(out / 'unit_predictions.csv')
    assert policy.payload_sha(policy.allowed_selection_rows(annotated)) == policy.payload_sha(allowed)
    truth = {(int(x['M']), x['id']): float(x['target']) for x in annotated}
    assert len(truth) == len(annotated) == len(allowed)
    refs = {(int(x['M']), int(x['R']), float(x['budget'])): x
            for x in policy.read_csv(out / 'fixed_budgets.csv') if x['method'] == 'model_prefix'}
    assert set(refs) == set(CASES)
    results = []; prefixes = []
    for case in CASES:
        m, r, b = case; name = recovery.case_name(case)
        choice = ev.read(dest / 'choices' / name)
        assert (choice['M'], choice['R'], choice['budget']) == case
        assert choice['raw_sha256'] == ev.sha(dest / 'raw' / name)
        ids, area, predictions = data[m]; pred = predictions[r]
        assert choice['selection_input_sha256'] == policy.payload_sha({'ids': ids, 'area': area.tolist(), 'predicted': pred.tolist(), 'budget': b})
        target = np.array([truth[(m, i)] for i in ids])
        assert np.isfinite(target).all() and np.all((target >= 0) & (target <= area + 1e-9))
        recovery.close(math.fsum(target), domain['target_m2'])
        current = policy.evaluate_selected(ids, choice['selected_ids'], area, pred, target)
        baseline_ids = policy.prefix_ids(ids, area, pred, b)
        baseline = policy.evaluate_selected(ids, baseline_ids, area, pred, target)
        for field, column in (('selected_count', 'k'), ('actual_area_m2', 'actual_area'),
                              ('captured_target_m2', 'captured_target'), ('capture_fraction', 'capture_fraction'),
                              ('zero_annotation_area_m2', 'zero_annotation_area')):
            recovery.close(baseline[field], refs[case][column])
        limit = b * math.fsum(area)
        assert current['actual_area_m2'] <= limit == choice['budget_area_m2']
        recovery.close(current['actual_area_m2'], choice['selected_area_m2'])
        recovery.close(current['predicted_objective_m2'], choice['predicted_primal_m2'])
        bound = choice['recomputed_bound']; raw = choice['raw_candidate_audit']
        delta = current['predicted_objective_m2'] - baseline['predicted_objective_m2']
        if choice['optimality_proven']:
            assert not choice['fallback_used'] and delta >= -max(1e-10, 1e-8 * math.fsum(pred))
        row = {'M': m, 'R': r, 'budget': b, 'policy_label': choice['policy_label'],
               'solution_file': ('choices/' + name), 'raw_origin': choice['raw_origin'],
               'budget_area_m2': limit, 'original_area_tolerance_m2': 0.0,
               'fallback_used': choice['fallback_used'], 'fallback_reasons': ';'.join(choice['fallback_reasons']),
               'prefix_tail_trim_count': choice['prefix_tail_trim_count'], 'final_status': choice['status'],
               'optimality_proven': choice['optimality_proven'], 'requested_relative_gap': policy.OPTIONS['mip_rel_gap'],
               'predicted_relative_gap': bound['relative_gap'], 'predicted_upper_m2': bound['upper_m2'],
               'bound_anomaly': bound['bound_anomaly'], 'bound_roundoff_clamp_m2': bound['roundoff_clamp_m2'],
               'raw_candidate_feasible': raw['recorded_feasible_flag'],
               'raw_candidate_budget_violation_m2': raw['budget_violation_m2_recomputed'],
               'raw_solver_status': raw['raw_solver_status'], 'raw_solver_gap': raw['raw_solver_gap'],
               **{'pmilp_' + k: v for k, v in current.items()}, **{'prefix_' + k: v for k, v in baseline.items()},
               'delta_capture_pp': 100 * (current['capture_fraction'] - baseline['capture_fraction']),
               'delta_predicted_objective_m2': delta, 'unused_budget_m2': limit - current['actual_area_m2'],
               'choice_sha256': ev.sha(dest / 'choices' / name)}
        results.append(row)
        prefixes.append({'M': m, 'R': r, 'budget': b, 'budget_area_m2': limit, **baseline,
                         'selected_ids_json': json.dumps(baseline_ids),
                         'strict_original_unit_feasible': baseline['actual_area_m2'] <= limit})
        ev.write_json(dest / (Path(name).stem + '_evaluation.json'), {'metrics': row, 'prefix_selected_ids': baseline_ids})
    ev.write_csv(dest / 'all_45_deltas.csv', results)
    ev.write_csv(dest / 'all_45_prefix_reference.csv', prefixes)
    verify_selection_seal(dest)
    answer = {'combinations': 45, 'solver_calls': 45, 'fallback_count': sum(x['fallback_used'] for x in results),
              'optimal_within_tolerance': sum(x['optimality_proven'] for x in results),
              'improved': sum(x['delta_capture_pp'] > 1e-8 for x in results),
              'worsened': sum(x['delta_capture_pp'] < -1e-8 for x in results),
              'unchanged': sum(abs(x['delta_capture_pp']) <= 1e-8 for x in results),
              'all_solutions_feasible': True, 'original_area_tolerance_m2': 0.0,
              'requested_relative_gap': policy.OPTIONS['mip_rel_gap'],
              'actual_certification_rule': 'non-fallback, frozen selector proven flag, recomputed final gap within recorded tolerance',
              'selection_seal_sha256': ev.sha(dest / 'selection_seal.json'),
              'no_target_annotations_in_selection': True, 'no_auc_from_sparse_budget_points': True}
    ev.write_json(dest / 'summary.json', answer)
    return answer


def run_policy(out, expected_sha):
    protocol, watched = load_protocol(expected_sha)
    assert out.parent == PA.resolve() and out.name.startswith('cmv2_')
    inputs = ev.read(out / 'inputs.json'); state = ev.read(out / 'status.json')
    assert state['state'] == 'running' and state['stage'] == 'policy_subprocess'
    assert inputs['pipeline_protocol_sha256'] == expected_sha
    assert inputs['run_id'] in protocol['remaining_run_ids']
    verify_hashes(inputs['file_sha256']); verify_hashes(watched)
    ledger = ev.read(out / 'source_ledger.json')
    assert ev.sha(out / 'source_ledger.json') == inputs['source_ledger_sha256']
    for record in ledger['records']: assert ev.sha(relative_path(out, record['copy'])) == record['sha256']
    assert ev.sha(out / 'unit_predictions.csv') == state['unit_predictions_sha256']
    assert ev.sha(out / 'fixed_budgets.csv') == state['fixed_budgets_sha256']
    dest = out / 'predicted_area_policy'; dest.mkdir(exist_ok=False)
    allowed = project_selection_inputs(out / 'unit_predictions.csv')
    context = {'pipeline_protocol_sha256': expected_sha, 'recovery_protocol_sha256': RECOVERY_SHA,
               'extension_protocol_sha256': EXTENSION_SHA, 'inputs_sha256': ev.sha(out / 'inputs.json'),
               'source_ledger_sha256': ev.sha(out / 'source_ledger.json'),
               'unit_predictions_sha256': state['unit_predictions_sha256'], 'run_id': inputs['run_id'],
               'checkpoint_sha256': inputs['checkpoint_sha256']}
    with threadpool_limits(limits=2):
        choose_all(dest, allowed, inputs['domain']['area_m2'], context)
        verify_hashes(inputs['file_sha256'])
        assert ev.sha(out / 'unit_predictions.csv') == state['unit_predictions_sha256']
        assert ev.sha(out / 'fixed_budgets.csv') == state['fixed_budgets_sha256']
        answer = evaluate_choices(out, dest, inputs['domain'])
    print(json.dumps(answer), flush=True)


def main(maps_dir, expected_sha, validate_only=False):
    start = time.time(); maps_dir = Path(maps_dir).resolve()
    protocol, checked, watched = validate(maps_dir, expected_sha)
    extension, geometry_protocol, maps, geometry, manifest, ownership, _ = checked
    if validate_only:
        print(json.dumps({'state': 'validated_only', 'model_run': maps['run_id'],
                          'checkpoint_epoch': maps['checkpoint_epoch'], 'source_sha256': watched}), flush=True)
        return
    out = PA / ('cmv2_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    claims = PA / 'common_rm_v2_claims'; claims.mkdir(exist_ok=True)
    claim_path = claims / (maps['run_id'] + '.json')
    # Exclusive permanent attempt claim prevents repeated solves after any failure.
    with claim_path.open('x', encoding='utf-8') as stream:
        json.dump({'model_run': maps['run_id'], 'output': str(out), 'pid': os.getpid(),
                   'checkpoint_sha256': maps['checkpoint_sha256'], 'pipeline_protocol_sha256': expected_sha,
                   'created_at': ev.stamp()}, stream, indent=2)
    out.mkdir(exist_ok=False)
    sp = out / 'status.json'
    state = {'state': 'running', 'stage': 'archive', 'model_run': maps['run_id'], 'pid': os.getpid(),
             'started_at': ev.stamp(), 'test_accessed': False, 'R6_accessed': False}
    ev.write_json(sp, state); print(json.dumps({'output': str(out), 'state': 'running'}), flush=True)
    try:
        with threadpool_limits(limits=2):
            archive_sources(out, protocol, watched)
            domain = {k: geometry[k] for k in ('native_valid_pixels', 'native_target_pixels', 'area_m2', 'target_m2', 'origin_utm_m', 'crs')}
            inputs = {'file_sha256': watched, 'pipeline_protocol_sha256': expected_sha,
                      'extension_protocol_sha256': EXTENSION_SHA, 'recovery_protocol_sha256': RECOVERY_SHA,
                      'run_id': maps['run_id'], 'checkpoint_sha256': maps['checkpoint_sha256'],
                      'checkpoint_epoch': maps['checkpoint_epoch'], 'maps_directory': str(maps_dir),
                      'source_ledger_sha256': ev.sha(out / 'source_ledger.json'), 'domain': domain,
                      'ownership': ownership, 'no_inference': True, 'no_test_or_R6': True,
                      'label_solver_threads': 2, 'policy_solver_threads_separate_process': 1,
                      'selection_note': 'Parent computes annotation references; child receives only A/P/ID/budget and seals all45 before target join.'}
            ev.write_json(out / 'inputs.json', inputs)
            cells, values, integration, windows = ev.integrate(geometry, maps, manifest, sp, state)
            inputs['windows'] = windows; ev.write_json(out / 'inputs.json', inputs)
            ev.write_json(out / 'integration_checks.json', integration)
            result = ev.evaluate_tables(cells, values, geometry_protocol, out, sp, state)
            state.update(stage='policy_subprocess', unit_predictions_sha256=ev.sha(out / 'unit_predictions.csv'),
                         fixed_budgets_sha256=ev.sha(out / 'fixed_budgets.csv'))
            ev.write_json(sp, state)
            command = [sys.executable, '-X', 'utf8', '-B', '-s', str(Path(__file__).resolve()),
                       '--policy-for', str(out), '--protocol-sha256', expected_sha]
            with (out / 'policy_stdout.log').open('w', encoding='utf-8') as stdout, (out / 'policy_stderr.log').open('w', encoding='utf-8') as stderr:
                job = subprocess.run(command, stdout=stdout, stderr=stderr, check=False,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            assert job.returncode == 0, 'Policy process failed; preserved raw/progress and policy_stderr.log'
            ps = ev.read(out / 'predicted_area_policy' / 'summary.json')
            summary = {'state': 'completed', 'created_at': ev.stamp(), 'model_run': maps['run_id'],
                       'checkpoint_epoch': maps['checkpoint_epoch'], 'checkpoint_sha256': maps['checkpoint_sha256'],
                       'domain': domain, 'results': result, 'predicted_area_policy': ps,
                       'integration': {k: v for k, v in integration.items() if k != 'per_window'},
                       'pipeline_protocol_sha256': expected_sha, 'extension_protocol_sha256': EXTENSION_SHA,
                       'recovery_protocol_sha256': RECOVERY_SHA, 'ownership': ownership, 'source_sha256': watched,
                       'versions': {'python': sys.version, 'numpy': np.__version__, 'scipy': ev.scipy.__version__},
                       'model_tuned': False, 'test_accessed': False, 'R6_accessed': False, 'elapsed_seconds': time.time() - start}
            verify_hashes(watched); verify_selection_seal(out / 'predicted_area_policy')
            ev.write_json(out / 'summary.json', summary); ev.make_report(summary, out)
            with (out / 'report.md').open('a', encoding='utf-8') as stream:
                stream.write('# Common R×M v2 — development numerical policy\n\n'
                    'Full geometry/prefix/random/label references are in summary.md; all45 final policy outcomes and prefix comparisons are in predicted_area_policy/.\n\n'
                    '## Numerical recovery and scope\n\n'
                    f"All45 raw returns were saved before recovery; final choices were sealed before joining target values. "
                    'Each actual SciPy solver return was persisted before the frozen selector audited it. Selector exceptions remain failures with evidence, without automatic retry. '
                    f"Fallbacks: {ps['fallback_count']}; recorded certification: {ps['optimal_within_tolerance']}/45. "
                    'Requested relative gap is distinct from achieved bounds; every fallback is not proven. '
                    'Strict final area tolerance is zero. Same maximum budget does not require equal actual selected area. '
                    'One-scene seed17 development evidence; no test/R6, field efficacy or significance claim.\n')
            state.update(state='completed', stage='complete', completed_at=ev.stamp(), policy_summary=ps,
                         files_sha256=file_hashes(out, [p for p in out.rglob('*') if p.is_file() and p != sp]))
            ev.write_json(sp, state)
            print(json.dumps({'state': 'completed', 'output': str(out), 'policy_summary': ps}), flush=True)
    except BaseException as error:
        state.update(state='failed', failed_at=ev.stamp(), error=repr(error), traceback=traceback.format_exc())
        ev.write_json(sp, state); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--maps-dir'); group.add_argument('--policy-for', help=argparse.SUPPRESS)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.policy_for:
        assert not args.validate_only
        run_policy(Path(args.policy_for).resolve(), args.protocol_sha256)
    else:
        main(args.maps_dir, args.protocol_sha256, args.validate_only)
