"""Post-hoc development diagnostic, frozen label-free predicted-area knapsack.

Selection is completed for all 45 cases before annotation/reference evaluation.
No raster/model/test/R6 loading; original v1 artifacts are read-only.
"""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, csv, datetime, hashlib, inspect, json, math, shutil, time, traceback, warnings
import numpy as np
import scipy
from scipy.optimize import milp, Bounds, LinearConstraint
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parent
PA = ROOT/'precision_agriculture'
PROTOCOL = PA/'RM_policy_diagnostic_protocol_v2.json'
LOCKED_PROTOCOL_SHA = '38f9452c242f4443e8c9c3811f0ffbe2081a1e07bc28ef932b9bf3322d035d10'
M_VALUES = (1, 2, 4)
R_VALUES = (1, 2, 4)
BUDGETS = (.1, .2, .3, .5, 1.)
OPTIONS = {'mip_rel_gap': 1e-8, 'time_limit': 30., 'presolve': True,
           'threads': 1, 'parallel': False, 'random_seed': 20260916,
           'mip_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def allowed_selection_rows(rows):
    """Projection deliberately ignores all target and reference columns."""
    keys = ('M', 'id', 'area', 'predicted_R1', 'predicted_R2', 'predicted_R4')
    return [{k: row[k] for k in keys} for row in rows]


def payload_sha(rows):
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def select_predicted_area(ids, area, predicted, budget):
    """Only IDs, costs and predictions enter this pure, annotation-free interface."""
    ids = np.asarray(ids, dtype=str)
    area = np.asarray(area, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    assert area.ndim == predicted.ndim == ids.ndim == 1 and len(ids) == len(area) == len(predicted)
    assert len(ids) > 0 and len(set(ids.tolist())) == len(ids)
    assert all(s.isascii() for s in ids)
    order = np.argsort(ids, kind='stable')
    ids, area, predicted = ids[order], area[order], predicted[order]
    assert np.isfinite(area).all() and np.all(area > 0)
    assert np.isfinite(predicted).all() and np.all(predicted >= 0)
    assert np.all(predicted <= area + 1e-9)
    assert 0 <= budget <= 1
    A = math.fsum(area)
    P = math.fsum(predicted)
    scale = P if P > 0 else 1.
    budget_area = float(budget)*A
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = milp(-predicted/scale, integrality=np.ones(len(ids), dtype=np.int8),
                      bounds=Bounds(0., 1.),
                      constraints=LinearConstraint((area/A)[None, :], -np.inf, float(budget)),
                      options=OPTIONS.copy())
    elapsed = time.time()-started
    candidate_ids = []
    candidate_area = candidate_objective = integrality_error = violation = None
    x = getattr(result, 'x', None)
    valid_binary = False
    chosen = np.zeros(len(ids), bool)
    if x is not None and np.isfinite(x).all():
        x = np.asarray(x)
        integrality_error = float(np.max(np.abs(x-np.rint(x))))
        valid_binary = bool(integrality_error <= 1e-6 and np.all((x >= -1e-6) & (x <= 1+1e-6)))
        if valid_binary:
            candidate = np.rint(x).astype(bool)
            candidate_ids = ids[candidate].tolist()
            candidate_area = math.fsum(area[candidate])
            candidate_objective = math.fsum(predicted[candidate])
            violation = max(0., candidate_area-budget_area)
            # Original-unit strict feasibility; no rounding budget extension.
            if violation == 0.:
                chosen = candidate
    candidate_feasible = bool(valid_binary and violation == 0.)
    selected_area = math.fsum(area[chosen])
    primal = math.fsum(predicted[chosen])
    raw_dual = finite(getattr(result, 'mip_dual_bound', None))
    raw_upper = -raw_dual*scale if raw_dual is not None else None
    upper = min(P, raw_upper) if raw_upper is not None else P
    inversion = max(0., primal-upper)
    assert inversion <= max(1e-10, 1e-10*P), 'Solver upper bound below feasible primal'
    upper = max(primal, upper)
    gap = (upper-primal)/max(abs(primal), 1e-30) if upper else 0.
    proven = bool(result.status == 0 and candidate_feasible and gap <= OPTIONS['mip_rel_gap']+1e-10)
    return {'selected_ids': ids[chosen].tolist(), 'solver_candidate_ids': candidate_ids,
            'selected_area_m2': selected_area, 'predicted_primal_m2': primal,
            'predicted_dual_upper_m2': upper, 'predicted_absolute_gap_m2': upper-primal,
            'predicted_relative_gap': gap, 'raw_solver_dual_upper_m2': raw_upper,
            'bound_roundoff_clamp_m2': inversion,
            'raw_minimization_fun': finite(getattr(result, 'fun', None)),
            'raw_minimization_dual': raw_dual, 'raw_solver_gap': finite(getattr(result, 'mip_gap', None)),
            'solver_status': int(result.status), 'solver_message': str(result.message),
            'status': 'optimal_within_recorded_gap' if proven else 'not_proven_optimal',
            'optimality_proven_within_tolerance': proven,
            'candidate_feasible_in_original_units': candidate_feasible,
            'candidate_area_m2': candidate_area, 'candidate_predicted_m2': candidate_objective,
            'candidate_budget_violation_m2': violation, 'candidate_integrality_error': integrality_error,
            'rejected_candidate_fallback': None if candidate_feasible else 'empty known-feasible set; not asserted to be solver optimum',
            'budget_fraction': float(budget), 'budget_area_m2': budget_area,
            'total_area_m2': A, 'total_predicted_m2': P, 'n_variables': len(ids),
            'elapsed_seconds': elapsed, 'mip_node_count': finite(getattr(result, 'mip_node_count', None)),
            'options': OPTIONS.copy(), 'original_unit_feasibility_tolerance_m2': 0.,
            'binary_audit_tolerance': 1e-6, 'normalization': 'cost/total_area; predicted/total_predicted unless zero',
            'no_secondary_objective': True, 'warnings': [str(w.message) for w in caught]}


def prefix_ids(ids, area, predicted, budget):
    ids = np.asarray(ids, dtype=str)
    order = np.lexsort((ids, -predicted/area))
    cumsum = np.r_[0., np.cumsum(area[order], dtype=np.float64)]
    total = math.fsum(area)
    cumsum[-1] = total
    k = int(np.searchsorted(cumsum, budget*total, side='right')-1)
    return ids[order[:k]].tolist()


def run_toys():
    ids = np.array(['a', 'b', 'c'])
    area = np.array([2., 3., 3.])
    pred = np.array([1.9, 2.7, 2.7])
    result = select_predicted_area(ids, area, pred, .75)
    assert result['optimality_proven_within_tolerance'] and result['selected_ids'] == ['b', 'c']
    assert abs(result['predicted_primal_m2']-5.4) < 1e-10
    selected_prefix = prefix_ids(ids, area, pred, .75)
    assert selected_prefix == ['a', 'b']
    # Labels change, but the allowed payload and result do not.
    rows = [{'M': '1', 'id': i, 'area': str(a), 'predicted_R1': str(p),
             'predicted_R2': str(p), 'predicted_R4': str(p), 'target': str(t)}
            for i, a, p, t in zip(ids, area, pred, [2., 0., 0.])]
    permuted = [dict(r, target=str(t)) for r, t in zip(rows, [0., 2., 0.])]
    assert payload_sha(allowed_selection_rows(rows)) == payload_sha(allowed_selection_rows(permuted))
    again = select_predicted_area(ids, area, pred, .75)
    assert again['selected_ids'] == result['selected_ids']
    # Predicted objective increases although this toy's true capture decreases.
    return {'passed': True, 'selector_signature': str(inspect.signature(select_predicted_area)),
            'unequal_cost_prefix_predicted_m2': 4.6, 'unequal_cost_milp_predicted_m2': 5.4,
            'label_permutation_selected_ids_equal': True, 'toy_predicted_milp_solution': result,
            'counterexample_true_captured_prefix_m2': 2., 'counterexample_true_captured_milp_m2': 0.}


def evaluate_selected(ids, selected, area, predicted, target):
    index = {i: j for j, i in enumerate(ids)}
    assert len(set(selected)) == len(selected) and set(selected).issubset(index)
    take = np.array([index[i] for i in selected], dtype=np.int64)
    A, T = math.fsum(area), math.fsum(target)
    a, t = math.fsum(area[take]), math.fsum(target[take])
    return {'selected_count': len(take), 'actual_area_m2': a, 'actual_area_fraction': a/A,
            'captured_target_m2': t, 'capture_fraction': t/T, 'missed_target_m2': T-t,
            'zero_annotation_area_m2': a-t,
            'selected_zero_target_unit_area_m2': math.fsum(area[take][target[take] == 0]),
            'predicted_objective_m2': math.fsum(predicted[take])}


def run():
    assert sha(PROTOCOL) == LOCKED_PROTOCOL_SHA
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8-sig'))
    assert tuple(protocol['management_grid_m']) == M_VALUES
    assert tuple(protocol['image_degradation_factors']) == R_VALUES
    assert tuple(protocol['budgets']) == BUDGETS
    parent = PA/protocol['parent_result']
    unit = parent/'unit_predictions.csv'
    fixed = parent/'fixed_budgets.csv'
    assert sha(unit) == protocol['unit_predictions_sha256']
    original_hashes = {str(f): sha(f) for f in (unit, fixed, parent/'inputs.json', parent/'summary.json', PROTOCOL)}
    output = PA/('RM_policy_diagnostic_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(); (output/'source').mkdir(); (output/'solutions').mkdir()
    for f in (Path(__file__), PROTOCOL):
        shutil.copyfile(f, output/'source'/f.name)
    shutil.copyfile(unit, output/'source'/'unit_predictions.csv')
    shutil.copyfile(fixed, output/'source'/'v1_fixed_budgets.csv')
    shutil.copyfile(parent/'inputs.json', output/'source'/'v1_inputs.json')
    inputs = {'created_local_before_solves': stamp(), 'status': 'post-hoc development diagnostic',
              'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(__file__),
              'original_input_hashes': original_hashes, 'parent_result': str(parent),
              'versions': {'python': __import__('sys').version, 'numpy': np.__version__, 'scipy': scipy.__version__},
              'solver': 'SciPy bundled HiGHS', 'solver_options': OPTIONS,
              'threadpool_info': threadpool_info(), 'threadpool_limit': 2, 'solver_threads': 1,
              'test_accessed': False, 'reserved_scale_accessed': False, 'gpu_used': False}
    write_json(output/'inputs.json', inputs)
    write_json(output/'status.json', {'state': 'running', 'stage': 'self_checks', 'output': str(output), 'pid': os.getpid()})
    print(json.dumps({'output': str(output)}, ensure_ascii=True), flush=True)
    start = time.time()
    try:
        checks = run_toys()
        write_json(output/'self_checks.json', checks)
        allowed = allowed_selection_rows(read_csv(unit))
        write_csv(output/'selection_inputs_without_annotations.csv', allowed)
        data = {}
        for M in M_VALUES:
            records = sorted([r for r in allowed if int(r['M']) == M], key=lambda r: r['id'])
            data[M] = {'ids': np.array([r['id'] for r in records]),
                       'area': np.array([float(r['area']) for r in records]),
                       'predicted': {R: np.array([float(r[f'predicted_R{R}']) for r in records]) for R in R_VALUES}}
        solutions = []
        # All choices precede annotation evaluation and loading any label reference.
        for M in M_VALUES:
            for R in R_VALUES:
                for b in BUDGETS:
                    d = data[M]
                    result = select_predicted_area(d['ids'], d['area'], d['predicted'][R], b)
                    result.update(M=M, R=R, completed_local=stamp(),
                                  selector_input_sha256=payload_sha({'ids': d['ids'].tolist(), 'area': d['area'].tolist(),
                                                                   'predicted': d['predicted'][R].tolist(), 'budget': b}))
                    result['file'] = f'solutions/M{M}_R{R}_B{int(b*100):03d}.json'
                    write_json(output/result['file'], result)
                    solutions.append(result)
                    write_json(output/'status.json', {'state': 'running', 'stage': 'label_free_selection',
                               'done': len(solutions), 'total': 45, 'seconds': time.time()-start, 'pid': os.getpid()})
                    print(json.dumps({'done': len(solutions), 'M': M, 'R': R, 'budget': b,
                                      'solver_status': result['solver_status'], 'gap': result['predicted_relative_gap'],
                                      'feasible': result['candidate_feasible_in_original_units'],
                                      'seconds': round(result['elapsed_seconds'], 3)}), flush=True)
        assert len(solutions) == 45
        # Annotation and v1 references are joined only after all selections exist.
        annotated = read_csv(unit)
        truth = {(int(r['M']), r['id']): float(r['target']) for r in annotated}
        reference = read_csv(fixed)
        label_refs = {(int(r['M']), float(r['budget'])): r for r in reference if r['method'] == 'label_milp'}
        prefix_refs = {(int(r['M']), int(r['R']), float(r['budget'])): r for r in reference if r['method'] == 'model_prefix'}
        rng = np.random.default_rng(271828)
        shuffled_labels = rng.permutation([r['target'] for r in annotated])
        permuted = [dict(r, target=t, q_true='REMOVED_FOR_INVARIANCE_CHECK') for r, t in zip(annotated, shuffled_labels)]
        assert payload_sha(allowed_selection_rows(annotated)) == payload_sha(allowed_selection_rows(permuted))
        # One real case is deterministically re-solved from the permuted rows' allowed projection.
        perm_allowed = allowed_selection_rows(permuted)
        perm_records = sorted([r for r in perm_allowed if int(r['M']) == 4], key=lambda r: r['id'])
        repeat = select_predicted_area([r['id'] for r in perm_records], [float(r['area']) for r in perm_records],
                                       [float(r['predicted_R1']) for r in perm_records], .1)
        original = next(s for s in solutions if (s['M'], s['R'], s['budget_fraction']) == (4, 1, .1))
        assert repeat['selected_ids'] == original['selected_ids']
        checks.update(real_labels_permuted_allowed_payload_unchanged=True,
                      real_M4_R1_B10_label_permutation_selected_ids_equal=True)
        write_json(output/'label_permutation_repeat_M4_R1_B10.json', repeat)
        rows = []
        for s in solutions:
            M, R, b = s['M'], s['R'], s['budget_fraction']
            d = data[M]; ids, area, pred = d['ids'], d['area'], d['predicted'][R]
            target = np.array([truth[(M, i)] for i in ids])
            assert np.all((target >= 0) & (target <= area+1e-9))
            chosen_prefix = prefix_ids(ids, area, pred, b)
            candidate = evaluate_selected(ids, s['selected_ids'], area, pred, target)
            baseline = evaluate_selected(ids, chosen_prefix, area, pred, target)
            old = prefix_refs[(M, R, b)]
            assert len(chosen_prefix) == int(old['k'])
            for our, prior in [('actual_area_m2', 'actual_area'), ('captured_target_m2', 'captured_target'),
                               ('capture_fraction', 'capture_fraction'), ('zero_annotation_area_m2', 'zero_annotation_area')]:
                assert math.isclose(baseline[our], float(old[prior]), rel_tol=1e-10, abs_tol=1e-9), (M, R, b, our)
            assert candidate['actual_area_m2'] <= s['budget_area_m2']
            assert math.isclose(candidate['predicted_objective_m2'], s['predicted_primal_m2'], rel_tol=1e-12, abs_tol=1e-12)
            objective_tol = max(1e-10, 1e-8*s['total_predicted_m2'])
            if s['optimality_proven_within_tolerance']:
                assert candidate['predicted_objective_m2'] >= baseline['predicted_objective_m2']-objective_tol
            if not s['candidate_feasible_in_original_units']:
                raise AssertionError('Solver candidate failed original-unit feasibility; retain audit, do not claim a policy result')
            curset, oldset = set(s['selected_ids']), set(chosen_prefix)
            xor = curset ^ oldset
            lookup = {i: j for j, i in enumerate(ids)}
            label = label_refs[(M, b)]
            row = {'M': M, 'R': R, 'budget': b, 'budget_area_m2': s['budget_area_m2'],
                   'solver_status': s['solver_status'], 'optimality_proven_within_tolerance': s['optimality_proven_within_tolerance'],
                   'predicted_dual_upper_m2': s['predicted_dual_upper_m2'], 'predicted_relative_gap': s['predicted_relative_gap'],
                   'solver_seconds': s['elapsed_seconds'], 'predicted_comparison_tolerance_m2': objective_tol,
                   **{f'pmilp_{k}': v for k, v in candidate.items()},
                   **{f'prefix_{k}': v for k, v in baseline.items()},
                   **{f'delta_{k}': candidate[k]-baseline[k] for k in candidate},
                   'delta_capture_percentage_points': 100*(candidate['capture_fraction']-baseline['capture_fraction']),
                   'pmilp_unused_budget_m2': s['budget_area_m2']-candidate['actual_area_m2'],
                   'prefix_unused_budget_m2': s['budget_area_m2']-baseline['actual_area_m2'],
                   'changed_ids_count': len(xor), 'added_ids_count': len(curset-oldset), 'removed_ids_count': len(oldset-curset),
                   'selected_id_jaccard': len(curset & oldset)/len(curset | oldset) if curset | oldset else 1.,
                   'changed_area_m2': math.fsum(area[[lookup[i] for i in sorted(xor)]]),
                   'label_reference_primal_target_m2': float(label['primal_target']),
                   'label_reference_dual_upper_m2': float(label['dual_target_bound']),
                   'label_reference_optimality_proven': label['optimality_proven'],
                   'gap_to_label_reference_primal_m2': float(label['primal_target'])-candidate['captured_target_m2'],
                   'solution_file': s['file']}
            rows.append(row)
            write_json(output/'solutions'/f'M{M}_R{R}_B{int(b*100):03d}_evaluation.json',
                       {'metrics': row, 'prefix_selected_ids': chosen_prefix, 'added_ids': sorted(curset-oldset),
                        'removed_ids': sorted(oldset-curset), 'label_reference_is_evaluation_only': True})
        checks.update(all_45_binary_original_area_objective_audits_passed=True,
                      all_matching_v1_prefix_metrics_reproduced=True,
                      proven_predicted_objectives_not_below_prefix_beyond_tolerance=True)
        write_json(output/'self_checks.json', checks)
        write_csv(output/'all_45_deltas.csv', rows)
        ten = [r for r in rows if r['budget'] == .1]
        write_csv(output/'comparison_10pct.csv', ten)
        deltas = np.array([r['delta_capture_percentage_points'] for r in rows])
        summary = {'status': 'completed', 'created_local': inputs['created_local_before_solves'], 'completed_local': stamp(),
                   'rows': len(rows), 'seconds': time.time()-start,
                   'solver_status_counts': {str(k): sum(s['solver_status'] == k for s in solutions) for k in sorted({s['solver_status'] for s in solutions})},
                   'optimal_within_tolerance_count': sum(s['optimality_proven_within_tolerance'] for s in solutions),
                   'true_capture_delta_pp_min': float(deltas.min()), 'true_capture_delta_pp_max': float(deltas.max()),
                   'true_capture_improved_rows': int((deltas > 1e-8).sum()),
                   'true_capture_worsened_rows': int((deltas < -1e-8).sum()),
                   'true_capture_unchanged_rows': int((np.abs(deltas) <= 1e-8).sum()),
                   'predicted_objective_delta_m2_min': min(r['delta_predicted_objective_m2'] for r in rows),
                   'predicted_objective_delta_m2_max': max(r['delta_predicted_objective_m2'] for r in rows),
                   'ten_percent_rows': ten,
                   'no_auc_computed': True, 'test_accessed': False, 'all_checks_passed': True,
                   'interpretation': 'Secondary post-hoc validation diagnostic; predicted-objective optimization can improve or reduce annotated target capture. No algorithm novelty or agronomic outcome claim.'}
        for path, expected in original_hashes.items():
            assert sha(path) == expected, 'Original input changed during diagnostic'
        assert sha(__file__) == inputs['source_sha256']
        write_json(output/'summary.json', summary)
        lines = ['# P-MILP 开发诊断', '', '选择先于标签评价；45 组完整组合，无 AUC 插值。原 v1 保持不变。', '',
                 '| M (m) | R | Prefix 捕获 (%) | P-MILP 捕获 (%) | Δ (pp) | Prefix 面积 (m²) | P-MILP 面积 (m²) |',
                 '|---|---|---|---|---|---|---|']
        for r in ten:
            lines.append(f"| {r['M']} | {r['R']} | {100*r['prefix_capture_fraction']:.6f} | {100*r['pmilp_capture_fraction']:.6f} | {r['delta_capture_percentage_points']:+.6f} | {r['prefix_actual_area_m2']:.6f} | {r['pmilp_actual_area_m2']:.6f} |")
        lines += ['', f"全部 45 行真实捕获：改善 {summary['true_capture_improved_rows']}，下降 {summary['true_capture_worsened_rows']}，不变 {summary['true_capture_unchanged_rows']}（±1e-8 pp 判等）。",
                  f"真实捕获差范围 [{deltas.min():.6f}, {deltas.max():.6f}] pp；{summary['optimal_within_tolerance_count']}/45 在记录的 1e-8 gap 容差内证明最优。",
                  '', '本诊断在查看 v1 后提出，是后验开发分析。选择目标是预测标注面积；实际标注捕获改善并非保证。zero_annotation_area 沿用 v1 的 A−T，另列 selected_zero_target_unit_area 区分整单元无目标的面积。']
        (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        products = {str(f.relative_to(output)): sha(f) for f in sorted(output.rglob('*')) if f.is_file() and f.name != 'status.json'}
        write_json(output/'product_hashes.json', products)
        write_json(output/'status.json', {'state': 'completed', 'rows': len(rows), 'seconds': time.time()-start,
                                        'product_hashes_sha256': sha(output/'product_hashes.json')})
        print(json.dumps({'completed': str(output), 'rows': len(rows), 'improved': summary['true_capture_improved_rows'],
                          'worsened': summary['true_capture_worsened_rows']}, ensure_ascii=True), flush=True)
        return output
    except BaseException as error:
        write_json(output/'status.json', {'state': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
