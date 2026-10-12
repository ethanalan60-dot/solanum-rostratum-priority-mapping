"""Preparation-stage local confirmation allocation; no execution without a root freeze.

Each of 24 segmentation fits has 60 cases and two policies. Only A/P enter
selection. All 24 local seals AND the global 30-model acceptance precede truth.
The CLI is a dedicated CPU process; no model forward or truth loader is present.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json, math, os, sys, traceback, types
import core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MS = (1, 2, 4)
RS = (1, 2, 4, 6)
BS = (.1, .2, .3, .5, 1.)
CASES = tuple((m, r, b) for m in MS for r in RS for b in BS)
FIELDS = ('M', 'id', 'area', 'predicted_R1', 'predicted_R2', 'predicted_R4', 'predicted_R6')
PINS = {
    'evaluate_common_management_v2.py': '99315d78bd31e721f024601227e032291962075336ed65a7e9bd157caf643d25',
    'evaluate_management.py': '29d7b8a8bd7e6762cdae18bde2d491ecbe9cd459bf401ef084a9f827e990ff8d',
    'geo_management.py': '16c6fb38e1c2c1c9c5ea9f5f25db317d0fa021505631ddef9f07dea528549d8f',
    'diagnose_predicted_knapsack.py': 'f45f3210bbef82aa3e0dcfcec58a2b06d2704164c50417819d982660c6674750',
    'recover_common_policy.py': '7fdc3fac309b8b6e98d7c53934ad676fc305670643fd933f5d762443ca550225',
    'audit_common_bound_consistency.py': '6b986367f9dc5ddc66f63baa26c408201110db21e39f9cb77e9bc2f9f6e3e183',
}


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def local_models(contract):
    rows = [x for x in contract['roster'] if x['dataset'] == 'local' and x['variant'] != 'regression']
    core.require(len(rows) == 24 and len({x['run_id'] for x in rows}) == 24, 'Fixed 24 local segmentation fits required')
    return rows


def extraction(path, names, namespace, expand_count=False):
    """Compile named function nodes only; never execute another file's top level."""
    source = Path(path).read_text(encoding='utf-8-sig'); tree = ast.parse(source)
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    core.require({x.name for x in nodes} == set(names), 'Extraction inventory differs')
    ledger = {x.name: hashlib.sha256(ast.get_source_segment(source, x).encode()).hexdigest() for x in nodes}
    changes = []
    class Expand(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 45:
                changes.append({'line': node.lineno, 'old': 45, 'new': 60}); node.value = 60
            elif node.value == 'all_45_final_choices_saved':
                changes.append({'line': node.lineno, 'old': node.value, 'new': 'all_60_final_choices_saved'})
                node.value = 'all_60_final_choices_saved'
            return node
    module = ast.Module(body=nodes, type_ignores=[])
    if expand_count: module = Expand().visit(module)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return {'source': core.binding(path), 'function_segment_sha256': ledger, 'count_only_AST_changes': changes}


def dependencies():
    """Lazy scientific dependencies. The side-effectful bound module is NEVER imported."""
    for name, digest in PINS.items(): core.bound_file(ROOT/name, digest)
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy
    from scipy import sparse
    import evaluate_management as ev
    import geo_management as geo
    import diagnose_predicted_knapsack as policy
    import recover_common_policy as recovery
    namespace = dict(Path=Path, math=math, traceback=traceback, np=np, ev=ev, policy=policy,
                     recovery=recovery, FROZEN=PINS, CASES=CASES, FIELDS=FIELDS,
                     __file__=str(ROOT/'evaluate_common_management_v2.py'))
    def prepare_data(allowed, domain_area):
        core.require(allowed and all(set(row) == set(FIELDS) for row in allowed), 'A/P whitelist differs')
        core.require({int(x['M']) for x in allowed} == set(MS), 'M inventory differs')
        result = {}
        for m in MS:
            rows = sorted((x for x in allowed if int(x['M']) == m), key=lambda x: x['id'])
            ids = [x['id'] for x in rows]; area = np.array([float(x['area']) for x in rows])
            pred = {r: np.array([float(x[f'predicted_R{r}']) for x in rows]) for r in RS}
            for r in RS: recovery.validate_arrays(ids, area, pred[r], .1)
            core.require(abs(math.fsum(area)-domain_area)/domain_area < 1e-9, 'Nested area mismatch')
            result[m] = (ids, area, pred)
        return result
    namespace['prepare_data'] = prepare_data
    reuse = extraction(ROOT/'evaluate_common_management_v2.py',
        ('relative_path', 'file_hashes', 'save_then_recover', 'encode_solver_value',
         'capture_selector_call', 'choose_all', 'verify_selection_seal'), namespace, True)
    bounds = {'math': math}
    bound_ledger = extraction(ROOT/'audit_common_bound_consistency.py', ('finite', 'audit_bounds'), bounds)
    metrics = {'np': np, 'math': math}
    metric_ledger = extraction(ROOT/'evaluate_management.py', ('ordered_prefix', 'step_auc'), metrics)
    return types.SimpleNamespace(np=np, scipy=scipy, sparse=sparse, geo=geo, policy=policy,
        recovery=recovery, v=types.SimpleNamespace(**namespace), audit_bounds=bounds['audit_bounds'],
        ordered_prefix=metrics['ordered_prefix'], step_auc=metrics['step_auc'],
        ledger={'selection': reuse, 'bound': bound_ledger, 'metrics': metric_ledger})


def validate_geometry(folder):
    core.completed_stage(folder); metadata = core.read(folder/'metadata.json')
    core.require(metadata['geometry_kind'] == 'native_alpha_only' and metadata['split'] == 'test_sealed', 'Wrong geometry domain')
    core.require(metadata['complete_declared_domain'] and metadata['expected_windows'] == 338, 'Partial geometry refused')
    core.require(len(metadata['windows']) == len({x['id'] for x in metadata['windows']}) == 338, 'Geometry member identity incomplete')
    core.require(metadata['annotation_arrays_opened'] == 0 and metadata['mask_paths_consumed'] is False, 'Geometry annotation boundary')
    core.require(metadata['crs'] == 'EPSG:32649' and metadata['base_grid_m'] == 1, 'Wrong metric geometry')
    core.require(metadata['native_pixels_per_prediction_cell'] == [2, 2], 'Prediction support changed')
    return metadata


def integrate_prediction(maps, geometry, folder, deps):
    """A/P-only sparse integration. This function accepts no target arguments."""
    np = deps.np; windows = geometry['windows']
    core.require([x['id'] for x in maps] == [x['id'] for x in windows], 'Prediction/geometry member order differs')
    all_cells = []; all_values = []; audits = []
    for record, window in zip(maps, windows):
        core.require(record['R'] == list(RS) and record['source_id'] == window['source_id'], 'Wrong map/source')
        core.require(record['native_window_xywh'] == [window['x'], window['y'], 512, 512], 'Native window changed')
        weight_path = core.resolve_under(folder, window['file']); core.bound_file(weight_path, window['sha256'])
        core.bound_file(record['path'], record['sha256'])
        with np.load(record['path'], allow_pickle=False) as z:
            core.require(set(z.files) == {'probabilities', 'valid', 'R', 'id'}, 'Unexpected prediction arrays')
            p, v = z['probabilities'], z['valid']
            core.require(str(z['id'].item()) == record['id'] and z['R'].tolist() == list(RS), 'NPZ identity differs')
        core.require(p.shape == (4, 256, 256) and p.dtype == np.float32 and np.isfinite(p).all() and ((p >= 0)&(p <= 1)).all(), 'Invalid p')
        core.require(v.shape == (256, 256) and v.dtype == np.float32 and np.isfinite(v).all(), 'Invalid validity')
        with np.load(weight_path, allow_pickle=False) as z:
            cells, area, raw = z['cells'], z['area'], z['raw_valid_by_support']
            weight = deps.sparse.csr_matrix((z['weights'], z['indices'], z['indptr']), shape=tuple(z['shape']))
            native_valid = z['raw_valid_by_native'].reshape(512, 512) > 0
        core.require(np.array_equal(v, native_valid.reshape(256, 2, 256, 2).mean(axis=(1, 3)).astype(np.float32)), 'Alpha/cache validity mismatch')
        core.require(weight.shape == (len(cells), 65536) and cells.shape == (len(area), 2), 'Wrong CSR support')
        core.require(np.isfinite(weight.data).all() and (weight.data > 0).all() and np.isfinite(area).all() and (area > 0).all(), 'Invalid area/weights')
        predicted = np.asarray(weight @ p.astype(np.float64).reshape(4, -1).T)
        expected = p.astype(np.float64).reshape(4, -1) @ raw
        obtained = np.array([math.fsum(predicted[:, j]) for j in range(4)])
        rel = np.abs(obtained-expected)/np.maximum(np.abs(expected), 1e-30)
        core.require((rel < 1e-9).all(), 'Native support prediction integral differs')
        core.require(abs(math.fsum(area)-math.fsum(raw))/math.fsum(raw) < 1e-9, 'Area conservation failure')
        core.require(np.all(predicted >= 0) and np.all(predicted <= area[:, None]+1e-9), 'Predicted area outside valid support')
        all_cells.append(cells); all_values.append(np.column_stack((area, predicted)))
        audits.append({'id': record['id'], 'relative_error_by_R': rel.tolist(), 'geometry_sha256': window['sha256'], 'map_sha256': record['sha256']})
        core.bound_file(weight_path, window['sha256']); core.bound_file(record['path'], record['sha256'])
    cells, inverse = np.unique(np.concatenate(all_cells), axis=0, return_inverse=True)
    values = np.concatenate(all_values)
    combined = np.column_stack([np.bincount(inverse, weights=values[:, k], minlength=len(cells)) for k in range(5)])
    reference = {(int(x['M']), x['id']): x for x in read_rows(folder/'area_units.csv')}
    rows = []; nested = {}
    for m in MS:
        keys, result = deps.geo.aggregate_nested(cells, combined, m); nested[m] = (keys, result)
        for (ix, iy), values in zip(keys, result):
            ident = f'M{m}_E{int(ix)}_N{int(iy)}'; ref = reference.pop((m, ident))
            core.require(int(ref['ix']) == ix and int(ref['iy']) == iy and math.isclose(values[0], float(ref['area_m2']), rel_tol=1e-10, abs_tol=1e-9), 'A-only unit identity changed')
            rows.append({'M': m, 'id': ident, 'area': float(values[0]), **{f'predicted_R{r}': float(values[j+1]) for j, r in enumerate(RS)}})
    core.require(not reference, 'Missing geometry unit')
    keys, via = deps.geo.aggregate_nested(nested[2][0], nested[2][1], 2)
    core.require(np.array_equal(keys, nested[4][0]) and np.allclose(via, nested[4][1], rtol=1e-10, atol=1e-9), 'Nested 1/2/4 integration differs')
    return rows, {'window_count': len(windows), 'per_window': audits, 'target_arrays_read': 0, 'native_relative_tolerance': 1e-9}


def strict_prefix(deps, ids, area, pred, budget):
    chosen = deps.policy.prefix_ids(ids, area, pred, budget); index = {s: j for j, s in enumerate(ids)}
    original = list(chosen); limit = budget*math.fsum(area)
    while math.fsum(area[index[s]] for s in chosen) > limit: chosen.pop()
    return {'selected_ids': chosen, 'area_m2': math.fsum(area[index[s]] for s in chosen),
            'predicted_m2': math.fsum(pred[index[s]] for s in chosen), 'budget_area_m2': limit,
            'tail_trim_count': len(original)-len(chosen), 'strict_original_budget': True}


def select_arrays(out, allowed, domain_area, context, deps, selector=None):
    """Internal pure A/P orchestration, also exercised with a synthetic backend."""
    core.require(set(context) == {'run_id', 'freeze_sha256', 'all_predictions_sha256', 'geometry_metadata_sha256', 'checkpoint_sha256'}, 'Selection context whitelist')
    data = deps.v.prepare_data(allowed, domain_area)
    core.write(out/'extraction_ledger.json', deps.ledger)
    policy_folder = out/'policy'; policy_folder.mkdir()
    deps.v.choose_all(policy_folder, allowed, domain_area, context, selector=selector or deps.policy.select_predicted_area)
    deps.v.verify_selection_seal(policy_folder)
    prefixes = []; rankings = []; bounds = []
    for m in MS:
        ids, area, pred = data[m]
        for r in RS:
            order = deps.np.lexsort((deps.np.asarray(ids), -pred[r]/area))
            rankings.append({'M': m, 'R': r, 'ordered_ids': [ids[i] for i in order], 'ASCII_ties': True})
    for m, r, budget in CASES:
        name = deps.recovery.case_name((m, r, budget)); choice = core.read(policy_folder/'choices'/name)
        ids, area, pred = data[m]; index = {s: j for j, s in enumerate(ids)}
        prefix = strict_prefix(deps, ids, area, pred[r], budget)
        final = choice['selected_ids']; a = math.fsum(area[index[s]] for s in final); p = math.fsum(pred[r][index[s]] for s in final)
        core.require(a <= budget*math.fsum(area), 'Strict candidate budget violation')
        deps.recovery.close(a, choice['selected_area_m2']); deps.recovery.close(p, choice['predicted_primal_m2'])
        bound = deps.audit_bounds(p, prefix['predicted_m2'], True, math.fsum(pred[r]),
            choice['raw_candidate_audit']['raw_solver_upper_m2'], choice['recomputed_bound']['upper_m2'],
            choice['optimality_proven'], choice['fallback_used'], deps.policy.OPTIONS['mip_rel_gap'])
        prefixes.append({'M': m, 'R': r, 'budget': budget, **prefix})
        bounds.append({'M': m, 'R': r, 'budget': budget, 'selected_ids': final, **bound,
                       'choice': core.binding(policy_folder/'choices'/name)})
    core.write(out/'prefix_choices.json', prefixes); core.write(out/'rankings.json', rankings); core.write(out/'bound_audit.json', bounds)
    core.write(out/'selection_seal.json', {'version': 'confirmation_local_selection_v1', **context,
        'case_order': [list(x) for x in CASES], 'case_count': 120, 'P_solver_calls': 60,
        'prefix_count': 60, 'ranking_count': 12, 'truth_used_for_choices': False,
        'selected_ids_unchanged_by_bound_audit': True, 'implementation_sha256': core.sha(__file__),
        'files': {name: core.binding(out/name) for name in ('prefix_choices.json', 'rankings.json', 'bound_audit.json', 'extraction_ledger.json', 'policy/selection_seal.json')},
        'A_P_only_inputs': core.binding(policy_folder/'selection_inputs_without_annotations.csv')})
    return verify_model_selection(out, deps)


def verify_model_selection(out, deps):
    seal = core.read(out/'selection_seal.json')
    core.require(seal['version'] == 'confirmation_local_selection_v1' and seal['case_order'] == [list(x) for x in CASES], 'Selection inventory differs')
    core.require(seal['case_count'] == 120 and seal['P_solver_calls'] == 60 and seal['truth_used_for_choices'] is False, 'Selection stage incomplete')
    core.require(seal['implementation_sha256'] == core.sha(__file__), 'Selection implementation differs')
    core.require(set(seal['files']) == {'prefix_choices.json', 'rankings.json', 'bound_audit.json', 'extraction_ledger.json', 'policy/selection_seal.json'}, 'Selection binding inventory differs')
    for item in [*seal['files'].values(), seal['A_P_only_inputs']]: core.bound_file(item['path'], item['sha256'])
    inner = deps.v.verify_selection_seal(out/'policy')
    core.require(inner['context'] == {k: seal[k] for k in ('run_id', 'freeze_sha256', 'all_predictions_sha256', 'geometry_metadata_sha256', 'checkpoint_sha256')}, 'Inner/outer model binding differs')
    for filename in ('prefix_choices.json', 'bound_audit.json'):
        records = core.read(out/filename)
        core.require([(x['M'], x['R'], x['budget']) for x in records] == list(CASES), 'Missing/duplicate selection case')
    ranking = core.read(out/'rankings.json')
    core.require([(x['M'], x['R']) for x in ranking] == [(m, r) for m in MS for r in RS], 'Ranking population differs')
    return seal


def execution_gate(run_id, root_review):
    contract, models, freeze = core.validate_freeze(); roster = local_models(contract)
    core.require(run_id in [x['run_id'] for x in roster], 'Undeclared/local-regression model refused')
    from pipeline import validate_global_predictions
    predictions = validate_global_predictions()
    review = core.read(root_review)
    core.require(review.get('passed') is True and review.get('scope') == 'confirmation_local_management', 'Root management execution review required')
    core.require(review['implementation_sha256'] == core.sha(__file__) and review['freeze_sha256'] == core.sha(HERE/'FREEZE.json'), 'Root review/source/freeze mismatch')
    core.require(review['run_ids'] == [x['run_id'] for x in roster] and review['case_count_per_model'] == 120, 'Root review population differs')
    core.require(review['all_predictions_sha256'] == core.sha(HERE/'stages'/'predictions_seal'/'all_predictions.json'), 'Prediction seal review differs')
    folder = HERE/'stages'/'local_geometry'; metadata = validate_geometry(folder)
    core.require(review['geometry_metadata_sha256'] == core.sha(folder/'metadata.json'), 'Unreviewed geometry')
    entry = next(x for x in models if x['run_id'] == run_id)
    predicted = next(x for x in predictions['models'] if x['run_id'] == run_id)
    return roster, entry, predicted, metadata


def execute_run(run_id, root_review):
    initial_implementation_sha256 = core.sha(__file__)
    roster, entry, predicted, geometry = execution_gate(run_id, root_review)
    slot = [x['run_id'] for x in roster].index(run_id)+1
    with core.attempt(f'local_select_{slot:02d}') as out:
        core.configure_cpu(out); deps = dependencies()
        from threadpoolctl import threadpool_limits
        watched = {str(Path(root_review).resolve()): core.sha(root_review), str(HERE/'FREEZE.json'): core.sha(HERE/'FREEZE.json'),
                   str(HERE/'SOURCE_LOCK.json'): core.sha(HERE/'SOURCE_LOCK.json'), str(HERE/'stages'/'predictions_seal'/'all_predictions.json'): core.sha(HERE/'stages'/'predictions_seal'/'all_predictions.json')}
        watched.update({str(Path(__file__).resolve()): initial_implementation_sha256, **{str(ROOT/name): digest for name, digest in PINS.items()}})
        bind_stage_identity(watched, HERE/'stages'/'local_geometry', ('metadata.json', 'artifact_manifest.json', 'area_units.csv'))
        bind_stage_identity(watched, Path(predicted['stage']), ('metadata.json', 'map_index.json', 'artifact_manifest.json'))
        core.write(out/'execution_binding.json', {'watched': watched, 'root_review': core.binding(root_review), 'prediction': predicted,
                   'geometry_artifact': core.binding(HERE/'stages'/'local_geometry'/'artifact_manifest.json'), 'original_checkpoint': core.binding(entry['checkpoint_path'])})
        with threadpool_limits(limits=2):
            allowed, audit = integrate_prediction(core.read(predicted['map_index']['path']), geometry, HERE/'stages'/'local_geometry', deps)
            core.write(out/'integration.json', audit)
        context = {'run_id': run_id, 'freeze_sha256': core.sha(HERE/'FREEZE.json'),
                   'all_predictions_sha256': core.sha(HERE/'stages'/'predictions_seal'/'all_predictions.json'),
                   'geometry_metadata_sha256': core.sha(HERE/'stages'/'local_geometry'/'metadata.json'), 'checkpoint_sha256': entry['checkpoint_sha256']}
        with threadpool_limits(limits=1): select_arrays(out, allowed, geometry['area_m2'], context, deps)
        verify_execution_hashes(watched)
        core.validate_source_lock(); core.completed_stage(predicted['stage']); validate_geometry(HERE/'stages'/'local_geometry')
    return out


def bind_stage_identity(watched, folder, names):
    """Retain initial consumed-stage bytes; completion revalidation alone is insufficient."""
    for name in names:
        path = core.resolve_under(folder, name)
        key = str(path); digest = core.sha(path)
        core.require(key not in watched or watched[key] == digest, 'Initial stage identity conflicts')
        watched[key] = digest


def verify_execution_hashes(watched):
    for path, digest in watched.items(): core.bound_file(path, digest)


def validate_population_identity(seal, row, model, freeze_sha, predictions_sha, geometry_sha):
    core.require(model['run_id'] == row['run_id'] == seal['run_id'] and seal['checkpoint_sha256'] == model['checkpoint_sha256'], 'Local checkpoint/model differs from final freeze')
    core.require(seal['freeze_sha256'] == freeze_sha and seal['all_predictions_sha256'] == predictions_sha and seal['geometry_metadata_sha256'] == geometry_sha, 'Local seal freeze/prediction/geometry differs')


def seal_population(root_review):
    contract, models, _ = core.validate_freeze(); roster = local_models(contract)
    by_model = {x['run_id']: x for x in models}
    execution_gate(roster[0]['run_id'], root_review)
    with core.attempt('local_selections_seal') as out:
        core.configure_cpu(out); deps = dependencies(); records = []
        for index, row in enumerate(roster, 1):
            folder = HERE/'stages'/f'local_select_{index:02d}'; core.completed_stage(folder)
            seal = verify_model_selection(folder, deps)
            validate_population_identity(seal, row, by_model[row['run_id']], core.sha(HERE/'FREEZE.json'), core.sha(HERE/'stages'/'predictions_seal'/'all_predictions.json'), core.sha(HERE/'stages'/'local_geometry'/'metadata.json'))
            records.append({'run_id': row['run_id'], 'case_count': 120, 'selection_seal': core.binding(folder/'selection_seal.json'), 'artifact_manifest': core.binding(folder/'artifact_manifest.json')})
        core.write(out/'all_local_selections.json', {'version': 'confirmation_all_local_selections_v1', 'models': records,
            'freeze_sha256': core.sha(HERE/'FREEZE.json'), 'all_predictions_sha256': core.sha(HERE/'stages'/'predictions_seal'/'all_predictions.json'),
            'P_solver_cases': 1440, 'policy_choices': 2880, 'truth_used_for_choices': False})
    return out


def truth_gate(choice_review):
    """Call before opening native labels or a target unit table; includes external models."""
    contract, _, _ = core.validate_freeze()
    from pipeline import validate_global_predictions, validate_choices
    validate_global_predictions(); accepted = validate_choices(choice_review)
    folder = HERE/'stages'/'local_selections_seal'; core.completed_stage(folder)
    seal = core.read(folder/'all_local_selections.json'); roster = local_models(contract)
    core.require([x['run_id'] for x in seal['models']] == [x['run_id'] for x in roster] and seal['P_solver_cases'] == 1440 and seal['policy_choices'] == 2880, 'Local selection population incomplete')
    core.require(seal['freeze_sha256'] == core.sha(HERE/'FREEZE.json') and seal['all_predictions_sha256'] == core.sha(HERE/'stages'/'predictions_seal'/'all_predictions.json'), 'Population seal drift')
    for item in seal['models']:
        for key in ('selection_seal', 'artifact_manifest'): core.bound_file(item[key]['path'], item[key]['sha256'])
        core.completed_stage(Path(item['selection_seal']['path']).parent)
        reviewed = next(x for x in accepted['models'] if x['run_id'] == item['run_id'])
        core.require(reviewed['selection_seal'] == item['selection_seal'], 'Accepted global/local selection seals differ')
    return seal


def evaluate_joined_arrays(selection_folder, truth_rows, deps):
    """Internal post-gate arithmetic. The caller supplies exact native-grid T, never coarse means.

    Production truth loading/reference construction is intentionally a separate, later
    reviewed stage. This internal helper is currently exercised with synthetic rows only.
    """
    seal = verify_model_selection(selection_folder, deps)
    data = deps.v.prepare_data(read_rows(selection_folder/'policy'/'selection_inputs_without_annotations.csv'),
        sum(float(x['area']) for x in read_rows(selection_folder/'policy'/'selection_inputs_without_annotations.csv') if int(x['M']) == 1))
    truth = {(int(x['M']), x['id']): float(x['target_m2']) for x in truth_rows}
    core.require(len(truth) == len(truth_rows) == sum(len(x[0]) for x in data.values()), 'Truth unit inventory differs')
    totals = []; curves = []; auc = []; outcomes = []
    rankings = {(x['M'], x['R']): x['ordered_ids'] for x in core.read(selection_folder/'rankings.json')}
    prefixes = {(x['M'], x['R'], x['budget']): x for x in core.read(selection_folder/'prefix_choices.json')}
    bounds = {(x['M'], x['R'], x['budget']): x for x in core.read(selection_folder/'bound_audit.json')}
    for m in MS:
        ids, area, predictions = data[m]; index = {s: j for j, s in enumerate(ids)}
        target = deps.np.array([truth[(m, s)] for s in ids]); total = math.fsum(target); totals.append(total)
        core.require(deps.np.isfinite(target).all() and (target >= 0).all() and (target <= area+1e-9).all(), 'Invalid native-grid target area')
        for r in RS:
            order = deps.np.array([index[s] for s in rankings[m, r]], dtype=int)
            curve = deps.ordered_prefix(area, target, order)
            for k in range(len(ids)+1):
                curves.append({'M': m, 'R': r, 'k': k, 'actual_area_m2': float(curve['area'][k]),
                    'area_fraction': float(curve['area_fraction'][k]), 'target_m2': float(curve['target'][k]),
                    'capture_fraction': float(curve['capture_fraction'][k]) if total else None})
            auc.append({'M': m, 'R': r, 'auc_0_1': deps.step_auc(curve), 'auc_0_half_normalized': deps.step_auc(curve, .5)})
            for budget in BS:
                case = (m, r, budget)
                for policy, selected in [('model_prefix', prefixes[case]['selected_ids']), ('predicted_area', bounds[case]['selected_ids'])]:
                    a = math.fsum(area[index[s]] for s in selected); t = math.fsum(target[index[s]] for s in selected)
                    core.require(a <= budget*math.fsum(area), 'Post-truth budget drift')
                    outcomes.append({'run_id': seal['run_id'], 'M': m, 'R': r, 'budget': budget, 'policy': policy,
                        'actual_area_m2': a, 'unused_area_m2': budget*math.fsum(area)-a,
                        'captured_target_m2': t, 'missed_target_m2': total-t, 'capture_fraction': t/total if total else None,
                        'selected_non_target_area_m2': a-t, 'selected_ids': selected})
    core.require(all(math.isclose(totals[0], x, rel_tol=1e-10, abs_tol=1e-9) for x in totals), 'Target total changed across M')
    return {'outcomes': outcomes, 'full_prefix_curves': curves, 'auc': auc, 'shared_reference_required': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-review', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--select-run'); group.add_argument('--seal-population', action='store_true')
    args = parser.parse_args()
    print(execute_run(args.select_run, args.root_review) if args.select_run else seal_population(args.root_review))


if __name__ == '__main__': main()
