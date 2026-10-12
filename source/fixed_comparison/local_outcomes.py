"""Post-seal local confirmation truth, model outcomes and one shared reference set.

Preparation only until a complete freeze and explicit root review exist. No old
entry point is called; native labels are opened only after all global gates.
"""
from pathlib import Path
import argparse, ast, hashlib, json, math, time, types, warnings
import core
import local_management as management

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RANDOM_SEED = 20260916
RANDOM_ORDERS = 1000


def dependencies():
    d = management.dependencies()
    source = ROOT/'evaluate_management.py'
    namespace = dict(np=d.np, math=math, time=time, warnings=warnings,
        milp=d.scipy.optimize.milp, Bounds=d.scipy.optimize.Bounds,
        LinearConstraint=d.scipy.optimize.LinearConstraint, RANDOM_SEED=RANDOM_SEED)
    text = source.read_text(encoding='utf-8-sig'); tree = ast.parse(text)
    names = ('safe_number', 'solve_label_knapsack')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    core.require({n.name for n in nodes} == set(names), 'Label function extraction differs')
    segments = {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest() for n in nodes}
    changes = []
    class Adapter(ast.NodeTransformer):
        def visit_Dict(self, node):
            self.generic_visit(node)
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'threads':
                    core.require(isinstance(node.values[i], ast.Constant) and node.values[i].value == 2, 'Unexpected label resource policy')
                    node.values[i] = ast.Constant(1); changes.append('label_solver_threads_2_to_1')
            return node
        def visit_If(self, node):
            if len(node.body) == 2 and isinstance(node.body[0], ast.Assert) and isinstance(node.body[0].msg, ast.Constant) and node.body[0].msg.value == 'MILP dual below feasible solution':
                # Original selected IDs stay unchanged. Large inversion is retained
                # in raw_solver_capture_upper and handled by the uniform audit.
                node.body = ast.parse('upper = capture if capture-upper <= 1e-10 else 1.0').body
                changes.append('large_raw_upper_inversion_uses_totalT_pending_uniform_downgrade')
            return self.generic_visit(node)
    module = Adapter().visit(ast.Module(body=nodes, type_ignores=[]))
    core.require(changes == ['label_solver_threads_2_to_1', 'large_raw_upper_inversion_uses_totalT_pending_uniform_downgrade'], 'Unexpected AST adapter change inventory')
    exec(compile(ast.fix_missing_locations(module), str(source)+':label_reference_adapter', 'exec'), namespace)
    d.label_namespace = namespace
    d.label_ledger = {'source': core.binding(source), 'function_segment_sha256': segments, 'explicit_AST_adapters': changes,
        'unchanged_options': {'time_limit': 30., 'mip_rel_gap': 1e-8, 'presolve': True, 'random_seed': RANDOM_SEED, 'mip_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9},
        'original_threads': 2, 'adapted_threads': 1, 'selected_IDs_never_changed_by_bound_amendment': True}
    return d


def native_target(mask, x, y, valid, np):
    """Frozen prepare_local/geo rule: grayscale >0, pad false, intersect alpha."""
    core.require(mask.ndim == 2 and valid.shape == (512, 512) and valid.dtype == np.bool_, 'Native label/validity shape')
    core.require(type(x) is int and type(y) is int and x >= 0 and y >= 0, 'Invalid native origin')
    core.require(np.issubdtype(mask.dtype, np.integer) or mask.dtype == np.bool_, 'PNG native labels must be integral grayscale')
    target = np.zeros((512, 512), dtype=bool)
    height = min(512, mask.shape[0]-y); width = min(512, mask.shape[1]-x)
    if height > 0 and width > 0: target[:height, :width] = mask[y:y+height, x:x+width] > 0
    target &= valid
    return target


def join_native_truth(truth, geometry, folder, summary, d, mask_loader):
    """Internal post-gate join; mask_loader is injectable for artificial labels."""
    np = d.np; windows = geometry['windows']; rows = truth['records']
    core.require([r['id'] for r in rows] == [w['id'] for w in windows], 'Native truth/geometry membership differs')
    sources = {x['source_id']: x for x in summary['sources']}
    headers = {x['source_id']: x for x in geometry['source_headers']}
    all_cells = []; all_values = []; audits = []; previous = None; mask = None
    for row, window in zip(rows, windows):
        native = row['native_reference']; sid = window['source_id']; source = sources[sid]
        core.require(set(native) == {'mask_source', 'source_id', 'native_window_xywh'}, 'Native truth reference whitelist differs')
        core.require(native['source_id'] == sid and native['native_window_xywh'] == [window['x'], window['y'], 512, 512], 'Truth native window swapped')
        core.require(Path(native['mask_source']).resolve() == Path(source['mask']).resolve(), 'Mask not in original bound source summary')
        if previous != sid:
            mask = None
            mask = mask_loader(source, headers[sid]); previous = sid
        core.require(mask.shape == (headers[sid]['height'], headers[sid]['width']), 'Mask/source dimensions differ')
        path = core.resolve_under(folder, window['file']); core.bound_file(path, window['sha256'])
        with np.load(path, allow_pickle=False) as z:
            cells, area, raw = z['cells'], z['area'], z['raw_valid_by_native']
            weight = d.sparse.csr_matrix((z['native_weights'], z['native_indices'], z['native_indptr']), shape=tuple(z['native_shape']))
        core.require(weight.shape == (len(area), 262144) and cells.shape == (len(area), 2), 'Native CSR shape differs')
        core.require(raw.shape == (262144,) and np.isfinite(raw).all() and (raw >= 0).all(), 'Invalid native valid-area weights')
        core.require(np.isfinite(weight.data).all() and (weight.data > 0).all() and np.isfinite(area).all() and (area > 0).all(), 'Invalid native overlaps')
        core.require(np.issubdtype(cells.dtype, np.integer) and len(np.unique(cells, axis=0)) == len(cells), 'Duplicate/noninteger geometry cells')
        core.require(np.allclose(np.asarray(weight.sum(axis=1)).ravel(), area, rtol=1e-9, atol=1e-14), 'Native CSR row area changed')
        core.require(np.max(np.abs(np.asarray(weight.sum(axis=0)).ravel()-raw)) <= 1e-9*math.fsum(raw), 'Native column conservation differs')
        valid = (raw > 0).reshape(512, 512); target = native_target(mask, window['x'], window['y'], valid, np)
        integrated = np.asarray(weight @ target.ravel().astype(np.float64)); expected = float(raw @ target.ravel())
        got = math.fsum(integrated); relative = abs(got-expected)/max(abs(expected), 1e-30)
        core.require(relative < 1e-9 and (integrated >= 0).all() and (integrated <= area+1e-9).all(), 'Native target conservation failed')
        all_cells.append(cells); all_values.append(np.column_stack((area, integrated)))
        audits.append({'id': row['id'], 'source_id': sid, 'mask_sha256': source['mask_sha256'], 'geometry_sha256': window['sha256'],
            'native_valid_pixels': int(valid.sum()), 'native_target_pixels': int(target.sum()), 'area_m2': math.fsum(area), 'target_m2': got, 'independent_native_target_m2': expected, 'target_relative_error': relative})
        core.bound_file(path, window['sha256'])
    mask = None
    cells, values = d.geo.aggregate_nested(np.concatenate(all_cells), np.concatenate(all_values), 1)
    reference = {(int(x['M']), x['id']): x for x in management.read_rows(folder/'area_units.csv')}
    unit = []; totals = []; nested = {}
    for m in management.MS:
        keys, result = d.geo.aggregate_nested(cells, values, m); nested[m] = (keys, result)
        totals.append([math.fsum(result[:, j]) for j in (0, 1)])
        for (ix, iy), (area, target) in zip(keys, result):
            ident = f'M{m}_E{int(ix)}_N{int(iy)}'; old = reference.pop((m, ident))
            core.require(int(old['ix']) == ix and int(old['iy']) == iy and math.isclose(area, float(old['area_m2']), rel_tol=1e-10, abs_tol=1e-9), 'Truth unit A differs from sealed geometry')
            unit.append({'M': m, 'id': ident, 'ix': int(ix), 'iy': int(iy), 'area_m2': float(area), 'target_m2': float(target)})
    core.require(not reference, 'Incomplete truth unit domain')
    core.require(all(math.isclose(totals[0][j], x[j], rel_tol=1e-10, abs_tol=1e-9) for x in totals for j in (0, 1)), 'Truth totals differ across nested M')
    keys, via = d.geo.aggregate_nested(nested[2][0], nested[2][1], 2)
    core.require(np.array_equal(keys, nested[4][0]) and np.allclose(via, nested[4][1], rtol=1e-10, atol=1e-9), 'Native truth 1->2->4 differs')
    core.require(math.isclose(totals[0][0], geometry['area_m2'], rel_tol=1e-9, abs_tol=1e-9), 'New confirmation area differs')
    return unit, {'windows': audits, 'area_m2': totals[0][0], 'target_m2': totals[0][1],
        'native_valid_pixels': sum(x['native_valid_pixels'] for x in audits), 'native_target_pixels': sum(x['native_target_pixels'] for x in audits),
        'target_source': 'native binary mask>0 intersect native alpha; native CSR integration', 'validation_T_reused': False}


def strict_order_budget(area, target, order, budget, d):
    curve = d.ordered_prefix(area, target, order); limit = budget*math.fsum(area)
    count = int(d.np.searchsorted(curve['area'], limit, side='right')-1); original = count
    while math.fsum(area[order[:count]]) > limit: count -= 1
    a = math.fsum(area[order[:count]]); t = math.fsum(target[order[:count]]); total = math.fsum(target)
    return {'budget': budget, 'k': count, 'actual_area_m2': a, 'actual_area_fraction': a/math.fsum(area), 'budget_area_m2': limit,
        'captured_target_m2': t, 'missed_target_m2': total-t, 'capture_fraction': t/total if total else None,
        'selected_non_target_area_m2': a-t, 'unused_budget_m2': limit-a, 'prefix_tail_trim_count': original-count}


def label_case(out, m, ids, area, target, budget, prefix, d):
    """Exactly one backend call for positive T; save return before adapter auditing."""
    name = f'M{m}_B{int(budget*100):03d}'; ns = d.label_namespace; backend = ns['milp']; calls = 0
    def capture(*args, **kwargs):
        nonlocal calls
        core.require(calls == 0, 'Unexpected label retry'); calls += 1
        result = backend(*args, **kwargs)
        core.write(out/(name+'_solver_raw.json'), {'M': m, 'budget': budget, 'actual_calls': calls,
            'options': kwargs['options'], 'result': d.v.encode_solver_value(dict(result)), 'persisted_before_selector_audit': True})
        x = getattr(result, 'x', None)
        if x is not None and d.np.isfinite(x).all():
            core.require(d.np.all((x >= -1e-6)&(x <= 1+1e-6)), 'Out-of-binary-bound solver candidate; raw preserved')
        return result
    ns['milp'] = capture
    try:
        result = ns['solve_label_knapsack'](area, target, d.np.asarray(ids), budget, time_limit=30., relative_gap=1e-8)
        core.require(calls == (1 if math.fsum(target) > 0 else 0), 'Wrong label solver call count')
        core.write(out/(name+'_adapted_return.json'), result)
        index = {s: i for i, s in enumerate(ids)}; selected = result['selected_ids']
        core.require(len(selected) == len(set(selected)) and set(selected) <= set(index), 'Invalid label selected IDs')
        a = math.fsum(area[index[s]] for s in selected); t = math.fsum(target[index[s]] for s in selected); total = math.fsum(target)
        core.require(a <= budget*math.fsum(area) and math.isclose(t, result['primal_target'], rel_tol=1e-12, abs_tol=1e-10), 'Strict label result infeasible/inconsistent')
        raw_upper = result.get('raw_solver_capture_upper'); raw_upper = raw_upper*total if raw_upper is not None else (0. if total == 0 else None)
        audit = d.audit_bounds(t, prefix['captured_target_m2'], True, total, raw_upper, result['dual_target_bound'], result['optimality_proven'], False, 1e-8)
        row = {'M': m, 'budget': budget, 'method': 'label_MILP_reference', 'selected_ids': selected, 'selected_ids_unchanged': True,
            'actual_area_m2': a, 'unused_budget_m2': budget*math.fsum(area)-a, 'captured_target_m2': t, 'missed_target_m2': total-t,
            'capture_fraction': t/total if total else None, 'target_m2': total, 'solver_calls': calls,
            'raw_solver_status': result['solver_status'], 'raw_solver_gap': result.get('raw_solver_relative_gap'),
            'candidate_feasible': result.get('candidate_feasible'), 'candidate_budget_violation': result.get('candidate_budget_violation'),
            'updated_upper_m2': audit['updated_upper_m2'], 'updated_relative_gap': audit['updated_relative_gap'] if total else None,
            'updated_proven_flag': audit['updated_proven_flag'], 'bound_anomaly': audit['bound_anomaly'], 'minimum_selected_area_claim': False,
            'bound_audit': audit, 'raw_return': core.binding(out/(name+'_adapted_return.json'))}
        core.write(out/(name+'_audited.json'), row); return row
    except BaseException as error:
        core.write(out/(name+'_failure.json'), {'error': repr(error), 'actual_solver_calls': calls, 'automatically_retried': False}); raise
    finally: ns['milp'] = backend


def shared_references(unit, out, d):
    """One domain reference set, independent of model count; no minimum-area goal."""
    np = d.np; out.mkdir(); (out/'label').mkdir(); fixed=[]; labels=[]; curves=[]; label_auc=[]; random_budget=[]; random_auc=[]; random_summary=[]
    for m in management.MS:
        rows = sorted((x for x in unit if int(x['M']) == m), key=lambda x: x['id'])
        ids = np.array([x['id'] for x in rows]); area=np.array([float(x['area_m2']) for x in rows]); target=np.array([float(x['target_m2']) for x in rows]); total=math.fsum(target)
        order = np.lexsort((ids, -target/area)); curve=d.ordered_prefix(area,target,order)
        for k in range(len(ids)+1): curves.append({'M':m,'method':'label_informed_prefix','k':k,'area_fraction':float(curve['area_fraction'][k]),'captured_target_m2':float(curve['target'][k]),'capture_fraction':float(curve['capture_fraction'][k]) if total else None})
        label_auc.append({'M':m,'auc_0_1':d.step_auc(curve),'auc_0_half_normalized':d.step_auc(curve,.5)})
        for budget in management.BS:
            prefix=strict_order_budget(area,target,order,budget,d); selected=ids[order[:prefix['k']]].tolist()
            fixed.append({'M':m,'method':'label_informed_prefix',**prefix,'selected_ids':selected})
            labels.append(label_case(out/'label',m,ids.tolist(),area,target,budget,prefix,d))
        rng=np.random.Generator(np.random.PCG64(RANDOM_SEED)); n=len(ids)
        all_s=np.zeros((RANDOM_ORDERS,n+1));all_c=np.zeros_like(all_s);all_a=np.zeros_like(all_s);all_t=np.zeros_like(all_s);orders=np.zeros((RANDOM_ORDERS,n),dtype=np.int32)
        for repeat in range(RANDOM_ORDERS):
            random_order=rng.permutation(n);c=d.ordered_prefix(area,target,random_order)
            all_s[repeat]=c['area_fraction'];all_c[repeat]=c['capture_fraction'];all_a[repeat]=c['area'];all_t[repeat]=c['target'];orders[repeat]=random_order
            random_auc.append({'M':m,'repetition':repeat+1,'auc_0_1':d.step_auc(c),'auc_0_half_normalized':d.step_auc(c,.5)})
            for budget in management.BS:random_budget.append({'M':m,'repetition':repeat+1,**strict_order_budget(area,target,random_order,budget,d)})
        path=out/f'random_full_curves_M{m}.npz';np.savez_compressed(path,area_fraction=all_s,capture_fraction=all_c,actual_area_m2=all_a,captured_target_m2=all_t,order_indices=orders,unit_ids=ids)
        for budget in management.BS:
            selected=[x for x in random_budget if x['M']==m and x['budget']==budget];row={'M':m,'budget':budget,'orders':RANDOM_ORDERS}
            for field in ('capture_fraction','actual_area_m2','captured_target_m2','missed_target_m2','unused_budget_m2'):
                values=[x[field] for x in selected]
                row[field+'_mean']=float(np.mean(values)) if values[0] is not None else None
                row[field+'_q025']=float(np.quantile(values,.025,method='linear')) if values[0] is not None else None
                row[field+'_q975']=float(np.quantile(values,.975,method='linear')) if values[0] is not None else None
            random_summary.append(row)
    for name,rows in [('label_prefix_budgets',fixed),('label_MILP_budgets',labels),('label_prefix_curves',curves),('label_prefix_auc',label_auc),('random_budgets',random_budget),('random_auc',random_auc),('random_summary',random_summary)]:
        core.write(out/(name+'.json'),rows)
    summary={'unique_label_cases':15,'actual_label_solver_calls':sum(x['solver_calls'] for x in labels),'label_bound_anomalies':sum(x['bound_anomaly'] for x in labels),
        'random_orders_per_M':1000,'random_seed':RANDOM_SEED,'RNG':'PCG64 reset for each M; initial ASCII IDs','all_steps_saved':True,'zero_target_capture_in_NPZ':'NaN means undefined; JSON ratios/AUC are null',
        'shared_across_all24_models':True,'reference_rows_not_model_replicates':True,'quantiles':'within-domain random-order variability, not cross-field confidence intervals',
        'minimum_selected_area_claim':False,'label_prefix_global_optimality_claim':False,'adapter':d.label_ledger}
    core.write(out/'summary.json',summary)
    core.write(out/'reference_seal.json',{'files':{p.relative_to(out).as_posix():core.sha(p) for p in out.rglob('*') if p.is_file()},'summary':summary})
    return summary


def model_tables(population, unit, out, d):
    out.mkdir(); all_fixed=[];all_auc=[];all_curves=[];bindings=[]
    lookup={(int(x['M']),x['id']):x for x in unit}
    for slot,item in enumerate(population['models'],1):
        folder=Path(item['selection_seal']['path']).parent;core.completed_stage(folder);core.bound_file(item['selection_seal']['path'],item['selection_seal']['sha256'])
        core.require(management.verify_model_selection(folder,d)['run_id']==item['run_id'],'Outcome model/selection identity differs')
        allowed=management.read_rows(folder/'policy/selection_inputs_without_annotations.csv')
        core.require(len(allowed)==len(lookup) and {(int(x['M']),x['id']) for x in allowed}==set(lookup),'Model/truth unit domain differs')
        for row in allowed:core.require(math.isclose(float(row['area']),float(lookup[int(row['M']),row['id']]['area_m2']),rel_tol=1e-10,abs_tol=1e-9),'Model/truth unit area differs')
        result=management.evaluate_joined_arrays(folder,unit,d);core.require(len(result['outcomes'])==120 and len(result['auc'])==12,'Model result population differs')
        bounds={(x['M'],x['R'],x['budget']):x for x in core.read(folder/'bound_audit.json')}
        for row in result['outcomes']:
            b=bounds[row['M'],row['R'],row['budget']];row['selection_seal_sha256']=item['selection_seal']['sha256']
            if row['policy']=='predicted_area':row.update({key:b[key] for key in ('fallback_used','bound_anomaly','updated_upper_m2','updated_relative_gap','updated_proven_flag','certification_downgraded')})
            else:row.update({key:None for key in ('fallback_used','bound_anomaly','updated_upper_m2','updated_relative_gap','updated_proven_flag','certification_downgraded')})
            row['selected_ids_json']=json.dumps(row.pop('selected_ids'),ensure_ascii=True)
        for name in ('auc','full_prefix_curves'):
            for row in result[name]:row['run_id']=item['run_id'];row['selection_seal_sha256']=item['selection_seal']['sha256']
        dest=out/f'{slot:02d}';dest.mkdir();core.table(dest/'fixed_budgets.csv',result['outcomes']);core.table(dest/'prefix_auc.csv',result['auc']);core.table(dest/'complete_prefix_curves.csv',result['full_prefix_curves'])
        all_fixed+=result['outcomes'];all_auc+=result['auc'];all_curves+=result['full_prefix_curves'];bindings.append({'run_id':item['run_id'],'source_selection':item['selection_seal'],'result_directory':str(dest)})
    core.require(len(population['models'])==24 and len(all_fixed)==2880 and len(all_auc)==288,'Full24 outcome matrix required')
    core.table(out/'all_fixed_budgets.csv',all_fixed);core.table(out/'all_prefix_auc.csv',all_auc);core.table(out/'all_complete_prefix_curves.csv',all_curves)
    core.write(out/'model_bindings.json',bindings)
    return {'models':24,'fixed_rows':2880,'auc_rows':288,'full_curve_rows':len(all_curves),'all_adverse_outcomes_retained':True}


def gate(root_review, choice_review):
    contract,models,freeze=core.validate_freeze();lock=core.validate_source_lock()
    for name in ('local_outcomes.py','local_management.py','local_geometry.py','core.py','pipeline.py','data_io.py'):
        core.require(lock['files'].get(name)==core.sha(HERE/name),'Final freeze must include outcome/selection sources: '+name)
    population=management.truth_gate(choice_review)
    review=core.read(root_review)
    core.require(review.get('passed') is True and review.get('scope')=='confirmation_local_outcomes','Root outcome authorization required')
    required={'implementation_sha256':core.sha(__file__),'freeze_sha256':core.sha(HERE/'FREEZE.json'),'choice_review_sha256':core.sha(choice_review),
        'local_population_sha256':core.sha(HERE/'stages/local_selections_seal/all_local_selections.json'),'geometry_metadata_sha256':core.sha(HERE/'stages/local_geometry/metadata.json')}
    for key,digest in required.items():core.require(review.get(key)==digest,'Root review does not bind '+key)
    core.require(review['local_run_ids']==[x['run_id'] for x in management.local_models(contract)] and review['label_case_count']==15 and review['random_orders_per_M']==1000,'Root outcome scope differs')
    core.completed_stage(HERE/'stages/truth_local');truth=core.read(HERE/'stages/truth_local/truth_manifest.json')
    core.require(truth['dataset']=='local' and truth['choice_review']==core.binding(choice_review),'Truth acceptance differs')
    core.require(truth['all_predictions_seal']==core.binding(HERE/'stages/predictions_seal/all_predictions.json'),'Truth prediction seal differs')
    core.require(truth['input_manifest']==core.binding(HERE/'stages/inputs_local/input_manifest.json'),'Truth input domain differs')
    return contract,population,truth


def execute(root_review,choice_review):
    initial=core.sha(__file__);contract,population,truth=gate(root_review,choice_review)
    with core.attempt('local_outcomes') as out:
        core.configure_cpu(out);d=dependencies()
        from threadpoolctl import threadpool_limits
        watched={str(Path(__file__).resolve()):initial,str(Path(root_review).resolve()):core.sha(root_review),str(Path(choice_review).resolve()):core.sha(choice_review),
            str(HERE/'FREEZE.json'):core.sha(HERE/'FREEZE.json'),str(HERE/'SOURCE_LOCK.json'):core.sha(HERE/'SOURCE_LOCK.json'),
            **{str(ROOT/n):h for n,h in management.PINS.items()}}
        for stage, names in (('truth_local', ('truth_manifest.json', 'artifact_manifest.json')),
                             ('local_selections_seal', ('all_local_selections.json', 'artifact_manifest.json')),
                             ('local_geometry', ('metadata.json', 'artifact_manifest.json', 'area_units.csv')),
                             ('predictions_seal', ('all_predictions.json', 'artifact_manifest.json'))):
            management.bind_stage_identity(watched, HERE/'stages'/stage, names)
        source=contract['datasets']['local'];core.bound_file(source['source_summary_path'],source['source_summary_sha256']);watched[source['source_summary_path']]=source['source_summary_sha256']
        summary=core.read(source['source_summary_path']);geometry=management.validate_geometry(HERE/'stages/local_geometry')
        for row in summary['sources']:watched[row['mask']]=row['mask_sha256']
        core.write(out/'begin_binding.json',{'watched_sha256':watched,'truth_manifest':core.binding(HERE/'stages/truth_local/truth_manifest.json'),
            'population':core.binding(HERE/'stages/local_selections_seal/all_local_selections.json'),'geometry':core.binding(HERE/'stages/local_geometry/artifact_manifest.json'),'label_adapter':d.label_ledger,'source_semantics':'Original grayscale mask>0 AND native alpha; exact native CSR, no256soft proxy'})
        def mask_loader(record,header):
            core.bound_file(record['mask'],record['mask_sha256'])
            from PIL import Image
            with Image.open(record['mask']) as im:im.load();array=d.np.asarray(im).copy()
            core.bound_file(record['mask'],record['mask_sha256']);return array
        with threadpool_limits(limits=2):
            unit,audit=join_native_truth(truth,geometry,HERE/'stages/local_geometry',summary,d,mask_loader)
            core.table(out/'unit_truth.csv',unit);core.write(out/'native_truth_audit.json',audit)
            model_summary=model_tables(population,unit,out/'models',d)
        with threadpool_limits(limits=1):reference=shared_references(unit,out/'shared_reference',d)
        management.verify_execution_hashes(watched);core.validate_source_lock();management.truth_gate(choice_review);core.completed_stage(HERE/'stages/truth_local');management.validate_geometry(HERE/'stages/local_geometry')
        core.write(out/'summary.json',{'version':'confirmation_local_outcomes_v1','models':model_summary,'domain':audit,'shared_reference':reference,
            'all_predictions_and_all30_choices_preceded_truth':True,'model_choice_IDs_modified':False,'validation_truth_reused':False,'field_independence_claim':False,'herbicide_or_yield_claim':False})
        core.write(out/'end_binding.json',{'watched_sha256':watched,'all_rechecked':True,'label_solver_calls':reference['actual_label_solver_calls'],'artifact_truth_sha256':core.sha(out/'unit_truth.csv')})
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-review',required=True);parser.add_argument('--choice-review',required=True)
    args=parser.parse_args();print(execute(args.root_review,args.choice_review))
