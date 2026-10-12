"""Independent bounded source/artificial-fixture review; never production data."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime
import argparse, ast, csv, hashlib, json, math, shutil, sys, traceback

ROOT = Path(__file__).resolve().parent
HOME = ROOT / 'confirmation_v1'
sys.path.insert(0, str(HOME))
import core
import local_management as m
import local_outcomes as q


def main():
    global ACTIVE_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--management-checks', required=True, type=Path)
    parser.add_argument('--outcomes-checks', required=True, type=Path)
    parser.add_argument('--identity-checks', required=True, type=Path)
    parser.add_argument('--previous-source', required=True, type=Path)
    args = parser.parse_args()
    out = ROOT / 'provenance' / ('confirmation_local_outcomes_independent_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    ACTIVE_OUT = out
    checks = []
    def check(name, condition=True):
        core.require(condition, name); checks.append(name)
    def reject(name, function):
        try: function()
        except (ValueError, AssertionError, FileNotFoundError): checks.append(name)
        else: raise AssertionError('failed to reject ' + name)
    source = {str(p): core.sha(p) for p in (Path(__file__), HOME/'local_management.py', HOME/'local_outcomes.py', HOME/'core.py', HOME/'pipeline.py')}
    identity = core.read(args.identity_checks)
    check('final identity supplement passed', identity['passed'] is True)
    prior = [core.binding(args.identity_checks)]
    for path, implementation in ((args.management_checks, HOME/'local_management.py'), (args.outcomes_checks, HOME/'local_outcomes.py')):
        path = path.resolve(); core.require(path.is_relative_to(ROOT/'provenance'), 'only preparation receipt allowed')
        receipt = core.read(path)
        archived = args.previous_source / implementation.name
        check('implementer original receipt exact archived source: ' + implementation.name,
              receipt['passed'] is True and receipt['implementation_sha256'] == core.sha(archived))
        check('final source pinned by identity supplement: ' + implementation.name,
              identity['implementation_sha256'][implementation.name] == core.sha(implementation))
        key = 'management43' if implementation.name == 'local_management.py' else 'outcomes35'
        check('identity supplement pins original receipt: ' + implementation.name,
              identity['previous_full_scientific_readiness'][key]['sha256'] == core.sha(path))
        def funcs(p):
            return {n.name: ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}
        old,new = funcs(archived),funcs(implementation)
        allowed = {'execute_run','bind_stage_identity'} if key=='management43' else {'execute'}
        check('all scientific and gate AST unchanged: ' + implementation.name,
              all(old[n]==new[n] for n in old if n not in allowed) and set(new)-set(old)<=allowed)
        source[str(archived.resolve())]=core.sha(archived)
        prior.append(core.binding(path))
    check('imports do not load model or array frameworks', 'torch' not in sys.modules and 'numpy' not in sys.modules)
    reject('absent production freeze denies before dataset', lambda: q.gate(out/'NO_REVIEW', out/'NO_CHOICES'))
    check('denied gate does not load scientific dependencies', 'numpy' not in sys.modules)
    # Gate call order in a fully mocked metadata-only population. A missing external
    # selection acceptance is not replaced by the local24 seal.
    events = []
    def freeze(): events.append('freeze'); return ({}, [], {})
    def predictions(): events.append('all37'); return {}
    def choices(_): events.append('all30'); raise ValueError('ARTIFICIAL missing external choices')
    with patch.object(core, 'validate_freeze', side_effect=freeze), patch.dict(sys.modules, {'pipeline': SimpleNamespace(validate_global_predictions=predictions, validate_choices=choices)}):
        reject('all30 acceptance refusal blocks truth gate', lambda: m.truth_gate(out/'NO_CHOICES'))
    check('all37 then all30 precede local truth', events == ['freeze', 'all37', 'all30'])
    fake_stage=out/'artificial_identity';fake_stage.mkdir()
    for name in ('metadata.json','artifact_manifest.json','area_units.csv'):
        (fake_stage/name).write_text('ARTIFICIAL BEFORE',encoding='utf-8')
    watched={};m.bind_stage_identity(watched,fake_stage,('metadata.json','artifact_manifest.json','area_units.csv'))
    m.verify_execution_hashes(watched)
    check('unchanged consumed-stage byte identity accepted',len(watched)==3)
    (fake_stage/'metadata.json').write_text('ARTIFICIAL AFTER',encoding='utf-8')
    reject('consumed metadata replacement detected at exit',lambda:m.verify_execution_hashes(watched))
    # Scientific functions are used with new small artificial arrays only. Disable
    # any real SciPy backend before extracting the label adapter.
    import numpy as np
    import scipy.optimize
    with patch.object(scipy.optimize, 'milp', side_effect=AssertionError('REAL SOLVER FORBIDDEN')):
        d = q.dependencies()
        check('bound module top level never imported', 'audit_common_bound_consistency' not in sys.modules)
        check('two explicit label adapters only', d.label_ledger['explicit_AST_adapters'] == [
            'label_solver_threads_2_to_1', 'large_raw_upper_inversion_uses_totalT_pending_uniform_downgrade'])
        for p, h in m.PINS.items(): core.bound_file(ROOT/p, h)
        check('all six immutable numerical dependency hashes')
        # Four native pixels inside a single output support, split asymmetrically
        # across two management cells, including a negative grid index.
        geometry = out/'artificial_geometry'; (geometry/'tiles').mkdir(parents=True)
        columns = [0, 1, 512, 513]
        small = np.array([[.3, .1, 0., .2], [0., .4, .5, 0.]])
        rr, cc = np.nonzero(small)
        W = d.sparse.csr_matrix((small[rr,cc], (rr, np.asarray(columns)[cc])), shape=(2,262144))
        raw = np.asarray(W.sum(axis=0)).ravel(); area = small.sum(axis=1)
        p = geometry/'tiles/artificial.npz'
        np.savez_compressed(p, cells=np.array([[-1,0],[0,0]]), area=area,
            native_weights=W.data, native_indices=W.indices, native_indptr=W.indptr,
            native_shape=np.asarray(W.shape), raw_valid_by_native=raw)
        unit_rows = [{'M':M,'id':f'M{M}_E{ix}_N0','ix':ix,'iy':0,'area_m2':float(a)}
                     for M in (1,2,4) for ix,a in zip((-1,0),area)]
        core.table(geometry/'area_units.csv', unit_rows)
        mask = np.zeros((512,512), np.uint8); mask[0,1]=1; mask[1,0]=255
        fake_mask_path = out/'ARTIFICIAL_MASK_ID_ONLY.png'
        window = dict(id='ARTIFICIAL',source_id='SYNTHETIC',x=0,y=0,file='tiles/artificial.npz',sha256=core.sha(p))
        meta = dict(windows=[window], source_headers=[dict(source_id='SYNTHETIC',width=512,height=512)], area_m2=math.fsum(area))
        truth = {'records':[{'id':'ARTIFICIAL','native_reference':{'source_id':'SYNTHETIC','mask_source':str(fake_mask_path),'native_window_xywh':[0,0,512,512]}}]}
        summary = {'sources':[{'source_id':'SYNTHETIC','mask':str(fake_mask_path),'mask_sha256':'0'*64}]}
        joined, evidence = q.join_native_truth(truth,meta,geometry,summary,d,lambda source, header: mask)
        actual = [r['target_m2'] for r in joined if r['M']==1]
        check('native CSR retains sub-support truth', np.allclose(actual,[.1,.9],rtol=0,atol=1e-15))
        check('coarse mean proxy gives different and wrong result', not np.allclose(actual, area*.5))
        check('negative-index nested truth totals', all(math.isclose(math.fsum(r['target_m2'] for r in joined if r['M']==M),1.,abs_tol=1e-15) for M in (1,2,4)))
        check('native raw-column reference agrees', evidence['windows'][0]['target_relative_error'] < 1e-14)
        valid = np.zeros((512,512), bool); valid[0,0]=True
        check('source-exterior target padded false', q.native_target(np.ones((1,1),np.uint8),0,0,valid,np).sum()==1)
        reject('RGB labels rejected', lambda:q.native_target(np.zeros((2,2,3),np.uint8),0,0,valid,np))
        reject('floating label interpretation rejected', lambda:q.native_target(np.zeros((2,2),np.float32),0,0,valid,np))
        a=np.ones(2);t=np.array([.7,.3]);prefix=q.strict_order_budget(a,t,np.array([0,1]),.5,d)
        calls=[]
        def fake(c, **kwargs):
            calls.append(kwargs['options'])
            return scipy.optimize.OptimizeResult(x=np.array([1.,0.]),fun=-.7,mip_dual_bound=-.1,mip_gap=0.,status=0,message='ARTIFICIAL BOUND CONTRADICTION')
        folder=out/'artificial_label';folder.mkdir()
        with patch.dict(d.label_namespace, {'milp':fake}):
            result=q.label_case(folder,1,['a','b'],a,t,.5,prefix,d)
            check('backend reference restored after call', d.label_namespace['milp'] is fake)
        check('one mock backend call and single-thread options', len(calls)==1 and calls[0]['threads']==1 and calls[0]['time_limit']==30. and calls[0]['mip_rel_gap']==1e-8)
        check('contradictory raw bound retains selected IDs', result['selected_ids']==['a'] and result['selected_ids_unchanged'])
        check('contradiction uses safe sumT, no certification', result['bound_anomaly'] and result['updated_upper_m2']==1. and not result['updated_proven_flag'])
        check('raw backend and adapted return preserved', (folder/'M1_B050_solver_raw.json').is_file() and (folder/'M1_B050_adapted_return.json').is_file())
        zero=out/'zero_target';zero.mkdir();zero_prefix=q.strict_order_budget(a,np.zeros(2),np.array([0,1]),.5,d)
        with patch.dict(d.label_namespace, {'milp':lambda *a,**kw: (_ for _ in ()).throw(AssertionError('ZERO T MUST NOT SOLVE'))}):
            z=q.label_case(zero,1,['a','b'],a,np.zeros(2),.5,zero_prefix,d)
        check('zero T preserves absolute values and NA ratios without solve', z['solver_calls']==0 and z['capture_fraction'] is None and z['updated_relative_gap'] is None and z['captured_target_m2']==0.)
    # Independently recalculate saved artificial full24 outputs; no evaluator,
    # integration or solver is re-run for these 24 models.
    fixture = args.outcomes_checks.resolve().parent/'h/stages/local_outcomes'
    core.completed_stage(fixture)
    check('saved artificial full fixture identity', fixture.is_relative_to(ROOT/'provenance'))
    units=m.read_rows(fixture/'unit_truth.csv')
    by={(int(r['M']),r['id']):(float(r['area_m2']),float(r['target_m2'])) for r in units}
    fixed=m.read_rows(fixture/'models/all_fixed_budgets.csv')
    check('all24x120 unique policy records', len(fixed)==2880 and len({(r['run_id'],r['M'],r['R'],r['budget'],r['policy']) for r in fixed})==2880)
    check('all models explicitly artificial', len({r['run_id'] for r in fixed})==24 and all(r['run_id'].startswith('SYNTHETIC_') for r in fixed))
    maximum=0.
    for row in fixed:
        M=int(row['M']); ids=json.loads(row['selected_ids_json']); A=math.fsum(by[M,i][0] for i in ids); T=math.fsum(by[M,i][1] for i in ids)
        D=math.fsum(a for (m0,_),(a,t0) in by.items() if m0==M); total=math.fsum(t0 for (m0,_),(a,t0) in by.items() if m0==M)
        core.require(len(ids)==len(set(ids)) and A<=float(row['budget'])*D,'artificial strict selection')
        errors=[abs(A-float(row['actual_area_m2'])),abs(T-float(row['captured_target_m2'])),abs(T/total-float(row['capture_fraction']))]
        maximum=max(maximum,*errors)
    check('all2880 artificial A/T/capture independently recomputed', maximum<1e-14)
    curves=m.read_rows(fixture/'models/all_complete_prefix_curves.csv');groups={}
    for row in curves:groups.setdefault((row['run_id'],row['M'],row['R']),[]).append(row)
    auc=m.read_rows(fixture/'models/all_prefix_auc.csv')
    check('all24x12 complete-prefix AUC inventory', len(auc)==len(groups)==288)
    max_auc=0.
    for row in auc:
        c=sorted(groups[row['run_id'],row['M'],row['R']],key=lambda x:int(x['k']))
        core.require([int(x['k']) for x in c]==list(range(len(c))),'missing artificial prefix step')
        for upper,key in ((1.,'auc_0_1'),(.5,'auc_0_half_normalized')):
            val=math.fsum(float(x['capture_fraction'])*max(0.,min(float(y['area_fraction']),upper)-float(x['area_fraction'])) for x,y in zip(c,c[1:]))/upper
            max_auc=max(max_auc,abs(val-float(row[key])))
    check('all576 artificial AUC values independent rectangle sum', max_auc<1e-14)
    for M in (1,2,4):
        path=fixture/f'shared_reference/random_full_curves_M{M}.npz'
        with np.load(path,allow_pickle=False) as zz:
            ids=zz['unit_ids'].tolist();order=zz['order_indices']; all_a=zz['actual_area_m2'];all_t=zz['captured_target_m2'];all_s=zz['area_fraction'];all_c=zz['capture_fraction']
        core.require(order.shape==(1000,len(ids)) and all_a.shape==(1000,len(ids)+1),'random whole-step shapes')
        areas=np.array([by[M,i][0] for i in ids]);targets=np.array([by[M,i][1] for i in ids]);D=math.fsum(areas);total=math.fsum(targets)
        for k in range(1000):
            core.require(sorted(order[k].tolist())==list(range(len(ids))),'not random permutation')
            ar=np.r_[0.,np.cumsum(areas[order[k]])];tr=np.r_[0.,np.cumsum(targets[order[k]])];ar[-1]=D;tr[-1]=total
            core.require(np.array_equal(ar,all_a[k]) and np.array_equal(tr,all_t[k]) and np.array_equal(ar/D,all_s[k]) and np.array_equal(tr/total,all_c[k]),'random curve mismatch')
        check(f'M{M} all1000 random fullsteps independently reconstructed')
    for p,h in source.items():core.bound_file(p,h)
    source['implementer_management_checks']=core.sha(args.management_checks)
    source['implementer_outcomes_checks']=core.sha(args.outcomes_checks)
    report={'passed':True,'review_type':'independent source and artificial-fixture review, not confirmation execution',
        'implementation_sha256':core.sha(HOME/'local_outcomes.py'),'management_sha256':core.sha(HOME/'local_management.py'),
        'checks':checks,'check_count':len(checks),'sources':source,'implementer_receipts':prior,
        'artificial_outcome_fixture':str(fixture),'max_artificial_outcome_error':maximum,'max_artificial_auc_error':max_auc,
        'real_test_or_label_arrays_read':0,'real_solver_calls':0,'real_model_forward_calls':0,'real_R6_forward_calls':0,
        'new_mock_label_backend_calls':1,'production_FREEZE_created':False,
        'limitations':['All numeric checks are artificial and do not validate the future338-window domain.',
            'Original core/pipeline all37/all30 validation also relies on its earlier independent component review.',
            'A later complete source freeze and root execution acceptance remain necessary.']}
    core.write(out/'review.json',report)
    for p in (Path(__file__),HOME/'local_management.py',HOME/'local_outcomes.py'):shutil.copyfile(p,out/p.name)
    (out/'review.md').write_text('# Independent local confirmation preparation review\n\n'
        f'{len(checks)} checks passed. Native512 truth uses the native CSR; an asymmetric four-pixel fixture rejects the coarse-mean substitute. '
        'The all37/all30 gate order, shared one-thread label backend and safe total-T bound, unchanged selected IDs, raw-return preservation and zero-T behavior were checked. '
        'All 2,880 saved artificial model outcomes and 576 AUC values were independently recalculated, and all 3,000 random complete-prefix curves reconstructed.\n\n'
        'No real test/label arrays, models, R6 forward or solver were used. This is preparation acceptance only; it does not certify future confirmation results.\n',encoding='utf-8')
    print(json.dumps({'directory':str(out),'sha256':core.sha(out/'review.json'),'checks':len(checks)}))


if __name__=='__main__':
    try: main()
    except BaseException:
        if 'ACTIVE_OUT' in globals():
            core.write(ACTIVE_OUT/'failure.json',{'passed':False,'traceback':traceback.format_exc(),
                'scope':'independent preparation checker; failure is not a production outcome',
                'management_sha256':core.sha(HOME/'local_management.py'),'outcomes_sha256':core.sha(HOME/'local_outcomes.py')})
        raise
