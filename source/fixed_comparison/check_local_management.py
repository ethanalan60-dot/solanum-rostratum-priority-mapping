"""Synthetic-only checks for local confirmation management preparation."""
from pathlib import Path
from unittest.mock import patch
import datetime, itertools, json, math, sys, traceback
import core
import local_management as m

OUT=m.ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_management_checks_%Y%m%d_%H%M%S_%f')
OUT.mkdir();checks=[]
def check(name,value=True):
    if not value:raise AssertionError(name)
    checks.append({'name':name,'passed':True})
def rejects(name,f):
    try:f()
    except (AssertionError,ValueError,FileNotFoundError,FileExistsError,RuntimeError,OSError):check(name);return
    raise AssertionError('Expected refusal: '+name)
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);core.write(p,x)

try:
    check('import_is_standard_library_only','numpy' not in sys.modules and 'torch' not in sys.modules)
    rejects('production_denied_before_dataset_without_freeze',lambda:m.execute_run('undeclared',OUT/'absent_root_review.json'))
    check('still_no_scientific_import_after_gate_rejection','numpy' not in sys.modules)
    watched_file=OUT/'watched.py';watched_file.write_text('original',encoding='utf-8');watch={str(watched_file):core.sha(watched_file)}
    m.verify_execution_hashes(watch);watched_file.write_text('changed',encoding='utf-8')
    rejects('end_execution_external_source_or_self_drift_rejected',lambda:m.verify_execution_hashes(watch))
    identity=dict(run_id='toy',checkpoint_sha256='a'*64,freeze_sha256='f',all_predictions_sha256='p',geometry_metadata_sha256='g')
    m.validate_population_identity(identity,{'run_id':'toy'},{'run_id':'toy','checkpoint_sha256':'a'*64},'f','p','g')
    rejects('population_checkpoint_not_final_freeze_rejected',lambda:m.validate_population_identity(identity,{'run_id':'toy'},{'run_id':'toy','checkpoint_sha256':'b'*64},'f','p','g'))
    d=m.dependencies();np=d.np
    check('never_import_side_effectful_bound_audit','audit_common_bound_consistency' not in sys.modules)
    check('no_torch_or_model_import','torch' not in sys.modules)
    check('exact60cases_12rankings',len(m.CASES)==60 and len({x[:2] for x in m.CASES})==12)
    ledger=d.ledger['selection']['count_only_AST_changes']
    check('only_explicit45to60_AST_literals',len(ledger)==5 and all((x['old'],x['new']) in [(45,60),('all_45_final_choices_saved','all_60_final_choices_saved')] for x in ledger))
    check('bound_and_metrics_AST_unmodified',not d.ledger['bound']['count_only_AST_changes'] and not d.ledger['metrics']['count_only_AST_changes'])
    allowed=[{'M':ms,'id':f'M{ms}_{s}','area':a,**{f'predicted_R{r}':p for r in m.RS}}
             for ms in m.MS for s,a,p in [('a',1.,.9),('b',2.,.4),('c',7.,.7)]]
    context={k:'0'*64 for k in ('freeze_sha256','all_predictions_sha256','geometry_metadata_sha256','checkpoint_sha256')};context['run_id']='SYNTHETIC_ONLY'
    rejects('target_field_rejected_by_selection_whitelist',lambda:d.v.prepare_data([{**x,'target':123} for x in allowed],10.))
    rejects('missing_R6_prediction_rejected',lambda:d.v.prepare_data([{k:v for k,v in x.items() if k!='predicted_R6'} for x in allowed],10.))
    rejects('duplicate_ID_rejected',lambda:d.v.prepare_data(allowed+[allowed[0]],10.))
    from scipy.optimize import OptimizeResult
    calls=[]
    def mock_milp(c,**kwargs):
        n=len(c);cost=np.asarray(kwargs['constraints'].A)[0];budget=float(kwargs['constraints'].ub[0])
        candidates=[np.asarray(z,dtype=float) for z in itertools.product((0,1),repeat=n) if math.fsum(cost*np.asarray(z))<=budget]
        x=min(candidates,key=lambda z:float(c@z));fun=float(c@x)
        calls.append({'options':dict(kwargs['options']),'x':x.tolist(),'objective':fun})
        return OptimizeResult(x=x,fun=fun,mip_dual_bound=fun,mip_gap=0.,mip_node_count=0,status=0,message='SYNTHETIC BRUTE FORCE BACKEND; NOT A SOLVER RUN')
    folder=OUT/'ok';folder.mkdir()
    with patch.object(d.policy,'milp',side_effect=mock_milp):seal=m.select_arrays(folder,allowed,10.,context,d)
    check('all60_mock_backend_returns_persisted',len(calls)==60 and len(list((folder/'policy/solver_raw').glob('*.json')))==60)
    check('all60_selector_raw_and_choice_records',len(list((folder/'policy/raw').glob('*.json')))==len(list((folder/'policy/choices').glob('*.json')))==60)
    check('unchanged_single_thread_solver_options',all(x['options']==d.policy.OPTIONS and x['options']['threads']==1 for x in calls))
    check('all120choices_and12rankings',seal['case_count']==120 and len(core.read(folder/'prefix_choices.json'))==60 and len(core.read(folder/'rankings.json'))==12)
    check('mock_backend_restored_after_full_population',getattr(d.policy.milp,'__name__','')=='milp')
    truth=[{'M':ms,'id':f'M{ms}_{s}','target_m2':t} for ms in m.MS for s,t in [('a',.1),('b',1.),('c',0.)]]
    evaluated=m.evaluate_joined_arrays(folder,truth,d)
    check('post_selection120rows12AUC',len(evaluated['outcomes'])==120 and len(evaluated['auc'])==12)
    check('all_prefix_points_retained',len(evaluated['full_prefix_curves'])==48)
    check('strict_budget_all_policies',all(x['actual_area_m2']<=x['budget']*10 for x in evaluated['outcomes']))
    zero=[{**x,'target_m2':0.} for x in truth];z=m.evaluate_joined_arrays(folder,zero,d)
    check('zero_target_ratios_and_auc_missing',all(x['capture_fraction'] is None for x in z['outcomes']) and all(x['auc_0_1'] is None and x['auc_0_half_normalized'] is None for x in z['auc']))
    core.write(OUT/'synthetic_outcomes.json',evaluated);core.write(OUT/'zero_target.json',z)
    rejects('truth_unit_inventory_mismatch_rejected',lambda:m.evaluate_joined_arrays(folder,truth[:-1],d))
    badpath=folder/'prefix_choices.json';before=badpath.read_bytes();badpath.write_bytes(before+b' ')
    rejects('tampered_seal_rejected_before_T_arithmetic',lambda:m.evaluate_joined_arrays(folder,truth,d));badpath.write_bytes(before)
    # Persisted toy raw evidence has an actually over-budget original candidate.
    rejected=OUT/'fallback';rejected.mkdir();fallback_calls=[]
    def overshoot(c,**kw):
        fallback_calls.append(1);x=np.ones(len(c));fun=float(c@x)
        return OptimizeResult(x=x,fun=fun,mip_dual_bound=fun,mip_gap=0.,mip_node_count=0,status=0,message='SYNTHETIC OVER-BUDGET CANDIDATE')
    with patch.object(d.policy,'milp',side_effect=overshoot):m.select_arrays(rejected,allowed,10.,context,d)
    b=core.read(rejected/'bound_audit.json')
    check('strict_fallback_retains_raw_and_never_proven',sum(x['fallback_used'] for x in b)==48 and all(not x['updated_proven_flag'] for x in b if x['fallback_used']))
    check('fallback60_no_retry',len(fallback_calls)==60)
    # Feasible but inferior candidate: accepted unchanged, contradictory upper downgraded.
    lower=OUT/'lower';lower.mkdir();lower_calls=[]
    def inferior(c,**kw):
        cost=np.asarray(kw['constraints'].A)[0];budget=float(kw['constraints'].ub[0]);x=np.zeros(len(c))
        if budget>=.2:x[1]=1.
        fun=float(c@x);lower_calls.append(1)
        return OptimizeResult(x=x,fun=fun,mip_dual_bound=fun,mip_gap=0.,mip_node_count=0,status=0,message='SYNTHETIC INCONSISTENT BOUND')
    with patch.object(d.policy,'milp',side_effect=inferior):m.select_arrays(lower,allowed,10.,context,d)
    b=core.read(lower/'bound_audit.json')
    check('feasible_prefix_witness_downgrades_false_bound',any(x['bound_anomaly'] and not x['updated_proven_flag'] for x in b))
    for row in b:
        old=core.read(row['choice']['path']);assert row['selected_ids']==old['selected_ids']
    check('bound_amendment_never_changes_IDs')
    # The original upper<primal assertion still fails, with raw backend return saved.
    broken=OUT/'backend_error';broken.mkdir()
    def raw_bad(c,**kw):
        x=np.zeros(len(c));return OptimizeResult(x=x,fun=0.,mip_dual_bound=1.,mip_gap=0.,status=0,message='SYNTHETIC IMPOSSIBLE UPPER')
    with patch.object(d.policy,'milp',side_effect=raw_bad):rejects('selector_error_is_not_swallowed_or_retried',lambda:m.select_arrays(broken,allowed,10.,context,d))
    check('raw_backend_evidence_precedes_selector_exception',(broken/'policy/solver_raw/M1_R1_B010.json').exists() and (broken/'policy/solver_raw/M1_R1_B010_exception.json').exists())
    check('backend_restored_on_exception',getattr(d.policy.milp,'__name__','')=='milp')
    # Original cumulative-sum policy receives only deterministic strict-budget tail trim.
    p=m.strict_prefix(d,[f'x{i}' for i in range(10)],np.full(10,.1),np.full(10,.05),.6)
    check('fsum_roundoff_prefix_tail_trim',p['tail_trim_count']==1 and len(p['selected_ids'])==5 and p['area_m2']<=.6)
    prefix=d.ordered_prefix(np.array([.2,.3,.5]),np.array([.1,.1,.3]),np.arange(3))
    check('full_and_normalized_half_step_AUC',math.isclose(d.step_auc(prefix),.26,abs_tol=1e-15) and math.isclose(d.step_auc(prefix,.5),.12,abs_tol=1e-15))
    # Synthetic native alpha/CSR and four probability fields. No real raster/cache access.
    geometry=OUT/'geo';geometry.mkdir();(geometry/'tiles').mkdir()
    raw_native=np.zeros((512,512));raw_native[:2,:4]=1.;raw=np.zeros(65536);raw[:2]=4
    weight=d.sparse.csr_matrix(([4.,4.],([0,1],[0,1])),shape=(2,65536));path=geometry/'tiles/toy.npz'
    np.savez_compressed(path,cells=np.array([[0,0],[1,0]]),area=np.array([4.,4.]),weights=weight.data,indices=weight.indices,indptr=weight.indptr,shape=np.array(weight.shape),raw_valid_by_support=raw,raw_valid_by_native=raw_native.ravel())
    window={'id':'toy','source_id':'SYNTHETIC','x':0,'y':0,'file':'tiles/toy.npz','sha256':core.sha(path)}
    areas=[{'M':1,'id':'M1_E0_N0','ix':0,'iy':0,'area_m2':4.},{'M':1,'id':'M1_E1_N0','ix':1,'iy':0,'area_m2':4.}]
    areas += [{'M':ms,'id':f'M{ms}_E0_N0','ix':0,'iy':0,'area_m2':8.} for ms in (2,4)];core.table(geometry/'area_units.csv',areas)
    prob=np.zeros((4,256,256),np.float32)
    for j in range(4):prob[j,0,:2]=[.1*(j+1),.2*(j+1)]
    valid=(raw_native>0).reshape(256,2,256,2).mean((1,3)).astype(np.float32);mp=OUT/'toy_map.npz'
    np.savez_compressed(mp,probabilities=prob,valid=valid,R=np.array(m.RS),id=np.asarray('toy'))
    rec={'id':'toy','source_id':'SYNTHETIC','native_window_xywh':[0,0,512,512],'R':list(m.RS),'path':str(mp),'sha256':core.sha(mp)}
    unit,audit=m.integrate_prediction([rec],{'windows':[window]},geometry,d)
    check('synthetic_four_R_native_integral_nested_M',len(unit)==4 and audit['target_arrays_read']==0 and max(audit['per_window'][0]['relative_error_by_R'])<1e-9)
    badprob=prob.copy();badprob[3,0,0]=1.01;np.savez_compressed(mp,probabilities=badprob,valid=valid,R=np.array(m.RS),id=np.asarray('toy'));rec['sha256']=core.sha(mp)
    rejects('out_of_range_probability_rejected',lambda:m.integrate_prediction([rec],{'windows':[window]},geometry,d))
    badvalid=valid.copy();badvalid[0,0]=.5;np.savez_compressed(mp,probabilities=prob,valid=badvalid,R=np.array(m.RS),id=np.asarray('toy'));rec['sha256']=core.sha(mp)
    rejects('alpha_validity_mismatch_rejected',lambda:m.integrate_prediction([rec],{'windows':[window]},geometry,d))
    # Gated entry initialization and claim failure checks patch all real-data access.
    fakehome=OUT/'home';fakehome.mkdir();write(fakehome/'FREEZE.json',{'synthetic':True});write(fakehome/'SOURCE_LOCK.json',{});write(fakehome/'contract.json',{});write(fakehome/'checkpoints.json',{})
    original_attempt=core.attempt
    def fake_attempt(name):return original_attempt(name,fakehome)
    with patch.object(m,'execution_gate',return_value=([{'run_id':'toy'}],{}, {}, {})),patch.object(m.core,'attempt',side_effect=fake_attempt),patch.object(m,'dependencies',side_effect=RuntimeError('CONTROLLED INITIALIZATION ERROR')):
        rejects('postclaim_initialization_failure_preserved',lambda:m.execute_run('toy',OUT/'fake_root'))
        check('permanent_claim_and_failure_artifacts',(fakehome/'claims/local_select_01.json').exists() and core.read(fakehome/'stages/local_select_01/status.json')['state']=='failed' and (fakehome/'stages/local_select_01/artifact_manifest.json').exists())
        rejects('permanent_claim_no_second_attempt',lambda:m.execute_run('toy',OUT/'fake_root'))
    # Global/local population gate, using completed synthetic byte-manifest stages.
    from types import SimpleNamespace
    population_home=OUT/'pop';population_home.mkdir();write(population_home/'FREEZE.json',{'synthetic':True})
    predicted=population_home/'stages/predictions_seal/all_predictions.json';write(predicted,{'synthetic':True})
    roster=[{'run_id':f'model{i:02d}','dataset':'local','variant':'soft_seg'} for i in range(24)]
    local=[]
    for i,row in enumerate(roster,1):
        stage=population_home/'stages'/f'local_select_{i:02d}';write(stage/'selection_seal.json',{'synthetic':True,'run_id':row['run_id']})
        write(stage/'status.json',{'state':'completed'});write(stage/'artifact_manifest.json',core.stage_manifest(stage))
        local.append({'run_id':row['run_id'],'case_count':120,'selection_seal':core.binding(stage/'selection_seal.json'),'artifact_manifest':core.binding(stage/'artifact_manifest.json')})
    popstage=population_home/'stages/local_selections_seal'
    write(popstage/'all_local_selections.json',{'models':local,'P_solver_cases':1440,'policy_choices':2880,'freeze_sha256':core.sha(population_home/'FREEZE.json'),'all_predictions_sha256':core.sha(predicted)})
    write(popstage/'status.json',{'state':'completed'});write(popstage/'artifact_manifest.json',core.stage_manifest(popstage))
    events=[]
    def global_predictions():events.append('predictions_global');return {}
    def global_choices(path):events.append('choices_global30');return {'models':local}
    fakepipeline=SimpleNamespace(validate_global_predictions=global_predictions,validate_choices=global_choices)
    with patch.object(m,'HERE',population_home),patch.object(core,'validate_freeze',return_value=({'roster':roster},[],{})),patch.dict(sys.modules,{'pipeline':fakepipeline}):
        result=m.truth_gate(OUT/'SYNTHETIC_GLOBAL_ACCEPTANCE');check('global_predictions_and_choices_checked_before_truth',events==['predictions_global','choices_global30'] and len(result['models'])==24)
        before=(popstage/'all_local_selections.json').read_bytes();bad=core.read(popstage/'all_local_selections.json');bad['models']=bad['models'][:-1];write(popstage/'all_local_selections.json',bad);write(popstage/'artifact_manifest.json',core.stage_manifest(popstage))
        rejects('missing_one_of24_local_models_blocks_truth',lambda:m.truth_gate(OUT/'SYNTHETIC_GLOBAL_ACCEPTANCE'))
        (popstage/'all_local_selections.json').write_bytes(before);write(popstage/'artifact_manifest.json',core.stage_manifest(popstage))
        with patch.object(fakepipeline,'validate_choices',side_effect=ValueError('GLOBAL EXTERNAL CHOICES NOT SEALED')):
            rejects('missing_external_population_blocks_truth',lambda:m.truth_gate(OUT/'SYNTHETIC_GLOBAL_ACCEPTANCE'))
        changed=[{**x,'selection_seal':dict(x['selection_seal'])} for x in local];changed[0]['selection_seal']['sha256']='0'*64
        with patch.object(fakepipeline,'validate_choices',return_value={'models':changed}):
            rejects('global_review_local_seal_swap_blocks_truth',lambda:m.truth_gate(OUT/'SYNTHETIC_GLOBAL_ACCEPTANCE'))
    report={'passed':True,'implementation_sha256':core.sha(m.__file__),'checker_sha256':core.sha(__file__),'checks':checks,'check_count':len(checks),
            'dependencies':d.ledger,'production_freeze_exists':(m.HERE/'FREEZE.json').exists(),'new_real_solver_calls':0,'mock_backend_calls':len(calls)+len(fallback_calls)+len(lower_calls)+1,
            'real_model_forward_calls':0,'real_test_input_or_label_files_opened':0,'real_R6_forward_calls':0,'real_probability_integrations':0,'synthetic_native_probability_integrations':1,
            'source_lock_created':False,'production_stage_created':False,'limitations':['Preparation only; later immutable root freeze and independent source review required.','Native target join and shared label/random reference construction require separately reviewed post-selection code and global30-model selection acceptance.']}
    write(OUT/'focused_checks.json',report);print(json.dumps({'passed':True,'directory':str(OUT),'checks':len(checks),'sha256':core.sha(OUT/'focused_checks.json')}))
except BaseException as error:
    write(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'checks':checks});raise
