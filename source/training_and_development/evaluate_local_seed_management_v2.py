"""Fixed remaining14 path-recovery local seed development R x M; explicit authorization, no inference/test.

Preparation and validate-only never integrate maps or call a solver. Execution
uses the frozen geometry and selector, then the accepted bound-consistency rule.
All45 choices AND their bound audit are sealed before annotation evaluation.
"""
from pathlib import Path
import argparse,ast,csv,datetime,hashlib,importlib,json,math,os,secrets,shutil,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture';VERSION=ROOT/'local_seed_stability_v1'
PROTOCOL=PA/'RM_local_seed_protocol_v2.json'
R_VALUES=(1,2,4);M_VALUES=(1,2,4);BUDGETS=(.1,.2,.3,.5,1.)
RUNTIME=None

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,data):
    assert data
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in data for k in x)));w.writeheader();w.writerows(data)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9),(a,b)
def verify(mapping):
    for path,digest in mapping.items():assert sha(path)==digest,('Changed input',path)
def checked_path(root,relative):
    parts=str(relative).replace('\\','/').split('/')
    assert all(x not in ('','.','..') and ':' not in x for x in parts)
    p=Path(root).joinpath(*parts).resolve();assert p.is_relative_to(Path(root).resolve());return p

def bound_segments(source):
    text=Path(source).read_text(encoding='utf-8');tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('finite','audit_bounds')]
    assert [n.name for n in nodes]==['finite','audit_bounds']
    segment='\n\n'.join(ast.get_source_segment(text,n) for n in nodes)
    return nodes,segment

def load_bound_function(protocol):
    # Never import the original audit module: its top level disables solvers.
    source=ROOT/'audit_common_bound_consistency.py'
    assert sha(source)==protocol['bound_rule']['source_sha256']
    nodes,segment=bound_segments(source)
    assert hashlib.sha256(segment.encode()).hexdigest()==protocol['bound_rule']['function_segments_sha256']
    ns={'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source)+':pure_bound_functions','exec'),ns)
    return ns['audit_bounds']

def load_protocol(expected):
    assert len(expected)==64 and sha(PROTOCOL)==expected
    p=read(PROTOCOL);assert p['schema']=='local_seed_management_v2' and p['implementation_sha256']==sha(__file__)
    assert p['R']==list(R_VALUES) and p['M_m']==list(M_VALUES) and p['budgets']==list(BUDGETS)
    assert p['strict_area_tolerance_m2']==0 and p['new_runs']==14
    assert p['no_test_or_R6'] and p['new_reference_solver_calls']==p['new_random_repetitions']==0
    verify(p['pinned_files_sha256'])
    plan=read(VERSION/'plan.json');assert p['declared_runs']==plan['runs'][2:]
    assert len({r['run_id'] for r in plan['runs']})==16
    assert p['preserved_completed_run_ids']==[r['run_id'] for r in plan['runs'][:2]]
    assert {(r['encoder'],r['lr'],r['variant'],r['seed']) for r in plan['runs']}=={(e,l,v,s) for e in ('resnet18','convnextv2_tiny') for l in (1e-4,3e-4) for v in ('hard_seg','soft_seg') for s in (29,43)}
    load_bound_function(p)
    return p,{**p['pinned_files_sha256'],str(PROTOCOL):expected,str(Path(__file__).resolve()):sha(__file__)}

def runtime(protocol):
    global RUNTIME
    if RUNTIME is None:
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import numpy as np
        import evaluate_common_management_v2 as v
        from threadpoolctl import threadpool_limits,threadpool_info
        exporter=importlib.import_module('export_local_seed_validation_v2')
        assert 'torch' not in sys.modules,'Metadata-only exporter must not import torch'
        assert v.policy.OPTIONS==protocol['solver_options'] and v.policy.OPTIONS['threads']==1
        assert list(v.CASES)==[(m,r,b) for m in M_VALUES for r in R_VALUES for b in BUDGETS]
        RUNTIME=(np,v,exporter,threadpool_limits,threadpool_info)
    return RUNTIME

def reject_existing(run_id):
    assert run_id in {x['run_id'] for x in read(PROTOCOL)['declared_runs']}
    claim=VERSION/'management_claims_v2'/(run_id+'.json')
    assert not claim.exists(),('Permanent prior attempt claim',claim)
    assert not (VERSION/'management_claims'/(run_id+'.json')).exists(),'Original spatial attempt must not be repeated'
    for folder in list(PA.glob('lsm_*'))+list(PA.glob('lsm2_*')):
        if (folder/'status.json').exists():assert read(folder/'status.json').get('run_id')!=run_id,('Prior output',folder)

def validate_maps(maps_dir,protocol,watched):
    np,v,exporter,_,_=runtime(protocol)
    maps_dir=Path(maps_dir).resolve();meta=read(maps_dir/'metadata.json');status=read(maps_dir/'status.json')
    rid=meta['run_id'];declared={x['run_id']:x for x in protocol['declared_runs']};assert rid in declared
    assert maps_dir.parent==exporter.output_parent(rid).resolve()
    assert meta['schema']=='local_validation_probability_maps_v1' and meta['model_family']=='local_seed_stability_v1'
    assert status['status']=='completed' and status['verification']['all_checks_passed']
    assert status['metadata_sha256']==sha(maps_dir/'metadata.json')
    assert meta['factors']==[1,2,4] and meta['spatial_prediction_available'] is True
    assert not meta['test_accessed'] and not meta['R6_accessed']
    run,cfg,state,manifest,best,checkpoint,model,hashes=exporter.validate_run(rid,load_model=False)
    assert model is None and 'torch' not in sys.modules
    run=Path(run).resolve();assert run==ROOT/'runs'/rid
    for key in ('encoder','variant','seed','lr'):
        assert cfg[key]==declared[rid][key] and meta[key]==cfg[key],key
    assert meta['paired_seed17_run_id']==declared[rid]['paired_seed17_run_id']
    assert Path(meta['config']).resolve()==run/'config.json' and sha(run/'config.json')==meta['config_sha256']
    assert Path(meta['checkpoint']).resolve()==Path(checkpoint).resolve()==run/'best.pt'
    assert sha(checkpoint)==meta['checkpoint_sha256'] and meta['checkpoint_epoch']==best['epoch']==state['best_epoch']
    assert cfg['n_train']==985 and cfg['n_val']==len(manifest)==357 and state['executed_epochs']==24
    assert cfg['validation_R']==[1,2,4] and cfg['selected_val_ids']==[r['id'] for r in manifest]
    assert meta['model_dependency_sha256']==cfg['source_sha256']
    assert sha(ROOT/'export_local_seed_validation_v2.py')==meta['export_source_sha256']
    assert set(meta['source_copies'])==set(meta['source_copy_bindings'])
    for name,path in meta['source_copies'].items():
        binding=meta['source_copy_bindings'][name];archive=Path(path).resolve();original=Path(binding['source_path']).resolve()
        assert archive.parent==maps_dir/'source' and archive==Path(binding['archive_path']).resolve()
        assert original.is_relative_to(ROOT) and original.is_file()
        if name=='export_v2_protocol.json':assert original==exporter.PROTOCOL.resolve()
        else:assert original.name==name
        assert sha(archive)==sha(original)==binding['sha256']
        watched[str(archive)]=binding['sha256'];watched[str(original)]=binding['sha256']
    assert set(meta['source_copies']).issuperset({'train_common.py','models_common.py','conservative_resample.py','export_local_seed_validation_v2.py'})
    assert meta['input_hashes']==hashes
    watched.update({str(Path(p).resolve()):h for p,h in hashes.items()})
    watched.update({str(Path(p).resolve()):h for p,h in meta['input_hashes'].items()});verify(watched)
    assert not (maps_dir/'failure.json').exists()
    claim=read(maps_dir/'EXECUTION_CLAIM.json');execution=meta['execution_bindings']
    assert execution['pre_execution_seal_sha256']==status['pre_execution_seal_sha256']==sha(maps_dir/'pre_execution_seal.json')
    assert exporter.check_execution_review(maps_dir,claim['review_path'])==execution['root_review_and_focuscheck_sha256']
    assert sha(claim['review_path'])==claim['review_sha256']
    watched.update(execution['root_review_and_focuscheck_sha256'])
    preseal=read(maps_dir/'pre_execution_seal.json')
    for rel,h in preseal['file_sha256'].items():assert sha(checked_path(maps_dir,rel))==h
    artifact=read(maps_dir/'artifact_manifest.json')['file_sha256']
    assert set(artifact)=={p.relative_to(maps_dir).as_posix() for p in maps_dir.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    for rel,h in artifact.items():
        p=checked_path(maps_dir,rel);assert sha(p)==h;watched[str(p)]=h
    watched[str(maps_dir/'artifact_manifest.json')]=sha(maps_dir/'artifact_manifest.json')
    for name,key in [('predictions.csv','scalar_csv_sha256'),('summary.json','scalar_summary_sha256')]:assert sha(maps_dir/name)==meta[key]
    assert read(maps_dir/'summary.json')['original_amp_validation']==best
    verification=read(maps_dir/'verification.json');assert verification==status['verification']
    assert verification['checked_pairs']==1071 and verification['expected_windows']==verification['all_npz_reopened']==357
    assert verification['original_selection_audited_24_epochs'] and verification['original_epoch_preserved']
    geometry=read(Path(protocol['geometry_directory'])/'metadata.json')
    assert geometry['state']=='completed' and not geometry['subset_diagnostic'] and not geometry['test_accessed']
    assert geometry['expected_windows']==357 and geometry['native_valid_pixels']==91061049 and geometry['native_target_pixels']==885226
    assert read(Path(protocol['geometry_directory'])/'status.json')['state']=='completed'
    mh=sha(ROOT/'local_data_v2'/'val.json')
    assert mh==meta['manifest_sha256']==geometry['manifest_sha256']==cfg['manifest_sha256']['val.json']
    assert geometry['crs']=='EPSG:32649' and geometry['base_grid_m']==1
    assert geometry['protocol_sha256']==sha(PA/'RM_development_protocol_v1.json')
    assert geometry['script_sha256']==sha(ROOT/'geo_management.py')
    for sid,g in geometry['sources'].items():
        assert g['tif_sha256']==meta['sources'][sid]['tif_sha256_from_dataset_summary']
        assert g['mask_sha256']==meta['sources'][sid]['mask_sha256_from_dataset_summary']
        assert np.allclose(g['transform'][:6],meta['sources'][sid]['affine'],rtol=0,atol=1e-15)
    ownership=v.ev.check_ownership(manifest,meta)
    expected={r['id']:r for r in manifest};windows={r['id']:r for r in meta['windows']}
    assert len(meta['windows'])==len(windows)==357 and set(windows)==set(expected)
    for key,m in windows.items():
        r=expected[key];p=Path(m['file']).resolve();assert p.is_relative_to(maps_dir) and p.suffix=='.npz'
        assert m['split']=='val' and m['factors']==[1,2,4] and m['source_id']==r['source_id']
        assert m['native_window_xywh']==[r['x'],r['y'],512,512] and m['native_pixels_per_output_pixel']==[2,2]
        assert m['checkpoint_sha256']==meta['checkpoint_sha256'] and m['manifest_sha256']==mh
        assert sha(p)==m['file_sha256'];watched[str(p)]=m['file_sha256']
    for g in geometry['windows']:assert sha(g['file'])==g['sha256'];watched[str(Path(g['file']).resolve())]=g['sha256']
    for name in ('metadata.json','status.json','verification.json','predictions.csv','summary.json'):watched[str(maps_dir/name)]=sha(maps_dir/name)
    receipt=VERSION/'completion_records'/(rid+'.json');assert receipt.is_file();watched[str(receipt)]=sha(receipt)
    verify(watched)
    return meta,geometry,manifest,ownership,watched

def build_units(base_cells,base_values):
    np,v,_,_,_=RUNTIME;unit=[];checks=[];total=[math.fsum(base_values[:,i]) for i in range(5)]
    for m in M_VALUES:
        cells,values=v.ev.gm.aggregate_nested(base_cells,base_values,m)
        totals=[math.fsum(values[:,i]) for i in range(5)]
        errors=[v.ev.relative_error(a,b) for a,b in zip(totals,total)];assert max(errors)<=1e-10
        A,T,P=values[:,0],values[:,1],values[:,2:];ids=v.ev.cell_ids(m,cells)
        assert len(set(ids))==len(ids) and np.all(A>0) and np.all((T>=0)&(T<=A+1e-12))
        assert np.isfinite(P).all() and np.all((P>=0)&(P<=A[:,None]+1e-12))
        for i,(ix,iy) in enumerate(cells):
            unit.append({'M':m,'ix':int(ix),'iy':int(iy),'id':str(ids[i]),'area':float(A[i]),'target':float(T[i]),'q_true':float(T[i]/A[i]),**{f'predicted_R{r}':float(P[i,j]) for j,r in enumerate(R_VALUES)},**{f'q_R{r}':float(P[i,j]/A[i]) for j,r in enumerate(R_VALUES)}})
        checks.append({'M':m,'units':len(A),'area':totals[0],'target':totals[1],'nested_relative_errors':errors})
    return unit,checks

def seal_bounds(dest,allowed,domain,protocol):
    np,v,_,_,_=RUNTIME;seal=v.verify_selection_seal(dest);data=v.prepare_data(allowed,domain['area_m2'])
    audit=load_bound_function(protocol);folder=dest/'bound_records';folder.mkdir();records=[]
    for m,r,b in v.CASES:
        case=(m,r,b);name=v.recovery.case_name(case);choice=read(dest/'choices'/name);raw=read(dest/'raw'/name)
        assert (choice['M'],choice['R'],choice['budget'])==case and choice['raw_sha256']==sha(dest/'raw'/name)
        ids,A,Ps=data[m];P=Ps[r];lookup={key:i for i,key in enumerate(ids)};selected=choice['selected_ids']
        assert len(selected)==len(set(selected)) and set(selected).issubset(lookup)
        area=math.fsum(A[lookup[i]] for i in selected);primal=math.fsum(P[lookup[i]] for i in selected);limit=b*math.fsum(A)
        assert area<=limit;close(area,choice['selected_area_m2']);close(primal,choice['predicted_primal_m2'])
        assert choice['selection_input_sha256']==v.policy.payload_sha({'ids':ids,'area':A.tolist(),'predicted':P.tolist(),'budget':b})
        prefix=v.policy.prefix_ids(ids,A,P,b);pa=math.fsum(A[lookup[i]] for i in prefix);pp=math.fsum(P[lookup[i]] for i in prefix)
        bound=audit(primal,pp,pa<=limit,math.fsum(P),raw.get('raw_solver_dual_upper_m2'),choice['recomputed_bound']['upper_m2'],choice['optimality_proven'],choice['fallback_used'],v.policy.OPTIONS['mip_rel_gap'])
        record={'M':m,'R':r,'budget':b,'selected_ids':selected,'prefix_ids':prefix,'selected_area_m2':area,'predicted_primal_m2':primal,'prefix_area_m2':pa,'prefix_predicted_m2':pp,'prefix_strictly_feasible':pa<=limit,'budget_area_m2':limit,'final_ids_unchanged':True,'choice_sha256':sha(dest/'choices'/name),'raw_sha256':sha(dest/'raw'/name),'bound_audit':bound}
        write(folder/name,record);records.append(record)
    assert len(records)==45
    bound_seal={'all_45_bound_records_saved':True,'annotations_joined':False,'selection_seal_sha256':sha(dest/'selection_seal.json'),'protocol_sha256':sha(PROTOCOL),'implementation_sha256':sha(__file__),'bound_rule':protocol['bound_rule'],'records_sha256':{p.relative_to(dest).as_posix():sha(p) for p in sorted(folder.glob('*.json'))}}
    write(dest/'bound_seal.json',bound_seal);v.verify_selection_seal(dest);return records

def verify_bound_seal(dest):
    _,v,_,_,_=RUNTIME;v.verify_selection_seal(dest);seal=read(dest/'bound_seal.json')
    assert seal['all_45_bound_records_saved'] and not seal['annotations_joined']
    assert seal['selection_seal_sha256']==sha(dest/'selection_seal.json') and seal['protocol_sha256']==sha(PROTOCOL) and seal['implementation_sha256']==sha(__file__)
    assert seal['bound_rule']==read(PROTOCOL)['bound_rule']
    assert set(seal['records_sha256'])=={'bound_records/'+v.recovery.case_name(c) for c in v.CASES}
    for rel,h in seal['records_sha256'].items():assert sha(checked_path(dest,rel))==h
    return seal

def reference_equivalence(unit,reference):
    a={(int(x['M']),x['id']):x for x in unit};b={(int(x['M']),x['id']):x for x in reference}
    assert len(a)==len(unit) and len(b)==len(reference) and set(a)==set(b)
    maxerr={'area':0.,'target':0.}
    for key,x in a.items():
        y=b[key];assert int(x['ix'])==int(y['ix']) and int(x['iy'])==int(y['iy'])
        for field in maxerr:
            close(x[field],y[field]);maxerr[field]=max(maxerr[field],abs(float(x[field])-float(y[field])))
    return {'matched_rows':len(a),'matched_by_M':{str(m):sum(k[0]==m for k in a) for m in M_VALUES},'max_absolute_difference_m2':maxerr,'rtol':1e-10,'atol_m2':1e-9}

def prefix_tables(unit):
    np,v,_,_,_=RUNTIME;curves=[];fixed=[];auc=[]
    for m in M_VALUES:
        subset=sorted((x for x in unit if int(x['M'])==m),key=lambda x:x['id']);ids=np.array([x['id'] for x in subset]);A=np.array([float(x['area']) for x in subset]);T=np.array([float(x['target']) for x in subset])
        for r in R_VALUES:
            P=np.array([float(x[f'predicted_R{r}']) for x in subset]);order=np.lexsort((ids,-P/A));prefix=v.ev.ordered_prefix(A,T,order)
            for k in range(len(A)+1):curves.append({'M':m,'method':'model_prefix','R':r,'k':k,'last_added_id':str(ids[order[k-1]]) if k else '', 'actual_area':float(prefix['area'][k]),'actual_area_fraction':float(prefix['area_fraction'][k]),'captured_target':float(prefix['target'][k]),'capture_fraction':float(prefix['capture_fraction'][k]),'missed_target':float(prefix['total_target']-prefix['target'][k]),'zero_annotation_area':float(prefix['area'][k]-prefix['target'][k])})
            for b in BUDGETS:fixed.append({'M':m,'method':'model_prefix','R':r,**v.ev.prefix_budget(prefix,b),'selected_ids_json':json.dumps(v.policy.prefix_ids(ids,A,P,b))})
            auc.append({'M':m,'R':r,'method':'model_prefix','auc_0_1':v.ev.step_auc(prefix),'auc_0_0p5_normalized':v.ev.step_auc(prefix,.5),'curve_points':len(A)+1})
    assert len(fixed)==45 and len(auc)==9;return curves,fixed,auc

def evaluate_annotations(out,protocol,domain):
    np,v,_,_,_=RUNTIME;dest=out/'predicted_area_policy';verify_bound_seal(dest) # first gate, before target/reference reads
    unit=rows(out/'unit_predictions.csv');allowed=rows(out/'selection_inputs_without_annotations.csv')
    assert v.policy.payload_sha(v.policy.allowed_selection_rows(unit))==v.policy.payload_sha(allowed)
    reference=Path(protocol['shared_reference']['directory']);equivalence=reference_equivalence(unit,rows(reference/'unit_predictions.csv'))
    curves,fixed,auc=prefix_tables(unit);data=v.prepare_data(allowed,domain['area_m2']);truth={(int(x['M']),x['id']):float(x['target']) for x in unit}
    cuts={(x['M'],x['R'],x['budget']):x for x in fixed};outcomes=[]
    for m,r,b in v.CASES:
        name=v.recovery.case_name((m,r,b));record=read(dest/'bound_records'/name);choice=read(dest/'choices'/name);raw=read(dest/'raw'/name)
        assert record['choice_sha256']==sha(dest/'choices'/name) and record['raw_sha256']==sha(dest/'raw'/name)
        ids,A,Ps=data[m];P=Ps[r];T=np.array([truth[(m,i)] for i in ids]);close(math.fsum(T),domain['target_m2'])
        assert np.isfinite(T).all() and np.all((T>=0)&(T<=A+1e-9))
        current=v.policy.evaluate_selected(ids,record['selected_ids'],A,P,T);prefix=v.policy.evaluate_selected(ids,record['prefix_ids'],A,P,T)
        assert current['actual_area_m2']<=record['budget_area_m2'];close(current['predicted_objective_m2'],record['predicted_primal_m2'])
        assert prefix['actual_area_m2']<=record['budget_area_m2'],'Original prefix not strict feasible; preserve attempt without silently modifying its policy'
        for field,column in (('selected_count','k'),('actual_area_m2','actual_area'),('captured_target_m2','captured_target'),('capture_fraction','capture_fraction'),('zero_annotation_area_m2','zero_annotation_area')):close(prefix[field],cuts[(m,r,b)][column])
        audit=record['bound_audit'];outcomes.append({'run_id':read(out/'inputs.json')['run_id'],'M':m,'R':r,'budget':b,'budget_area_m2':record['budget_area_m2'],'selected_ids_json':json.dumps(record['selected_ids']),'prefix_ids_json':json.dumps(record['prefix_ids']),'final_ids_unchanged':True,'strict_original_area_feasible':True,'fallback_used':choice['fallback_used'],'fallback_reasons':';'.join(choice['fallback_reasons']),'raw_solver_status':raw['solver_status'],'raw_solver_gap':raw['raw_solver_gap'],'original_proven':audit['original_proven_flag'],'optimality_proven':audit['updated_proven_flag'],'predicted_upper_m2':audit['updated_upper_m2'],'predicted_relative_gap':audit['updated_relative_gap'],'bound_anomaly':audit['bound_anomaly'],'certification_downgraded':audit['certification_downgraded'],'roundoff_clamp_m2':audit['roundoff_clamp_m2'],'bound_audit_json':json.dumps(audit,sort_keys=True),**{'pmilp_'+k:z for k,z in current.items()},**{'prefix_'+k:z for k,z in prefix.items()},'delta_capture_pp':100*(current['capture_fraction']-prefix['capture_fraction']),'delta_predicted_objective_m2':current['predicted_objective_m2']-prefix['predicted_objective_m2'],'unused_budget_m2':record['budget_area_m2']-current['actual_area_m2'],'choice_sha256':record['choice_sha256']})
    write_csv(out/'decision_curves.csv',curves);write_csv(out/'fixed_budgets.csv',fixed);write_csv(out/'prefix_auc.csv',auc);write_csv(out/'all_45_policy_outcomes.csv',outcomes)
    reference_rows=[x for x in rows(reference/'fixed_budgets.csv') if x['method']!='model_prefix'];assert len(reference_rows)==45
    write_csv(out/'shared_reference_fixed_budgets.csv',reference_rows)
    ref={'shared_across_models':True,'not_model_replicates':True,'directory':str(reference),'files_sha256':protocol['shared_reference']['files_sha256'],'domain_equivalence':equivalence,'new_label_solver_calls':0,'new_random_orders':0,'interpretation':'Random quantiles describe within-scene order variation; label prefix is not a global optimum. Saved label MILP retains its original primal/dual/gap; no secondary minimum-area objective.'};write(out/'shared_reference.json',ref)
    verify_bound_seal(dest)
    return {'model_prefix_budget_rows':len(fixed),'predicted_area_budget_rows':len(outcomes),'model_full_curves':9,'curve_points':len(curves),'prefix_auc_rows':len(auc),'fallback_count':sum(x['fallback_used'] for x in outcomes),'updated_certified_count':sum(x['optimality_proven'] for x in outcomes),'bound_anomaly_count':sum(x['bound_anomaly'] for x in outcomes),'certification_downgraded_count':sum(x['certification_downgraded'] for x in outcomes),'vs_prefix_higher':sum(x['delta_capture_pp']>1e-8 for x in outcomes),'vs_prefix_lower':sum(x['delta_capture_pp']< -1e-8 for x in outcomes),'vs_prefix_tie':sum(abs(x['delta_capture_pp'])<=1e-8 for x in outcomes),'shared_reference':ref}

def preserve_failure(out,state,error,child=None,command=None):
    state.update(state='failed',failed_at=stamp(),error=repr(error),traceback=traceback.format_exc(),claim_retained=True,no_retry=True,child_pid=None if child is None else child.pid,child_command=command)
    if child is not None:
        cleanup={'owned_pid':child.pid}
        try:
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=20)
            cleanup['returncode']=child.returncode
        except BaseException as e:cleanup['error']=repr(e)
        state['owned_child_cleanup']=cleanup
    try:out.mkdir(exist_ok=True);write(out/'status.json',state)
    except BaseException as e:
        state['status_write_error']=repr(e)
        try:
            with (PA/('lsm_failure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')).open('x',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False,indent=2)
        except BaseException:print(json.dumps(state,ensure_ascii=False),file=sys.stderr,flush=True)

def run_policy(out,expected,token):
    protocol,_=load_protocol(expected);out=Path(out).resolve();assert out.parent==PA.resolve() and out.name.startswith('lsm2_')
    state=read(out/'status.json');assert state['state']=='running' and state['stage']=='policy_subprocess'
    assert sha(out/'child_inputs.json')==state['child_inputs_sha256']
    inputs=read(out/'child_inputs.json')
    assert set(inputs)=={'run_id','protocol_sha256','checkpoint_sha256','file_sha256','area_m2'}
    assert isinstance(inputs['area_m2'],(int,float)) and math.isfinite(inputs['area_m2']) and inputs['area_m2']>0
    assert state['child_token_sha256']==hashlib.sha256(token.encode()).hexdigest() and inputs['protocol_sha256']==expected
    claim=read(VERSION/'management_claims_v2'/(inputs['run_id']+'.json'));assert Path(claim['output']).resolve()==out
    assert inputs['run_id'] in {r['run_id'] for r in protocol['declared_runs']};verify(inputs['file_sha256'])
    assert sha(out/'selection_inputs_without_annotations.csv')==state['selection_inputs_sha256']
    childstate={'state':'running','pid':os.getpid(),'stage':'initialization','test_accessed':False,'R6_accessed':False}
    claim_stream=(out/'POLICY_CLAIM.json').open('x',encoding='utf-8')
    try:
        with claim_stream as f:json.dump({'pid':os.getpid(),'created_at':stamp(),'protocol_sha256':expected},f)
        np,v,_,limits,_=runtime(protocol);dest=out/'predicted_area_policy';dest.mkdir()
        with limits(limits=2):
            allowed=rows(out/'selection_inputs_without_annotations.csv');assert all(set(x)==set(v.FIELDS) for x in allowed)
            context={'run_id':inputs['run_id'],'protocol_sha256':expected,'inputs_sha256':sha(out/'child_inputs.json'),'checkpoint_sha256':inputs['checkpoint_sha256'],'new_reference_solver_calls':0}
            v.choose_all(dest,allowed,inputs['area_m2'],context)
            seal_bounds(dest,allowed,{'area_m2':inputs['area_m2']},protocol);verify_bound_seal(dest);verify(inputs['file_sha256'])
        childstate.update(state='completed',selection_seal_sha256=sha(dest/'selection_seal.json'),bound_seal_sha256=sha(dest/'bound_seal.json'),solver_calls=45,annotations_joined=False)
        write(out/'policy_status.json',childstate)
    except BaseException as e:
        childstate.update(state='failed',error=repr(e),traceback=traceback.format_exc(),no_retry=True)
        try:write(out/'policy_status.json',childstate)
        except BaseException:print(json.dumps(childstate),file=sys.stderr,flush=True)
        raise

def execute(maps_dir,expected,root_review):
    protocol,watched=load_protocol(expected);maps_dir=Path(maps_dir).resolve();meta=read(maps_dir/'metadata.json');rid=meta['run_id'];reject_existing(rid)
    review=read(root_review);assert review['passed'] and review['execution_scope']=='local_seed_management_v2_remaining14'
    assert review['protocol_sha256']==expected and review['implementation_sha256']==sha(__file__) and not review.get('test_or_R6_authorized',False)
    out=PA/('lsm2_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));claims=VERSION/'management_claims_v2';claims.mkdir(exist_ok=True)
    state={'state':'running','stage':'initialization','run_id':rid,'pid':os.getpid(),'created_at':stamp(),'output_directory':str(out),'protocol_sha256':expected,'test_accessed':False,'R6_accessed':False};child=None;command=None;started=time.time()
    claim_stream=(claims/(rid+'.json')).open('x',encoding='utf-8')
    try:
        with claim_stream as f:json.dump({**state,'output':str(out),'root_review':str(Path(root_review).resolve()),'root_review_sha256':sha(root_review)},f,ensure_ascii=False,indent=2)
        out.mkdir();write(out/'status.json',state)
        np,v,_,limits,info=runtime(protocol)
        with limits(limits=2):
            meta,geometry,manifest,ownership,watched=validate_maps(maps_dir,protocol,watched)
            watched[str(Path(root_review).resolve())]=sha(root_review)
            sources=out/'source';sources.mkdir();copies=[]
            for i,p in enumerate([Path(__file__),PROTOCOL]+[Path(x) for x in protocol['archive_source_paths']]):
                dest=sources/(f'{i:02d}'+p.suffix);shutil.copy2(p,dest);assert sha(dest)==sha(p);copies.append({'original':str(p),'copy':dest.relative_to(out).as_posix(),'sha256':sha(p)})
            write(out/'source_ledger.json',{'records':copies,'file_sha256':watched,'threadpools':info(),'python':sys.version,'numpy':np.__version__,'scipy':v.ev.scipy.__version__,'no_model_import':True})
            domain={k:geometry[k] for k in ('native_valid_pixels','native_target_pixels','area_m2','target_m2','origin_utm_m','crs')}
            inputs={'run_id':rid,'seed':meta['seed'],'encoder':meta['encoder'],'variant':meta['variant'],'lr':meta['lr'],'paired_seed17_run_id':meta['paired_seed17_run_id'],'checkpoint_epoch':meta['checkpoint_epoch'],'checkpoint_sha256':meta['checkpoint_sha256'],'protocol_sha256':expected,'maps_directory':str(maps_dir),'file_sha256':watched,'source_ledger_sha256':sha(out/'source_ledger.json'),'domain':domain,'ownership':ownership,'test_accessed':False,'R6_accessed':False}
            write(out/'inputs.json',inputs)
            cells,values,integration,windows=v.ev.integrate(geometry,meta,manifest,out/'status.json',state)
            write(out/'integration_checks.json',integration);inputs['windows']=windows;write(out/'inputs.json',inputs)
            unit,nested=build_units(cells,values);write_csv(out/'unit_predictions.csv',unit);write(out/'nested_checks.json',nested)
            # Re-open CSV so the child and later target join see identical string payloads.
            allowed=v.policy.allowed_selection_rows(rows(out/'unit_predictions.csv'));write_csv(out/'selection_inputs_without_annotations.csv',allowed)
            write(out/'child_inputs.json',{'run_id':rid,'protocol_sha256':expected,'checkpoint_sha256':meta['checkpoint_sha256'],'file_sha256':watched,'area_m2':domain['area_m2']})
            token=secrets.token_hex(24);state.update(stage='policy_subprocess',unit_predictions_sha256=sha(out/'unit_predictions.csv'),selection_inputs_sha256=sha(out/'selection_inputs_without_annotations.csv'),child_inputs_sha256=sha(out/'child_inputs.json'),child_token_sha256=hashlib.sha256(token.encode()).hexdigest());write(out/'status.json',state)
            command=[sys.executable,'-X','utf8','-B','-s',str(Path(__file__).resolve()),'--policy-for',str(out),'--protocol-sha256',expected,'--child-token',token]
            with (out/'policy_stdout.log').open('x',encoding='utf-8') as stdout,(out/'policy_stderr.log').open('x',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state['child_pid']=child.pid;write(out/'status.json',state);code=child.wait()
            assert code==0 and read(out/'policy_status.json')['state']=='completed','Child failed; preserve evidence, no retry'
            ps=read(out/'policy_status.json')
            assert ps['selection_seal_sha256']==sha(out/'predicted_area_policy'/'selection_seal.json') and ps['bound_seal_sha256']==sha(out/'predicted_area_policy'/'bound_seal.json')
            assert sha(out/'unit_predictions.csv')==state['unit_predictions_sha256'] and sha(out/'selection_inputs_without_annotations.csv')==state['selection_inputs_sha256']
            result=evaluate_annotations(out,protocol,domain);verify(watched)
            summary={'state':'completed',**{k:inputs[k] for k in ('run_id','seed','encoder','variant','lr','paired_seed17_run_id','checkpoint_epoch','checkpoint_sha256','protocol_sha256','domain')},'results':result,'integration':{k:z for k,z in integration.items() if k!='per_window'},'elapsed_seconds':time.time()-started,'new_predicted_area_solver_calls':45,'new_label_solver_calls':0,'new_random_orders':0,'test_accessed':False,'R6_accessed':False,'model_tuned':False}
            write(out/'summary.json',summary)
            (out/'report.md').write_text('# Local fixed-seed spatial development\n\nComplete nine prefix curves, nine full/half AUC rows,45 fixed prefix budgets and45 predicted-area choices are retained. Original checkpoint selection is unchanged. All45 raw solver,selector and recovered choices were sealed before label-free bound consistency audit; annotation metrics follow both seals.\n\nShared label/random references use the identical domain and unit A/T, not additional model replicates. Same budget can use different actual area. No minimum-area optimality, field independence, significance, treatment efficacy or test/R6 claim.\n\n'+json.dumps(summary['results'],ensure_ascii=False,indent=2),encoding='utf-8')
            state.update(state='completed',stage='complete',completed_at=stamp(),summary_sha256=sha(out/'summary.json'),selection_seal_sha256=sha(out/'predicted_area_policy'/'selection_seal.json'),bound_seal_sha256=sha(out/'predicted_area_policy'/'bound_seal.json'))
            state['files_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'status.json'};write(out/'status.json',state)
            print(json.dumps({'state':'completed','output':str(out),'run_id':rid,'results':result}),flush=True)
    except BaseException as e:preserve_failure(out,state,e,child,command);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--maps-dir');g.add_argument('--policy-for',help=argparse.SUPPRESS)
    p.add_argument('--protocol-sha256',required=True);p.add_argument('--validate-only',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--root-review');p.add_argument('--child-token',help=argparse.SUPPRESS);a=p.parse_args()
    if a.policy_for:
        assert not a.execute and not a.validate_only and a.child_token;run_policy(a.policy_for,a.protocol_sha256,a.child_token)
    elif a.validate_only:
        assert not a.execute and not a.root_review;protocol,watched=load_protocol(a.protocol_sha256);rid=read(Path(a.maps_dir)/'metadata.json')['run_id'];reject_existing(rid)
        meta,_,_,_,watch=validate_maps(a.maps_dir,protocol,watched);print(json.dumps({'state':'validated_only','run_id':meta['run_id'],'checkpoint_epoch':meta['checkpoint_epoch'],'file_sha256':watch,'new_solver_calls':0,'probability_integrations':0}),flush=True)
    else:
        assert a.execute and a.root_review;execute(a.maps_dir,a.protocol_sha256,a.root_review)
if __name__=='__main__':main()
