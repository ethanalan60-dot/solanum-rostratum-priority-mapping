"""Independent recomputation from saved validation outputs. No models, raw labels or solvers."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,math,os,traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).resolve().read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,atol=1e-9,rtol=1e-9):
    a=float(a);b=float(b)
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*abs(b),(a,b)
def verify_outputs(folder,kind):
    manifest=read(folder/('artifact_manifest.json' if kind=='export' else 'status.json'))['file_sha256' if kind=='export' else 'files_sha256']
    for name,h in manifest.items():assert sha(folder/name)==h,(folder,name)
    return len(manifest)
def audit(receipt):
    record=read(receipt);export=Path(record['export_directory']);spatial=Path(record['spatial_directory'])
    for key,path in [('export_status_sha256',export/'status.json'),('export_manifest_sha256',export/'artifact_manifest.json'),('spatial_status_sha256',spatial/'status.json'),('spatial_summary_sha256',spatial/'summary.json')]:assert sha(path)==record[key]
    es=read(export/'status.json');ss=read(spatial/'status.json');meta=read(export/'metadata.json');summary=read(spatial/'summary.json')
    assert es['status']=='completed' and ss['state']=='completed' and record['run_id']==meta['run_id']==summary['run_id']
    files=verify_outputs(export,'export')+verify_outputs(spatial,'spatial')
    assert meta['factors']==[1,2,4] and not meta['test_accessed'] and not meta['R6_accessed']
    config=read(meta['config']);best=read(Path(meta['config']).parent/'best_validation.json')
    assert meta['checkpoint_epoch']==best['epoch'] and meta['checkpoint_sha256']==sha(meta['checkpoint'])
    predicted=rows(export/'predictions.csv');pred={(x['id'],int(x['scale_factor'])):x for x in predicted}
    assert len(pred)==len(predicted)==1071
    proto=read(ROOT/'precision_agriculture'/('RM_local_seed_protocol_v2.json' if spatial.name.startswith('lsm2_') else 'RM_local_seed_protocol_v1.json'))
    geometry=read(Path(proto['geometry_directory'])/'metadata.json');geos={x['id']:x for x in geometry['windows']}
    assert len(meta['windows'])==len(geos)==357
    assert set(geos)=={x['id'] for x in meta['windows']}==set(config['selected_val_ids'])
    base={};max_scalar=0.;max_mass=0.
    for item in sorted(meta['windows'],key=lambda x:x['id']):
        key=item['id'];g=geos[key];assert sha(item['file'])==item['file_sha256'] and sha(g['file'])==g['sha256']
        with np.load(item['file'],allow_pickle=False) as z:p=z['p'];v=z['valid']
        assert p.dtype==np.float32 and p.shape==(3,256,256) and v.shape==(256,256)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
        with np.load(g['file'],allow_pickle=False) as z:
            cells=z['cells'];area=z['area'];target=z['target'];data=z['weights'];columns=z['indices'];ptr=z['indptr'];raw=z['raw_valid_by_support']
        rowids=np.repeat(np.arange(len(cells)),np.diff(ptr))
        independent=np.stack([np.bincount(rowids,weights=data*p[j].ravel()[columns].astype(np.float64),minlength=len(cells)) for j in range(3)],axis=1)
        rowmass=np.bincount(rowids,weights=data,minlength=len(cells))
        err=float(np.max(np.abs(rowmass-area)));max_mass=max(max_mass,err);assert err<=1e-9
        assert np.array_equal(raw>0,v.ravel()>0) and float(v.sum())*4==g['native_valid_pixels']
        for j,r in enumerate((1,2,4)):
            estimate=float(np.sum(p[j].astype(np.float64)*v,dtype=np.float64)/np.sum(v,dtype=np.float64));stored=pred[(key,r)]
            error=abs(estimate-float(stored['coverage_pred']));max_scalar=max(max_scalar,error);assert error<=1e-6
            close(float(stored['coverage_pred'])-float(stored['coverage_true']),stored['signed_error'],atol=1e-10)
            close(abs(float(stored['coverage_pred'])-float(stored['coverage_true'])),stored['absolute_error'],atol=1e-10)
            close(math.fsum(independent[:,j]),math.fsum(raw*p[j].ravel()),atol=1e-10)
        for i,(x,y) in enumerate(cells):
            key=(int(x),int(y));base.setdefault(key,[]).append([float(area[i]),float(target[i]),*map(float,independent[i])])
    base={key:[math.fsum(v[i] for v in values) for i in range(5)] for key,values in base.items()}
    units=rows(spatial/'unit_predictions.csv');look={(int(x['M']),x['id']):x for x in units};max_integration=0.
    assert len(look)==len(units)==2183
    for m in (1,2,4):
        aggregate={}
        for (x,y),values in base.items():aggregate.setdefault((x//m,y//m),[]).append(values)
        for (x,y),vs in aggregate.items():
            value=[math.fsum(v[i] for v in vs) for i in range(5)];identifier=f'rmv1_m{m}_x{x:+07d}_y{y:+07d}'
            row=look[(m,identifier)]
            for i,column in enumerate(('area','target','predicted_R1','predicted_R2','predicted_R4')):
                max_integration=max(max_integration,abs(value[i]-float(row[column])));close(value[i],row[column],atol=1e-10)
    fixed=rows(spatial/'fixed_budgets.csv');curves=rows(spatial/'decision_curves.csv');aucs=rows(spatial/'prefix_auc.csv');outcomes=rows(spatial/'all_45_policy_outcomes.csv')
    assert (len(fixed),len(curves),len(aucs),len(outcomes))==(45,6558,9,45)
    expected_cases={(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in (.1,.2,.3,.5,1.)}
    assert {(int(x['M']),int(x['R']),float(x['budget'])) for x in outcomes}==expected_cases
    assert all(x['run_id']==record['run_id'] for x in outcomes)

    fixed_idx={(int(x['M']),int(x['R']),float(x['budget'])):x for x in fixed}
    auc_idx={(int(x['M']),int(x['R'])):x for x in aucs}
    curve_idx={}
    for row in curves:curve_idx.setdefault((int(row['M']),int(row['R'])),[]).append(row)
    order_by={}
    for m in (1,2,4):
        group={x['id']:x for x in units if int(x['M'])==m};A=math.fsum(float(x['area']) for x in group.values());T=math.fsum(float(x['target']) for x in group.values())
        close(A,summary['domain']['area_m2']);close(T,summary['domain']['target_m2'])
        for r in (1,2,4):
            order=sorted(group,key=lambda i:(-float(group[i]['predicted_R'+str(r)])/float(group[i]['area']),i));order_by[(m,r)]=order
            saved=curve_idx[(m,r)];assert len(saved)==len(group)+1 and [int(x['k']) for x in saved]==list(range(len(group)+1))
            for k,row in enumerate(saved):
                ids=order[:k];area=math.fsum(float(group[i]['area']) for i in ids);target=math.fsum(float(group[i]['target']) for i in ids)
                for field,value in [('actual_area',area),('actual_area_fraction',area/A),('captured_target',target),('capture_fraction',target/T),('missed_target',T-target),('zero_annotation_area',area-target)]:close(value,row[field],atol=1e-8)
                assert row['last_added_id']==(order[k-1] if k else '')
            for upper,column in [(1,'auc_0_1'),(.5,'auc_0_0p5_normalized')]:
                exact=math.fsum(float(left['capture_fraction'])*max(0,min(float(right['actual_area_fraction']),upper)-float(left['actual_area_fraction'])) for left,right in zip(saved,saved[1:]))/upper
                close(exact,auc_idx[(m,r)][column],atol=1e-12)
            for b in (.1,.2,.3,.5,1.):
                row=fixed_idx[(m,r,b)];ids=json.loads(row['selected_ids_json']);k=int(row['k'])
                assert ids==order[:k] and len(ids)==k
                area=math.fsum(float(group[i]['area']) for i in ids);target=math.fsum(float(group[i]['target']) for i in ids)
                assert area<=b*A
                if k<len(order):assert area+float(group[order[k]]['area'])>b*A-1e-9
                close(area,row['actual_area']);close(target,row['captured_target']);close(target/T,row['capture_fraction'])
    certification=fallback=anomaly=downgraded=0
    for row in outcomes:
        m,r,b=int(row['M']),int(row['R']),float(row['budget']);group={x['id']:x for x in units if int(x['M'])==m}
        A=math.fsum(float(x['area']) for x in group.values());T=math.fsum(float(x['target']) for x in group.values());limit=b*A
        for prefix,key in [('pmilp','selected_ids_json'),('prefix','prefix_ids_json')]:
            ids=json.loads(row[key]);assert len(ids)==len(set(ids)) and set(ids)<=set(group)
            if prefix=='prefix':assert ids==json.loads(fixed_idx[(m,r,b)]['selected_ids_json'])
            area=math.fsum(float(group[i]['area']) for i in ids);target=math.fsum(float(group[i]['target']) for i in ids);prediction=math.fsum(float(group[i]['predicted_R'+str(r)]) for i in ids)
            assert area<=limit
            for key,value in [('selected_count',len(ids)),('actual_area_m2',area),('actual_area_fraction',area/A),('captured_target_m2',target),('capture_fraction',target/T),('missed_target_m2',T-target),('zero_annotation_area_m2',area-target),('predicted_objective_m2',prediction)]:close(value,row[prefix+'_'+key],atol=1e-9)
        close(float(row['pmilp_capture_fraction'])-float(row['prefix_capture_fraction']),float(row['delta_capture_pp'])/100,atol=1e-12)
        bound=json.loads(row['bound_audit_json']);upper=float(row['predicted_upper_m2']);primal=float(row['pmilp_predicted_objective_m2'])
        feasible=max(primal,float(row['prefix_predicted_objective_m2']))
        assert upper+1e-9>=feasible
        close(max(0.,upper-primal)/max(abs(primal),1e-30),row['predicted_relative_gap'],atol=1e-9)
        certification+=row['optimality_proven']=='True';fallback+=row['fallback_used']=='True';anomaly+=row['bound_anomaly']=='True';downgraded+=row['certification_downgraded']=='True'
        name=f'M{m}_R{r}_B{round(b*100):03d}.json';choice=spatial/'predicted_area_policy/choices'/name
        assert sha(choice)==row['choice_sha256']
        check=read(choice);assert check['selected_ids']==json.loads(row['selected_ids_json'])
        raw=read(spatial/'predicted_area_policy/raw'/name)
        total_prediction=math.fsum(float(u['predicted_R'+str(r)]) for u in group.values())
        tolerance=max(1e-10,1e-10*total_prediction)
        raw_upper=raw.get('raw_solver_dual_upper_m2');retained=check['recomputed_bound']['upper_m2']
        finite=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
        contradictions=[key for key,value in [('raw_solver_upper',raw_upper),('retained_upper',retained)] if finite(value) and value<feasible-tolerance]
        reasons=contradictions+([] if finite(retained) else ['missing_or_nonfinite_retained_upper'])
        if reasons:expected_upper=total_prediction;clamp=0.
        else:
            unclamped=min([total_prediction,float(retained)]+([float(raw_upper)] if finite(raw_upper) else []))
            expected_upper=max(unclamped,feasible);clamp=max(0.,feasible-unclamped)
            assert clamp<=tolerance
        expected_gap=(expected_upper-primal)/max(abs(primal),1e-30) if expected_upper else 0.
        qualifies=bool(check['optimality_proven'] and not check['fallback_used'] and not reasons and expected_gap<=proto['solver_options']['mip_rel_gap']+1e-10)
        close(expected_upper,upper,atol=1e-10)
        close(expected_gap,row['predicted_relative_gap'],atol=1e-10)
        close(clamp,row['roundoff_clamp_m2'],atol=1e-10)
        assert (row['optimality_proven']=='True')==qualifies
        assert (row['bound_anomaly']=='True')==bool(reasons)
        assert (row['certification_downgraded']=='True')==bool(check['optimality_proven'] and not qualifies)
        assert bound['contradicted_upper_fields']==contradictions and bound['bound_anomaly_reasons']==reasons

    result=summary['results']
    assert (certification,fallback,anomaly,downgraded)==tuple(result[x] for x in ('updated_certified_count','fallback_count','bound_anomaly_count','certification_downgraded_count'))
    return {'run_id':record['run_id'],'receipt':str(receipt.resolve()),'receipt_sha256':sha(receipt),'verified_files':files,'maps':357,'scalar_pairs':1071,'management_units':2183,'prefix_curve_points':6558,'fixed_prefix_rows':45,'P_policy_rows':45,'auc_rows':9,'complete_unique_policy_cases':45,'independently_recomputed_uniform_bound_records':45,'max_scalar_map_difference_fraction':max_scalar,'max_geometry_row_mass_error_m2':max_mass,'max_independent_bincount_integration_difference_m2':max_integration,'recorded_bound_qualified':certification,'fallbacks':fallback,'bound_anomalies':anomaly,'bound_downgrades':downgraded,'no_new_model_forward':True,'no_new_solver':True,'test_or_R6_accessed':False,'limitations':['Independent recalculation from previously accepted geometry weights, not a new ground survey or raw-source polygon reconstruction','Numerical bound consistency is not an independent proof of global optimality']}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',action='append',type=Path,required=True);a=ap.parse_args()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=ROOT/'provenance'/('completed_seed_outputs_independent_'+stamp);out.mkdir()
    results=[]
    try:
        for receipt in a.receipt:
            result=audit(receipt);results.append(result);print(json.dumps({'completed_audit':result['run_id'],'maps':357}),flush=True)
        review={'passed':True,'reviewer':'root independently recomputed new agent-authored exporter/spatial outputs','models':results,'auditor_sha256':sha(__file__),'test_or_R6_accessed':False,'new_model_forward_calls':0,'new_solver_calls':0}
    except BaseException as error:
        review={'passed':False,'models':results,'error':repr(error),'traceback':traceback.format_exc(),'auditor_sha256':sha(__file__)}
        (out/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8');raise
    (out/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'output':str(out)},ensure_ascii=False))
if __name__=='__main__':main()
