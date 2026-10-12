"""Bounded second review of root's first-two completed seed-output audit.

Counts/hashes are checked in full. Only two predetermined map windows per model
and four predetermined management cases are numerically sampled. No production
auditor, model, solver, raw source image, mask or test data is executed/opened.
"""
from pathlib import Path
import csv,datetime,hashlib,json,math,os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np
ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT/'provenance/completed_seed_outputs_independent_20260916_095939_048943/review.json'
WINDOWS=('4_7_10_10','4_7_11_6')
CASES=((2,1,.1),(2,4,.1),(4,4,.1),(1,1,1.))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):assert abs(float(a)-float(b))<=tol,(a,b)
def main():
    out=ROOT/'provenance'/('first2_seed_audit_bounded_review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    audit=read(ORIGINAL);assert audit['passed']
    assert sha(ROOT/'audit_completed_seed_outputs.py')==audit['auditor_sha256']
    plan=read(ROOT/'local_seed_stability_v1/plan.json')
    assert [x['run_id'] for x in audit['models']]==[x['run_id'] for x in plan['runs'][:2]]
    geometry_root=ROOT/'precision_agriculture/geometry_20260916_v1'
    geometry=read(geometry_root/'metadata.json');geos={x['id']:x for x in geometry['windows']}
    results=[]
    for model in audit['models']:
        assert sha(model['receipt'])==model['receipt_sha256'];receipt=read(model['receipt'])
        export,spatial=Path(receipt['export_directory']),Path(receipt['spatial_directory'])
        for key,path in [('export_status_sha256',export/'status.json'),('export_manifest_sha256',export/'artifact_manifest.json'),('spatial_status_sha256',spatial/'status.json'),('spatial_summary_sha256',spatial/'summary.json')]:assert sha(path)==receipt[key]
        verified=0
        for folder,filename,field in ((export,'artifact_manifest.json','file_sha256'),(spatial,'status.json','files_sha256')):
            listed=read(folder/filename)[field]
            actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!=filename}
            assert actual==set(listed)
            for relative,digest in listed.items():assert sha(folder/relative)==digest
            verified+=len(listed)
        assert verified==model['verified_files']==615
        meta=read(export/'metadata.json');summary=read(spatial/'summary.json')
        assert meta['checkpoint_epoch']==summary['checkpoint_epoch']
        maps={x['id']:x for x in meta['windows']};assert len(maps)==357 and set(maps)==set(geos)
        predicted=rows(export/'predictions.csv');pidx={(x['id'],int(x['scale_factor'])):x for x in predicted}
        assert len(predicted)==len(pidx)==1071 and set(pidx)=={(i,r) for i in maps for r in (1,2,4)}
        units=rows(spatial/'unit_predictions.csv');u={(int(x['M']),x['id']):x for x in units}
        assert len(u)==len(units)==2183
        assert {m:sum(int(x['M'])==m for x in units) for m in (1,2,4)}=={1:1614,2:443,4:126}
        fixed=rows(spatial/'fixed_budgets.csv');outcomes=rows(spatial/'all_45_policy_outcomes.csv')
        expected={(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in (.1,.2,.3,.5,1.)}
        key=lambda x:(int(x['M']),int(x['R']),float(x['budget']))
        assert len(fixed)==len(outcomes)==45 and {key(x) for x in fixed}=={key(x) for x in outcomes}==expected
        oidx={key(x):x for x in outcomes};fidx={key(x):x for x in fixed}
        curves=rows(spatial/'decision_curves.csv');aucs=rows(spatial/'prefix_auc.csv')
        assert len(curves)==6558 and len(aucs)==9
        curve_keys={(int(x['M']),int(x['R']),int(x['k'])) for x in curves}
        expected_curve={(m,r,k) for m,n in ((1,1614),(2,443),(4,126)) for r in (1,2,4) for k in range(n+1)}
        assert len(curve_keys)==6558 and curve_keys==expected_curve
        assert {(int(x['M']),int(x['R'])) for x in aucs}=={(m,r) for m in (1,2,4) for r in (1,2,4)}
        map_samples=[]
        for identifier in WINDOWS:
            item,g=maps[identifier],geos[identifier]
            assert sha(item['file'])==item['file_sha256'] and sha(g['file'])==g['sha256']
            with np.load(item['file'],allow_pickle=False) as z:p=z['p'];v=z['valid']
            with np.load(g['file'],allow_pickle=False) as z:
                data=z['weights'];col=z['indices'];ptr=z['indptr'];A=z['area'];raw=z['raw_valid_by_support']
            assert p.shape==(3,256,256) and p.dtype==np.float32 and v.dtype==np.float32
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            rr=np.repeat(np.arange(len(A)),np.diff(ptr));mass=np.bincount(rr,weights=data,minlength=len(A))
            mass_error=float(abs(mass-A).max());assert mass_error<1e-9
            pairs=[]
            for j,r in enumerate((1,2,4)):
                q=math.fsum((p[j].astype(np.float64)*v).ravel())/math.fsum(v.ravel())
                error=abs(q-float(pidx[(identifier,r)]['coverage_pred']));assert error<1e-6
                cell=np.bincount(rr,weights=data*p[j].ravel()[col].astype(np.float64),minlength=len(A))
                integral_error=abs(math.fsum(cell)-math.fsum(raw*p[j].ravel()))
                assert integral_error<1e-10
                pairs.append({'R':r,'scalar_difference_fraction':error,'column_integral_difference_m2':integral_error})
            map_samples.append({'id':identifier,'source_sha256':item['file_sha256'],'mass_difference_m2':mass_error,'pairs':pairs})
        selected_samples=[]
        for m,r,b in CASES:
            row=oidx[(m,r,b)];group={x['id']:x for x in units if int(x['M'])==m}
            A=math.fsum(float(x['area']) for x in group.values());T=math.fsum(float(x['target']) for x in group.values())
            order=sorted(group,key=lambda i:(-float(group[i][f'predicted_R{r}'])/float(group[i]['area']),i))
            prefix_ids=json.loads(row['prefix_ids_json']);assert prefix_ids==order[:len(prefix_ids)]
            sample={'M':m,'R':r,'budget':b}
            for policy,idkey in (('prefix','prefix_ids_json'),('pmilp','selected_ids_json')):
                ids=json.loads(row[idkey]);assert len(ids)==len(set(ids)) and set(ids)<=set(group)
                area=math.fsum(float(group[i]['area']) for i in ids);target=math.fsum(float(group[i]['target']) for i in ids);P=math.fsum(float(group[i][f'predicted_R{r}']) for i in ids)
                assert area<=b*A
                close(area,row[policy+'_actual_area_m2']);close(target,row[policy+'_captured_target_m2']);close(P,row[policy+'_predicted_objective_m2']);close(target/T,row[policy+'_capture_fraction'])
                sample[policy]={'selected_count':len(ids),'area_m2':area,'target_m2':target,'predicted_m2':P,'capture':target/T}
            name=f'M{m}_R{r}_B{round(b*100):03d}.json';choice=spatial/'predicted_area_policy/choices'/name
            assert sha(choice)==row['choice_sha256'] and read(choice)['selected_ids']==json.loads(row['selected_ids_json'])
            bound=read(spatial/'predicted_area_policy/bound_records'/name)['bound_audit']
            assert bound==json.loads(row['bound_audit_json'])
            assert (row['optimality_proven']=='True')==bound['updated_proven_flag']
            assert (row['bound_anomaly']=='True')==bound['bound_anomaly']
            assert (row['certification_downgraded']=='True')==bound['certification_downgraded']
            upper=float(row['predicted_upper_m2']);primal=sample['pmilp']['predicted_m2']
            close(max(0.,upper-primal)/max(abs(primal),1e-30),row['predicted_relative_gap'])
            selected_samples.append(sample)
        # Integrate saved complete staircases, not sparse fixed-budget points.
        auc_samples=[]
        for m,r in ((2,1),(4,4)):
            curve=[x for x in curves if int(x['M'])==m and int(x['R'])==r]
            saved=next(x for x in aucs if int(x['M'])==m and int(x['R'])==r)
            for upper,field in ((1.,'auc_0_1'),(.5,'auc_0_0p5_normalized')):
                value=math.fsum(float(a['capture_fraction'])*max(0.,min(float(b['actual_area_fraction']),upper)-float(a['actual_area_fraction'])) for a,b in zip(curve,curve[1:]))/upper
                close(value,saved[field],1e-12);auc_samples.append({'M':m,'R':r,'upper':upper,'AUC':value})
        counts={'recorded_bound_qualified':sum(x['optimality_proven']=='True' for x in outcomes),'fallbacks':sum(x['fallback_used']=='True' for x in outcomes),'bound_anomalies':sum(x['bound_anomaly']=='True' for x in outcomes),'bound_downgrades':sum(x['certification_downgraded']=='True' for x in outcomes)}
        assert all(value==model[key] for key,value in counts.items())
        results.append({'run_id':model['run_id'],'all_keys_counts_and_artifact_inventory_verified':True,'verified_files':verified,
                        'counted_flags':counts,'map_samples':map_samples,'management_case_samples':selected_samples,'saved_step_AUC_samples':auc_samples})
    result={'passed':True,'review_kind':'Bounded independent second review of root auditor and its first2 artifacts',
            'parent_review':str(ORIGINAL),'parent_review_sha256':sha(ORIGINAL),'auditor_sha256':audit['auditor_sha256'],
            'reviewer_source_sha256':sha(__file__),'models':results,'actual_map_NPZ_decoded':4,'scalar_sample_pairs':12,
            'management_samples':8,'AUC_samples':8,'full_714_map_integration_repeated':False,
            'test_or_R6_accessed':False,'new_model_or_solver_calls':0,
            'static_scope_notes':['Root audit recomputes all714 map integrations and all saved staircases; this review does not repeat that work',
              'Root policy length45 is supplemented here by exact unique45-case sets for each artifact',
              'Root bound flags are counted and checked against sealed sample bound records here; this is not a new independent MILP optimality proof',
              'For future auditor versions, explicit complete policy-key inventory and full bound-flag field comparisons would strengthen automated coverage']}
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'review.md').write_text('# First two seed-output audits: bounded second review\n\nPassed: all artifact hashes/file inventories and complete scalar, unit, curve and45-case key sets. Fixed samples comprised2 windows/model,4 management cases/model and2 complete saved curves/model, with both AUC intervals. No full714-map integration was repeated.\n\nThe original auditor source matches its published receipt. No blocking numerical discrepancy was found. Recorded bound flags agree with selected sealed records; neither audit establishes agronomic efficacy or independent global optimality. Future full-auditor revisions can explicitly check unique policy keys and every sealed bound flag; current two artifacts passed these count/sample checks.\n',encoding='utf-8')
    print(json.dumps({'passed':True,'output':str(out),'sha256':sha(out/'review.json')}))
if __name__=='__main__':main()
