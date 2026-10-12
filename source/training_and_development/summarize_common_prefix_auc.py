"""Complete eight-model prefix AUC supplement; standard library, no models/solvers."""
from pathlib import Path
import csv,datetime,hashlib,json,math,shutil,sys

ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
BCA=PA/'bca_20260916_050803_135549'
PROTOCOL=PA/'RM_development_protocol_v1.json'
AUDIT=PA/'bound_closure_independent_20260916_050846_426215/audit.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def step_auc(x,y,upper):
    """F(b)=y[i] for x[i]<=b<x[i+1]; integrate rectangles, divide by upper."""
    assert 0<upper<=1 and len(x)==len(y)>=2
    assert x[0]==y[0]==0 and x[-1]==y[-1]==1
    assert all(math.isfinite(a) and 0<=a<=1 for a in x)
    # Preserve the original cumsum values, including sub-ULP/ULP excursions;
    # the endpoint was fixed by the original producer. Do not clip y for AUC.
    assert all(math.isfinite(a) and -1e-12<=a<=1+1e-12 for a in y)
    assert all(b>=a for a,b in zip(x,x[1:]))
    assert all(b>=a-1e-12 for a,b in zip(y,y[1:]))
    terms=[y[i]*max(0.,min(x[i+1],upper)-max(x[i],0.)) for i in range(len(x)-1)]
    answer=math.fsum(terms)/upper
    assert 0<=answer<=1
    return answer


def self_checks():
    x=[0.,.2,.5,1.];y=[0.,.3,.8,1.]
    assert math.isclose(step_auc(x,y,1.),.49,abs_tol=1e-15)
    assert math.isclose(step_auc(x,y,.5),.18,abs_tol=1e-15)
    assert math.isclose(step_auc([0.,.2,.8,1.],y,.5),.18,abs_tol=1e-15)
    assert step_auc([0.,1.],[0.,1.],1.)==0. # Endpoint capture has zero measure.
    return {'passed':True,'checks':['full_rectangle_integral','half_interval_normalization',
                                  'partial_interval_crossing','endpoint_has_zero_measure'],
            'model_or_solver_calls':0}


def main():
    assert sha(PROTOCOL)=='527f716fe7db0125ceca70a76479d8f3337fc448ff197d5e16517f40bbe8ab8d'
    assert sha(ROOT/'evaluate_management.py')=='29d7b8a8bd7e6762cdae18bde2d491ecbe9cd459bf401ef084a9f827e990ff8d'
    assert sha(AUDIT)=='cd88fe934e9c9d5dcee01a38a7571257f23d1ab8838ad9a1811e6e1f72913da2'
    audit=read(AUDIT);assert audit['passed'] and audit['all_model_prefix_curve_points_recomputed']==52464
    assert sha(BCA/'manifest.json')=='21622e9a3a0e970354816e0876c718f472d29a054f009cf2be343479c4d45512'
    models=read(BCA/'manifest.json')['models'];assert len(models)==8
    original=read(PROTOCOL)
    assert original['primary_management_grid_m']==2 and original['management_grid_m']==[1,2,4]
    assert original['image_degradation_factors']==[1,2,4]
    watched={str(p):sha(p) for p in (PROTOCOL,AUDIT,BCA/'manifest.json',Path(__file__),ROOT/'evaluate_management.py')}
    out=PA/('common_prefix_auc_full8_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False);(out/'source').mkdir();(out/'source'/'complete_curves').mkdir()
    rows=[];summaries_matched=0;max_summary_error=0.;curve_points=0
    for model in models:
        parent=Path(model['unit_tables_directory']);path=parent/'decision_curves.csv'
        digest=sha(path);watched[str(path)]=digest
        copy=out/'source'/'complete_curves'/f"{model['index']:02d}_decision_curves.csv"
        shutil.copy2(path,copy);assert sha(copy)==digest
        records=table(path);groups={}
        for record in records:
            if record['method']=='model_prefix':
                groups.setdefault((int(record['M']),int(record['R'])),[]).append(record)
        assert set(groups)=={(m,r) for m in (1,2,4) for r in (1,2,4)}
        stored_path=parent/'summary.json';stored=None
        if stored_path.exists():
            stored=read(stored_path);assert stored['state']=='completed'
            assert stored['model_run']==model['run_id'] and stored['checkpoint_epoch']==model['checkpoint_epoch']
            watched[str(stored_path)]=sha(stored_path)
        for (m,r),group in sorted(groups.items()):
            ordered=sorted(group,key=lambda x:int(x['k']))
            assert [int(x['k']) for x in ordered]==list(range(len(ordered)))
            ids=[x['last_added_id'] for x in ordered[1:]]
            assert ordered[0]['last_added_id']=='' and len(set(ids))==len(ids) and all(x.isascii() for x in ids)
            x=[float(a['actual_area_fraction']) for a in ordered]
            y=[float(a['capture_fraction']) for a in ordered]
            auc=step_auc(x,y,1.);half=step_auc(x,y,.5);curve_points+=len(ordered)
            saved_full=saved_half=error_full=error_half=None
            if stored is not None:
                previous=[q for q in stored['results']['by_M'][str(m)]['prefix_results'] if q['method']=='model_prefix' and q['R']==r]
                assert len(previous)==1
                saved_full=previous[0]['auc_0_1'];saved_half=previous[0]['auc_0_0p5_normalized']
                error_full=abs(auc-saved_full);error_half=abs(half-saved_half)
                assert max(error_full,error_half)<1e-12
                summaries_matched+=1;max_summary_error=max(max_summary_error,error_full,error_half)
            rows.append({'model_index':model['index'],'run_id':model['run_id'],'encoder':model['encoder'],
                         'lr':model['lr'],'variant':model['variant'],'seed':17,'checkpoint_epoch':model['checkpoint_epoch'],
                         'checkpoint_sha256':model['checkpoint_sha256'],'M':m,'R':r,'curve_points':len(ordered),
                         'management_units':len(ordered)-1,'auc_0_1':auc,'auc_0_0p5_normalized':half,
                         'max_input_capture_fraction_above_one':max(0.,max(y)-1.),'input_capture_clipped':False,
                         'stored_parent_auc_available':stored is not None,'stored_auc_0_1':saved_full,
                         'stored_auc_0_0p5_normalized':saved_half,'abs_error_vs_stored_auc_0_1':error_full,
                         'abs_error_vs_stored_auc_0_0p5':error_half,'parent_directory':str(parent),
                         'curve_csv_sha256':digest,'curve_csv_copy':copy.relative_to(out).as_posix(),
                         'original_model_output_state':model['original_output_state'],
                         'auc_source_note':'Complete curve; original stored summary matched' if stored is not None else 'Complete curve retained; parent summary absent after historical failure'})
    assert len(rows)==72 and curve_points==52464
    lookup={(x['encoder'],x['lr'],x['variant'],x['M'],x['R']):x for x in rows}
    pairs=[]
    for row in rows:
        if row['variant']!='hard_seg':continue
        soft=lookup[(row['encoder'],row['lr'],'soft_seg',row['M'],row['R'])]
        pairs.append({'encoder':row['encoder'],'lr':row['lr'],'M':row['M'],'R':row['R'],
                      'hard_run':row['run_id'],'soft_run':soft['run_id'],'hard_checkpoint_epoch':row['checkpoint_epoch'],
                      'soft_checkpoint_epoch':soft['checkpoint_epoch'],'hard_auc_0_1':row['auc_0_1'],
                      'soft_auc_0_1':soft['auc_0_1'],'soft_minus_hard_auc_0_1':soft['auc_0_1']-row['auc_0_1'],
                      'hard_auc_0_0p5_normalized':row['auc_0_0p5_normalized'],
                      'soft_auc_0_0p5_normalized':soft['auc_0_0p5_normalized'],
                      'soft_minus_hard_auc_0_0p5_normalized':soft['auc_0_0p5_normalized']-row['auc_0_0p5_normalized'],
                      'hard_curve_csv_sha256':row['curve_csv_sha256'],'soft_curve_csv_sha256':soft['curve_csv_sha256']})
    assert len(pairs)==36
    counts={}
    for m in (1,2,4):
        for metric in ('auc_0_1','auc_0_0p5_normalized'):
            ds=[p['soft_minus_hard_'+metric] for p in pairs if p['M']==m]
            counts[f'M{m}_{metric}']={'pairs':len(ds),'soft_higher':sum(d>1e-12 for d in ds),
                                    'soft_lower':sum(d< -1e-12 for d in ds),'tied':sum(abs(d)<=1e-12 for d in ds),
                                    'min_difference':min(ds),'max_difference':max(ds)}
    write_csv(out/'all72_model_prefix_auc.csv',rows);write_csv(out/'all36_hard_soft_auc_pairs.csv',pairs)
    for p in (Path(__file__),PROTOCOL,AUDIT,ROOT/'evaluate_management.py'):
        shutil.copy2(p,out/'source'/p.name)
    write(out/'analytic_checks.json',self_checks())
    for path,digest in watched.items():assert sha(path)==digest
    summary={'state':'completed','created_at':datetime.datetime.now().astimezone().isoformat(),
             'models':8,'model_curves':72,'hard_soft_pairs':36,'complete_model_curve_points':curve_points,
             'original_protocol_metric':True,'integration':'fsum y[i] * max(0,min(x[i+1],u)-x[i]), then divide by u; u=1 or0.5',
             'no_linear_interpolation':True,'no_sparse_budget_auc':True,
             'curves_with_saved_parent_auc_matched':summaries_matched,'curves_without_parent_summary':72-summaries_matched,
             'max_abs_difference_from_saved_parent_auc':max_summary_error,'by_M_direction_counts':counts,
             'max_original_curve_roundoff_above_one':max(x['max_input_capture_fraction_above_one'] for x in rows),
             'original_curve_values_clipped':False,
             'direction_tie_tolerance_auc':1e-12,'input_source_sha256':watched,'primary_M':2,'sensitivity_M':[1,4],
             'new_budget_points_added':0,'new_checkpoint_selection':False,'solver_calls':0,'model_calls':0,
             'test_accessed':False,'R6_accessed':False,'central_report_modified':False,'figure_created':False}
    write(out/'summary.json',summary)
    report=['# Complete-prefix AUC supplement: all eight common models','',
            'This supplement aggregates the two complete-prefix AUC metrics defined in the original development protocol. It retains all72 model curves and all36 matched hard/soft comparisons. Checkpoints remain those selected by the original stratified MAE. No new optimization, inference, geometry, budget point or model selection was performed; the locked central report remains unchanged.','',
            'For adjacent complete-prefix points (xᵢ,yᵢ), capture equals yᵢ on [xᵢ,xᵢ₊₁). AUC is the sum of rectangle areas over [0,1]. The [0,0.5] metric clips the crossing rectangle at0.5 and divides by0.5. No interpolation or AUC from the five sparse reported budgets is used. Values are dimensionless integrated capture, not percentage-point differences or field efficacy.','',
            f'All{curve_points:,} model-curve points were read from eight parent CSVs. {summaries_matched} curve pairs of stored AUC values matched to a maximum absolute difference of {max_summary_error:.3g}. The other{72-summaries_matched} curves were complete in historically failed parent runs but had no persisted parent summary; their original-protocol AUCs are recomputed from the preserved curves and identified as such.','',
            '## Primary analysis: M = 2 m','',
            '|Encoder|LR|R|Hard AUC[0,1]|Soft AUC[0,1]|Soft−hard|Hard normalized[0,.5]|Soft normalized[0,.5]|Soft−hard|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        if p['M']==2:
            report.append(f"|{p['encoder']}|{p['lr']:.4g}|{p['R']}|{p['hard_auc_0_1']:.8f}|{p['soft_auc_0_1']:.8f}|{p['soft_minus_hard_auc_0_1']:+.8f}|{p['hard_auc_0_0p5_normalized']:.8f}|{p['soft_auc_0_0p5_normalized']:.8f}|{p['soft_minus_hard_auc_0_0p5_normalized']:+.8f}|")
    report+=['','## Full primary/sensitivity direction counts','',
             '|M|Metric|Soft higher|Soft lower|Tie|Minimum difference|Maximum difference|',
             '|---:|---|---:|---:|---:|---:|---:|']
    for m in (2,1,4):
        for metric in ('auc_0_1','auc_0_0p5_normalized'):
            c=counts[f'M{m}_{metric}']
            report.append(f"|{m}|{metric}|{c['soft_higher']}|{c['soft_lower']}|{c['tied']}|{c['min_difference']:+.8f}|{c['max_difference']:+.8f}|")
    report+=['','These are correlated conditions within one scene and one seed, not independent replicates. AUC assigns uniform weight over its declared nominal-budget interval; it does not establish an operational budget distribution or optimal management resolution. Both complete intervals and every hard/soft contrast are retained. This is completion of predeclared descriptive metrics, not a new hypothesis or checkpoint-selection criterion.',
             '', '## Files','', '- all72_model_prefix_auc.csv: full metrics, checkpoint/source identity and persisted-summary availability.',
             '- all36_hard_soft_auc_pairs.csv: all paired differences for both metrics.',
             '- source/complete_curves/: byte-identical copies of all eight original CSVs.',
             '- summary.json, analytic_checks.json, manifest.json and status.json: formula, checks and hashes.',
             '', 'Tables are the primary supplement; no figure was generated.']
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write(out/'manifest.json',{'created_at':summary['created_at'],'source_sha256':watched,
          'metric_protocol_sha256':sha(PROTOCOL),'implementation_sha256':sha(__file__),
          'independent_prior_point_audit_sha256':sha(AUDIT),'no_central_lock_changes':True})
    status={'state':'completed','summary':summary,'files_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}}
    write(out/'status.json',status)
    print(json.dumps({'output':str(out),'curves':72,'pairs':36,'stored_auc_curves_matched':summaries_matched,
                      'max_stored_auc_error':max_summary_error,'directions':counts},ensure_ascii=True))


if __name__=='__main__':main()
