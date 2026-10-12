"""Artificial-only cross-implementation fixtures for the independent saved-result auditor."""
from pathlib import Path
from unittest.mock import patch
import datetime,itertools,json,math,sys,traceback
import review_confirmation_external_scalar as a
import core

OUT=a.ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_ext_scalar_audit_checks_%Y%m%d_%H%M%S_%f');OUT.mkdir();checks=[]
def check(name,value=True):
    assert value,name
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,FileNotFoundError,KeyError):check(name);return
    raise AssertionError(name)
def writecsv(path,rows):path.parent.mkdir(parents=True,exist_ok=True);core.table(path,rows)
def model_roster():
    models={}
    for ei,e in enumerate(('resnet18','convnextv2_tiny')):
        for li,lr in enumerate((.0001,.0003)):
            for variant in ('regression','hard_seg','soft_seg'):
                for s in ((17,) if variant=='regression' else (17,29,43)):
                    rid=f'LOCAL_{ei}_{li}_{variant}_{s}';models[rid]=dict(run_id=rid,dataset='local',encoder=e,lr=lr,variant=variant,seed=s,best_epoch=4+ei+li,checkpoint_sha256='a'*64,config_sha256='c'*64)
    for variant,s in itertools.product(('regression','hard_seg','soft_seg'),(17,29,43)):
        rid=f'WEEDS_{variant}_{s}';models[rid]=dict(run_id=rid,dataset='weeds',encoder='resnet18',lr=.0001,variant=variant,seed=s,best_epoch=9,checkpoint_sha256='b'*64,config_sha256='c'*64)
    return models

def main():
    check('auditor_import_is_stdlib',not any(n in sys.modules for n in ('numpy','torch','scipy')))
    with patch.object(core,'validate_freeze',side_effect=FileNotFoundError('SYNTHETIC missing freeze')):
        rejects('freeze_refusal_precedes_any_array_import',lambda:a.gate(OUT/'missing',OUT/'missing2'))
    check('denied_gate_still_no_numpy','numpy' not in sys.modules)
    # Scalar fixture emits the original producer's pure arithmetic on an entirely artificial full roster.
    import scalar_outcomes as producer
    models=model_roster();ids={'local':[f'L{i:03}' for i in range(338)],'weeds':[f'W{i:02}' for i in range(26)]}
    truth={'local':{i:(0. if n%4==0 else (n%19+1)/30) for n,i in enumerate(ids['local'])},'weeds':{i:(n%7+1)/12 for n,i in enumerate(ids['weeds'])}}
    folder=OUT/'scalar';folder.mkdir();samples=[];summaries=[];means=[];exported={}
    for rid,m in models.items():
        own=[];meta={k:m[k] for k in ('run_id','dataset','encoder','variant','lr','seed','best_epoch')}
        for R in a.RS:
            pp=[];tt=[]
            for j,i in enumerate(ids[m['dataset']]):
                t=truth[m['dataset']][i];p=max(0.,min(1.,t+.007*R+.00004*m['seed']**2-.012*(m['variant']=='soft_seg')+.003*((j%5)-2)))
                exported[rid,R,i]=p;pp.append(p);tt.append(t);samples.append(dict(meta,R=R,id=i,prediction=p,truth=t,signed_error=p-t,absolute_error=abs(p-t)))
            row=dict(meta,R=R,**producer.summarize(pp,tt));summaries.append(row);own.append(row)
        row=dict(meta,R_levels='1,2,4,6',checkpoint_reselected=False)
        for metric in a.METRICS:row['mean_four_R_'+metric]=sum(x[metric] for x in own)/4 if all(x[metric] is not None for x in own) else None
        means.append(row)
    baselines={'datasets':{ds:{'zero':0.,'train_mean':.217 if ds=='local' else .319} for ds in ids}}
    constants=[dict(dataset=ds,baseline=name,constant_prediction=v,R_applicability='identical at R1,2,4,6; counted once',**producer.summarize([v]*len(truth[ds]),list(truth[ds].values()))) for ds in ids for name,v in baselines['datasets'][ds].items()]
    for name,rows in [('all_scalar_samples',samples),('all_model_R_metrics',summaries),('all_model_four_R_means',means),('hard_soft_pairs',producer.paired(summaries)),('descriptive_seed_summaries',producer.seed_summaries(summaries)),('constant_baselines',constants)]:writecsv(folder/(name+'.csv'),rows)
    result=a.audit_scalar(folder,models,ids,truth,exported,baselines)
    check('full38792_148_37_60_60_4_scalar_population',tuple(result.values())==(38792,148,37,60,60,4))
    check('external_absent_zero_stratum_is_NA',a.independent_metrics([.2,.3],[.1,.2])['balanced_mae'] is None)
    check('allzero_absent_positive_is_NA',a.independent_metrics([.2,.3],[0.,0.])['balanced_mae'] is None)
    saved=(folder/'all_model_R_metrics.csv').read_bytes();bad=[dict(x) for x in summaries];bad[0]['mae']+=.01;writecsv(folder/'all_model_R_metrics.csv',bad)
    rejects('wrong_scalar_metric_detected',lambda:a.audit_scalar(folder,models,ids,truth,exported,baselines));(folder/'all_model_R_metrics.csv').write_bytes(saved)
    # External fixture: two synthetic frames (zero weed / zero crop), two paired fits,
    # four R and all three grids. Producer's pure functions generate expected saved tables.
    import numpy as np
    original=OUT/'artificial_local_source.json';core.write(original,[{'id':'a','coverage':1/3},{'id':'b','coverage':0.}])
    original_inputs={'local':{'source_manifest':core.binding(original),'records':[{'id':'a'},{'id':'b'}]}}
    values=a.local_source_coverage(original_inputs,np)
    check('original_float32_source_coverage_not_soft_grid_mean',values=={'a':float(np.float32(1/3)),'b':0.})
    wrong={'local':{**original_inputs['local'],'records':[{'id':'b'},{'id':'a'}]}}
    rejects('original_source_scalar_membership_order_rejected',lambda:a.local_source_coverage(wrong,np))
    import external_arithmetic as ar
    import external_spatial as ep
    check('no_model_or_solver_import','torch' not in sys.modules and 'scipy' not in sys.modules)
    frames=[{'id':'frameA','acquisition_date':'dateA'},{'id':'frameB','acquisition_date':'dateB'}]
    ext={rid:m for rid,m in models.items() if m['dataset']=='weeds' and m['variant']!='regression' and m['seed']==17}
    grids={};cell_truth={};pred={};selections={};metas={};full_curves=[];out=OUT/'external';out.mkdir()
    yy,xx=np.indices((256,256));x=(xx//16)/15;y=(yy//16)/15
    for frame in frames:
        soft=np.zeros((3,256,256),np.float32)
        if frame['id']=='frameA':soft[1]=((xx<128)&(yy<192))*.5;soft[2]=0
        else:soft[1]=0;soft[2]=((xx<192)&(yy<128))*.5
        soft[0]=1-soft[1]-soft[2]
        for g in a.GRIDS:
            values=np.stack([a.aggregate(soft[c],g,np) for c in range(3)],1);grids[frame['id'],g]=values
            cell_truth[frame['id'],g]={i:{'A':1/g**2,'B':values[j,0],'C':values[j,1],'W':values[j,2]} for j,i in enumerate(a.ids_for(frame['id'],g))}
    for rid,m in ext.items():
        selected=OUT/rid;selected.mkdir();selections[rid]=selected;meta=a.metadata(m,'d'*64,'e'*64);metas[rid]=meta;ranks=[];choices=[];units=[]
        for frame in frames:
            for R in a.RS:
                plane=(.11+.31*x+.19*y+.003*R*((xx//16)%3)+(.08*(1-x) if m['variant']=='soft_seg' else .02*x)).astype(np.float32)
                for g in a.GRIDS:
                    rank,choice=ar.ranking_record(meta,frame,R,g,plane);ranks.append(rank);choices.extend(choice)
                    pred[rid,R,frame['id'],g]=np.array([math.fsum(map(float,plane[r*(256//g):(r+1)*(256//g),c*(256//g):(c+1)*(256//g)].ravel()))/65536 for r in range(g) for c in range(g)])
                    units.extend(dict(run_id=rid,R=R,frame_id=frame['id'],grid_g=g,**u) for u in rank['units'])
        ep.lines(selected/'rankings.jsonl',ranks);ep.lines(selected/'choices.jsonl',choices);writecsv(selected/'prediction_units.csv',units)
        full_curves.extend(ar.make_frame_curve(r,cell_truth) for r in ranks)
    writecsv(out/'truth_units.csv',[dict(frame_id=f,grid_g=g,id=i,**v) for (f,g),cells in sorted(cell_truth.items()) for i,v in sorted(cells.items())])
    learned=ep.save_family(out,'learned',full_curves)
    for scope,result in learned.items():
        writecsv(out/f'hard_soft_{scope}_budget_pairs.csv',ar.paired_rows(result['budget_rows']));writecsv(out/f'hard_soft_{scope}_auc_pairs.csv',ar.paired_auc_rows(result['auc_rows']))
    for policy in ('label_oracle','random_expectation'):
        ranks=[r for g in a.GRIDS for r in ep.reference_rankings(frames,cell_truth,g,policy,'e'*64)];ep.lines(out/(policy+'_reference_rankings.jsonl'),ranks)
        ep.save_family(out,policy,[ar.make_frame_curve(r,cell_truth,policy,policy=='random_expectation') for r in ranks])
    result=a.audit_external(out,ext,frames,pred,grids,selections,metas,np,False)
    check('all_external_fourR_threegrid_bases_and_references',result['rankings']==48 and result['choices']==480 and result['truth_units']==672)
    check('zeroT_C_and_zero_selected_fullcurve_cases_exercised',all(grids['frameA',g][:,2].sum()==0 and grids['frameB',g][:,1].sum()==0 for g in a.GRIDS))
    check('analytic_random_full_and_half_checked',a.COUNTS['random_expectation_frame_auc.csv']==6)
    # Reject a changed saved choice without producing alternative choices or calling a solver.
    first=next(iter(selections.values()))/'choices.jsonl';saved=first.read_bytes();rows=list(a.jsonlines(first));rows[0]['selected_ids']=list(reversed(rows[0]['selected_ids']))+['FAKE'];first.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    # Clear only the artificial audit read ledger so semantic, not hash, mismatch is tested.
    a.WATCH.pop(str(first.resolve()),None)
    rejects('altered_selected_IDs_rejected',lambda:a.audit_external(out,ext,frames,pred,grids,selections,metas,np,False));first.write_bytes(saved);a.WATCH.pop(str(first.resolve()),None)
    # Source gate refuses before global validators if any producer outcome is absent.
    home=OUT/'gate';home.mkdir()
    for n in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json','scalar_baselines.json'):core.write(home/n,{'synthetic':True})
    choice=home/'choices.json';core.write(choice,{'synthetic':True});grant=home/'grant.json'
    core.write(grant,{'passed':True,'scope':'confirmation_external_scalar_independent','implementation_sha256':core.sha(a.__file__),'freeze_sha256':core.sha(home/'FREEZE.json'),'choice_review_sha256':core.sha(choice),'outcome_artifact_sha256':{x:'0'*64 for x in ('scalar_outcomes','local_outcomes','external_outcomes')}})
    with patch.object(a,'HOME',home),patch.object(core,'validate_freeze',return_value=({},list(models.values()),{})),patch.object(core,'validate_source_lock',return_value={}):
        rejects('three_actual_outcome_stages_required',lambda:a.gate(grant,choice))
    def changed_during_validator(**kwargs):
        core.write(home/'contract.json',{'synthetic':'changed after initial pin'})
        return {},list(models.values()),{}
    with patch.object(a,'HOME',home),patch.object(core,'validate_freeze',side_effect=changed_during_validator),patch.object(core,'validate_source_lock',return_value={}):
        rejects('validator_time_metadata_drift_rejected',lambda:a.gate(grant,choice))
    check('no_real_confirmation_stage_or_claim_created',not (a.HOME/'stages').exists())
    report={'passed':True,'implementation_sha256':core.sha(a.__file__),'checker_sha256':core.sha(__file__),'check_count':len(checks),'checks':checks,'scalar_full_matrix':{'samples':38792,'model_R':148,'means':37,'pairs':60,'seed_groups':60,'baselines':4},
        'external_fixture':'two paired models, two artificial frames, four R, all three grids and both budget bases; two dates, zero weed/crop; all27 outcome/reference families and paired rows',
        'numeric_checks':a.NUMERIC,'max_errors':dict(a.ERRORS),'real_result_or_label_files_opened':0,'model_calls':0,'solver_calls':0,'actual_execution':False,
        'limitations':['Synthetic cross-implementation preparation, not actual reserved-result acceptance.','Raw NPZ loader and full real-source execution gate require subsequent independent static review and a source-bound root grant.']}
    core.write(OUT/'focused_checks.json',report);print(json.dumps({'passed':True,'path':str(OUT/'focused_checks.json'),'sha256':core.sha(OUT/'focused_checks.json'),'checks':len(checks)}))

if __name__=='__main__':
    try:main()
    except BaseException as error:core.write(OUT/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'checks':checks});raise
