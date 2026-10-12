"""Synthetic-only checks of report arithmetic, completeness and NA propagation."""
from pathlib import Path
from datetime import datetime
import itertools, math, sys, shutil, traceback
import core
import report_tables as q


def fixtures():
    models={}
    for e,lr,v in itertools.product(('resnet18','convnextv2_tiny'),(.0001,.0003),('regression','hard_seg','soft_seg')):
        for seed in ((17,) if v=='regression' else (17,29,43)):
            rid=f'ART_LOCAL_{e}_{lr}_{v}_{seed}'
            models[rid]=dict(run_id=rid,dataset='local',encoder=e,lr=lr,variant=v,seed=seed,best_epoch=7,checkpoint_sha256='a'*64)
    for v,seed in itertools.product(('regression','hard_seg','soft_seg'),(17,29,43)):
        rid=f'ART_WEEDS_{v}_{seed}'
        models[rid]=dict(run_id=rid,dataset='weeds',encoder='resnet18',lr=.0001,variant=v,seed=seed,best_epoch=8,checkpoint_sha256='b'*64)
    scalar=[];local=[];la=[];external=[];ea=[]
    for model in models.values():
        for R in q.RS:
            local_ds=model['dataset']=='local';n=338 if local_ds else 26;nz=100 if local_ds else 0
            scalar.append(dict(model,R=R,n=n,n_zero=nz,n_positive=n-nz,mae=.1,bias=-.01,mae_zero=.1 if nz else None,mae_positive=.1,balanced_mae=.1 if nz else None))
        if model['variant']=='regression':continue
        if model['dataset']=='local':
            for M,R in itertools.product(q.MS,q.RS):
                la.append(dict(model,M=M,R=R,auc_0_1=.4,auc_0_half_normalized=.2))
                for b,policy in itertools.product(q.BS,('model_prefix','predicted_area')):
                    capture=b+(.03 if model['variant']=='soft_seg' else -.02)*b*(1-b)-.002*R*b*(1-b)
                    local.append(dict(model,M=M,R=R,budget=b,policy=policy,actual_area_m2=10*b,unused_area_m2=0.,
                        captured_target_m2=capture,missed_target_m2=1-capture,capture_fraction=capture,
                        selected_non_target_area_m2=10*b-capture,selected_ids_json='[]',fallback_used=False if policy=='predicted_area' else None,
                        bound_anomaly=False if policy=='predicted_area' else None,updated_proven_flag=False if policy=='predicted_area' else None,
                        certification_downgraded=False if policy=='predicted_area' else None,updated_upper_m2=1 if policy=='predicted_area' else None,
                        updated_relative_gap=1 if policy=='predicted_area' else None))
        else:
            scopes=[('frame',f'f{i:02d}',[i]) for i in range(26)]+[('date','date0',list(range(13))),('date','date1',list(range(13,26))),('pooled','all26',list(range(26)))]
            for scope,ident,members in scopes:
                n=len(members);tw=.1*sum(i!=0 for i in members);tc=.2*sum(i!=1 for i in members)
                for g,R in itertools.product(q.GRIDS,q.RS):
                    meta=dict(model,original_selected_epoch=model['best_epoch'],R=R,grid_g=g,policy='learned',scope=scope,scope_id=ident,
                        frame_id=ident if scope=='frame' else '',acquisition_date=('date0' if members[0]<13 else 'date1') if scope=='frame' else ident if scope=='date' else '',frame_n=n)
                    ea.append(dict(meta,total_weed_FAE=tw,auc_0_1=.4 if tw else None,auc_0_0p5_normalized=.2 if tw else None))
                    for basis,(code,b) in itertools.product(q.BASES,zip(q.CODES,q.BS)):
                        N=g*g;k=int(N*b) if basis==q.BASES[0] else N*int(16*b)//16;x=k/N;effective=b if basis==q.BASES[0] else int(16*b)/16
                        wc=x+(.02 if model['variant']=='soft_seg' else -.01)*x*(1-x)-.003*R*x*(1-x)
                        cc=x-.002*R*x*(1-x);sw=tw*wc;sc=tc*cc;A=x*n
                        external.append(dict(meta,budget_basis=basis,parent_nominal_budget_code=code,parent_nominal_budget_fraction=b,effective_budget_fraction=effective,
                            k=k,N=N,actual_fraction=x,selected_area_FAE=A,selected_weed_FAE=sw,selected_crop_FAE=sc,total_weed_FAE=tw,total_crop_FAE=tc,
                            selected_background_FAE=A-sw-sc,unselected_weed_FAE=tw-sw,unused_effective_budget_area_FAE=(effective-x)*n,
                            weed_capture_ratio=wc if tw else None,crop_capture_ratio=cc if tc else None,crop_fraction_of_selected_area=sc/A if A else None,
                            duplicate_selection_alias='shared-original-selection' if basis==q.BASES[1] and b in (.5,1.) else ''))
    return models,scalar,local,la,external,ea


def main():
    out=q.HERE.parent/'provenance'/('confirmation_report_checks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    checks=[]
    def check(name,condition=True):core.require(condition,name);checks.append(name)
    def reject(name,function):
        try:function()
        except (ValueError,KeyError,FileNotFoundError):checks.append(name)
        else:raise AssertionError('Expected refusal: '+name)
    try:
        check('stdlib import only',all(x not in sys.modules for x in ('numpy','torch','scipy')))
        reject('missing FREEZE refuses production before any saved outcome',lambda:q.gate(out/'absent_review'))
        check('denied production creates no real stage or claim',not (q.HERE/'stages/report_tables').exists() and not (q.HERE/'claims/report_tables.json').exists())
        models,s,l,a,e,ea=fixtures()
        s=q.normalize_scalar(s,models);l=q.normalize_local(l,models);a=q.normalize_local(a,models,True);e=q.normalize_external(e,models);ea=q.normalize_external(ea,models,True)
        frames=q.validate_population(s,l,a,e,ea,models)
        check('full37 scalar and24 local6 external inventories',len(s)==148 and len(l)==2880 and len(a)==288 and len(frames)==26)
        check('full26 frame plus2 dates plus pooled',len(e)==20880 and len(ea)==2088)
        for ds,count in (('local',112),('weeds',36)):
            family=q.family_tables([r for r in s if r['dataset']==ds],('R',),q.SCALAR_METRICS,models)
            check(ds+' scalar complete R matrix',len(family['all'])==count)
            check(ds+' all five R contrast directions',set(r['contrast'] for r in family['R_contrasts'])=={'R4_minus_R1','R2_minus_R1','R4_minus_R2','R6_minus_R1','R6_minus_R4'})
            if ds=='local':
                reg=[r for r in family['seed_mean_sample_sd'] if r['variant']=='regression']
                check('local regression17 only never fake SD',len(reg)==16 and all(r['n_models']==1 and r['seeds']=='17' and r['mae_pp_sample_sd'] is None for r in reg))
            else:
                check('external three regression seeds are used',all(r['n_models']==3 for r in family['seed_mean_sample_sd']))
                check('absent zero stratum remains NA',all(r['balanced_mae_pp_mean'] is None and r['balanced_mae_pp_sample_sd'] is None for r in family['seed_mean_sample_sd']))
        family=q.family_tables(l,('M','R','budget','policy'),q.LOCAL_METRICS,models,('M',q.M_PAIRS))
        check('local hardsoft same-seed1440 rows',len(family['hard_soft_pairs'])==1440)
        check('all local R3600 and M2880 contrasts',len(family['R_contrasts'])==3600 and len(family['M_contrasts'])==2880)
        check('all adverse/zero signs retained without filtering',len(family['all'])==2880)
        ac=q.family_tables(a,('M','R'),q.AUC_METRICS,models,('M',q.M_PAIRS))
        check('all local144 paired AUC and360 R contrasts',len(ac['hard_soft_pairs'])==144 and len(ac['R_contrasts'])==360)
        pooled=[r for r in e if r['scope']=='pooled'];ex=q.family_tables(pooled,('scope','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),q.EXTERNAL_METRICS,models,('grid_g',q.G_PAIRS))
        check('external720 pooled rows and360 pairs',len(ex['all'])==720 and len(ex['hard_soft_pairs'])==360)
        check('external both budget bases never collapsed',{r['budget_basis'] for r in ex['grid_g_contrasts']}==set(q.BASES) and len(ex['grid_g_contrasts'])==720)
        matched=[r for r in ex['grid_g_contrasts'] if r['budget_basis']==q.BASES[1]]
        check('matched-actual grid area differences zero',all(r['difference_selected_area_FAE']==0 for r in matched))
        nominal=[r for r in ex['grid_g_contrasts'] if r['budget_basis']==q.BASES[0] and r['parent_nominal_budget_code']=='b10']
        check('nominal grid actual-area differences retained',any(r['difference_selected_area_FAE']!=0 for r in nominal))
        zero=[r for r in e if r['frame_id']=='f00'];zero_pairs=q.pair_rows(zero,('scope','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),q.EXTERNAL_METRICS)
        check('zero weed paired difference is NA',all(r['soft_minus_hard_weed_capture_pct'] is None for r in zero_pairs))
        cropzero=[r for r in e if r['frame_id']=='f01'];crop_pairs=q.pair_rows(cropzero,('scope','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),q.EXTERNAL_METRICS)
        check('zero crop capture NA while composition zero remains defined',all(r['soft_minus_hard_crop_capture_pct'] is None and r['soft_minus_hard_crop_fraction_pct']==0 for r in crop_pairs))
        check('budget alias evidence retained',all(r['duplicate_selection_alias']=='shared-original-selection' for r in e if r['budget_basis']==q.BASES[1] and r['parent_nominal_budget_code'] in ('b50','b100')))
        toy=[dict(dataset='weeds',encoder='resnet18',lr=.0001,variant='soft_seg',seed=seed,x=x) for seed,x in zip((17,29,43),(1.,2.,3.))]
        stats=q.seed_stats(toy,('dataset','encoder','variant','lr'),('x',),models)[0]
        check('sample SD n-minus1',stats['x_mean']==2. and stats['x_sample_sd']==1.)
        toy[0]['x']=None;stats=q.seed_stats(toy,('dataset','encoder','variant','lr'),('x',),models)[0]
        check('partial NA blocks complete-seed mean and SD',stats['x_defined_models']==2 and stats['x_mean'] is None and stats['x_sample_sd'] is None)
        reject('missing declared seed rejected',lambda:q.seed_stats(toy[:2],('dataset','encoder','variant','lr'),('x',),models))
        reject('missing report model cell rejected',lambda:q.validate_population(s[:-1],l,a,e,ea,models))
        reject('duplicate report cell rejected',lambda:q.validate_population(s[:-1]+[s[0]],l,a,e,ea,models))
        reject('missing R6 contrast rejected',lambda:q.contrast_rows([r for r in s if r['R']!=6],'R',(),q.R_PAIRS,q.SCALAR_METRICS))
        reject('missing grid case rejected',lambda:q.validate_population(s,l,a,e[:-1],ea,models))
        reject('nonfinite numeric guard',lambda:q.number(float('nan')))
        # No combined cross-dataset seed statistics: dataset is a mandatory output key.
        check('all scalar means remain dataset-specific',all(r['dataset']=='weeds' for r in q.family_tables([r for r in s if r['dataset']=='weeds'],('R',),q.SCALAR_METRICS,models)['seed_mean_sample_sd']))
        for p in (Path(__file__),Path(q.__file__)):shutil.copyfile(p,out/p.name)
        core.write(out/'focused_checks.json',{'passed':True,'source_sha256':core.sha(q.__file__),'checker_sha256':core.sha(__file__),'checks':checks,'count':len(checks),
            'scope':'synthetic report preparation only','real_confirmation_tables_read':0,'raw_labels_or_arrays_read':0,'model_calls':0,'solver_calls':0,
            'FREEZE_created':False,'real_claim_created':False,'full_synthetic_counts':{'models':37,'scalar':148,'local_budget':2880,'local_auc':288,'external_budget':20880,'external_auc':2088,'dates':2}})
        print(out)
    except BaseException:
        core.write(out/'failure.json',{'passed':False,'checks':checks,'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
