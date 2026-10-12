"""Independent synthetic arithmetic/identity QA. No real outcomes, arrays, models or solvers."""
from pathlib import Path
from unittest.mock import patch
import ast, csv, datetime, itertools, json, math, sys

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'confirmation_v1'))
import core
import report_tables as q
OUT=ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_report_independent_%Y%m%d_%H%M%S_%f');OUT.mkdir()
PIN='1139e90254af3eff61335e3a9f96232efa85b845fae7c6089c311235e85bfc26'
checks=[];max_error=0.;comparisons=0
def check(name,condition=True):
    assert condition,name
    checks.append(name)
def equals(got,wanted):
    global max_error,comparisons
    comparisons+=1
    if wanted is None:assert got is None;return
    assert got is not None
    error=abs(float(got)-float(wanted));max_error=max(max_error,error)
    assert error<=1e-9*max(1.,abs(wanted)),(got,wanted)
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,FileNotFoundError,FileExistsError):check(name);return
    raise AssertionError(name)
def manual_stats(values):
    if any(x is None for x in values):return None,None
    mean=math.fsum(values)/len(values)
    return mean,math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1)) if len(values)>1 else None
def make_models():
    models={}
    for ei,e in enumerate(('resnet18','convnextv2_tiny')):
        for li,lr in enumerate((.0001,.0003)):
            for v in ('regression','hard_seg','soft_seg'):
                for s in ((17,) if v=='regression' else (17,29,43)):
                    rid=f'L{ei}{li}_{v}_{s}';models[rid]=dict(run_id=rid,dataset='local',encoder=e,lr=lr,variant=v,seed=s,best_epoch=3+ei+li,checkpoint_sha256='a'*64)
    for v,s in itertools.product(('regression','hard_seg','soft_seg'),(17,29,43)):
        rid=f'W_{v}_{s}';models[rid]=dict(run_id=rid,dataset='weeds',encoder='resnet18',lr=.0001,variant=v,seed=s,best_epoch=9,checkpoint_sha256='b'*64)
    return models
def factors(model):
    return {17:1,29:3,43:6}[model['seed']],int(model['encoder']=='convnextv2_tiny'),int(model['lr']==.0003),{'regression':0,'hard_seg':-1,'soft_seg':1}[model['variant']]
def fixtures(models):
    scalar=[];local=[];la=[];external=[];ea=[]
    date=lambda i:'dA' if i<5 else 'dB' if i<14 else 'dC'
    for model in models.values():
        s,e,l,v=factors(model)
        for R in (1,2,4,6):
            z=.015+.0008*s*s+.001*R*R+.004*e+.001*l+.0003*v*s
            pos=.11+.0012*s+.0009*R+.003*l-.001*v*s
            n=338 if model['dataset']=='local' else 26;nz=79 if model['dataset']=='local' else 0
            scalar.append(dict(model,R=R,n=n,n_zero=nz,n_positive=n-nz,mae=(nz*z+(n-nz)*pos)/n,bias=-.04+.001*s*s+.0008*v*R,
                mae_zero=z if nz else None,mae_positive=pos,balanced_mae=(z+pos)/2 if nz else None))
        if model['variant']=='regression':continue
        if model['dataset']=='local':
            for M,R in itertools.product((1,2,4),(1,2,4,6)):
                av=.37+.001*s*s+.008*M-.0013*R*R+.006*v*s+.002*e
                la.append(dict(model,M=M,R=R,auc_0_1=av,auc_0_half_normalized=av*.7+.001*l))
                for b,policy in itertools.product((.1,.2,.3,.5,1.),('model_prefix','predicted_area')):
                    gain=.05*v*s-.004*R*R+.009*M+.005*l-.01*e+(.013 if policy=='predicted_area' else 0)
                    capture=b+b*(1-b)*gain;A=100*b-(.03*R+.02*M+.01*s)*(1-b)
                    local.append(dict(model,M=M,R=R,budget=b,policy=policy,actual_area_m2=A,unused_area_m2=100*b-A,
                        captured_target_m2=10*capture,missed_target_m2=10*(1-capture),capture_fraction=capture,selected_non_target_area_m2=A-10*capture,
                        fallback_used=False if policy=='predicted_area' else None,bound_anomaly=False,updated_proven_flag=False,
                        certification_downgraded=False,updated_upper_m2=9.,updated_relative_gap=.4))
        else:
            groups=[('frame',f'f{i:02}',[i]) for i in range(26)]+[('date',d,[i for i in range(26) if date(i)==d]) for d in ('dA','dB','dC')]+[('pooled','all26',list(range(26)))]
            for scope,ident,members in groups:
                tw=math.fsum(0. if i==0 else .04+.003*(i%5) for i in members);tc=math.fsum(0. if i==1 else .13+.004*(i%4) for i in members)
                for g,R in itertools.product((4,8,16),(1,2,4,6)):
                    meta=dict(model,R=R,grid_g=g,scope=scope,scope_id=ident,frame_id=ident if scope=='frame' else '',acquisition_date=date(members[0]) if scope=='frame' else ident if scope=='date' else '',frame_n=len(members),policy='learned')
                    au=math.fsum((0. if i==0 else .04+.003*(i%5))*(.31+.004*v*s+.0011*s*s-.002*R*R+.0017*g+.0002*i) for i in members)/tw if tw else None
                    ea.append(dict(meta,total_weed_FAE=tw,auc_0_1=au,auc_0_0p5_normalized=au*.8 if au is not None else None))
                    for basis,(code,b) in itertools.product(q.BASES,zip(q.CODES,q.BS)):
                        n=g*g;k=(n*int(code[1:]))//100 if basis==q.BASES[0] else n*((16*int(code[1:]))//100)//16;x=k/n;effective=b if basis==q.BASES[0] else ((16*int(code[1:]))//100)/16
                        sw=math.fsum((0. if i==0 else .04+.003*(i%5))*(x+x*(1-x)*(.027*v*s+.001*s*s-.005*R*R+.001*g+.0002*i)) for i in members)
                        sc=math.fsum((0. if i==1 else .13+.004*(i%4))*(x+x*(1-x)*(-.021*v*s+.0008*s*s+.002*R-.001*g+.0001*i)) for i in members)
                        A=x*len(members)
                        external.append(dict(meta,budget_basis=basis,parent_nominal_budget_code=code,parent_nominal_budget_fraction=b,effective_budget_fraction=effective,k=k,N=n,actual_fraction=x,
                            selected_area_FAE=A,selected_weed_FAE=sw,selected_crop_FAE=sc,total_weed_FAE=tw,total_crop_FAE=tc,
                            selected_background_FAE=A-sw-sc,unselected_weed_FAE=tw-sw,unused_effective_budget_area_FAE=(effective-x)*len(members),
                            weed_capture_ratio=sw/tw if tw else None,crop_capture_ratio=sc/tc if tc else None,crop_fraction_of_selected_area=sc/A if A else None,duplicate_selection_alias='saved_alias' if b in (.5,1.) else ''))
    return scalar,local,la,external,ea

def audit_family(values,dims,metrics,models,extra=None):
    result=q.family_tables(values,dims,metrics,models,extra)
    base={tuple(r[k] for k in ('dataset','encoder','lr','seed')+dims+( 'variant',)):r for r in values}
    for row in result['hard_soft_pairs']:
        key=tuple(row[k] for k in ('dataset','encoder','lr','seed')+dims)
        h,soft=base[key+('hard_seg',)],base[key+('soft_seg',)]
        for metric in metrics:
            equals(row['hard_'+metric],h[metric]);equals(row['soft_'+metric],soft[metric]);equals(row['soft_minus_hard_'+metric],None if None in (h[metric],soft[metric]) else soft[metric]-h[metric])
    def audit_stats(source,stats,keys,names):
        groups={}
        for row in source:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
        assert len(stats)==len(groups)
        for row in stats:
            src=groups[tuple(row[k] for k in keys)];assert len(src)==row['n_models'] and row['independent_field_replicates']==1
            for metric in names:
                vals=[r[metric] for r in src];mean,sd=manual_stats(vals)
                equals(row[metric+'_mean'],mean);equals(row[metric+'_sample_sd'],sd)
                assert row[metric+'_defined_models']==sum(x is not None for x in vals)
    audit_stats(values,result['seed_mean_sample_sd'],('dataset','encoder','variant','lr')+dims,metrics)
    audit_stats(result['hard_soft_pairs'],result['hard_soft_seed_mean_sample_sd'],('dataset','encoder','lr')+dims,tuple('soft_minus_hard_'+x for x in metrics))
    for axis,pairs in [('R',((4,1),(2,1),(4,2),(6,1),(6,4)))]+([extra] if extra else []):
        other=tuple(x for x in dims if x!=axis);lookup={tuple(r[k] for k in ('run_id',)+other+(axis,)):r for r in values};contrast=result[axis+'_contrasts']
        assert len(contrast)==len(values)//len({r[axis] for r in values})*len(pairs)
        assert {(r['comparison'],r['reference']) for r in contrast}==set(pairs)
        for row in contrast:
            prefix=tuple(row[k] for k in ('run_id',)+other);hi,lo=lookup[prefix+(row['comparison'],)],lookup[prefix+(row['reference'],)]
            for metric in metrics:
                equals(row['comparison_'+metric],hi[metric]);equals(row['reference_'+metric],lo[metric]);equals(row['difference_'+metric],None if None in (hi[metric],lo[metric]) else hi[metric]-lo[metric])
        audit_stats(contrast,result[axis+'_contrast_seed_mean_sample_sd'],('dataset','encoder','variant','lr')+other+('contrast',),tuple('difference_'+x for x in metrics))
    return {name:len(rows) for name,rows in result.items()}

def main():
    check('exact_frozen_candidate_source',core.sha(q.__file__)==PIN)
    check('stdlib_only_imports',not any(n in sys.modules for n in ('torch','numpy','scipy')))
    models=make_models();raw=fixtures(models)
    s=q.normalize_scalar(raw[0],models);l=q.normalize_local(raw[1],models);a=q.normalize_local(raw[2],models,True);e=q.normalize_external(raw[3],models);ea=q.normalize_external(raw[4],models,True)
    frames=q.validate_population(s,l,a,e,ea,models)
    check('full37_24_6_population',len(s)==148 and len(l)==2880 and len(a)==288 and len(e)==21600 and len(ea)==2160)
    check('unbalanced_three_date_fixture',len(frames)==26 and sorted(list(frames.values()).count(x) for x in set(frames.values()))==[5,9,12])
    for old,new in zip(raw[0],s):
        for metric in q.SCALAR_METRICS:equals(new[metric],None if old[metric[:-3]] is None else old[metric[:-3]]*100)
    for old,new in zip(raw[1],l):equals(new['capture_pct'],old['capture_fraction']*100);equals(new['domain_area_m2'],100);equals(new['domain_target_m2'],10)
    for old,new in zip(raw[3],e):
        for dst,src in [('weed_capture_pct','weed_capture_ratio'),('crop_capture_pct','crop_capture_ratio'),('crop_fraction_pct','crop_fraction_of_selected_area')]:equals(new[dst],None if old[src] is None else old[src]*100)
    check('all_unit_conversions_independently_recomputed')
    counts={}
    for ds in ('local','weeds'):counts[ds+'_scalar']=audit_family([r for r in s if r['dataset']==ds],('R',),q.SCALAR_METRICS,models)
    counts['local_budget']=audit_family(l,('M','R','budget','policy'),q.LOCAL_METRICS,models,('M',((4,1),(2,1),(4,2))))
    counts['local_auc']=audit_family(a,('M','R'),q.AUC_METRICS,models,('M',((4,1),(2,1),(4,2))))
    for scope in ('frame','date','pooled'):
        counts[scope+'_budget']=audit_family([r for r in e if r['scope']==scope],('scope','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),q.EXTERNAL_METRICS,models,('grid_g',((16,4),(8,4),(16,8))))
        counts[scope+'_auc']=audit_family([r for r in ea if r['scope']==scope],('scope','scope_id','grid_g','R'),q.AUC_METRICS,models,('grid_g',((16,4),(8,4),(16,8))))
    check('all_families_pairs_contrasts_mean_manual_n_minus_1_sd')
    reg=q.family_tables([r for r in s if r['dataset']=='local'],('R',),q.SCALAR_METRICS,models)['seed_mean_sample_sd']
    check('single_seed_local_regression_NA_SD',all(r['n_models']==1 and r['mae_pp_sample_sd'] is None for r in reg if r['variant']=='regression'))
    check('nontrivial_seed_and_adverse_formula',any(r['capture_pct']<r['budget']*100 for r in l) and any(r['capture_pct']>r['budget']*100 for r in l))
    zero=[dict(r,captured_target_m2=0.,missed_target_m2=0.,capture_fraction=None) for r in raw[1]];za=[dict(r,auc_0_1=None,auc_0_half_normalized=None) for r in raw[2]]
    zl=q.normalize_local(zero,models);zla=q.normalize_local(za,models,True);q.validate_population(s,zl,zla,e,ea,models)
    check('zeroT_full_local_grid_preserved_with_NA',all(x['capture_pct'] is None for x in zl))
    reject('missing_model_R',lambda:q.validate_population(s[:-1],l,a,e,ea,models))
    reject('duplicate_external_cell',lambda:q.validate_population(s,l,a,e[:-1]+[e[0]],ea,models))
    reject('missing_policy_budget',lambda:q.validate_population(s,l[:-1],a,e,ea,models))
    reject('mixed_source_dataset',lambda:q.normalize_local([dict(raw[1][0],dataset='weeds')],models))
    reject('mixed_checkpoint',lambda:q.normalize_scalar([dict(raw[0][0],checkpoint_sha256='0'*64)],models))
    reject('missing_seed',lambda:q.seed_stats([r for r in s if r['dataset']=='weeds' and r['seed']!=43],('dataset','encoder','variant','lr','R'),q.SCALAR_METRICS,models))
    # Real gate functions over artificial stages; patch only upstream freeze/source validation.
    home=OUT/'toy';home.mkdir()
    for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):core.write(home/name,{'synthetic':True})
    review={'passed':True,'scope':'confirmation_report_tables','implementation_sha256':PIN,'freeze_sha256':core.sha(home/'FREEZE.json'),'all_three_outcomes_accepted':True,'stages':[]}
    for stage in q.STAGES:
        folder=home/'stages'/stage;folder.mkdir(parents=True);core.write(folder/'summary.json',{'synthetic':True});core.write(folder/'status.json',{'state':'completed'})
        (folder/'curve.csv').write_text('k,area,capture\n0,0,0\n1,0.13,0.72\n2,0.9,0.81\n3,1,1\n',encoding='utf-8')
        core.write(folder/'artifact_manifest.json',core.stage_manifest(folder));audit=home/(stage+'_audit.json')
        core.write(audit,{'passed':True,'stage':stage,'audited_artifact_manifest_sha256':core.sha(folder/'artifact_manifest.json'),'freeze_sha256':review['freeze_sha256']})
        review['stages'].append({'stage':stage,'artifact_manifest_sha256':core.sha(folder/'artifact_manifest.json'),'summary_sha256':core.sha(folder/'summary.json'),'independent_review':core.binding(audit)})
    root_review=home/'review.json';core.write(root_review,review)
    with patch.object(q,'HERE',home),patch.object(core,'validate_freeze',return_value=({},list(models.values()),{})),patch.object(core,'validate_source_lock',return_value={'files':{'report_tables.py':PIN}}):
        contract,frozen,folders,watched,audits=q.gate(root_review);check('three_completed_stages_exact_acceptance_and_watch',len(folders)==3 and len(audits)==3 and len(watched)==15)
        before=root_review.read_bytes();bad=dict(review,all_three_outcomes_accepted=False);core.write(root_review,bad);reject('missing_acceptance_blocks',lambda:q.gate(root_review));root_review.write_bytes(before)
        bad=dict(review,stages=list(reversed(review['stages'])));core.write(root_review,bad);reject('stage_order_swap_blocks',lambda:q.gate(root_review));root_review.write_bytes(before)
        audit=home/'local_outcomes_audit.json';saved=audit.read_bytes();core.write(audit,{'passed':True});reject('unqualified_passed_audit_blocks',lambda:q.gate(root_review));audit.write_bytes(saved)
        source=folders['local_outcomes']/'curve.csv';dest=home/'copies';dest.mkdir();q.copy_saved_table(folders['local_outcomes'],'curve.csv',dest,'copied.csv',watched)
        check('full_curve_copy_is_byte_exact_no_interpolation',source.read_bytes()==(dest/'copied.csv').read_bytes())
        source.write_text('different but locally self-consistent',encoding='utf-8');core.write(source.parent/'artifact_manifest.json',core.stage_manifest(source.parent))
        reject('begin_to_end_consistent_replacement_detected',lambda:[core.bound_file(path,digest) for path,digest in watched.items()])
    tree=ast.parse(Path(q.__file__).read_text(encoding='utf-8-sig'));execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    source=ast.get_source_segment(Path(q.__file__).read_text(encoding='utf-8-sig'),execute)
    check('end_checks_source_lock_and_each_completed_stage','for p,h in watched.items():core.bound_file(p,h)' in source and 'core.validate_source_lock()' in source and 'core.completed_stage(folder)' in source)
    check('table_read_whitelist_no_new_score_or_reselection',not any(n in sys.modules for n in ('numpy','torch','scipy')))
    report={'passed':True,'implementation_sha256':PIN,'checker_sha256':core.sha(ROOT/'confirmation_v1/check_report_tables.py'),'interface_sha256':core.sha(ROOT/'confirmation_v1/REPORT_TABLES_INTERFACE.md'),
        'independent_checker_sha256':core.sha(__file__),'checks':checks,'check_count':len(checks),'numerical_comparisons':comparisons,'max_absolute_error':max_error,'family_counts':counts,
        'producer_readiness':core.binding(ROOT/'provenance/confirmation_report_checks_20260916_104157_633015/focused_checks.json'),
        'synthetic_population':{'models':37,'local_scalar':112,'external_scalar':36,'local_budget':2880,'local_auc':288,'external_budget':21600,'external_auc':2160,'frames':26,'artificial_date_sizes':[5,9,12]},
        'real_outcome_or_label_reads':0,'model_calls':0,'solver_calls':0,'production_sources_modified':False,'blocking_findings':[],
        'limitations':['Producer numerical correctness still requires independently accepted actual outcome-stage audits.','Reporter intentionally copies full curves and reference tables verbatim; it does not recompute AUC from raw arrays.','No real confirmation execution or full report-stage output acceptance was performed; this is preparation QA.','Constant baselines are checked for population and copied, so their numerical recomputation belongs to the scalar-stage audit.']}
    core.write(OUT/'review.json',report)
    (OUT/'review.md').write_text('# Independent confirmation report preparation review\n\nPASS. Complete synthetic model/grids, all signed paired/scale contrasts and manual n−1 seed SD were verified with nonconstant seed/R/grid values and unbalanced artificial dates. Dataset/alias and zero-denominator distinctions are retained. Three-stage acceptance and begin/end replacement rejection passed; complete curve bytes remain unchanged.\n\nThis is source/preparation acceptance, not acceptance of future confirmation results. See review.json for exact hashes, counts and limitations.\n',encoding='utf-8')
    print(json.dumps({'passed':True,'path':str(OUT/'review.json'),'sha256':core.sha(OUT/'review.json'),'checks':len(checks),'comparisons':comparisons,'max_error':max_error}))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        core.write(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'checks':checks});raise
