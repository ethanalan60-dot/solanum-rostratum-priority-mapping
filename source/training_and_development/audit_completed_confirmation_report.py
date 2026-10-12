"""Independent audit of completed confirmation report tables; stdlib, saved tables only.

No producer imports, tensor/array/image reads, truth-table parsing, model calls,
solver calls, epoch selection, or source-stage execution. CLI requires --execute.
"""
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,itertools,json,math,shutil,sys,traceback

ROOT=Path(__file__).resolve().parent
REPORTER_SHA='1139e90254af3eff61335e3a9f96232efa85b845fae7c6089c311235e85bfc26'
STAGES=('scalar_outcomes','local_outcomes','external_outcomes')
RS=(1,2,4,6);MS=(1,2,4);GS=(4,8,16);BS=(.1,.2,.3,.5,1.)
CODES=('b10','b20','b30','b50','b100');BASES=('nominal_primary','matched_actual_secondary')
RP=((4,1),(2,1),(4,2),(6,1),(6,4));MP=((4,1),(2,1),(4,2));GP=((16,4),(8,4),(16,8))
META=('run_id','dataset','encoder','variant','lr','seed','best_epoch','checkpoint_sha256')
SM=('mae_pp','bias_pp','mae_zero_pp','mae_positive_pp','balanced_mae_pp')
LM=('capture_pct','captured_target_m2','actual_area_m2','unused_budget_m2','missed_target_m2','selected_non_target_area_m2')
EM=('weed_capture_pct','crop_capture_pct','crop_fraction_pct','selected_area_FAE','selected_weed_FAE','selected_crop_FAE','unselected_weed_FAE','selected_background_FAE','unused_effective_budget_area_FAE')
AM=('auc_0_1','auc_0_0p5_normalized')
MAIN_INPUTS={'scalar_outcomes':['all_model_R_metrics.csv'],'local_outcomes':['models/all_fixed_budgets.csv','models/all_prefix_auc.csv'],
 'external_outcomes':[f'learned_{s}_{k}.csv' for s in ('frame','date','pooled') for k in ('budget_metrics','auc')]}
LOCAL_COPIES=[('models/all_complete_prefix_curves.csv','complete_prefix_curves.csv')]+[(f'shared_reference/{n}.json',f'shared_{n}.json') for n in ('label_prefix_budgets','label_MILP_budgets','label_prefix_curves','label_prefix_auc','random_budgets','random_auc','random_summary','summary','reference_seal')]
EXT_COPIES=[(f'{a}_{b}_{c}.csv',f'{a}_{b}_{c}.csv') for a,b,c in itertools.product(('learned','label_oracle','random_expectation'),('frame','date','pooled'),('complete_prefix_curves','budget_metrics','auc')) if a!='learned' or c=='complete_prefix_curves']

def report_file_names():
    names={'status.json','input_seal.json','begin_binding.json','end_binding.json','summary.json','model_metadata.json','saved_curve_reference_index.json'}
    stems=[('local','scalar',None),('weeds','scalar',None),('local','spatial_budget','M'),('local','prefix_auc','M')]+[('weeds',scope+'_'+kind,'grid_g') for scope in ('frame','date','pooled') for kind in ('spatial_budget','prefix_auc')]
    common=('all','hard_soft_pairs','seed_mean_sample_sd','hard_soft_seed_mean_sample_sd','R_contrasts','R_contrast_seed_mean_sample_sd')
    for dataset,stem,extra in stems:
        suffixes=common+((extra+'_contrasts',extra+'_contrast_seed_mean_sample_sd') if extra else ())
        names.update(f'{dataset}/{stem}_{suffix}.csv' for suffix in suffixes)
    names.add('local/primary_M2_budget10_prefix.csv')
    names.update('weeds/primary_g8_pooled_'+basis+'.csv' for basis in BASES)
    names.update(ds+'/constant_scalar_baselines.csv' for ds in ('local','weeds'))
    names.update('local/'+dest for _,dest in LOCAL_COPIES);names.update('weeds/'+dest for _,dest in EXT_COPIES)
    return names

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def beneath(folder,name):
    p=(Path(folder)/name).resolve();require(p.is_relative_to(Path(folder).resolve()),'Escaped artifact path');return p
def num(x,optional=False):
    if x is None or x=='':require(optional,'Missing number');return None
    v=float(x);require(math.isfinite(v),'Nonfinite number');return v
def flag(x):
    if x is None or x=='':return None
    require(x in (True,False,'True','False','true','false'),'Invalid saved boolean')
    return x in (True,'True','true')
def integer(x):
    v=num(x);require(v==int(v),'Noninteger field');return int(v)
def subtract(a,b):return None if a is None or b is None else a-b
def mean_sd(values):
    if any(x is None for x in values):return None,None
    mu=math.fsum(values)/len(values)
    sd=math.sqrt(math.fsum((x-mu)**2 for x in values)/(len(values)-1)) if len(values)>1 else None
    return mu,sd

class Reader:
    """Explicit allowlist, captured before parsing; no implicit recursive payload hashing."""
    def __init__(self):self.allowed=set();self.watched={};self.inventories={};self.parsed=[]
    def allow(self,p):
        p=Path(p).resolve();require(p.suffix.lower() in ('.json','.csv','.py'), 'Non-table/source file forbidden')
        self.allowed.add(p);return p
    def pin(self,p,expected=None):
        p=Path(p).resolve();require(p in self.allowed,'Read outside explicit allowlist: '+str(p))
        h=digest(p);require(expected is None or h==expected,'SHA mismatch: '+str(p))
        require(str(p) not in self.watched or self.watched[str(p)]==h,'Source drift: '+str(p));self.watched[str(p)]=h
        return {'path':str(p),'sha256':h}
    def json(self,p,expected=None):
        self.pin(p,expected);require(Path(p).suffix=='.json','JSON only');self.parsed.append(str(Path(p).resolve()))
        return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    def csv(self,p,expected=None):
        self.pin(p,expected);require(Path(p).suffix=='.csv','CSV only');self.parsed.append(str(Path(p).resolve()))
        with Path(p).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);require(reader.fieldnames and len(set(reader.fieldnames))==len(reader.fieldnames),'Duplicate/missing CSV header')
            data=list(reader);require(all(None not in row for row in data),'Malformed CSV width');return data
    def finish(self):
        for p,h in self.watched.items():require(digest(p)==h,'Source changed during audit: '+p)
        for p,names in self.inventories.items():require(inventory(p)==names,'Report inventory changed')

def inventory(folder):return {p.relative_to(folder).as_posix() for p in Path(folder).rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
def stage_metadata(reader,folder,full_report=False,expected_manifest=None):
    folder=Path(folder).resolve()
    manifest=reader.json(reader.allow(folder/'artifact_manifest.json'),expected_manifest)
    require(isinstance(manifest.get('files'),dict),'Wrong stage artifact schema')
    for name,h in manifest['files'].items():
        beneath(folder,name);require(isinstance(h,str) and len(h)==64,'Invalid artifact digest')
    status=reader.json(reader.allow(folder/'status.json'),manifest['files']['status.json'])
    require(status['state']=='completed','Stage incomplete: '+str(folder))
    if full_report:
        names=inventory(folder);require(names==set(manifest['files']),'Report artifact inventory differs');reader.inventories[str(folder)]=names
        require(names==report_file_names(),'Report filename allowlist differs; no unknown payload opened')
        require(all(Path(x).suffix.lower() in ('.json','.csv') for x in names),'Unexpected non-table report artifact')
        for name,h in manifest['files'].items():reader.pin(reader.allow(beneath(folder,name)),h)
    return manifest

def acceptance_file(reader,stage,pin):
    """Respect the actual frozen auditors' different output locations."""
    p=Path(pin['path']).resolve()
    if stage=='local_outcomes':
        expected=(ROOT/'confirmation_v1/stages/audit_local_outcomes/review.json').resolve()
        require(p==expected,'Local acceptance must be the canonical completed audit stage')
        manifest=stage_metadata(reader,p.parent)
        require(manifest['files'].get('review.json')==pin['sha256'],'Local acceptance absent from completed audit manifest')
    else:
        require(stage in ('scalar_outcomes','external_outcomes') and p.is_relative_to((ROOT/'provenance').resolve()),'Outcome acceptance outside its declared provenance location')
    return reader.json(reader.allow(p),pin['sha256'])

def indexed(rows,keys):
    output={}
    for row in rows:
        key=tuple(row[k] for k in keys);require(key not in output,'Duplicate table key: '+str(key));output[key]=row
    return output
def grid(rows,keys,expected):require(set(indexed(rows,keys))==set(expected),'Incomplete table population: '+','.join(keys))

def model_fields(row,models):
    m=models[row['run_id']];out={k:m[k] for k in META}
    for k in ('dataset','encoder','variant','checkpoint_sha256'):
        if k in row:require(row[k]==m[k],'Model identity mismatch: '+k)
    for k in ('lr','seed'):
        if k in row:require(num(row[k])==m[k],'Model setting mismatch: '+k)
    for k in ('best_epoch','original_selected_epoch'):
        if k in row:require(integer(row[k])==m['best_epoch'],'Original checkpoint epoch mismatch')
    return out

def normalize(raw,kind,models):
    """Independent, explicit source-to-report unit/field mapping; no producer calls."""
    result=[]
    for row in raw:
        x=dict(row);x.update(model_fields(row,models));x['R']=integer(row['R'])
        if kind=='scalar':
            for k in ('n','n_zero','n_positive'):x[k]=integer(row[k])
            require(x['n']==(338 if x['dataset']=='local' else 26) and x['n_zero']+x['n_positive']==x['n'],'Scalar population mismatch')
            for k in SM:
                v=num(row[k[:-3]],True);x[k]=None if v is None else v*100
            require(x['mae_pp'] is not None and x['bias_pp'] is not None,'Missing scalar metric')
            require((x['mae_zero_pp'] is None)==(x['n_zero']==0) and (x['mae_positive_pp'] is None)==(x['n_positive']==0),'Scalar stratum NA mismatch')
            if None in (x['mae_zero_pp'],x['mae_positive_pp']):require(x['balanced_mae_pp'] is None,'Undefined balanced error filled')
            else:require(math.isclose(x['balanced_mae_pp'],(x['mae_zero_pp']+x['mae_positive_pp'])/2,rel_tol=1e-10,abs_tol=1e-9),'Balanced arithmetic differs')
        elif kind.startswith('local'):
            require(x['dataset']=='local' and x['variant']!='regression','Wrong local spatial model');x['M']=integer(row['M'])
            if kind=='local_auc':
                x.update(policy='prefix',budget_basis='complete_prefix_once',auc_0_1=num(row['auc_0_1'],True),auc_0_0p5_normalized=num(row['auc_0_half_normalized'],True))
            else:
                require(row['policy'] in ('model_prefix','predicted_area'),'Wrong policy');x.update(budget=num(row['budget']),original_policy=row['policy'],policy='prefix' if row['policy']=='model_prefix' else 'predicted_area')
                for k in LM:
                    v=num(row[{'capture_pct':'capture_fraction','unused_budget_m2':'unused_area_m2'}.get(k,k)],k=='capture_pct');x[k]=None if v is None else v*(100 if k=='capture_pct' else 1)
                x['budget_area_m2']=x['actual_area_m2']+x['unused_budget_m2'];x['domain_area_m2']=x['budget_area_m2']/x['budget'];x['domain_target_m2']=x['captured_target_m2']+x['missed_target_m2'];x['actual_area_fraction']=x['actual_area_m2']/x['domain_area_m2']
                require(x['actual_area_m2']>=0 and x['unused_budget_m2']>=0,'Negative saved budget area')
                require((x['capture_pct'] is None)==(x['domain_target_m2']==0),'Local capture NA mismatch')
                if x['domain_target_m2']:require(math.isclose(x['capture_pct'],100*x['captured_target_m2']/x['domain_target_m2'],rel_tol=1e-10,abs_tol=1e-9),'Local capture ratio differs')
                x['capture_na_reason']='zero_total_target' if x['capture_pct'] is None else ''
                for k in ('fallback_used','bound_anomaly','updated_proven_flag','certification_downgraded'):x[k]=flag(row.get(k))
                for k in ('updated_upper_m2','updated_relative_gap'):x[k]=num(row.get(k),True)
                require(not x['fallback_used'] or x['updated_proven_flag'] is False,'Fallback incorrectly certified')
        else:
            require(x['dataset']=='weeds' and x['variant']!='regression','Wrong external spatial model');x['grid_g']=integer(row['grid_g']);x['frame_n']=integer(row['frame_n'])
            if kind=='external_auc':
                x['budget_basis']='complete_prefix_once'
                for k in AM:x[k]=num(row[k],True)
                require(all((x[k] is None)==(num(row['total_weed_FAE'])==0) for k in AM),'External AUC NA mismatch')
            else:
                for k in ('k','N'):x[k]=integer(row[k])
                for k in ('effective_budget_fraction','parent_nominal_budget_fraction','actual_fraction'):x[k]=num(row[k])
                for k in EM:
                    src={'weed_capture_pct':'weed_capture_ratio','crop_capture_pct':'crop_capture_ratio','crop_fraction_pct':'crop_fraction_of_selected_area'}.get(k,k)
                    v=num(row[src],k.endswith('_pct'));x[k]=None if v is None else v*(100 if k.endswith('_pct') else 1)
                require(x['N']==x['grid_g']**2 and x['actual_fraction']==x['k']/x['N'],'External integer area mismatch')
                require(x['budget_basis'] in BASES and x['parent_nominal_budget_code'] in CODES,'Unknown budget basis/code')
                percent=int(x['parent_nominal_budget_code'][1:]);expected_k=x['N']*percent//100 if x['budget_basis']==BASES[0] else x['N']*((16*percent)//100)//16
                require(x['k']==expected_k,'Whole-unit budget differs')
                require(math.isclose(x['selected_area_FAE'],x['actual_fraction']*x['frame_n'],rel_tol=1e-10,abs_tol=1e-12),'External area denominator differs')
                x['duplicate_selection_alias']=row.get('duplicate_selection_alias') or ''
                for k,a,b in [('weed_capture_pct','selected_weed_FAE','total_weed_FAE'),('crop_capture_pct','selected_crop_FAE','total_crop_FAE'),('crop_fraction_pct','selected_crop_FAE','selected_area_FAE')]:
                    den=num(row[b]);require((x[k] is None)==(den==0),'External ratio NA mismatch')
                    if den:require(math.isclose(x[k],100*num(row[a])/den,rel_tol=1e-10,abs_tol=1e-10),'External ratio differs')
        result.append(x)
    return result

def validate_population(s,l,la,e,ea,models):
    lids=[k for k,m in models.items() if m['dataset']=='local' and m['variant']!='regression'];wids=[k for k,m in models.items() if m['dataset']=='weeds' and m['variant']!='regression']
    require(len(models)==37 and len(lids)==24 and len(wids)==6,'Wrong 37/24/6 model scope')
    grid(s,('run_id','R'),itertools.product(models,RS));grid(l,('run_id','M','R','budget','policy'),itertools.product(lids,MS,RS,BS,('prefix','predicted_area')));grid(la,('run_id','M','R'),itertools.product(lids,MS,RS))
    totals={(x['run_id'],x['M'],x['R']):x['domain_target_m2'] for x in l}
    for x in la:require(all((x[k] is None)==(totals[x['run_id'],x['M'],x['R']]==0) for k in AM),'Local AUC NA population differs')
    dates={}
    for x in e:
        if x['scope']=='frame':
            require(x['scope_id']==x['frame_id'] and x['frame_n']==1 and x['acquisition_date'],'External frame identity differs')
            require(x['frame_id'] not in dates or dates[x['frame_id']]==x['acquisition_date'],'Frame date changed');dates[x['frame_id']]=x['acquisition_date']
    require(len(dates)==26,'Not all26 external frames')
    scopes={'frame':sorted(dates),'date':sorted(set(dates.values())),'pooled':['all26']}
    require({x['scope'] for x in e}==set(scopes) and {x['scope'] for x in ea}==set(scopes),'Wrong external scopes')
    for scope,ids in scopes.items():
        z=[x for x in e if x['scope']==scope];a=[x for x in ea if x['scope']==scope]
        grid(z,('run_id','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),itertools.product(wids,ids,GS,RS,BASES,CODES));grid(a,('run_id','scope_id','grid_g','R'),itertools.product(wids,ids,GS,RS))
        for x in z+a:require(x['frame_n']==(1 if scope=='frame' else 26 if scope=='pooled' else list(dates.values()).count(x['scope_id'])),'Grouped frame count mismatch')
    return dates

def grouped(values,keys):
    out=collections.defaultdict(list)
    for row in values:out[tuple(row[k] for k in keys)].append(row)
    return out
def pairs(values,dims,metrics):
    keys=('dataset','encoder','lr','seed')+dims;out=[]
    for key,group in sorted(grouped([r for r in values if r['variant']!='regression'],keys).items()):
        by=indexed(group,('variant',));require(set(by)=={('hard_seg',),('soft_seg',)},'Hard/soft pair incomplete');h,s=by[('hard_seg',)],by[('soft_seg',)]
        r=dict(zip(keys,key));r.update(hard_run_id=h['run_id'],soft_run_id=s['run_id'])
        for k in metrics:r.update({f'hard_{k}':h[k],f'soft_{k}':s[k],f'soft_minus_hard_{k}':subtract(s[k],h[k])})
        out.append(r)
    return out
def contrasts(values,dims,metrics,axis,levels):
    keys=META+tuple(k for k in dims if k!=axis);out=[]
    for key,group in sorted(grouped(values,keys).items()):
        by=indexed(group,(axis,));require(set(by)=={(v,) for pair in levels for v in pair},'Scale population incomplete')
        for a,b in levels:
            x,y=by[(a,)],by[(b,)];r=dict(zip(keys,key));r.update(contrast=f'{axis}{a}_minus_{axis}{b}',comparison=a,reference=b)
            for k in metrics:r.update({f'comparison_{k}':x[k],f'reference_{k}':y[k],f'difference_{k}':subtract(x[k],y[k])})
            out.append(r)
    return out
def statistics_rows(values,keys,metrics,models):
    out=[]
    for key,group in sorted(grouped(values,keys).items()):
        by=indexed(group,('seed',));f=group[0]
        seeds={x['seed'] for x in models.values() if all(x[k]==f[k] for k in ('dataset','encoder','lr')) and (x['variant']==f['variant'] if 'variant' in f else x['variant']!='regression')}
        require(set(by)=={(s,) for s in seeds},'Missing declared fitting seed')
        row=dict(zip(keys,key));row.update(seeds=';'.join(str(s) for s in sorted(seeds)),n_models=len(group),independent_field_replicates=1)
        for k in metrics:
            v=[x[k] for x in group];mu,sd=mean_sd(v);row.update({k+'_defined_models':sum(x is not None for x in v),k+'_mean':mu,k+'_sample_sd':sd})
        out.append(row)
    return out
def families(values,dims,metrics,models,extra=None):
    """Yield independently reconstructed rows with their complete unique-key definition."""
    yield 'all',values,('run_id',)+dims
    pair=pairs(values,dims,metrics);pk=('dataset','encoder','lr','seed')+dims
    yield 'hard_soft_pairs',pair,pk
    sk=('dataset','encoder','variant','lr')+dims
    yield 'seed_mean_sample_sd',statistics_rows(values,sk,metrics,models),sk
    psk=('dataset','encoder','lr')+dims;pm=tuple('soft_minus_hard_'+k for k in metrics)
    yield 'hard_soft_seed_mean_sample_sd',statistics_rows(pair,psk,pm,models),psk
    for axis,levels in [('R',RP)]+([extra] if extra else []):
        c=contrasts(values,dims,metrics,axis,levels);other=tuple(k for k in dims if k!=axis)
        yield axis+'_contrasts',c,('run_id',)+other+('contrast',)
        ck=('dataset','encoder','variant','lr')+other+('contrast',);cm=tuple('difference_'+k for k in metrics)
        yield axis+'_contrast_seed_mean_sample_sd',statistics_rows(c,ck,cm,models),ck

class Compare:
    def __init__(self):self.cells=0;self.numeric=0;self.na=0;self.max_error=0.;self.exact_zero_differences=0;self.rows=0
    def cell(self,got,want,label):
        self.cells+=1
        if want is None:self.na+=1;require(got=='','NA changed: '+label);return
        if isinstance(want,bool):require(got==str(want),'Boolean changed: '+label);return
        if isinstance(want,(int,float)):
            v=num(got);err=abs(v-want);self.numeric+=1;self.max_error=max(self.max_error,err)
            require(err<=max(1e-9,1e-10*abs(want)),'Arithmetic differs: '+label+f' ({v}, {want})')
            if ('difference_' in label or 'minus_hard_' in label) and want==0:self.exact_zero_differences+=1
        else:require(got==str(want),'Saved field changed: '+label)
    def table(self,got,wanted,keys,label):
        require(bool(wanted),'Empty expected table');want_keys=list(dict.fromkeys(k for r in wanted for k in r));require(all(set(r)==set(want_keys) for r in got),'Missing/extra output columns: '+label)
        # Canonical keys are typed from the independent expectation, not coerced with eval.
        key_examples={k:next(r[k] for r in wanted if r[k] is not None) for k in keys}
        def key(row):return tuple(integer(row[k]) if isinstance(key_examples[k],int) else num(row[k]) if isinstance(key_examples[k],float) else row[k] for k in keys)
        observed={}
        for row in got:
            k=key(row);require(k not in observed,'Duplicate output key: '+label);observed[k]=row
        expected=indexed(wanted,keys);require(set(observed)==set(expected),'Missing/extra output key: '+label);self.rows+=len(got)
        for k,row in expected.items():
            for col in want_keys:self.cell(observed[k][col],row.get(col),label+'/'+col)

def audit(folder,grant_path,grant_sha,reader,compare):
    home=ROOT/'confirmation_v1';folder=Path(folder).resolve();require(folder==(home/'stages/report_tables').resolve(),'Only canonical completed report is allowed')
    reader.pin(reader.allow(__file__))
    report_manifest=stage_metadata(reader,folder,True)
    for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):reader.allow(home/name)
    freeze=reader.json(home/'FREEZE.json');require(freeze.get('passed') is True and freeze.get('frozen') is True and freeze.get('scope')=='confirmation_v1','Formal freeze missing')
    lock=reader.json(home/'SOURCE_LOCK.json',freeze['source_lock_sha256']);contract=reader.json(home/'contract.json',freeze['contract_sha256']);checkpoints=reader.json(home/'checkpoints.json',freeze['checkpoints_sha256'])
    reader.pin(reader.allow(home/'report_tables.py'),REPORTER_SHA);require(lock['files']['report_tables.py']==REPORTER_SHA,'Reporter differs from accepted frozen source')
    require(checkpoints['frozen'] is True and len(checkpoints['models'])==37,'Unfrozen/incomplete37 checkpoint metadata')
    model_list=checkpoints['models'];models={x['run_id']:x for x in model_list};require(len(models)==37,'Duplicate frozen model')
    require([{k:r[k] for k in c} for r,c in zip(model_list,contract['roster'])]==contract['roster'] and len(contract['roster'])==37,'Frozen roster/order differs')
    require(contract['R']==freeze['R']==list(RS) and freeze['run_ids']==list(models),'Wrong R/population freeze')
    require(contract['version']=='confirmation_v1' and contract['intermediate_sizes']==[256,128,64,42] and contract['output_size']==256 and contract['CPU_threads']==2 and contract['batch_size']==4 and contract['dtype']=='float32','Frozen resource/support contract differs')
    require(contract['val_R6'] is False and contract['norm_diagnostics_included'] is False,'Wrong confirmation scope')
    require(freeze['truth_after_all_predictions_and_selections'] is True,'Wrong sequencing contract')
    for ds,total,seg in [('local',28,24),('weeds',9,6)]:
        own=[m for m in models.values() if m['dataset']==ds];require(len(own)==total and sum(m['variant']!='regression' for m in own)==seg,'Dataset model counts differ')
    expected_settings={('local',e,lr,v,s) for e in ('resnet18','convnextv2_tiny') for lr in (.0001,.0003) for v in ('regression','hard_seg','soft_seg') for s in ((17,) if v=='regression' else (17,29,43))}|{('weeds','resnet18',.0001,v,s) for v in ('regression','hard_seg','soft_seg') for s in (17,29,43)}
    require({tuple(m[k] for k in ('dataset','encoder','lr','variant','seed')) for m in model_list}==expected_settings,'Frozen model-setting inventory differs')
    grant_path=Path(grant_path).resolve();require(grant_path.is_relative_to(ROOT),'Grant outside campaign');grant=reader.json(reader.allow(grant_path),grant_sha)
    freeze_sha=reader.watched[str((home/'FREEZE.json').resolve())]
    require(grant.get('passed') is True and grant.get('scope')=='confirmation_report_tables' and grant['implementation_sha256']==REPORTER_SHA and grant['freeze_sha256']==freeze_sha,'Wrong report grant')
    require(grant['all_three_outcomes_accepted'] is True and [x['stage'] for x in grant['stages']]==list(STAGES),'Three exact outcome acceptances required')
    producer_watch={str((home/'report_tables.py').resolve()):REPORTER_SHA,str(grant_path):grant_sha}
    for n in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):producer_watch[str((home/n).resolve())]=reader.watched[str((home/n).resolve())]
    folders={};manifests={};accepted=[]
    for entry in grant['stages']:
        name=entry['stage'];f=home/'stages'/name;manifests[name]=stage_metadata(reader,f,False,entry['artifact_manifest_sha256']);folders[name]=f
        reader.json(reader.allow(f/'summary.json'),entry['summary_sha256']);require(manifests[name]['files']['summary.json']==entry['summary_sha256'],'Source summary not in accepted manifest')
        pin=entry['independent_review'];p=Path(pin['path']).resolve();a=acceptance_file(reader,name,pin)
        require(a.get('passed') is True and a.get('stage')==name and a.get('audited_artifact_manifest_sha256')==entry['artifact_manifest_sha256'] and a.get('freeze_sha256')==freeze_sha,'Source acceptance identity differs')
        accepted.append(pin)
        for p0,h in [(f/'artifact_manifest.json',entry['artifact_manifest_sha256']),(f/'summary.json',entry['summary_sha256']),(p,pin['sha256'])]:producer_watch[str(Path(p0).resolve())]=h
    def source(stage,name,parse=True):
        allowed=set(MAIN_INPUTS[stage])
        if stage=='scalar_outcomes':allowed.add('constant_baselines.csv')
        if stage=='local_outcomes':allowed.update(n for n,_ in LOCAL_COPIES)
        if stage=='external_outcomes':allowed.update(n for n,_ in EXT_COPIES)
        require(name in allowed,'Source table not explicitly allowed');p=beneath(folders[stage],name);h=manifests[stage]['files'][name]
        reader.pin(reader.allow(p),h);producer_watch[str(p)]=h
        return reader.csv(p,h) if parse else p
    s=normalize(source('scalar_outcomes','all_model_R_metrics.csv'),'scalar',models)
    l=normalize(source('local_outcomes','models/all_fixed_budgets.csv'),'local_budget',models);la=normalize(source('local_outcomes','models/all_prefix_auc.csv'),'local_auc',models)
    e=[];ea=[]
    for scope in ('frame','date','pooled'):
        e.extend(normalize(source('external_outcomes',f'learned_{scope}_budget_metrics.csv'),'external_budget',models));ea.extend(normalize(source('external_outcomes',f'learned_{scope}_auc.csv'),'external_auc',models))
    frame_dates=validate_population(s,l,la,e,ea,models)
    begin=reader.json(folder/'begin_binding.json');require(begin['watched_sha256']==producer_watch and begin['root_review']=={'path':str(grant_path),'sha256':grant_sha} and begin['independent_reviews']==accepted,'Report begin sources differ from accepted inputs')
    required={'status.json','input_seal.json','begin_binding.json','end_binding.json','summary.json','model_metadata.json','saved_curve_reference_index.json'}
    counts={};table_count=0
    def check(name,wanted,keys):
        nonlocal table_count
        require(name in report_manifest['files'],'Missing report table');compare.table(reader.csv(folder/name),wanted,keys,name);required.add(name);table_count+=1
    specs=[('local','scalar',[r for r in s if r['dataset']=='local'],('R',),SM,None),('weeds','scalar',[r for r in s if r['dataset']=='weeds'],('R',),SM,None),('local','spatial_budget',l,('M','R','budget','policy'),LM,('M',MP)),('local','prefix_auc',la,('M','R'),AM,('M',MP))]
    for scope in ('frame','date','pooled'):
        specs.extend([('weeds',scope+'_spatial_budget',[r for r in e if r['scope']==scope],('scope','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),EM,('grid_g',GP)),('weeds',scope+'_prefix_auc',[r for r in ea if r['scope']==scope],('scope','scope_id','grid_g','R'),AM,('grid_g',GP))])
    for ds,stem,values,dims,metrics,extra in specs:
        for suffix,expected,keys in families(values,dims,metrics,models,extra):
            name=ds+'/'+stem+'_'+suffix+'.csv';check(name,expected,keys);counts[name]=len(expected)
    check('local/primary_M2_budget10_prefix.csv',[r for r in l if r['M']==2 and r['budget']==.1 and r['policy']=='prefix'],('run_id','R'))
    for basis in BASES:check('weeds/primary_g8_pooled_'+basis+'.csv',[r for r in e if r['scope']=='pooled' and r['grid_g']==8 and r['budget_basis']==basis],('run_id','R','parent_nominal_budget_code'))
    ledger=[]
    for stage,ds,entries in [('local_outcomes','local',LOCAL_COPIES),('external_outcomes','weeds',EXT_COPIES)]:
        for name,dest in entries:
            p=source(stage,name,False);name_out=ds+'/'+dest;require(name_out in report_manifest['files'],'Missing copied reference/curve');h=reader.watched[str(p)]
            require(report_manifest['files'][name_out]==h,'Reference/complete-curve copy changed');reader.pin(folder/name_out,h);required.add(name_out)
            ledger.append({'file':dest,'source':str(p),'sha256':h,'format':'unaltered saved producer table'})
    constants=source('scalar_outcomes','constant_baselines.csv')
    require(len(constants)==4 and {r['dataset'] for r in constants}=={'local','weeds'},'Constant baseline inventory differs')
    for ds in ('local','weeds'):
        current=[r for r in constants if r['dataset']==ds];grid(current,('baseline',),[('zero',),('train_mean',)]);check(ds+'/constant_scalar_baselines.csv',current,('dataset','baseline'))
    require(reader.json(folder/'model_metadata.json')==model_list,'Report model metadata differs from original checkpoints')
    require(reader.json(folder/'saved_curve_reference_index.json')==ledger,'Complete curve/reference index differs')
    summary=reader.json(folder/'summary.json')
    expected_summary={'version':'confirmation_report_tables_v1','table_rows':counts,'models':{'local':28,'weeds':9},'local_spatial_models':24,'external_spatial_models':6,'R':list(RS),'R_pairs':[list(x) for x in RP],'local_M_pairs':[list(x) for x in MP],'external_grid_pairs':[list(x) for x in GP],'external_frames':26,'external_dates':sorted(set(frame_dates.values())),'primary_local':{'M':2,'budget':.1,'policy':'prefix'},'primary_external':{'grid_g':8,'scope':'pooled','budget_basis':list(BASES)},'datasets_pooled':False,'new_model_calls':0,'new_solver_calls':0,'raw_labels_or_arrays_loaded':0,'checkpoint_reselected':False,'independent_output_review':'pending'}
    for k,v in expected_summary.items():require(summary[k]==v,'Report summary differs: '+k)
    for k,v in {'scalar_unit':'percentage points in *_pp columns; source fractional columns preserved','capture_unit':'percentage points in *_pct columns; AUC remains a fraction','seed_statistics':'Complete declared seed groups; sample SD only for >=2 defined values. Any NA keeps group mean/SD NA; defined count retained.','replication':'Fitting-seed variability within each dataset scene; frames/dates/grids/budgets are not independent fields.'}.items():require(summary[k]==v,'Reporting interpretation changed: '+k)
    require(set(summary)==set(expected_summary)|{'scalar_unit','capture_unit','seed_statistics','replication'},'Summary field inventory differs')
    end=reader.json(folder/'end_binding.json');require(end['watched_sha256']==producer_watch and end['all_begin_and_consumed_sources_unchanged'] is True,'Report end source ledger differs')
    seal=reader.json(folder/'input_seal.json')
    for key,name in [('freeze','FREEZE.json'),('source_lock','SOURCE_LOCK.json'),('contract','contract.json'),('checkpoints','checkpoints.json')]:require(seal[key]=={'path':str((home/name).resolve()),'sha256':reader.watched[str((home/name).resolve())]},'Stage input seal differs')
    require(set(report_manifest['files'])==required,'Unexpected/missing report files')
    reader.finish()
    return {'passed':True,'scope':'confirmation_report_tables_independent','stage':'report_tables','output_directory':str(folder),'freeze_sha256':freeze_sha,'audited_artifact_manifest_sha256':reader.watched[str((folder/'artifact_manifest.json').resolve())],'summary_sha256':reader.watched[str((folder/'summary.json').resolve())],'reporter_sha256':REPORTER_SHA,'reviewer_sha256':reader.watched[str(Path(__file__).resolve())],'root_report_grant':{'path':str(grant_path),'sha256':grant_sha},'source_acceptances':accepted,'run_ids':list(models),'models':37,'local_spatial_models':24,'external_spatial_models':6,'external_frames':26,'external_dates':sorted(set(frame_dates.values())),'table_rows':counts,'numeric_tables_checked':table_count,'byte_exact_curve_reference_copies':len(ledger),'numeric_comparisons':compare.numeric,'comparisons':compare.__dict__,'source_and_output_hashes':reader.watched,'source_stages_hashed_only_for_allowlisted_CSV_JSON':True,'source_truth_tables_parsed':False,'npz_tensor_image_reads':0,'model_solver_subprocess_calls':0,'checkpoint_reselection':False,'limitations':['Copied curves, reference results and baseline metrics are checked for exact identity to their independently accepted producer outputs; their underlying scientific calculations are not rerun.','Acceptance is confined to this completed report and the explicitly bound three outcome stages.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--report-grant',type=Path,required=True);p.add_argument('--report-grant-sha256',required=True);p.add_argument('--execute',action='store_true',required=True);a=p.parse_args()
    out=ROOT/'provenance'/('completed_confirmation_report_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    reader=Reader();compare=Compare();reader.pin(reader.allow(__file__))
    try:
        out.mkdir();shutil.copy2(__file__,out/'auditor_source.py')
        require(digest(out/'auditor_source.py')==reader.watched[str(Path(__file__).resolve())],'Auditor changed during initial snapshot')
        report=audit(a.output_directory,a.report_grant,a.report_grant_sha256,reader,compare);dump(out/'review.json',report)
        dump(out/'artifact_manifest.json',{'files_sha256':{x.name:digest(x) for x in out.iterdir() if x.is_file()}})
        print(json.dumps({'passed':True,'out':str(out),'review_sha256':digest(out/'review.json'),'numeric_comparisons':compare.numeric},ensure_ascii=False))
    except BaseException as error:
        failure={'passed':False,'scope':'confirmation_report_tables_independent','error':repr(error),'traceback':traceback.format_exc(),'comparisons':compare.__dict__,'captured_sha256':reader.watched,'no_automatic_retry':True,'model_solver_calls':0}
        try:out.mkdir(exist_ok=True);dump(out/'failure.json',failure)
        except BaseException as secondary:
            failure['recording_error']=repr(secondary)
            try:dump(ROOT/'provenance'/(out.name+'_failure.json'),failure)
            except BaseException as last:print(json.dumps({**failure,'fallback_error':repr(last)}),file=sys.stderr)
        raise

if __name__=='__main__':main()
