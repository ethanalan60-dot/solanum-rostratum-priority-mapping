"""Independent artifact/array audit. Does not import or execute the evaluator."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import collections,csv,datetime,hashlib,json,math,shutil,traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
SPATIAL=ROOT/'external_weeds_v1/spatial_v1'
RESULT=SPATIAL/'results_20260916_065643_813785'
PINS={'source':'af13294cdff8e5da059cb4f8f3bec4cb4b0c2f7c24c05226278724a8a79e1810',
      'protocol':'a96c28da40f5b2b336651af4826ff8b1476ac56233d08ef2c19e00596d9410e8',
      'acceptance':'649ea0dd7e385967f9e09653d2acad2c4b5413be24483db69d0537818c591bc0',
      'selection':'1bd058e82b8758d2f2e98e99d242ae598564c29a2a0e970b20cacc36f8c72cb0'}
GRIDS=(4,8,16);RS=(1,2,4);BUDGETS=(('b10',1,10),('b20',1,5),('b30',3,10),('b50',1,2),('b100',1,1))
WATCH={};COUNTS={};ERRORS=collections.defaultdict(float);NULLS=collections.Counter();CSV_KEYS={}

def rawsha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def sha(path):
    path=Path(path).resolve();d=rawsha(path);WATCH[str(path)]=d;return d
def read(path):sha(path);return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def iter_json(path):
    sha(path)
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:yield json.loads(line)
def iter_csv(name):
    path=RESULT/name;sha(path)
    with path.open(encoding='utf-8-sig',newline='') as stream:yield from csv.DictReader(stream)
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check(actual,expected,key='numeric',atol=2e-11):
    if expected is None:
        assert actual is None or actual=='',(key,actual,'expected NA');NULLS[key]+=1;return
    if isinstance(expected,(bool,str)):
        assert str(actual)==str(expected),(key,actual,expected);return
    assert actual is not None and actual!='',(key,actual,expected)
    a=float(actual);b=float(expected);assert math.isfinite(a) and math.isfinite(b),(key,a,b)
    error=abs(a-b);ERRORS[key]=max(ERRORS[key],error)
    assert error<=atol+1e-12*abs(b),(key,a,b,error)
def check_fields(row,expected,tag):
    for k,v in expected.items():assert k in row,(tag,'missing',k);check(row[k],v,tag+'/'+k)
def ids_for(frame,g):return [f'{frame}:g{g:02d}:r{r:02d}:c{c:02d}' for r in range(g) for c in range(g)]
def aggregate(array,g):
    # Independent vectorized reduction; original evaluator uses sorted cell dictionaries and math.fsum.
    a=np.asarray(array,dtype=np.float64);assert a.shape==(256,256) and np.isfinite(a).all()
    side=256//g
    return a.reshape(g,side,g,side).sum(axis=(1,3),dtype=np.float64).reshape(-1)/(256*256)
def budgets(g):
    result=[];seen={};N=g*g
    for basis in ('nominal_primary','matched_actual_secondary'):
        for code,num,den in BUDGETS:
            k=N*num//den if basis=='nominal_primary' else (16*num//den)*(N//16)
            effective=num/den if basis=='nominal_primary' else (16*num//den)/16
            result.append({'budget_basis':basis,'parent_nominal_budget_code':code,
                'parent_nominal_budget_fraction':num/den,'effective_budget_fraction':effective,
                'k':k,'N':N,'actual_fraction':k/N,'unused_effective_fraction':effective-k/N,
                'prior_alias':seen.get(k)})
            seen.setdefault(k,(basis,code))
    return result
def ratio(n,d):return None if d==0 else n/d

def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('external_spatial_independent_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    try:
        # Authenticate ALL selections and original bindings before any annotation arrays.
        assert sha(RESULT/'source/evaluator.py')==sha(ROOT/'evaluate_external_weeds_spatial.py')==PINS['source']
        assert sha(RESULT/'source/protocol.json')==sha(SPATIAL/'protocol.json')==PINS['protocol']
        assert sha(RESULT/'source/acceptance.json')==PINS['acceptance']
        assert sha(RESULT/'selection_seal.json')==PINS['selection']
        protocol=read(RESULT/'source/protocol.json');accepted=read(RESULT/'source/acceptance.json')
        auth=read(RESULT/'source/authorization.json');seal=read(RESULT/'selection_seal.json')
        status=read(RESULT/'status.json');manifest=read(RESULT/'artifact_manifest.json');binding=read(RESULT/'input_binding.json')
        summary=read(RESULT/'summary.json')
        assert status['state']=='completed' and status['source_sha256']==PINS['source'] and status['selection_seal_sha256']==PINS['selection']
        assert status['protocol_sha256']==PINS['protocol'] and status['acceptance_sha256']==PINS['acceptance']
        assert auth['passed'] and auth['production_execution_authorized'] and not auth['test_or_R6_authorized']
        assert auth['source_sha256']==PINS['source'] and auth['protocol_sha256']==PINS['protocol'] and auth['acceptance_sha256']==PINS['acceptance']
        assert sha(auth['readiness'])==auth['readiness_sha256'] and read(auth['readiness'])['passed']
        assert accepted['passed'] and accepted['valid_all_ones_attested']
        assert seal['phase']=='all_predictions_sealed_before_labels' and seal['label_arrays_opened'] is False
        assert binding['CPU_only'] and binding['label_arrays_opened'] is False
        assert seal['rankings']==1404 and seal['budget_rows']==14040 and seal['unique_choices']==9360
        for rel,digest in seal['file_sha256'].items():
            p=(RESULT/rel).resolve();assert p.is_relative_to(RESULT.resolve()) and sha(p)==digest,rel
        assert seal['source_input_sha256']==binding['source_input_sha256']
        for rel,digest in seal['source_input_sha256'].items():
            p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT.resolve()) and sha(p)==digest,rel
        allfiles={p.relative_to(RESULT).as_posix() for p in RESULT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        assert allfiles==set(manifest['file_sha256'])
        for rel,digest in manifest['file_sha256'].items():assert sha(RESULT/rel)==digest,rel
        assert not (RESULT/'failure.json').exists()
        phaseb=read(RESULT/'phase_b_started.json');assert phaseb['selection_seal_sha256']==PINS['selection']
        assert datetime.datetime.fromisoformat(seal['created_utc'])<datetime.datetime.fromisoformat(phaseb['created_utc'])
        frames=protocol['dataset']['frame_metadata'];frameids=[x['id'] for x in frames]
        assert frameids==accepted['val_ids'] and len(frameids)==len(set(frameids))==26
        assert all(x['split']=='val' for x in frames)
        assert sha(ROOT/'weedsgalore_data_v1/val.json')==accepted['val_manifest_sha256']
        framedict={x['id']:x for x in frames};dates=sorted({x['acquisition_date'] for x in frames})
        groups={'frame':{i:[i] for i in frameids},'date':{d:[x['id'] for x in frames if x['acquisition_date']==d] for d in dates},'pooled':{'all26':frameids}}
        models={x['run_id']:x for x in accepted['segmentation_models']};assert len(models)==6
        for rid,m in models.items():
            assert m['completed24'] and sha(m['config_path'])==m['config_sha256'] and sha(m['checkpoint_path'])==m['checkpoint_sha256']
            bm=binding['models'][rid]
            assert bm['checkpoint_sha256']==m['checkpoint_sha256'] and bm['original_selected_epoch']==m['best_epoch']
        export=Path(accepted['export_root']);assert sha(export/'artifact_manifest.json')==accepted['artifact_manifest_sha256']
        assert sha(RESULT/'source/map_index.json')==sha(export/'map_index.json')==accepted['map_index_sha256']
        maps=read(RESULT/'source/map_index.json');assert len(maps)==156
        write(out/'phase_A_integrity_verified_before_truth.json',{'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'selection_seal_sha256':PINS['selection'],'source_sha256':PINS['source'],'all_expected_rankings':1404,
            'all_expected_budget_choices':14040,'label_arrays_opened_by_auditor':False})
        print('Source, manifest and complete pre-label selection seal verified.',flush=True)
        # Only now access the 26 allowed validation caches; no source RGB or test decoding.
        truth={};class_checks=[];nested=[]
        for frame in frames:
            p=Path(frame['cache']);assert sha(p)==frame['cache_sha256']
            with np.load(p,allow_pickle=False) as z:
                soft3=z['soft3'];valid=z['valid'];soft=z['soft'];counts=z['native_class_counts']
            assert soft3.shape==(3,256,256) and soft3.dtype==np.float32 and np.isfinite(soft3).all() and np.all((soft3>=0)&(soft3<=1))
            assert valid.shape==(256,256) and valid.dtype==np.float32 and np.all(valid==1)
            assert np.array_equal(soft,soft3[2]) and counts.shape==(3,) and counts.dtype==np.uint32 and counts.sum()==360000
            errs=np.abs(soft3.astype(np.float64).mean(axis=(1,2))-counts.astype(np.float64)/360000)
            assert np.max(errs)<=1e-7 and np.max(np.abs(soft3.astype(np.float64).sum(axis=0)-1))<=1e-6
            class_checks.append({'id':frame['id'],'class_fraction_max_error':float(errs.max())})
            for g in GRIDS:truth[frame['id'],g]=np.stack([aggregate(soft3[c],g) for c in range(3)],axis=1)
            for g in (4,8):
                coarse=truth[frame['id'],g].reshape(g,g,3)
                child=truth[frame['id'],g*2].reshape(g,2,g,2,3).sum(axis=(1,3))
                residual=np.abs(coarse-child);assert residual.max()<=1e-12
                nested.extend(residual.ravel().tolist())
        assert len(nested)==6240
        truthcheck=read(RESULT/'truth_validation_after_seal.json')
        assert truthcheck['selection_seal_sha256']==PINS['selection'] and truthcheck['nested_cell_conservation']['scalar_comparisons']==6240
        check(truthcheck['nested_cell_conservation']['max_absolute_residual_FAE'],max(nested),'nested_cell_residual',1e-12)
        keys=set()
        for row in iter_csv('truth_units.csv'):
            fid=row['frame_id'];g=int(row['grid_g']);ident=row['id'];idx=ids_for(fid,g).index(ident)
            assert (fid,g,ident) not in keys;keys.add((fid,g,ident))
            check_fields(row,{'A':1/g**2,'B':truth[fid,g][idx,0],'C':truth[fid,g][idx,1],'W':truth[fid,g][idx,2]},'truth')
        assert len(keys)==8736;COUNTS['truth_units']=len(keys)
        # Recompute prediction cells from NPZ arrays, and validate complete immutable rankings.
        pred={};rankings={};order_arrays={}
        for item in maps:
            path=export/item['path'];assert sha(path)==item['sha256']
            with np.load(path,allow_pickle=False) as z:
                assert set(z.files)=={'probabilities','R','id'}
                assert str(z['id'])==item['id'] and np.array_equal(z['R'],RS)
                arr=z['probabilities'];assert arr.shape==(3,256,256) and arr.dtype==np.float32 and np.isfinite(arr).all()
                assert np.all((arr>=0)&(arr<=1))
            for ix,r in enumerate(RS):
                for g in GRIDS:pred[item['run_id'],r,item['id'],g]=aggregate(arr[ix],g)
        for row in iter_json(RESULT/'learned_rankings.jsonl'):
            key=(row['run_id'],row['R'],row['frame_id'],row['grid_g']);assert key not in rankings
            rid,r,fid,g=key;cellids=ids_for(fid,g);p=pred[key]
            expected=sorted(range(g*g),key=lambda ix:(-p[ix],cellids[ix]))
            assert row['ordered_ids']==[cellids[i] for i in expected],('ranking mismatch',key)
            assert row['ranking_key']==f'{rid}|R{r}|{fid}|g{g:02d}'
            assert row['acquisition_date']==framedict[fid]['acquisition_date']
            for k,v in binding['models'][rid].items():check(row[k],v,'rankingmeta/'+k)
            assert [u['id'] for u in row['units']]==cellids
            for i,u in enumerate(row['units']):
                assert set(u)=={'id','A','P'};check(u['A'],1/g**2,'rank/A');check(u['P'],p[i],'rank/P',1e-13)
            rankings[key]=row;order_arrays[key]=np.asarray(expected,dtype=np.int64)
        expected_ranks={(rid,r,fid,g) for rid in models for r in RS for fid in frameids for g in GRIDS}
        assert set(rankings)==expected_ranks;COUNTS['learned_rankings']=len(rankings)
        pkeys=set()
        for row in iter_csv('prediction_units.csv'):
            key=(row['run_id'],int(row['R']),row['frame_id'],int(row['grid_g']));rid,r,fid,g=key
            idx=ids_for(fid,g).index(row['id']);k=key+(idx,);assert k not in pkeys;pkeys.add(k)
            check(row['P'],pred[key][idx],'prediction_units/P',1e-13);check(row['A'],1/g**2,'prediction_units/A')
        assert len(pkeys)==157248;COUNTS['prediction_units']=len(pkeys)
        choicekeys=set();uniques=set();aliascount=0
        for row in iter_json(RESULT/'learned_fixed_budget_choices.jsonl'):
            key=(row['run_id'],row['R'],row['frame_id'],row['grid_g']);rid,r,fid,g=key
            bd=next(x for x in budgets(g) if x['budget_basis']==row['budget_basis'] and x['parent_nominal_budget_code']==row['parent_nominal_budget_code'])
            for k,v in bd.items():
                if k!='prior_alias':check(row[k],v,'choice/'+k)
            base=f'{rid}|R{r}|{fid}|g{g:02d}';ck=base+'|'+bd['budget_basis']+'|'+bd['parent_nominal_budget_code']
            assert row['choice_key']==ck and ck not in choicekeys;choicekeys.add(ck)
            assert row['selection_key']==base+f"|k{bd['k']:03d}";uniques.add(row['selection_key'])
            alias=None if bd['prior_alias'] is None else base+'|'+'|'.join(bd['prior_alias'])
            assert row['duplicate_selection_alias']==alias;aliascount+=alias is not None
            selected=order_arrays[key][:bd['k']]
            assert row['selected_ids']==[ids_for(fid,g)[i] for i in selected]
            check(row['selected_area_FAE'],bd['k']/g**2,'choice/area')
            check(row['selected_predicted_weed_FAE'],float(pred[key][selected].sum(dtype=np.float64)),'choice/P')
            assert row['selected_area_FAE']<=row['effective_budget_fraction']
        assert len(choicekeys)==14040 and len(uniques)==9360
        COUNTS.update(learned_fixed_budget_choices=len(choicekeys),unique_choices=len(uniques),duplicate_alias_rows=aliascount)
        print('All prediction cells, 1404 orders, 14040 choices, and 8736 truth cells recomputed.',flush=True)
        # Independent cumulative arrays [background,crop,weed,predicted] for every frame/choice.
        curves={};orders={}
        for key,order in order_arrays.items():
            rid,r,fid,g=key;arr=np.column_stack((truth[fid,g],pred[key]))[order]
            curves[rid,r,g,'frame',fid]=np.vstack((np.zeros((1,4)),np.cumsum(arr,axis=0,dtype=np.float64)))
            orders[rid,r,g,fid]=order
        for family in ('label_oracle','random_expectation'):
            for fid in frameids:
                for g in GRIDS:
                    t=truth[fid,g];total=t.sum(axis=0);cellids=ids_for(fid,g)
                    order=np.asarray(sorted(range(g*g),key=lambda i:(-t[i,2],cellids[i])),dtype=np.int64)
                    orders[family,None,g,fid]=order
                    c=np.zeros((g*g+1,4),dtype=np.float64)
                    c[:,0:3]=(np.arange(g*g+1)/(g*g))[:,None]*total if family=='random_expectation' else np.vstack((np.zeros((1,3)),np.cumsum(t[order],axis=0)))
                    curves[family,None,g,'frame',fid]=c
        refseen=set()
        for row in iter_json(RESULT/'label_oracle_rankings_after_seal.jsonl'):
            fid=row['frame_id'];g=row['grid_g'];key=(fid,g);assert key not in refseen;refseen.add(key)
            assert row['ordered_ids']==[ids_for(fid,g)[i] for i in orders['label_oracle',None,g,fid]]
            assert row['run_id']=='label_oracle' and row['R'] is None and row['seed'] is None
            assert row['units']==[{'id':i,'A':1/g**2,'P':0.} for i in ids_for(fid,g)]
        assert len(refseen)==78;COUNTS['shared_oracle_rankings']=78
        allruns=[(rid,r) for rid in models for r in RS]+[(p,None) for p in ('label_oracle','random_expectation')]
        for rid,r in allruns:
            for g in GRIDS:
                for scope in ('date','pooled'):
                    for scopeid,fs in groups[scope].items():curves[rid,r,g,scope,scopeid]=np.stack([curves[rid,r,g,'frame',fid] for fid in fs]).sum(axis=0,dtype=np.float64)
        metric_cache={};auc_cache={};fixed_cache={}
        def expected_metrics(key,k):
            rid,r,g,scope,scopeid=key;fs=groups[scope][scopeid];n=len(fs);N=g*g;c=curves[key]
            b,crop,w,p=c[k];tw=c[-1,2];tc=c[-1,1];area=n*k/N
            wp=sum(truth[fid,g][:,2].sum()>0 for fid in fs);cp=sum(truth[fid,g][:,1].sum()>0 for fid in fs)
            return {'k':k,'N':N,'actual_fraction':k/N,'selected_area_FAE':area,
                'selected_predicted_weed_FAE':p if rid in models else None,
                'selected_weed_FAE':w,'total_weed_FAE':tw,'unselected_weed_FAE':tw-w,
                'selected_crop_FAE':crop,'total_crop_FAE':tc,'selected_background_FAE':b,
                'weed_capture_ratio':ratio(w,tw),'weed_capture_na_reason':'' if tw>0 else 'zero_total_weed',
                'crop_capture_ratio':ratio(crop,tc),'crop_capture_na_reason':'' if tc>0 else 'zero_total_crop',
                'crop_fraction_of_selected_area':ratio(crop,area),'crop_composition_na_reason':'' if area>0 else 'zero_selected_area',
                'frame_n':n,'weed_eligible_frame_n':int(wp),'crop_eligible_frame_n':int(cp),'zero_weed_frame_n':int(n-wp),'zero_crop_frame_n':int(n-cp)}
        def metakey(row):return row['run_id'],(int(row['R']) if row['R']!='' else None),int(row['grid_g']),row['scope'],row['scope_id']
        def checkmeta(row,key,family):
            rid,r,g,scope,scopeid=key;assert key in curves
            assert row['policy']==family and row['frame_id']==(scopeid if scope=='frame' else '')
            assert row['acquisition_date']==(framedict[scopeid]['acquisition_date'] if scope=='frame' else scopeid if scope=='date' else '')
            assert row['val_manifest_sha256']==accepted['val_manifest_sha256']
            if rid in models:
                for k,v in binding['models'][rid].items():check(row[k],v,'metricmeta/'+k)
            else:
                assert row['variant']=='reference' and all(row[k]=='' for k in ('seed','original_selected_epoch','checkpoint_sha256','config_sha256','export_sha256','R'))
        for family in ('learned','label_oracle','random_expectation'):
            for scope in ('frame','date','pooled'):
                expectedkeys={key for key in curves if key[3]==scope and ((key[0] in models) if family=='learned' else key[0]==family)}
                seen=set();name=f'{family}_{scope}_complete_prefix_curves.csv'
                for row in iter_csv(name):
                    key=metakey(row);k=int(row['k']);checkmeta(row,key,family);assert key in expectedkeys
                    assert (key,k) not in seen;seen.add((key,k));expected=expected_metrics(key,k)
                    check_fields(row,expected,'curves')
                assert seen=={(key,k) for key in expectedkeys for k in range(key[2]**2+1)}
                COUNTS[name]=len(seen)
                seen=set();name=f'{family}_{scope}_budget_metrics.csv'
                for row in iter_csv(name):
                    key=metakey(row);checkmeta(row,key,family);assert key in expectedkeys
                    basis,code=row['budget_basis'],row['parent_nominal_budget_code'];sk=(key,basis,code);assert sk not in seen;seen.add(sk)
                    bd=next(x for x in budgets(key[2]) if x['budget_basis']==basis and x['parent_nominal_budget_code']==code)
                    expected=expected_metrics(key,bd['k']);check_fields(row,expected,'fixed')
                    check_fields(row,{k:v for k,v in bd.items() if k!='prior_alias'},'fixed_budget')
                    rid,r,g,sc,sid=key;base=f'{rid}|R{r}|'+(sid if sc=='frame' else sc+'|'+sid)+f'|g{g:02d}'
                    selection=base+f"|k{bd['k']:03d}";choice=base+'|'+basis+'|'+code
                    alias='' if bd['prior_alias'] is None else base+'|'+'|'.join(bd['prior_alias'])
                    assert row['selection_key']==selection and row['choice_key']==choice and row['duplicate_selection_alias']==alias
                    check(row['unused_effective_budget_area_FAE'],bd['unused_effective_fraction']*expected['frame_n'],'fixed_unused')
                    fixed_cache[sk]={**expected,**bd,'selection_key':selection,'duplicate_selection_alias':alias}
                assert seen=={(key,b['budget_basis'],b['parent_nominal_budget_code']) for key in expectedkeys for b in budgets(key[2])};COUNTS[name]=len(seen)
                seen=set();name=f'{family}_{scope}_auc.csv'
                for row in iter_csv(name):
                    key=metakey(row);checkmeta(row,key,family);assert key in expectedkeys and key not in seen;seen.add(key)
                    N=key[2]**2;tw=curves[key][-1,2];c=curves[key][:,2]
                    # Uniform complete-cell steps allow simple means, independently of evaluator interval clipping.
                    full=None if tw==0 else float(c[:-1].mean()/tw)
                    half=None if tw==0 else float(c[:N//2].mean()/tw)
                    v={'prefix_points':N+1,'frame_n':len(groups[scope][key[4]]),'weed_eligible_frame_n':expected_metrics(key,0)['weed_eligible_frame_n'],
                       'total_weed_FAE':tw,'auc_0_1':full,'auc_0_0p5_normalized':half,'auc_na_reason':'' if tw>0 else 'zero_total_weed'}
                    check_fields(row,v,'auc');auc_cache[key]=v
                    if family=='random_expectation' and tw>0:
                        check(full,(N-1)/(2*N),'random_full_auc',1e-12);check(half,.25-1/(2*N),'random_half_auc',1e-12)
                assert seen==expectedkeys;COUNTS[name]=len(seen)
            print(f'Checked all {family} frame/date/pooled curves, fixed budgets and AUCs.',flush=True)
        # Pair every seed/R/g/scope/basis/code; no significance or seed-as-field inference.
        primary=[];pairsout=[];signs=collections.Counter()
        def modelid(variant,seed):return next(r for r,m in models.items() if m['variant']==variant and m['seed']==seed)
        metrics3=('weed_capture_ratio','crop_capture_ratio','crop_fraction_of_selected_area')
        for scope in ('frame','date','pooled'):
            seen=set();name=f'hard_soft_paired_{scope}_budget_metrics.csv'
            for row in iter_csv(name):
                seed=int(row['seed']);r=int(row['R']);g=int(row['grid_g']);sid=row['scope_id'];basis=row['budget_basis'];code=row['parent_nominal_budget_code']
                key=(seed,r,g,scope,sid,basis,code);assert key not in seen;seen.add(key)
                hard=modelid('hard_seg',seed);soft=modelid('soft_seg',seed)
                h=fixed_cache[(hard,r,g,scope,sid),basis,code];s=fixed_cache[(soft,r,g,scope,sid),basis,code]
                assert row['hard_run_id']==hard and row['soft_run_id']==soft and row['hard_selection_key']==h['selection_key'] and row['soft_selection_key']==s['selection_key']
                assert row['duplicate_budget_selection']==str(bool(h['duplicate_selection_alias']))
                check_fields(row,{k:h[k] for k in ('effective_budget_fraction','k','N','frame_n','selected_area_FAE')},'pairmeta')
                for metric in metrics3:
                    delta=None if h[metric] is None or s[metric] is None else 100*(s[metric]-h[metric])
                    check(row['hard_'+metric],h[metric],'pairhard');check(row['soft_'+metric],s[metric],'pairsoft');check(row['soft_minus_hard_'+metric+'_pp'],delta,'pairdelta')
                    if scope=='pooled':signs[(basis,metric,'NA' if delta is None else 'positive' if delta>1e-8 else 'negative' if delta<-1e-8 else 'tie')]+=1
                if scope=='pooled':pairsout.append(row)
                if scope=='pooled' and g==8 and basis=='nominal_primary':
                    primary.append({'seed':seed,'R':r,'nominal_budget':code,'actual_fraction':h['k']/h['N'],
                        'hard_weed_capture_pct':100*h['weed_capture_ratio'],'soft_weed_capture_pct':100*s['weed_capture_ratio'],
                        'soft_minus_hard_weed_pp':100*(s['weed_capture_ratio']-h['weed_capture_ratio']),
                        'hard_crop_capture_pct':100*h['crop_capture_ratio'],'soft_crop_capture_pct':100*s['crop_capture_ratio'],
                        'hard_crop_fraction_selected_pct':100*h['crop_fraction_of_selected_area'],'soft_crop_fraction_selected_pct':100*s['crop_fraction_of_selected_area']})
            assert len(seen)==3*3*3*len(groups[scope])*10;COUNTS[name]=len(seen)
            seen=set();name=f'hard_soft_paired_{scope}_auc.csv'
            for row in iter_csv(name):
                seed=int(row['seed']);r=int(row['R']);g=int(row['grid_g']);sid=row['scope_id'];key=seed,r,g,scope,sid
                assert key not in seen;seen.add(key);hkey=(modelid('hard_seg',seed),r,g,scope,sid);skey=(modelid('soft_seg',seed),r,g,scope,sid)
                assert row['hard_run_id']==hkey[0] and row['soft_run_id']==skey[0] and row['budget_basis']=='complete_prefix_once'
                for metric in ('auc_0_1','auc_0_0p5_normalized'):
                    h=auc_cache[hkey][metric];s=auc_cache[skey][metric];d=None if h is None or s is None else s-h
                    check_fields(row,{'hard_'+metric:h,'soft_'+metric:s,'soft_minus_hard_'+metric:d},'paired_auc')
            assert len(seen)==3*3*3*len(groups[scope]);COUNTS[name]=len(seen)
        nestedseen=set()
        for row in iter_csv('nested_oracle_matched_actual_checks.csv'):
            scope=row['scope'];sid=row['frame_id'] if scope=='frame' else 'all26';code=row['parent_nominal_budget_code']
            key=scope,sid,code;assert key not in nestedseen;nestedseen.add(key)
            _,num,den=next(x for x in BUDGETS if x[0]==code);fraction=(16*num//den)/16;values=[]
            check(row['matched_fraction'],fraction,'oracle_nested_fraction')
            for g in GRIDS:
                k=int(fraction*g*g);v=curves['label_oracle',None,g,scope,sid][k,2];values.append(v);check(row[f'g{g}_W'],v,'oracle_nested_W')
            assert values[1]>=values[0]-1e-12 and values[2]>=values[1]-1e-12
        assert len(nestedseen)==135;COUNTS['nested_oracle_checks']=135;COUNTS['nested_scalar_conservation_checks']=len(nested)
        randseen=set()
        for row in iter_csv('random_grid_budget_expectations.csv'):
            g=int(row['grid_g']);basis=row['budget_basis'];code=row['parent_nominal_budget_code'];key=g,basis,code;assert key not in randseen;randseen.add(key)
            bd=next(x for x in budgets(g) if x['budget_basis']==basis and x['parent_nominal_budget_code']==code)
            check_fields(row,{k:v for k,v in bd.items() if k!='prior_alias'},'random_budget')
            check_fields(row,{'expected_capture_if_denominator_positive':bd['k']/g**2,'auc_0_1_if_denominator_positive':(g*g-1)/(2*g*g),
                'auc_0_0p5_normalized_if_denominator_positive':.25-1/(2*g*g),'Monte_Carlo_draws':0},'random_expectation')
        assert len(randseen)==30;COUNTS['random_grid_budget_rows']=30
        assert len(primary)==45 and len(pairsout)==270
        assert ratio(0,0) is None and ratio(0,1)==0 and ratio(1,2)==.5
        # Verify no byte changed while reading; outputs remain untouched.
        for p,digest in WATCH.items():assert rawsha(p)==digest,('Changed input',p)
        primary.sort(key=lambda x:(x['seed'],x['R'],next(i for i,b in enumerate(BUDGETS) if b[0]==x['nominal_budget'])))
        csvsave(out/'all45_primary_g8_nominal_pairs.csv',primary);csvsave(out/'all270_pooled_pairs_verified.csv',pairsout)
        signrows=[{'budget_basis':b,'metric':m,'direction':d,'count':n} for (b,m,d),n in sorted(signs.items())]
        csvsave(out/'all_pooled_pair_signs.csv',signrows)
        aucrows=[]
        for key,v in sorted(auc_cache.items(),key=lambda kv:str(kv[0])):
            if key[3]=='pooled':aucrows.append({'run_id':key[0],'R':key[1],'grid_g':key[2],**v})
        csvsave(out/'all60_pooled_AUC_rows.csv',aucrows)
        result={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output_directory':str(RESULT),
            'protocol_sha256':PINS['protocol'],'evaluator_sha256':PINS['source'],'acceptance_sha256':PINS['acceptance'],
            'authorization_sha256':sha(RESULT/'source/authorization.json'),'selection_seal_sha256':PINS['selection'],
            'output_artifact_manifest_sha256':sha(RESULT/'artifact_manifest.json'),'source_code_sha256':rawsha(__file__),
            'counts_verified':COUNTS,'numeric_comparison_max_errors':dict(ERRORS),'NA_comparisons_verified':dict(NULLS),
            'prediction_ranking_discrepancies':0,'source_binding_discrepancies':0,'metric_discrepancies':0,'alias_budget_discrepancies':0,
            'all_seed_primary_pairs':primary,'pooled_pair_signs':signrows,'val_ids':frameids,'models':list(models.values()),
            'unique_choices':9360,'budget_basis_rows':14040,'primary_nominal_grid':8,
            'cache_class_conservation_max_error':max(x['class_fraction_max_error'] for x in class_checks),
            'nested_cell_max_error':max(nested),'verified_probability_unit_count':157248,'verified_truth_unit_count':8736,
            'independent_method':'NumPy float64 reshape sums, vector cumulative sums, ratio-of-sums and means of all complete-step ordinates; no evaluator imports/calls.',
            'zero_denominator_scope':'All actual denominators and reason fields checked; every k0 crop composition is NA. Pure arithmetic zero-denominator fixtures checked; no artificial field observations added.',
            'test_accessed':False,'R6_accessed':False,'inference_or_training':False,'optimization':False,'GPU_used':False,
            'input_sha256':WATCH}
        write(out/'audit.json',result)
        lines=['# WeedsGalore spatial results: independent full-output audit','',
            '**PASS. Complete frozen outputs were independently recomputed; no source results were changed.**','',
            '- Source/authorization/acceptance/all-file manifest and the complete six-model selection seal were verified before reading allowed validation label arrays.',
            '- 157,248 prediction units, 1,404 exact rankings, 14,040 budget rows (9,360 unique selections), 8,736 truth units, 6,240 parent–child scalar checks and 135 matched-area oracle checks passed.',
            '- Every learned/reference frame/date/pooled complete curve, fixed budget, both staircase AUCs and every hard–soft pair was recomputed. Both budget bases and aliases are retained.',
            '- Units are frame-area equivalents; repeated or overlapping field support is not a geographic union. Crop inclusion is annotation composition/exposure, not measured crop damage. All findings remain same-dataset development evidence.',
            '', '## Primary g8 nominal-budget results: all 45 paired conditions','',
            'Weed/crop capture are percentages of total corresponding annotation. Crop composition is percentage of selected image area. 10% nominal budget uses 6/64=9.375% actual image area.',
            '', '| Seed | R | Nominal | Actual % | Hard weed % | Soft weed % | Soft−hard pp | Hard crop capture % | Soft crop capture % | Hard crop composition % | Soft crop composition % |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for x in primary:lines.append(f"|{x['seed']}|{x['R']}|{x['nominal_budget']}|{100*x['actual_fraction']:.4f}|{x['hard_weed_capture_pct']:.4f}|{x['soft_weed_capture_pct']:.4f}|{x['soft_minus_hard_weed_pp']:+.4f}|{x['hard_crop_capture_pct']:.4f}|{x['soft_crop_capture_pct']:.4f}|{x['hard_crop_fraction_selected_pct']:.4f}|{x['soft_crop_fraction_selected_pct']:.4f}|")
        lines+=['','## Full condition direction counts','', '| Budget basis | Metric | Direction | Count |','|---|---|---|---:|']
        for x in signrows:lines.append(f"|{x['budget_basis']}|{x['metric']}|{x['direction']}|{x['count']}|")
        lines+=['','Direction counts are correlated condition descriptions, not independent replicate counts or significance tests. The exact complete values are retained in the accompanying CSVs. No optimum operating resolution, agronomic efficacy, herbicide saving, crop injury or field-level generalization is inferred.']
        (out/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(out/'artifact_manifest.json',{'passed':True,'files_sha256':{p.name:rawsha(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(json.dumps({'passed':True,'directory':str(out),'audit_sha256':rawsha(out/'audit.json'),'counts':COUNTS,'pair_signs':signrows},ensure_ascii=False))
    except BaseException as error:
        write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'completed_counts':COUNTS,
                                 'source_code_sha256':rawsha(__file__),'inputs_sha256':WATCH});raise

if __name__=='__main__':main()
