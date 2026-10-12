"""Post-development grid-origin sensitivity using archived predictions only.

No fitting, threshold selection or source-data modification. Run --freeze first;
--run refuses if the protocol, implementation or pinned inputs have changed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, importlib.util, json, math, os, sys, time, traceback
ROOT = Path(__file__).resolve().parent
ARC = Path(os.environ.get('SRPA_RECORDS_ROOT', 'records'))
CAM = ARC/'03_computation/campaign'
ST = CAM/'confirmation_v1/stages'
ORIGINAL_CAM = Path(os.environ.get('SRPA_ORIGINAL_CAMPAIGN', str(CAM)))
sys.path.insert(0, str(ORIGINAL_CAM/'geo_dependencies'))
sys.dont_write_bytecode = True
RS = (1,2,4,6)
MS = (1,2,4)
BS = (.1,.2,.3,.5)
SHIFTS = ((0,0),(1,0),(0,1),(1,1))

def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def table(p,x):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def event(kind,**kw):
    item={'utc':utc(),'kind':kind,**kw}
    with (ROOT/'execution_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(item,ensure_ascii=False),flush=True)
def archive_path(path):
    p=Path(path)
    if p.is_relative_to(ORIGINAL_CAM): return CAM/p.relative_to(ORIGINAL_CAM)
    if p.suffix in ('.png','.tif') and p.stem.startswith('4_'): return ARC/'02_raw/local_scene'/p.name
    raise ValueError('Unmapped source path '+str(p))
def models(): return [x['run_id'] for x in read(ST/'local_selections_seal/all_local_selections.json')['models']]

def freeze():
    assert not (ROOT/'analysis_plan.json').exists(), 'Freeze already exists; do not overwrite'
    started=utc(); md=models(); assert len(md)==24 and len(set(md))==24
    geo=read(ST/'local_geometry/metadata.json')
    manifest={}
    def pin(path,expected=None):
        path=Path(path); digest=sha(path)
        assert expected is None or digest==expected,(str(path),digest,expected)
        manifest[str(path)]={'sha256':digest,'bytes':path.stat().st_size}
    pin(__file__)
    for p in [CAM/'geo_management.py',CAM/'diagnose_predicted_knapsack.py',CAM/'confirmation_v1/local_geometry.py',CAM/'confirmation_v1/local_management.py',ST/'local_geometry/metadata.json',ST/'local_outcomes/unit_truth.csv',ST/'local_selections_seal/all_local_selections.json',ST/'report_tables/local/spatial_budget_all.csv',ST/'truth_local/truth_manifest.json',CAM/'local_sampling_completeness.json']:
        pin(p)
    for path,digest in geo['source_tif_sha256'].items(): pin(archive_path(path),digest)
    truth=read(ST/'truth_local/truth_manifest.json');native_masks={x['native_reference']['mask_source'] for x in truth['records']}
    source_audit={x['mask']:x['mask_sha256'] for x in read(CAM/'local_sampling_completeness.json')['sources']}
    for p in native_masks: pin(archive_path(p),source_audit[p])
    for w in geo['windows']: pin(ST/'local_geometry'/w['file'],w['sha256'])
    for i,model in enumerate(md):
        p=ST/('predict_'+model)/'map_index.json';pin(p)
        maps=read(p);assert len(maps)==338 and {x['id'] for x in maps}=={x['id'] for x in geo['windows']}
        for x in maps: pin(archive_path(x['path']),x['sha256'])
        pin(ST/f'local_select_{i+1:02d}/policy/selection_inputs_without_annotations.csv')
        event('input_hash_progress',model=i+1,total=24)
    plan={'record_type':'Prospective freeze for a post-development sensitivity analysis; not preregistration of original study',
          'freeze_started_utc':started,'freeze_completed_utc':utc(),'models':md,'R':RS,'M_m':MS,'budgets':BS,
          'shift_fractions_of_M':[[a/2,b/2] for a,b in SHIFTS],
          'expected_rows':24*4*3*4*4,'domain_windows':338,'base_origin_utm_m':geo['origin_utm_m'],
          'domain':'Same native-alpha-valid support in all conditions; padded/background-invalid pixels excluded',
          'target':'Original coarse mask > 0 intersect alpha; annotated-patch area, not plant leaf area',
          'selection':'descending integrated predicted proportion; ASCII unit-ID tie break; strict whole-unit prefix; stop at first unit exceeding budget; no fill-in',
          'geometry':'Use archived exact 1 m integration for integer-metre origins (M2 and M4 shifts). Recompute projected native-pixel quadrilateral overlap for the three half-metre origins at M1. No display rasters or smoothing.',
          'verification':['Reproduce all 1152 original zero-origin selected-ID lists and numeric outcomes before shifted evaluation','Independently integrate all saved predictions on archived zero-origin CSR and reproduce original unit predictions','Area and target total conservation across all grid origins, tolerance absolute 1e-7 m2 and relative 1e-9','No fitting or model/setting selection'],
          'summary':'For each model/R/M/budget report min,max,range across all four origins; differences from original for all three shifts; grid-size comparison at each normalized origin; descriptive summaries only, no independent-field inference',
          'stopping_rule':'All 24 models unless missing/corrupt inputs or runtime failure; retain failure and report changed scope explicitly, never select a favorable subset',
          'input_manifest':'input_manifest.json','input_manifest_sha256':None,'script_sha256':sha(__file__),
          'no_training':True,'no_new_independent_scene':True,'historical_results_previously_seen':True}
    write(ROOT/'input_manifest.json',manifest);plan['input_manifest_sha256']=sha(ROOT/'input_manifest.json')
    write(ROOT/'analysis_plan.json',plan);write(ROOT/'freeze_receipt.json',{'utc':utc(),'analysis_plan_sha256':sha(ROOT/'analysis_plan.json'),'input_manifest_sha256':sha(ROOT/'input_manifest.json'),'script_sha256':sha(__file__)})
    event('freeze_complete',files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),analysis_plan_sha256=sha(ROOT/'analysis_plan.json'))

def dependencies():
    import numpy as np
    import scipy
    from scipy import sparse
    import shapely,rasterio,pyproj
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    spec=importlib.util.spec_from_file_location('archived_geometry_read_only',CAM/'geo_management.py')
    geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
    return np,sparse,geo,Image,{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'shapely':shapely.__version__,'rasterio':rasterio.__version__,'pyproj':pyproj.__version__}

def select(np,keys,values,m,model,rindex,budget,sx,sy):
    ids=np.array([f'M{m}_E{int(x)}_N{int(y)}' for x,y in keys])
    a,t,p=values[:,0],values[:,1],values[:,2+rindex]
    order=np.lexsort((ids,-p/a));total=math.fsum(a);limit=budget*total
    cumulative=np.r_[0.,np.cumsum(a[order],dtype=np.float64)];cumulative[-1]=total
    k=int(np.searchsorted(cumulative,limit,side='right')-1)
    while math.fsum(a[order[:k]])>limit:k-=1
    ix=order[:k];selected_area=math.fsum(a[ix]);target=math.fsum(t);captured=math.fsum(t[ix])
    return {'run_id':model,'R':RS[rindex],'M':m,'shift_x_fraction':sx/2,'shift_y_fraction':sy/2,'shift_x_m':m*sx/2,'shift_y_m':m*sy/2,'budget':budget,
        'domain_area_m2':total,'domain_target_m2':target,'units':len(ids),'selected_units':k,'actual_area_m2':selected_area,'actual_area_fraction':selected_area/total,'unused_budget_m2':limit-selected_area,
        'captured_target_m2':captured,'capture_fraction':captured/target,'capture_pct':100*captured/target,'selected_ids_json':json.dumps(ids[ix].tolist())}

def run():
    receipt=read(ROOT/'freeze_receipt.json');assert sha(ROOT/'analysis_plan.json')==receipt['analysis_plan_sha256']
    assert sha(__file__)==receipt['script_sha256'];plan=read(ROOT/'analysis_plan.json')
    assert sha(ROOT/'input_manifest.json')==plan['input_manifest_sha256']
    pins=read(ROOT/'input_manifest.json');assert not (ROOT/'results_all.csv').exists(),'Results already exist'
    np,sparse,geo,Image,versions=dependencies();started=utc();start=time.time()
    write(ROOT/'run_environment.json',{'started_utc':started,'argv':sys.argv,'versions':versions,'pid':os.getpid(),'executable':sys.executable,'geo_dependency_path':str(ORIGINAL_CAM/'geo_dependencies')})
    event('run_started',versions=versions)
    for i,(p,v) in enumerate(pins.items()):
        assert sha(p)==v['sha256'],('Frozen input changed',p)
        if i and i%2000==0:event('verify_inputs',done=i,total=len(pins))
    md=plan['models'];g=read(ST/'local_geometry/metadata.json');nmodels=len(md)
    truthrows=rows(ST/'local_outcomes/unit_truth.csv');tlookup={(int(x['M']),x['id']):float(x['target_m2']) for x in truthrows}
    old=[x for x in rows(ST/'report_tables/local/spatial_budget_all.csv') if x['policy']=='prefix' and float(x['budget']) in BS]
    oldmap={(x['run_id'],int(x['M']),int(x['R']),float(x['budget'])):x for x in old};assert len(oldmap)==1152
    originals=[];ap_by_model={};precheck=[]
    for i,model in enumerate(md):
        ap=rows(ST/f'local_select_{i+1:02d}/policy/selection_inputs_without_annotations.csv');ap_by_model[model]=ap
        for m in MS:
            chosen=sorted((x for x in ap if int(x['M'])==m),key=lambda x:x['id'])
            keys=np.array([[int(x['id'].split('_E')[1].split('_N')[0]),int(x['id'].split('_N')[1])] for x in chosen])
            values=np.array([[float(x['area']),tlookup[m,x['id']],*[float(x[f'predicted_R{r}']) for r in RS]] for x in chosen])
            for ri in range(4):
                for b in BS:
                    result=select(np,keys,values,m,model,ri,b,0,0);ref=oldmap[model,m,RS[ri],b]
                    assert json.loads(result['selected_ids_json'])==json.loads(ref['selected_ids_json']),('Zero-origin ID mismatch',model,m,RS[ri],b)
                    diffs={k:abs(result[k]-float(ref[k])) for k in ('actual_area_m2','captured_target_m2','capture_fraction')}
                    assert max(diffs.values())<1e-9,('Zero-origin numeric mismatch',model,m,RS[ri],b,diffs)
                    originals.append(result);precheck.append({'run_id':model,'M':m,'R':RS[ri],'budget':b,'ids_exact':True,**diffs})
    table(ROOT/'zero_origin_selector_verification.csv',precheck);event('zero_origin_verified',cases=len(precheck),max_abs_numeric_difference=max(max(x[k] for k in ('actual_area_m2','captured_target_m2','capture_fraction')) for x in precheck))
    # Native-pixel support and target; no resized display maps are used.
    native=read(ST/'truth_local/truth_manifest.json')['records'][0]['native_reference'];maskpath=archive_path(native['mask_source'])
    mask=Image.open(maskpath).convert('L');header=g['source_headers'][0];affine=geo.rasterio.Affine(*header['transform']);transformer=geo.Transformer.from_crs(4326,32649,always_xy=True)
    maplook=[{x['id']:x for x in read(ST/('predict_'+model)/'map_index.json')} for model in md]
    acc={shift:{'keys':[],'values':[]} for shift in SHIFTS};audit=[]
    parts=ROOT/'window_integrals';parts.mkdir(exist_ok=False)
    for wi,w in enumerate(g['windows']):
        with np.load(ST/'local_geometry'/w['file'],allow_pickle=False) as z:
            raw=z['raw_valid_by_native'];valid=(raw>0).reshape(512,512);basekeys=z['cells'];basearea=z['area']
            weight=sparse.csr_matrix((z['weights'],z['indices'],z['indptr']),shape=tuple(z['shape']))
            nativeweight=sparse.csr_matrix((z['native_weights'],z['native_indices'],z['native_indptr']),shape=tuple(z['native_shape']))
        x,y=w['x'],w['y'];target=(np.asarray(mask.crop((x,y,x+512,y+512)))>0)&valid
        probs=np.empty((65536,nmodels*4),dtype=np.float64)
        for mi,lookup in enumerate(maplook):
            item=lookup[w['id']]
            with np.load(archive_path(item['path']),allow_pickle=False) as z:
                assert z['id'].item()==w['id'] and z['R'].tolist()==list(RS)
                pr=z['probabilities'];assert pr.shape==(4,256,256) and np.isfinite(pr).all() and pr.min()>=0 and pr.max()<=1
                assert np.array_equal(z['valid'],valid.reshape(256,2,256,2).mean((1,3)))
                probs[:,mi*4:(mi+1)*4]=pr.reshape(4,-1).T
        lattice=geo.projected_lattice(transformer,affine,x,y,512,512,g['origin_utm_m'])
        rawT=float(raw@target.ravel());rawA=math.fsum(raw)
        saved={}
        for sx,sy in SHIFTS:
            if (sx,sy)==(0,0):
                keys=basekeys;area=basearea;target_integral=np.asarray(nativeweight@target.ravel());W=weight
            else:
                calc=geo.overlap_native(lattice-np.array([sx/2,sy/2]),valid,target,support=2)
                keys=calc['cells'];area=calc['area'];target_integral=calc['target'];W=calc['weight']
            values=np.column_stack((area,target_integral,W@probs))
            assert abs(math.fsum(area)-rawA)<=1e-7 and abs(math.fsum(target_integral)-rawT)<=1e-7
            assert np.all(values[:,2:]>=0) and np.all(values[:,2:]<=area[:,None]+1e-9)
            key=(sx,sy);acc[key]['keys'].append(keys);acc[key]['values'].append(values)
            saved[f'keys_{sx}{sy}']=keys;saved[f'values_{sx}{sy}']=values
            audit.append({'id':w['id'],'shift_x_m':sx/2,'shift_y_m':sy/2,'native_area_m2':rawA,'native_target_m2':rawT,'area_m2':math.fsum(area),'target_m2':math.fsum(target_integral),'area_abs_error_m2':abs(math.fsum(area)-rawA),'target_abs_error_m2':abs(math.fsum(target_integral)-rawT),'units':len(keys)})
        np.savez_compressed(parts/(w['id']+'.npz'),**saved)
        if (wi+1)%10==0 or wi+1==len(g['windows']):event('native_overlap_progress',windows=wi+1,total=len(g['windows']),elapsed_seconds=round(time.time()-start,1))
    mask.close();table(ROOT/'native_window_conservation.csv',audit)
    global_data={}
    for shift,container in acc.items():
        keys,val=geo.aggregate_nested(np.concatenate(container['keys']),np.concatenate(container['values']),1)
        global_data[shift]=(keys,val)
        np.savez_compressed(ROOT/f'integrated_base_shift_{shift[0]}{shift[1]}.npz',keys=keys,values=val,models=np.array(md),R=np.array(RS))
    del acc
    checks=[];basekeys,basevals=global_data[(0,0)];baseindex={tuple(k):i for i,k in enumerate(basekeys)}
    for mi,model in enumerate(md):
        for row in ap_by_model[model]:
            if int(row['M'])!=1:continue
            key=(int(row['id'].split('_E')[1].split('_N')[0]),int(row['id'].split('_N')[1]));idx=baseindex[key]
            ds=[abs(basevals[idx,2+mi*4+ri]-float(row[f'predicted_R{r}'])) for ri,r in enumerate(RS)]
            da=abs(basevals[idx,0]-float(row['area']));dt=abs(basevals[idx,1]-tlookup[1,row['id']])
            assert max(ds+[da,dt])<1e-8,('Independent zero-origin unit integration mismatch',model,row['id'],ds,da,dt)
            checks.append({'run_id':model,'unit_id':row['id'],'max_abs_prediction_error_m2':max(ds),'area_abs_error_m2':da,'target_abs_error_m2':dt})
    table(ROOT/'independent_zero_origin_integration_verification.csv',checks)
    event('independent_zero_origin_integration_verified',unit_model_comparisons=len(checks),max_prediction_abs_error_m2=max(x['max_abs_prediction_error_m2'] for x in checks))
    out=[];conservation=[];unitdir=ROOT/'unit_tables';unitdir.mkdir(exist_ok=False)
    for m in MS:
        for sx,sy in SHIFTS:
            if m==1: keys,vals=global_data[(sx,sy)]
            else: keys,vals=geo.aggregate_nested(basekeys-np.array([m*sx//2,m*sy//2]),basevals,m)
            totalA=math.fsum(vals[:,0]);totalT=math.fsum(vals[:,1]);ref=originals[0]
            assert math.isclose(totalA,ref['domain_area_m2'],rel_tol=1e-9,abs_tol=1e-7) and math.isclose(totalT,ref['domain_target_m2'],rel_tol=1e-9,abs_tol=1e-7)
            conservation.append({'M':m,'shift_x_m':m*sx/2,'shift_y_m':m*sy/2,'area_m2':totalA,'target_m2':totalT,'area_abs_error_m2':abs(totalA-ref['domain_area_m2']),'target_abs_error_m2':abs(totalT-ref['domain_target_m2']),'units':len(keys)})
            np.savez_compressed(unitdir/f'M{m}_S{sx}{sy}.npz',keys=keys,values=vals,models=np.array(md),R=np.array(RS))
            for mi,model in enumerate(md):
                v=np.column_stack((vals[:,:2],vals[:,2+mi*4:2+(mi+1)*4]))
                for ri in range(4):
                    for b in BS:
                        result=select(np,keys,v,m,model,ri,b,sx,sy)
                        if (sx,sy)==(0,0):
                            refold=oldmap[model,m,RS[ri],b]
                            assert json.loads(result['selected_ids_json'])==json.loads(refold['selected_ids_json'])
                            assert abs(result['capture_fraction']-float(refold['capture_fraction']))<1e-9
                        out.append(result)
    assert len(out)==plan['expected_rows'];table(ROOT/'results_all.csv',out);table(ROOT/'domain_conservation.csv',conservation)
    write(ROOT/'execution_complete.json',{'started_utc':started,'completed_utc':utc(),'elapsed_seconds':time.time()-start,'rows':len(out),'models':24,'no_training':True,'scope':'post-development within-scene sensitivity analysis','zero_origin_cases_exact_selected_IDs':1152,'all_inputs_frozen':True,'max_area_conservation_error_m2':max(x['area_abs_error_m2'] for x in conservation),'max_target_conservation_error_m2':max(x['target_abs_error_m2'] for x in conservation)})
    event('completed',rows=len(out),elapsed_seconds=time.time()-start)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    try:
        assert args.freeze != args.run
        freeze() if args.freeze else run()
    except BaseException as e:
        event('failure',error=repr(e),traceback=traceback.format_exc());raise
