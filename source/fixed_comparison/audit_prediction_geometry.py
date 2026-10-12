"""Independent saved-prediction and geometry arithmetic; no model or annotation access."""
from pathlib import Path
import argparse,csv,json,math
import core
HERE=Path(__file__).resolve().parent

def compare_map(probability,valid,scalars,expected_R,identifier,stored_id,stored_R,tolerance=1e-6):
    import numpy as np
    core.require(probability.dtype==np.float32 and probability.shape==(4,256,256),'Invalid map array')
    core.require(valid.dtype==np.float32 and valid.shape==(256,256),'Invalid validity array')
    core.require(np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'Invalid map probabilities')
    core.require(np.isfinite(valid).all() and ((valid>=0)&(valid<=1)).all() and valid.sum(dtype=np.float64)>0,'Invalid observed support')
    core.require(stored_id.shape==() and str(stored_id.item())==identifier,'Wrong map member')
    core.require(stored_R.dtype==np.int64 and np.array_equal(stored_R,expected_R),'Wrong R array')
    values=(probability.astype(np.float64)*valid[None].astype(np.float64)).sum(axis=(1,2))/valid.sum(dtype=np.float64)
    residual=np.abs(values-np.asarray(scalars,dtype=np.float64))
    core.require(float(residual.max())<=tolerance,'Map scalar integration mismatch')
    return float(residual.max())

def geometry_arrays(z):
    import numpy as np
    from scipy.sparse import csr_matrix
    allowed={'cells','area','weights','indices','indptr','shape','raw_valid_by_support','native_weights','native_indices','native_indptr','native_shape','raw_valid_by_native'}
    core.require(set(z)==allowed,'Geometry keys changed')
    cells=z['cells'];area=z['area'];n=len(area)
    core.require(cells.shape==(n,2) and np.issubdtype(cells.dtype,np.integer) and len(set(map(tuple,cells)))==n,'Cell identities changed')
    core.require(area.ndim==1 and area.dtype==np.float64 and np.isfinite(area).all() and (area>0).all(),'Invalid unit areas')
    for prefix,columns in (('',65536),('native_',262144)):
        shape=z[prefix+'shape'];indices=z[prefix+'indices'];indptr=z[prefix+'indptr'];weights=z[prefix+'weights']
        core.require(shape.shape==(2,) and np.issubdtype(shape.dtype,np.integer) and np.array_equal(shape,[n,columns]),'Wrong CSR shapes')
        core.require(indices.ndim==weights.ndim==1 and np.issubdtype(indices.dtype,np.integer) and weights.dtype==np.float64 and len(indices)==len(weights),'CSR coefficient/index schema')
        core.require(indptr.shape==(n+1,) and np.issubdtype(indptr.dtype,np.integer),'CSR pointer schema')
    native=csr_matrix((z['native_weights'],z['native_indices'],z['native_indptr']),shape=(n,262144))
    support=csr_matrix((z['weights'],z['indices'],z['indptr']),shape=(n,65536))
    for matrix in (native,support):
        matrix.check_format(full_check=True)
        core.require(matrix.has_canonical_format and np.isfinite(matrix.data).all() and (matrix.data>0).all(),'Invalid sparse coefficients')
    raw=z['raw_valid_by_native'];coarse=z['raw_valid_by_support']
    core.require(raw.shape==(262144,) and coarse.shape==(65536,) and raw.dtype==coarse.dtype==np.float64,'Area vector schema')
    core.require(np.isfinite(raw).all() and (raw>=0).all() and np.isfinite(coarse).all() and (coarse>=0).all(),'Invalid native/support area vector')
    rowerror=max(float(np.abs(np.asarray(x.sum(axis=1)).ravel()-area).max()) for x in (native,support))
    colerror=float(np.abs(np.asarray(native.sum(axis=0)).ravel()-raw).max())
    coarse_expected=raw.reshape(256,2,256,2).sum(axis=(1,3)).ravel()
    coarseerror=float(np.abs(coarse_expected-coarse).max())
    coo=native.tocoo();column=(coo.col//1024)*256+(coo.col%512)//2
    independent=np.zeros((n,65536),dtype=np.float64)
    np.add.at(independent,(coo.row,column),coo.data)
    coefficienterror=float(np.abs(independent-support.toarray()).max())
    core.require(max(rowerror,colerror,coarseerror,coefficienterror)<max(1e-12,float(area.sum())*1e-10),'CSR conservation/coarsening mismatch')
    valid=(raw.reshape(256,2,256,2)>0).sum(axis=(1,3)).astype(np.float32)/4
    return {'row_error':rowerror,'native_column_error':colerror,'support_area_error':coarseerror,'coefficient_error':coefficienterror,
      'native_valid_pixels':int((raw>0).sum()),'area_m2':float(raw.sum(dtype=np.float64))},valid

def execution_bindings(mode,review_path):
    # Capture before validating grants to reject changes across that interval.
    watched={str(Path(__file__).resolve()):core.sha(__file__),str(Path(review_path).resolve()):core.sha(review_path),
      str(HERE/'FREEZE.json'):core.sha(HERE/'FREEZE.json'),str(HERE/'SOURCE_LOCK.json'):core.sha(HERE/'SOURCE_LOCK.json')}
    contract,models,_=core.validate_freeze();grant=core.read(review_path)
    core.require(grant.get('passed') is True and grant['scope']=='confirmation_prediction_geometry_audit' and grant['mode']==mode,'Audit execution grant missing')
    core.require(grant['implementation_sha256']==core.sha(__file__) and grant['freeze_sha256']==core.sha(HERE/'FREEZE.json'),'Audit source/freeze changed')
    core.require(grant['annotation_access_authorized'] is False and grant['model_forward_authorized'] is False,'Wrong audit phase')
    target=HERE/'stages'/('predictions_seal' if mode=='predictions' else 'local_geometry')
    core.completed_stage(target);manifest=target/'artifact_manifest.json'
    core.require(core.sha(manifest)==grant['target_artifact_manifest_sha256'],'Wrong target stage')
    watched[str(manifest)]=grant['target_artifact_manifest_sha256']
    for ds in (('local','weeds') if mode=='predictions' else ('local',)):
        for filename in ('input_manifest.json','artifact_manifest.json'):
            path=HERE/'stages'/('inputs_'+ds)/filename;watched[str(path)]=core.sha(path)
    for path,digest in watched.items():core.bound_file(path,digest)
    return contract,models,watched

def validate_model_provenance(model,metadata,start_seal,ids,input_binding,input_artifact_binding,freeze_sha):
    core.require(all(metadata[k]==v for k,v in model.items()),'Prediction frozen model identity')
    core.require(metadata['member_ids']==ids and metadata['R']==[1,2,4,6] and metadata['input_manifest']==input_binding,'Prediction input membership/binding changed')
    core.require(metadata['freeze_sha256']==freeze_sha and metadata['truth_read_by_prediction'] is False and metadata['checkpoint_reselected'] is False,'Prediction phase changed')
    core.require(metadata['CPU_dtype']=='float32' and metadata['batch_size']==4 and metadata['CPU_threads']==2,'Prediction resource/precision changed')
    core.require(metadata['scalar_rows']==len(ids)*4 and metadata['map_count']==(0 if model['variant']=='regression' else len(ids)),'Prediction metadata counts')
    core.require(start_seal['input_manifest']==input_binding and start_seal['input_artifact_manifest']==input_artifact_binding,'Prediction start input binding changed')
    core.require(start_seal['checkpoint']['sha256']==model['checkpoint_sha256'] and start_seal['config']['sha256']==model['config_sha256'],'Prediction start model binding changed')
    core.require(start_seal['original_best_epoch']==model['best_epoch'] and start_seal['truth_read_by_prediction'] is False and start_seal['checkpoint_reselected'] is False,'Prediction original epoch/phase changed')

def validate_map_provenance(row,model,record,folder):
    core.require(Path(row['path']).resolve()==(Path(folder)/'maps'/(row['id']+'.npz')).resolve(),'Map path identity changed')
    core.require(row['run_id']==model['run_id'] and row['dataset']==model['dataset'] and row['id']==record['id'],'Map model/member identity changed')
    core.require(row['R']==[1,2,4,6] and row['shape']==[4,256,256] and row['dtype']=='float32','Map metadata schema')
    for key in ('input_cache_sha256','source_id','native_window_xywh','valid_semantics'):core.require(row[key]==record[key],'Map source provenance changed: '+key)

def audit_predictions(contract,models,out):
    import numpy as np
    from pipeline import validate_global_predictions
    from data_io import validate_input_manifest
    global_seal=validate_global_predictions();inputs={ds:validate_input_manifest(ds) for ds in ('local','weeds')}
    validity={};inputcount=0
    for ds,manifest in inputs.items():
        validity[ds]={}
        for record in manifest['records']:
            with np.load(record['input_cache'],allow_pickle=False) as z:
                core.require(set(z.files)=={'image','valid'},'Input contains label keys')
                x,v=z['image'],z['valid']
            core.require(x.dtype==np.float32 and x.shape==(3,256,256) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all(),'Image schema')
            core.require(v.dtype==np.float32 and v.shape==(256,256) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all() and v.sum()>0,'Input validity')
            validity[ds][record['id']]=v;inputcount+=1
    results=[];totalmaps=totalscalars=0
    for model,entry in zip(models,global_seal['models']):
        ds=model['dataset'];folder=Path(entry['stage']);metadata=core.read(folder/'metadata.json')
        ids=[r['id'] for r in inputs[ds]['records']];byid={r['id']:r for r in inputs[ds]['records']}
        input_binding=core.binding(HERE/'stages'/('inputs_'+ds)/'input_manifest.json')
        start_seal=core.read(folder/'prediction_input_seal.json')
        validate_model_provenance(model,metadata,start_seal,ids,input_binding,core.binding(HERE/'stages'/('inputs_'+ds)/'artifact_manifest.json'),core.sha(HERE/'FREEZE.json'))
        with (folder/'predicted_scalars.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        core.require([(r['id'],int(r['R'])) for r in rows]==[(identifier,R) for R in (1,2,4,6) for identifier in ids],'Scalar ordered grid')
        lookup={(r['id'],int(r['R'])):float(r['prediction_cpu']) for r in rows}
        for row in rows:
            core.require(all(str(row[k])==str(model[k]) for k in ('run_id','dataset','encoder','variant','seed','best_epoch')),'Scalar model identity')
            value=float(row['prediction_cpu']);core.require(math.isfinite(value) and 0<=value<=1,'Scalar value')
        maps=core.read(folder/'map_index.json');expected=ids if model['variant']!='regression' else []
        core.require([r['id'] for r in maps]==expected,'Map population')
        largest=0
        for row in maps:
            record=byid[row['id']]
            validate_map_provenance(row,model,record,folder)
            core.bound_file(row['path'],row['sha256'])
            with np.load(row['path'],allow_pickle=False) as z:
                core.require(set(z.files)=={'probabilities','valid','R','id'},'Prediction array keys')
                largest=max(largest,compare_map(z['probabilities'],z['valid'],[lookup[row['id'],R] for R in (1,2,4,6)],(1,2,4,6),row['id'],z['id'],z['R'],1e-7 if ds=='weeds' else 1e-6))
                core.require(np.array_equal(z['valid'],validity[ds][row['id']]),'Saved map validity differs from original prepared input')
        results.append({'run_id':model['run_id'],'checkpoint_sha256':model['checkpoint_sha256'],'scalar_rows':len(rows),'maps':len(maps),'maximum_map_scalar_difference':largest})
        totalscalars+=len(rows);totalmaps+=len(maps)
        core.write(out/'progress.json',{'models_completed':len(results),'expected':37})
    core.require(len(results)==37 and inputcount==364 and totalscalars==38792 and totalmaps==8268,'Incomplete global prediction review')
    validate_global_predictions()
    for ds in ('local','weeds'):validate_input_manifest(ds)
    return {'passed':True,'scope':'confirmation_predictions_independent','all_predictions_sha256':core.sha(HERE/'stages/predictions_seal/all_predictions.json'),
      'models':results,'input_windows':inputcount,'scalar_rows':totalscalars,'map_count':totalmaps,'annotation_arrays_accessed':0,'model_forward_calls':0}

def audit_geometry(contract,models,out):
    import numpy as np
    from data_io import validate_input_manifest
    folder=HERE/'stages/local_geometry';metadata=core.read(folder/'metadata.json');inputs=validate_input_manifest('local')
    ids=[r['id'] for r in inputs['records']];byid={r['id']:r for r in inputs['records']}
    core.require(metadata['state']=='completed' and metadata['geometry_kind']=='native_alpha_only' and metadata['complete_declared_domain'] is True,'Wrong geometry scope')
    core.require(metadata['expected_windows']==338 and [r['id'] for r in metadata['windows']]==ids,'Geometry population/order')
    core.require(metadata['annotation_arrays_opened']==0 and metadata['prediction_arrays_opened']==0 and metadata['mask_paths_consumed'] is False,'Non-geometric inputs')
    cells={};results=[]
    for item in metadata['windows']:
        path=core.resolve_under(folder,item['file']);core.bound_file(path,item['sha256'])
        with np.load(path,allow_pickle=False) as z:
            result,valid=geometry_arrays({k:z[k] for k in z.files})
            for key,A in zip(z['cells'],z['area']):cells.setdefault(tuple(int(x) for x in key),[]).append(float(A))
        record=byid[item['id']]
        with np.load(record['input_cache'],allow_pickle=False) as z:core.require(np.array_equal(z['valid'],valid),'Native positive-area support vs prepared alpha')
        core.require(result['native_valid_pixels']==item['native_valid_pixels'],'Native valid pixel count')
        core.require(math.isclose(result['area_m2'],item['raw_area_m2'],rel_tol=1e-10,abs_tol=1e-9),'Native area mismatch')
        results.append(dict(id=item['id'],**result));core.write(out/'progress.json',{'windows_completed':len(results),'expected':338})
    with (folder/'area_units.csv').open(encoding='utf-8',newline='') as f:units=list(csv.DictReader(f))
    expected={}
    for M in (1,2,4):
        groups={}
        for (ix,iy),values in cells.items():groups.setdefault((ix//M,iy//M),[]).extend(values)
        for (ix,iy),values in groups.items():expected[M,f'M{M}_E{ix}_N{iy}']=(ix,iy,math.fsum(values))
    core.require(len(units)==len(expected) and len({(int(r['M']),r['id']) for r in units})==len(units),'Area-unit population')
    maximum=0
    for row in units:
        key=int(row['M']),row['id'];ix,iy,A=expected[key];error=abs(float(row['area_m2'])-A)
        core.require(int(row['ix'])==ix and int(row['iy'])==iy and math.isclose(float(row['area_m2']),A,rel_tol=1e-10,abs_tol=1e-9),'Nested area mismatch');maximum=max(maximum,error)
    total=math.fsum(r['area_m2'] for r in results)
    core.require(math.isclose(total,metadata['area_m2'],rel_tol=1e-10,abs_tol=1e-9) and sum(r['native_valid_pixels'] for r in results)==metadata['native_valid_pixels'],'Domain total mismatch')
    core.completed_stage(folder);validate_input_manifest('local')
    return {'passed':True,'scope':'confirmation_geometry_independent','metadata_sha256':core.sha(folder/'metadata.json'),'windows':results,'area_m2':total,
      'native_valid_pixels':sum(r['native_valid_pixels'] for r in results),'management_units':len(units),'maximum_nested_area_difference_m2':maximum,
      'annotation_arrays_accessed':0,'model_forward_calls':0,'scope_limit':'Independent sparse conservation, native-to-support coefficients and nested area aggregation; not independent land survey or reprojection.'}

def execute(mode,review_path):
    contract,models,watched=execution_bindings(mode,review_path)
    with core.attempt('audit_'+mode) as out:
        core.configure_cpu(out);core.write(out/'begin_binding.json',watched)
        review=(audit_predictions if mode=='predictions' else audit_geometry)(contract,models,out)
        for path,digest in watched.items():core.bound_file(path,digest)
        core.validate_source_lock();review.update(freeze_sha256=core.sha(HERE/'FREEZE.json'),implementation_sha256=core.sha(__file__),root_review=core.binding(review_path))
        core.write(out/'review.json',review)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True,choices=['predictions','geometry']);p.add_argument('--root-review',required=True);a=p.parse_args();print(execute(a.mode,a.root_review))
if __name__=='__main__':main()
