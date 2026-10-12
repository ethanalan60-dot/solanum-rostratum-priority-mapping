"""Input-only caches first; reference caches only after global prediction/choice seals."""
from pathlib import Path
import hashlib,io,zipfile
from core import HERE,attempt,binding,bound_file,completed_stage,configure_cpu,read,require,safe_id,sha,validate_freeze,write

def scientific():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    return np,torch

def resize(array):
    np,torch=scientific()
    from frozen_resample import area_resize
    return area_resize(torch.from_numpy(np.ascontiguousarray(array)).to(torch.float64)[None],(256,256))[0].numpy().astype(np.float32)

def local_input(image,valid):
    np,_=scientific()
    require(image.shape==(256,256,3) and image.dtype==np.uint8,'Local RGB schema')
    require(valid.shape==(256,256) and np.isfinite(valid).all() and ((valid>=0)&(valid<=1)).all(),'Local valid schema')
    x=image.transpose(2,0,1).astype(np.float32)/np.float32(255)
    v=valid.astype(np.float32)
    require(v.sum(dtype=np.float64)>0,'Empty local valid support')
    return x,v

def weeds_input(bands):
    np,_=scientific()
    require(len(bands)==3 and all(a.dtype==np.uint16 and a.shape==(600,600) for a in bands),'Weeds native RGB schema')
    native=np.stack([a.astype(np.float32)/np.float32(65535) for a in bands])
    x=resize(native);require(np.max(np.abs(x.mean((1,2),dtype=np.float64)-native.mean((1,2),dtype=np.float64)))<1e-7,'RGB mean not conserved')
    return x,np.ones((256,256),np.float32)

def weeds_truth(label):
    np,_=scientific()
    require(label.dtype==np.uint8 and label.shape==(600,600) and np.isin(label,range(6)).all(),'Weeds label schema')
    native=np.stack([label==0,label==1,label>1]);counts=native.sum((1,2),dtype=np.uint32)
    soft3=resize(native.astype(np.float64));truth=counts.astype(np.float64)/360000
    require(np.max(np.abs(soft3.mean((1,2),dtype=np.float64)-truth))<1e-7,'Class mean not conserved')
    require(np.max(np.abs(soft3.sum(0,dtype=np.float64)-1))<1e-6,'Class sum mismatch')
    return {'soft':soft3[2].copy(),'soft3':soft3,'native_class_counts':counts,'coverage':np.asarray(np.float32(truth[2]))}

def decode_png(archive,member):
    from PIL import Image
    np,_=scientific();raw=archive.read(member)
    with Image.open(io.BytesIO(raw)) as image:image.load();a=np.asarray(image).copy()
    return a,hashlib.sha256(raw).hexdigest()

def load_members(contract,dataset):
    """Called only after validate_freeze. Legacy local JSON contains labels: explicitly acknowledged."""
    source=contract['datasets'][dataset]
    bound_file(source['dataset_lock_path'],source['dataset_lock_sha256'])
    bound_file(source['manifest_path'],source['manifest_sha256']);rows=read(source['manifest_path'])
    require(len(rows)==source['count'] and len({x['id'] for x in rows})==source['count'],'Wrong complete member list')
    for row in rows:
        safe_id(row['id']);require(row['split']==('test_sealed' if dataset=='local' else 'test'),'Not reserved split')
    return rows

def make_geometry(rows,source):
    """Header/ID geometry only. No source mask or alpha band decoding here."""
    import rasterio
    bound_file(source['source_summary_path'],source['source_summary_sha256'])
    sources={x['source_id']:x for x in read(source['source_summary_path'])['sources']};result=[];windows=[]
    for row in rows:
        require(row['global_y']>=20480+384,'Test boundary violated')
        require(all(isinstance(row[k],int) for k in ('x','y','global_x','global_y')),'Noninteger native window')
        windows.append({**{k:row[k] for k in ('id','source_id','x','y','global_x','global_y')},'native_size':512})
    require(len({(x['source_id'],x['x'],x['y']) for x in windows})==len(windows),'Duplicate native ownership')
    for sid in sorted({x['source_id'] for x in rows}):
        row=sources[sid];bound_file(row['tif'],row['tif_sha256'])
        offsets={(x['global_x']-x['x'],x['global_y']-x['y']) for x in rows if x['source_id']==sid};require(len(offsets)==1,'Inconsistent source offsets')
        with rasterio.open(row['tif']) as ds:
            require(ds.count>=4 and ds.crs is not None,'Source alpha/CRS missing')
            for w in windows:
                if w['source_id']==sid:validate_native_window(w['x'],w['y'],ds.width,ds.height)
            result.append({'source_id':sid,'tif':row['tif'],'tif_sha256':row['tif_sha256'],'width':ds.width,'height':ds.height,'transform':list(ds.transform)[:6],'crs':ds.crs.to_string(),'global_offset':list(next(iter(offsets))),'valid_band':4})
    return result,windows

def validate_native_window(x,y,width,height):
    # Original caches preserve nominal512 support at source edges, padding outside with invalid.
    # Geometry clips native ownership at source bounds and checks that padded alpha/valid is zero.
    require(all(isinstance(v,int) for v in (x,y,width,height)),'Noninteger source/window metadata')
    require(0<=x<width and 0<=y<height,'Window origin outside source')

def prepare_inputs(dataset):
    contract,models,freeze=validate_freeze();require(dataset in ('local','weeds'),'Unknown dataset')
    with attempt('inputs_'+dataset) as out:
        configure_cpu(out);np,_=scientific();rows=load_members(contract,dataset);source=contract['datasets'][dataset]
        sources,windows=make_geometry(rows,source) if dataset=='local' else ([],[])
        lookup={x['source_id']:x for x in sources};records=[];(out/'cache').mkdir()
        archive=None
        try:
            if dataset=='weeds':bound_file(source['archive_path'],source['archive_sha256']);archive=zipfile.ZipFile(source['archive_path'])
            for row in rows:
                identifier=row['id'];record={'id':identifier,'dataset':dataset,'split':row['split'],'scene_group':row['scene_group'],'native_shape':[512,512] if dataset=='local' else [600,600],'cache_shape':[256,256],'same_full_frame_footprint':True}
                if dataset=='local':
                    bound_file(row['cache'],row['cache_sha256'])
                    with np.load(row['cache'],allow_pickle=False) as z:x,v=local_input(z['image'],z['valid'])
                    src=lookup[row['source_id']]
                    record.update({k:row[k] for k in ('source_id','x','y','global_x','global_y')})
                    record.update(native_window_xywh=[row['x'],row['y'],512,512],affine=src['transform'],CRS=src['crs'],source_cache_path=row['cache'],source_cache_sha256=row['cache_sha256'],valid_semantics='original cached native alpha area fraction per 2x2 support')
                else:
                    date=row['acquisition_date'];members=[f'weedsgalore-dataset/{date}/images/{identifier}_{band}.png' for band in ('R','G','B')]
                    listed={m['name']:m for m in row['archive_members']};require(all(m in listed for m in members),'RGB member not in sealed metadata')
                    bands=[];hashes={}
                    for m in members:
                        info=archive.getinfo(m);require(f'{info.CRC:08x}'==listed[m]['crc32'] and info.file_size==listed[m]['uncompressed_bytes'],'Archive member metadata changed')
                        a,d=decode_png(archive,m);bands.append(a);hashes[m]=d
                    x,v=weeds_input(bands)
                    record.update(acquisition_date=date,source_id=identifier,native_window_xywh=[0,0,600,600],affine=None,CRS=None,valid_semantics='all native and cached pixels valid; class0 is background',source_RGB_sha256=hashes)
                path=out/'cache'/(identifier+'.npz');np.savez_compressed(path,image=x,valid=v)
                with np.load(path,allow_pickle=False) as z:require(set(z.files)=={'image','valid'} and np.array_equal(z['image'],x) and np.array_equal(z['valid'],v),'Input cache roundtrip')
                record.update(input_cache=str(path),input_cache_sha256=sha(path));records.append(record)
        finally:
            if archive is not None:archive.close()
        write(out/'input_manifest.json',{'version':'confirmation_input_only_v1','dataset':dataset,'split':rows[0]['split'],'expected_count':len(rows),'source_manifest':{'path':source['manifest_path'],'sha256':source['manifest_sha256']},'records':records,'geometry_sources':sources,'geometry_windows':windows,'freeze_sha256':sha(HERE/'FREEZE.json'),'legacy_label_bearing_JSON_parsed':dataset=='local','label_arrays_decoded':False,'truth_fields_in_prediction_records':False})
    return out

def validate_input_manifest(dataset):
    out=HERE/'stages'/('inputs_'+dataset);completed_stage(out);manifest=read(out/'input_manifest.json')
    contract=read(HERE/'contract.json');source=contract['datasets'][dataset]
    require(manifest['freeze_sha256']==sha(HERE/'FREEZE.json'),'Input freeze mismatch')
    validate_input_rows(manifest,source,dataset)
    for row in manifest['records']:bound_file(row['input_cache'],row['input_cache_sha256'])
    return manifest

def validate_input_rows(manifest,source,dataset):
    require(manifest['dataset']==dataset and manifest['expected_count']==source['count'] and len(manifest['records'])==source['count'],'Input population mismatch')
    require(manifest['source_manifest']=={'path':source['manifest_path'],'sha256':source['manifest_sha256']},'Source manifest mismatch')
    require(len({x['id'] for x in manifest['records']})==source['count'],'Duplicate input ID')
    allowed={'id','dataset','split','scene_group','native_shape','cache_shape','same_full_frame_footprint','source_id','x','y','global_x','global_y','native_window_xywh','affine','CRS','source_cache_path','source_cache_sha256','valid_semantics','acquisition_date','source_RGB_sha256','input_cache','input_cache_sha256'}
    for row in manifest['records']:
        require(set(row)<=allowed,'Unknown/non-whitelisted prediction input field')
        safe_id(row['id'])
    return manifest

def materialize_truth(dataset,choice_review):
    # Import metadata gates lazily to avoid a module cycle. No test references before both global seals.
    from pipeline import validate_global_predictions,validate_choices
    contract,models,freeze=validate_freeze();validate_global_predictions();validate_choices(choice_review)
    manifest=validate_input_manifest(dataset)
    with attempt('truth_'+dataset) as out:
        configure_cpu(out);np,_=scientific();rows=load_members(contract,dataset);source=contract['datasets'][dataset]
        require([r['id'] for r in rows]==[r['id'] for r in manifest['records']],'Truth join order mismatch')
        (out/'cache').mkdir();records=[];archive=None
        try:
            if dataset=='weeds':bound_file(source['archive_path'],source['archive_sha256']);archive=zipfile.ZipFile(source['archive_path'])
            for row in rows:
                if dataset=='local':
                    bound_file(row['cache'],row['cache_sha256'])
                    with np.load(row['cache'],allow_pickle=False) as z:arrays={'soft':z['soft'].astype(np.float32),'valid':z['valid'].astype(np.float32),'coverage':np.asarray(np.float32(row['coverage']))}
                    require(arrays['soft'].shape==(256,256) and np.isfinite(arrays['soft']).all() and ((arrays['soft']>=0)&(arrays['soft']<=1)).all(),'Local truth schema')
                    native_label={'mask_source':row['mask_source'],'source_id':row['source_id'],'native_window_xywh':[row['x'],row['y'],512,512]}
                else:
                    member=f"weedsgalore-dataset/{row['acquisition_date']}/semantics/{row['id']}.png"
                    require(member in {x['name'] for x in row['archive_members']},'Unlisted semantic member')
                    label,digest=decode_png(archive,member);arrays=weeds_truth(label);arrays['valid']=np.ones((256,256),np.float32);native_label={'semantic_member':member,'sha256':digest}
                path=out/'cache'/(row['id']+'.npz');np.savez_compressed(path,**arrays)
                records.append({'id':row['id'],'truth_cache':str(path),'truth_cache_sha256':sha(path),'native_reference':native_label})
        finally:
            if archive is not None:archive.close()
        write(out/'truth_manifest.json',{'dataset':dataset,'records':records,'source_manifest_sha256':source['manifest_sha256'],'input_manifest':binding(HERE/'stages'/('inputs_'+dataset)/'input_manifest.json'),'all_predictions_seal':binding(HERE/'stages'/'predictions_seal'/'all_predictions.json'),'choice_review':binding(choice_review),'note':'Truth created after every declared prediction and label-free selection seal. Scalar truth is float32 source coverage; local native spatial truth remains a separate geometry join.'})
    return out
