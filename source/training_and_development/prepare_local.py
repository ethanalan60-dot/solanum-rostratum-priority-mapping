"""Prepare audited, spatially separated local development tiles; never edit sources."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, os, sys, time
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

ROOT=Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def spatial_split(y0,y1):
    if y1<=15360-384:return 'train'
    if y0>=15360+384 and y1<=20480-384:return 'val'
    if y0>=20480+384:return 'test_sealed'
    return 'buffer_excluded'

def main():
    t0=time.time()
    out=ROOT/'local_data_v2'
    if (out/'DATASET_LOCK.json').exists():raise FileExistsError('Dataset version is sealed; use a new version for revisions.')
    cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
    audit_path=ROOT.parent/'solanum_audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    original_rows=[r for r in audit['tiles'] if r['height_label']=='12m']
    unique={}
    for r in original_rows:
        key=(r['source_id'],r['grid_x_1based'],r['grid_y_1based'])
        if key not in unique:unique[key]=dict(r,legacy_paths=[],legacy_categories=[])
        unique[key]['legacy_paths'].append(r['path'])
        unique[key]['legacy_categories'].append(r['category_directory'])
    rows=list(unique.values())
    source_dir=Path(audit['root'])/'大图'/'12m'/'4刺萼龙葵'
    records=[];excluded=[];sources=[]
    for source in ['4_5','4_6','4_7','4_8']:
        tif=source_dir/f'{source}.tif';png=tif.with_suffix('.png')
        sources.append({'source_id':source,'tif':str(tif),'tif_sha256':digest(tif),'mask':str(png),'mask_sha256':digest(png)})
        with Image.open(png) as im:
            im.load();mask=np.asarray(im).copy()
        yoff=14677 if source in ['4_7','4_8'] else 0
        xoff=22320 if source in ['4_6','4_8'] else 0
        with rasterio.open(tif) as ds:
            for r in [x for x in rows if x['source_id']==source]:
                x=(r['grid_x_1based']-1)*512;y=(r['grid_y_1based']-1)*512
                split=spatial_split(y+yoff,y+yoff+512)
                rec={'id':Path(r['path']).stem,'source_id':source,'scene_group':'solanum_adjacent_grid_A',
                     'rgb_source':r['path'],'mask_source':str(png),'x':x,'y':y,
                     'global_x':x+xoff,'global_y':y+yoff,'split':split,'legacy_category':r['category_directory'],
                     'legacy_paths':r['legacy_paths'],'legacy_categories':r['legacy_categories'],
                     'legacy_label_conflict':len(set(r['legacy_categories']))>1}
                if split=='buffer_excluded':excluded.append(rec);continue
                rgba=ds.read(window=Window(x,y,512,512),boundless=True,fill_value=0)
                valid=rgba[3]>0
                target=np.zeros((512,512),dtype=bool)
                hh=min(512,mask.shape[0]-y);ww=min(512,mask.shape[1]-x)
                if hh>0 and ww>0:target[:hh,:ww]=mask[y:y+hh,x:x+ww]>0
                rec['valid_fraction']=float(valid.mean())
                rec['mask_pixels_outside_valid']=int((target&~valid).sum())
                if valid.mean()<.25:
                    rec['split']='low_valid_excluded';excluded.append(rec);continue
                target &= valid
                rec['coverage']=float(target.sum()/valid.sum())
                # Preserve coverage by area pooling, including fractional valid pixels.
                v=valid.reshape(256,2,256,2).mean(axis=(1,3)).astype(np.float32)
                a=target.reshape(256,2,256,2).mean(axis=(1,3)).astype(np.float32)
                soft=np.divide(a,v,out=np.zeros_like(a),where=v>0)
                # RGB and validity use the same area kernel, so NoData is not mixed into valid color.
                rgb=np.moveaxis(rgba[:3],0,-1).astype(np.float32)*valid[:,:,None]
                rgb_mean=rgb.reshape(256,2,256,2,3).mean(axis=(1,3))
                image=np.divide(rgb_mean,v[:,:,None],out=np.zeros_like(rgb_mean),where=v[:,:,None]>0)
                image=np.clip(np.rint(image),0,255).astype(np.uint8)
                hard=np.asarray(Image.fromarray(target.astype(np.uint8)*255).resize((256,256),Image.Resampling.NEAREST))>0
                cached=cache/f"{rec['id']}.npz"
                np.savez_compressed(cached,image=image,soft=soft.astype(np.float16),valid=v.astype(np.float16),hard=hard)
                rec['cache']=str(cached)
                reconstructed=float((soft*v).sum()/v.sum())
                assert abs(reconstructed-rec['coverage'])<1e-6
                records.append(rec)
        del mask
        print(json.dumps({'source':source,'kept':len(records),'excluded':len(excluded),'elapsed_s':round(time.time()-t0)},ensure_ascii=False),flush=True)
    for split in ['train','val','test_sealed']:
        (out/f'{split}.json').write_text(json.dumps([r for r in records if r['split']==split],ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'excluded.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8')
    assert len({(r['source_id'],r['x'],r['y']) for r in records})==len(records)
    summary={'protocol':'local_v2','source_audit_sha256':digest(audit_path),'script_sha256':digest(__file__),
             'legacy_file_records':len(original_rows),'unique_source_locations_before_exclusion':len(rows),
             'duplicate_file_records_removed':len(original_rows)-len(rows),
             'legacy_label_conflict_locations':sum(len(set(r['legacy_categories']))>1 for r in rows),
             'sources':sources,'counts':dict(Counter(r['split'] for r in records)),
             'positive_counts':dict(Counter(r['split'] for r in records if r['coverage']>0)),
             'exclusions':dict(Counter(r['split'] for r in excluded)),
             'valid_fraction_min':.25,'spatial_boundaries':[15360,20480],'half_buffer_pixels':384,
             'mask_positive_outside_alpha_pixels':sum(r.get('mask_pixels_outside_valid',0) for r in records+excluded),
             'runtime_seconds':time.time()-t0,
             'notes':['All scale versions of one tile stay in its spatial split.',
                      'Adjacent quadrants are a single candidate scene, not four independent acquisitions.',
                      'test_sealed is not read by train_local.py; only development validation is used for selection.',
                      'Existing RGB/mask/alpha are used without new manual or AI-generated ground truth.']}
    (out/'manifest_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
