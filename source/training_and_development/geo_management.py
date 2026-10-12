"""Validation-domain native pixel integration onto nested metric management grids.

Projected native pixel edges are straight segments. The same quadrilaterals
define source area and overlap area; no post-hoc area renormalization is used.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time, traceback
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'geo_dependencies'))
import numpy as np
import scipy
from scipy import sparse
import shapely
from pyproj import Transformer, __version__ as pyproj_version
import rasterio
from rasterio.windows import Window
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, obj):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def quad_areas(quads):
    q = np.asarray(quads, dtype=np.float64)
    r = q - q[:, :1, :]
    return np.abs(r[:,1,0]*r[:,2,1] - r[:,1,1]*r[:,2,0]
                  + r[:,2,0]*r[:,3,1] - r[:,2,1]*r[:,3,0]) * .5

def projected_lattice(transformer, affine, x0, y0, h, w, origin):
    xx, yy = np.meshgrid(np.arange(w+1, dtype=np.float64)+x0,
                         np.arange(h+1, dtype=np.float64)+y0)
    lon = affine.a*xx + affine.b*yy + affine.c
    lat = affine.d*xx + affine.e*yy + affine.f
    east, north = transformer.transform(lon, lat)
    return np.stack([east-origin[0], north-origin[1]], axis=-1)

def overlap_native(lattice, valid, target, support=2):
    """Return sparse area weights from model cells to 1 m ground cells.

    Row keys are (east_index,north_index). Only alpha-positive native pixels
    enter, retaining their positions even within partially valid model cells.
    """
    h,w = valid.shape
    assert h % support == 0 and w % support == 0
    assert lattice.shape == (h+1,w+1,2) and target.shape == valid.shape
    native_index = np.flatnonzero(valid.ravel())
    quads = np.stack([lattice[:-1,:-1],lattice[:-1,1:],
                      lattice[1:,1:],lattice[1:,:-1]],axis=2).reshape(-1,4,2)[native_index]
    native_area = quad_areas(quads)
    assert np.isfinite(quads).all() and np.all(native_area > 0)
    low = np.floor(quads.min(axis=1)).astype(np.int64)
    high = np.floor(np.nextafter(quads.max(axis=1), -np.inf)).astype(np.int64)
    assert np.all(high-low <= 1), 'Native pixel too large for the configured 1 m base grid'
    inside = np.all(low == high, axis=1)
    indices = [np.flatnonzero(inside)]
    keys = [low[inside]]
    areas = [native_area[inside]]
    crossing = np.flatnonzero(~inside)
    polygons = shapely.polygons(quads[crossing])
    for dx in (0,1):
        for dy in (0,1):
            candidate = low[crossing] + np.array([dx,dy])
            keep = np.all(candidate <= high[crossing],axis=1)
            if not keep.any(): continue
            k = candidate[keep]
            box = shapely.box(k[:,0],k[:,1],k[:,0]+1,k[:,1]+1)
            a = shapely.area(shapely.intersection(polygons[keep],box,grid_size=0))
            if not np.isfinite(a).all(): raise ValueError('Nonfinite intersection area')
            nonzero = a > 0
            indices.append(crossing[keep][nonzero]); keys.append(k[nonzero]); areas.append(a[nonzero])
    ix = np.concatenate(indices); key = np.concatenate(keys); a = np.concatenate(areas)
    cells, row = np.unique(key,axis=0,return_inverse=True)
    pix = native_index[ix]
    col = (pix//w//support)*(w//support)+(pix%w//support)
    weight = sparse.coo_matrix((a,(row,col)),shape=(len(cells),(h//support)*(w//support))).tocsr()
    A = np.bincount(row,weights=a,minlength=len(cells))
    T = np.bincount(row,weights=a*target.ravel()[pix],minlength=len(cells))
    raw_A = float(native_area.sum(dtype=np.float64))
    raw_T = float(np.dot(native_area,target.ravel()[native_index]))
    per_pixel = np.bincount(ix,weights=a,minlength=len(native_index))
    error = float(np.max(np.abs(per_pixel-native_area))) if len(native_area) else 0.
    relA = abs(float(A.sum())-raw_A)/max(raw_A,1e-30)
    relT = abs(float(T.sum())-raw_T)/max(raw_T,1e-30) if raw_T else 0.
    assert relA < 1e-9 and relT < 1e-9, (relA,relT)
    assert np.all(T >= 0) and np.all(T <= A+1e-12)
    raw_valid_by_support = np.bincount(
        (native_index//w//support)*(w//support)+(native_index%w//support),
        weights=native_area,minlength=(h//support)*(w//support))
    column_error = float(np.max(np.abs(np.asarray(weight.sum(axis=0)).ravel()-raw_valid_by_support)))
    assert column_error < 1e-9 * max(raw_A,1e-30)
    return dict(cells=cells,weight=weight,area=A,target=T,
                raw_valid_by_support=raw_valid_by_support,
                audit={'native_valid_pixels':int(valid.sum()),'native_target_pixels':int((valid&target).sum()),
                       'raw_area_m2':raw_A,'raw_target_m2':raw_T,
                       'area_relative_error':relA,'target_relative_error':relT,
                       'max_pixel_conservation_abs_m2':error,
                       'max_support_area_abs_error_m2':column_error,
                       'crossing_native_pixels':int(len(crossing)),
                       'base_cells':int(len(cells)),'sparse_entries':int(weight.nnz)})

def aggregate_nested(cells, values, size):
    assert size in (1,2,4)
    parents,inv = np.unique(cells//size,axis=0,return_inverse=True)
    arr = np.asarray(values)
    if arr.ndim == 1: arr=arr[:,None]
    out = np.stack([np.bincount(inv,weights=arr[:,j],minlength=len(parents)) for j in range(arr.shape[1])],axis=1)
    assert np.allclose(out.sum(axis=0),arr.sum(axis=0),rtol=1e-10,atol=1e-10)
    return parents,out

def midpoint_error(transformer, affine, x0,y0,origin,lattice):
    # Fixed regular sample on each tile, no dependence on annotation or predictions.
    sample=np.arange(0,512,32,dtype=float)
    xx,yy=np.meshgrid(sample,sample)
    maximum=0.
    for dx,dy in ((.5,0.),(0.,.5)):
        x=xx+x0+dx; y=yy+y0+dy
        east,north=transformer.transform(affine.a*x+affine.b*y+affine.c,affine.d*x+affine.e*y+affine.f)
        exact=np.stack([east-origin[0],north-origin[1]],axis=-1)
        i=yy.astype(int);j=xx.astype(int)
        end=lattice[i+(dy>0),j+(dx>0)]
        chord=(lattice[i,j]+end)*.5
        maximum=max(maximum,float(np.max(np.linalg.norm(exact-chord,axis=-1))))
    return maximum

def build(output, limit=None):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Choose a new geometry output directory')
    output.mkdir(parents=True)
    (output/'tiles').mkdir()
    started=time.time()
    protocol=ROOT/'precision_agriculture/RM_development_protocol_v1.json'
    manifest=ROOT/'local_data_v2/val.json'
    assert sha(manifest)==read(protocol)['manifest_sha256']
    rows=sorted(read(manifest),key=lambda r:r['id'])
    assert len(rows)==357 and all(r['split']=='val' for r in rows)
    assert len({(r['source_id'],r['x'],r['y']) for r in rows})==len(rows)
    audit=read(ROOT/'local_sampling_completeness.json')
    all_sources={s['source_id']:s for s in audit['sources']}
    all_rows=list(rows)
    # Padded parts outside a source image have alpha=0 and own no ground.
    rectangles=[(r['global_x'],r['global_y'],
                 r['global_x']+min(512,all_sources[r['source_id']]['width']-r['x']),
                 r['global_y']+min(512,all_sources[r['source_id']]['height']-r['y'])) for r in all_rows]
    for i,(x0,y0,x1,y1) in enumerate(rectangles):
        for xx0,yy0,xx1,yy1 in rectangles[:i]:
            assert min(x1,xx1)<=max(x0,xx0) or min(y1,yy1)<=max(y0,yy0), 'Overlapping native ownership'
    if limit: rows=rows[:limit]
    sources={s['source_id']:s for s in audit['sources'] if s['source_id'] in {r['source_id'] for r in rows}}
    transformer=Transformer.from_crs('EPSG:4326','EPSG:32649',always_xy=True)
    ds={sid:rasterio.open(s['tif']) for sid,s in sources.items()}
    reference=all_sources[all_rows[0]['source_id']]
    reference_transform=rasterio.Affine(*reference['transform'][:6])
    alignment_error=0.
    for sid in {r['source_id'] for r in all_rows}:
        source=all_sources[sid]
        offset=np.array(source['global_offset'])-np.array(reference['global_offset'])
        expected=reference_transform*rasterio.Affine.translation(*offset)
        actual=rasterio.Affine(*source['transform'][:6])
        x=np.array([0.,float(source['width'])]);y=np.array([0.,float(source['height'])])
        ex,ey=transformer.transform(expected.a*x+expected.b*y+expected.c,expected.d*x+expected.e*y+expected.f)
        ax,ay=transformer.transform(actual.a*x+actual.b*y+actual.c,actual.d*x+actual.e*y+actual.f)
        alignment_error=max(alignment_error,float(np.max(np.hypot(ex-ax,ey-ay))))
    assert alignment_error < 1e-7, 'Source offsets do not form one consistent native grid'
    all_bounds=[]
    for r in read(manifest):
        # Determine a fixed origin from all validation geometry, even in smoke mode.
        s=next(x for x in audit['sources'] if x['source_id']==r['source_id'])
        tr=rasterio.Affine(*s['transform'][:6])
        xx=np.array([r['x'],r['x']+512,r['x']+512,r['x']],dtype=float)
        yy=np.array([r['y'],r['y'],r['y']+512,r['y']+512],dtype=float)
        east,north=transformer.transform(tr.a*xx+tr.b*yy+tr.c,tr.d*xx+tr.e*yy+tr.f)
        all_bounds.append(np.stack([east,north],axis=-1))
    origin=(np.floor(np.concatenate(all_bounds).min(axis=0)/4)*4).tolist()
    info={'state':'running','created_utc':datetime.now(timezone.utc).isoformat(),
          'pid':__import__('os').getpid(),'protocol_sha256':sha(protocol),'manifest_sha256':sha(manifest),
          'script_sha256':sha(__file__),'origin_utm_m':origin,'crs':'EPSG:32649','base_grid_m':1,
          'subset_diagnostic':bool(limit),'expected_windows':len(rows),'windows':[],
          'test_accessed':False,'source_grid_alignment_error_m':alignment_error,
          'native_ownership':'source-clipped rectangles have disjoint global native pixel coordinates; padded alpha=0 parts excluded',
          'versions':{'numpy':np.__version__,'scipy':scipy.__version__,
                                          'shapely':shapely.__version__,'pyproj':pyproj_version}}
    (output/'geo_management_source.py').write_bytes(Path(__file__).read_bytes())
    write(output/'metadata.json',info)
    write(output/'status.json',{'state':'running','done':0,'total':len(rows),'pid':info['pid']})
    try:
        truth={} # one source PNG decoded at a time; no resampling of annotations
        source_hashes={}
        for sid,s in sources.items():
            source_hashes[sid]={'tif_sha256':sha(s['tif']),'mask_sha256':sha(s['mask']),
                                'transform':list(ds[sid].transform),'crs':str(ds[sid].crs)}
            assert source_hashes[sid]['mask_sha256']==s['mask_sha256']
        info['sources']=source_hashes
        previous=None
        for index,r in enumerate(rows):
            sid=r['source_id']; dataset=ds[sid]
            if previous!=sid:
                truth.clear()
                with Image.open(sources[sid]['mask']) as im: truth[sid]=np.asarray(im).copy()
                previous=sid
            valid=dataset.read(4,window=Window(r['x'],r['y'],512,512),boundless=True,fill_value=0)>0
            target=np.zeros((512,512),dtype=bool)
            mask=truth[sid]
            hh=min(512,mask.shape[0]-r['y']);ww=min(512,mask.shape[1]-r['x'])
            if hh>0 and ww>0: target[:hh,:ww]=mask[r['y']:r['y']+hh,r['x']:r['x']+ww]>0
            target &= valid
            assert abs(target.sum()/valid.sum()-r['coverage'])<1e-12
            lattice=projected_lattice(transformer,dataset.transform,r['x'],r['y'],512,512,origin)
            deviation=midpoint_error(transformer,dataset.transform,r['x'],r['y'],origin,lattice)
            assert deviation<1e-7, deviation
            result=overlap_native(lattice,valid,target)
            result['audit']['midpoint_chord_error_m']=deviation
            result['audit']['id']=r['id']
            weight=result['weight']
            path=output/'tiles'/f"{r['id']}.npz"
            np.savez_compressed(path,cells=result['cells'],area=result['area'],target=result['target'],
                                weights=weight.data,indices=weight.indices,indptr=weight.indptr,
                                shape=np.array(weight.shape),raw_valid_by_support=result['raw_valid_by_support'])
            info['windows'].append(dict(result['audit'],file=str(path),sha256=sha(path),source_id=sid,x=r['x'],y=r['y']))
            if (index+1)%10==0 or index+1==len(rows):
                write(output/'status.json',{'state':'running','done':index+1,'total':len(rows),'pid':info['pid'],'seconds':time.time()-started})
                print(json.dumps({'geometry_done':index+1,'total':len(rows),'seconds':round(time.time()-started,1)}),flush=True)
        info['state']='completed';info['seconds']=time.time()-started
        info['native_valid_pixels']=sum(r['native_valid_pixels'] for r in info['windows'])
        info['native_target_pixels']=sum(r['native_target_pixels'] for r in info['windows'])
        info['area_m2']=sum(r['raw_area_m2'] for r in info['windows'])
        info['target_m2']=sum(r['raw_target_m2'] for r in info['windows'])
        if not limit:
            assert info['native_valid_pixels']==91061049 and info['native_target_pixels']==885226
        write(output/'metadata.json',info)
        write(output/'status.json',{k:info[k] for k in ('state','seconds','native_valid_pixels','native_target_pixels','area_m2','target_m2')})
        return info
    except BaseException as exc:
        info.update(state='failed',error=repr(exc),traceback=traceback.format_exc())
        write(output/'metadata.json',info);write(output/'status.json',{'state':'failed','error':repr(exc)})
        raise
    finally:
        for dataset in ds.values(): dataset.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    build(args.output,args.limit)
