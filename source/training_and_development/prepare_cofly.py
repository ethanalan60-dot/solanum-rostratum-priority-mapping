"""Build CoFly v1 with location-only PCA splits and validity-normalized caches.

Run on CPU. Source files and other preparation/training scripts are unchanged.
The split is fixed before images or masks are decoded; no model is evaluated.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv,hashlib,json,math,os,re,time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'cofly_data_v1'
INDEX=ROOT/'external_data'/'cofly'/'prepare_index.csv'
SPLITS=('train','val','test_sealed')
HALF_BUFFER_M=10.0
QUANTILES=(.55,.80)
MIN_VALID=.25
SCENE='cofly_larissa_20190703_single_flight'

def stamp():return datetime.now(timezone.utc).isoformat()

def io_path(path):
    p=str(path)
    if os.name=='nt' and not p.startswith('\\\\?\\'):p='\\\\?\\'+str(Path(p).resolve())
    return p

def sha(path):
    h=hashlib.sha256()
    with open(io_path(path),'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def local_coordinates(sources):
    lat=np.array([r['latitude'] for r in sources],dtype=np.float64)
    lon=np.array([r['longitude'] for r in sources],dtype=np.float64)
    lat0=float(lat.mean());lon0=float(lon.mean());phi=math.radians(lat0)
    # Local WGS84 east/north linearization, adequate for this sub-kilometre field.
    a=6378137.;e2=6.6943799901413165e-3
    n=a/math.sqrt(1-e2*math.sin(phi)**2)
    m=a*(1-e2)/(1-e2*math.sin(phi)**2)**1.5
    coords=np.c_[np.radians(lon-lon0)*n*math.cos(phi),np.radians(lat-lat0)*m]
    coords-=coords.mean(axis=0)
    covariance=np.cov(coords,rowvar=False)
    values,vectors=np.linalg.eigh(covariance)
    axis=vectors[:,np.argmax(values)]
    # PCA sign is otherwise arbitrary. Fix it using only axis geometry.
    if axis[np.argmax(np.abs(axis))]<0:axis=-axis
    perpendicular=np.array([-axis[1],axis[0]])
    along=coords@axis;across=coords@perpendicular
    boundaries=np.quantile(along,QUANTILES,method='linear')
    metadata={'origin_latitude':lat0,'origin_longitude':lon0,'coordinate_definition':'Local WGS84 curvature-based east/north metres; altitude ignored.',
       'prime_vertical_radius_m':n,'meridional_radius_m':m,'pca_axis_east_north':axis.tolist(),
       'pca_axis_sign':'Largest-absolute east/north component is positive.',
       'pca_explained_variance_ratio':float(values.max()/values.sum()),'quantiles':list(QUANTILES),
       'quantile_method':'linear','boundaries_pca_m':boundaries.tolist(),'half_buffer_m':HALF_BUFFER_M,
       'split_inputs':'Only latitude and longitude of the 201 explicit-mask, nonduplicate source images. Target fractions, masks, model scores and official random splits are not inputs.'}
    return coords,along,across,boundaries,metadata

def choose_split(score,boundaries):
    a,b=boundaries
    if score<=a-HALF_BUFFER_M:return 'train'
    if a+HALF_BUFFER_M<=score<=b-HALF_BUFFER_M:return 'val'
    if score>=b+HALF_BUFFER_M:return 'test_sealed'
    return 'buffer_excluded'

def camera_distances(sources):
    result=[]
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            aa=[r for r in sources if r['split']==a];bb=[r for r in sources if r['split']==b]
            if not aa or not bb:
                result.append({'split_a':a,'split_b':b,'distance_m':None});continue
            xy1=np.array([[r['east_m'],r['north_m']] for r in aa]);xy2=np.array([[r['east_m'],r['north_m']] for r in bb])
            dist=np.linalg.norm(xy1[:,None,:]-xy2[None,:,:],axis=2);ix,iy=np.unravel_index(np.argmin(dist),dist.shape)
            result.append({'split_a':a,'split_b':b,'distance_m':float(dist[ix,iy]),'source_a':aa[ix]['source_id'],'source_b':bb[iy]['source_id']})
    return result

def plot_sources(sources,bounds,distances):
    colors={'train':'#0072B2','val':'#E69F00','test_sealed':'#CC79A7','buffer_excluded':'#8b8b8b'}
    labels={'train':'Train','val':'Validation','test_sealed':'Sealed test','buffer_excluded':'Buffer excluded'}
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.4),constrained_layout=True)
    for split in [*SPLITS,'buffer_excluded']:
        rows=[r for r in sources if r['split']==split]
        marker='x' if split=='buffer_excluded' else 'o'
        for ax,xkey,ykey in [(axes[0],'east_m','north_m'),(axes[1],'pca_first_m','pca_second_m')]:
            ax.scatter([r[xkey] for r in rows],[r[ykey] for r in rows],s=27,c=colors[split],marker=marker,alpha=.85,label=f'{labels[split]} (n={len(rows)})')
    for boundary in bounds:
        axes[1].axvspan(boundary-HALF_BUFFER_M,boundary+HALF_BUFFER_M,color='#777777',alpha=.15,zorder=-1)
        axes[1].axvline(boundary,color='#444444',ls='--',lw=1)
    axes[0].set(xlabel='Local east (m)',ylabel='Local north (m)',title='Source-image camera positions')
    axes[1].set(xlabel='First PCA coordinate (m)',ylabel='Second PCA coordinate (m)',title='55% / 80% quantiles; +/-10 m buffers')
    for ax in axes:
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='best',fontsize=8,framealpha=.9)
    lines=[f"{r['split_a']} / {r['split_b']}: {r['distance_m']:.2f} m" for r in distances if r['distance_m'] is not None]
    fig.suptitle('CoFly v1: location-only source grouping\nCamera separation does not establish non-overlapping image footprints',fontsize=12)
    axes[1].text(.01,-.19,'Nearest cross-split camera distances: '+ '; '.join(lines),transform=axes[1].transAxes,fontsize=7.5,ha='left',wrap=True)
    fig.savefig(OUT/'spatial_distribution.png',dpi=190,bbox_inches='tight')
    fig.savefig(OUT/'spatial_distribution.pdf',bbox_inches='tight')
    plt.close(fig)

def pool_window(rgb,target,valid):
    v=valid.reshape(256,2,256,2).mean(axis=(1,3)).astype(np.float32)
    a=(target&valid).reshape(256,2,256,2).mean(axis=(1,3)).astype(np.float32)
    soft=np.divide(a,v,out=np.zeros_like(a),where=v>0)
    weighted_rgb=rgb.astype(np.float32)*valid[:,:,None]
    rgb_mean=weighted_rgb.reshape(256,2,256,2,3).mean(axis=(1,3))
    normalized=np.divide(rgb_mean,v[:,:,None],out=np.zeros_like(rgb_mean),where=v[:,:,None]>0)
    image=np.clip(np.rint(normalized),0,255).astype(np.uint8)
    hard=np.asarray(Image.fromarray((target&valid).astype(np.uint8)*255).resize((256,256),Image.Resampling.NEAREST))>0
    return image,soft.astype(np.float16),v.astype(np.float16),hard

def main():
    start=time.monotonic()
    if (OUT/'DATASET_LOCK.json').exists():raise FileExistsError('cofly_data_v1 is sealed; use a new version for changes.')
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Output exists and is nonempty; refusing to overwrite an existing data version.')
    with INDEX.open(encoding='utf-8-sig',newline='') as f:input_rows=list(csv.DictReader(f))
    sources=[];removed=[];seen=set()
    for row in input_rows:
        if row['mask_available'].strip().lower()!='true' or not row['mask_path'].strip():continue
        pixel_hash=row['pixel_sha256'].strip()
        if row.get('duplicate_of_sample_id','').strip() or pixel_hash in seen:
            removed.append({'sample_id':row['sample_id'],'reason':'duplicate_explicit_mask_source'});continue
        seen.add(pixel_hash)
        match=re.search(r'ID_(\d+)',row['sample_id'])
        if not match:raise ValueError('Source ID not found')
        sources.append({'source_id':f'cf_{int(match.group(1)):05d}','scene_group':SCENE,
           'index_sample_id':row['sample_id'],'rgb_source':row['image_path'],'mask_source':row['mask_path'],
           'rgb_pixel_sha256':pixel_hash,'latitude':float(row['latitude']),'longitude':float(row['longitude']),
           'altitude_m':float(row['altitude_m']),'acquisition_date':row['acquisition_date'],
           'acquisition_local_unzoned':row['acquisition_local_unzoned']})
    sources.sort(key=lambda r:r['source_id'])
    assert len(sources)==201,('Expected exactly 201 explicit nonduplicate sources',len(sources))
    assert len({r['source_id'] for r in sources})==len(sources)
    xy,pc1,pc2,bounds,split_metadata=local_coordinates(sources)
    for i,r in enumerate(sources):
        r.update({'east_m':float(xy[i,0]),'north_m':float(xy[i,1]),'pca_first_m':float(pc1[i]),'pca_second_m':float(pc2[i]),'split':choose_split(pc1[i],bounds)})
    parent_counts=dict(Counter(r['split'] for r in sources));nearest=camera_distances(sources)
    OUT.mkdir(parents=True,exist_ok=False)
    split_metadata.update({'protocol':'cofly_data_v1','created_utc':stamp(),'source_index':str(INDEX),'source_index_sha256':sha(INDEX),
       'script_sha256':sha(__file__),'source_count':len(sources),'source_counts':parent_counts,'nearest_cross_split_camera_distances':nearest,
       'camera_buffer_caveat':'Each boundary excludes +/-10m along a unit PCA axis. Kept cross-boundary camera positions are at least 20m apart along that axis; pixel footprint non-overlap has not been established.'})
    save_json(OUT/'split_protocol.json',split_metadata)
    save_json(OUT/'source_assignments.json',sources)
    plot_sources(sources,bounds,nearest)
    if any(parent_counts.get(s,0)==0 for s in SPLITS):
        save_json(OUT/'manifest_summary.json',{'status':'stopped_empty_split','source_counts':parent_counts,'reason':'Fixed metadata-only boundaries produce an empty split. No boundary adjustment was made.'})
        print(json.dumps({'status':'stopped_empty_split','source_counts':parent_counts},ensure_ascii=False),flush=True)
        return
    cache=OUT/'cache';cache.mkdir()
    records=[];excluded=[];source_hashes=[];max_coverage_error=0.
    for source_number,source in enumerate(sources):
        if source['split']=='buffer_excluded':
            excluded.append(dict(source,exclusion_type='source_buffer'));continue
        with Image.open(io_path(source['rgb_source'])) as im:rgb=np.asarray(im.convert('RGB')).copy()
        observed=hashlib.sha256(rgb.tobytes()).hexdigest()
        assert observed==source['rgb_pixel_sha256'],('RGB differs from audited source index',source['source_id'])
        with Image.open(io_path(source['mask_source'])) as im:mask=np.asarray(im).copy()
        assert mask.ndim==2 and rgb.shape[:2]==mask.shape
        assert set(map(int,np.unique(mask))).issubset({0,1,2,3})
        target_full=np.isin(mask,[1,2,3]);height,width=mask.shape
        source_hashes.append({'source_id':source['source_id'],'rgb_pixel_sha256':observed,'mask_file_sha256':sha(source['mask_source']),'width':width,'height':height})
        for y in range(0,height,512):
            for x in range(0,width,512):
                hh=min(512,height-y);ww=min(512,width-x)
                valid=np.zeros((512,512),dtype=bool);valid[:hh,:ww]=True
                rec={k:source[k] for k in ['source_id','scene_group','split','latitude','longitude','altitude_m','east_m','north_m','pca_first_m','pca_second_m','rgb_source','mask_source','acquisition_date','acquisition_local_unzoned']}
                rec.update({'id':f"{source['source_id']}_x{x:04d}_y{y:04d}",'x':x,'y':y,'source_width':width,'source_height':height,
                    'window_width':512,'window_height':512,'valid_width':ww,'valid_height':hh,'valid_pixels':int(valid.sum()),'valid_fraction':float(valid.mean())})
                if valid.mean()<MIN_VALID:
                    rec.update({'split':'low_valid_excluded','source_split':source['split'],'exclusion_type':'tile_geometry'});excluded.append(rec);continue
                target=np.zeros((512,512),dtype=bool);target[:hh,:ww]=target_full[y:y+hh,x:x+ww]
                color=np.zeros((512,512,3),dtype=np.uint8);color[:hh,:ww]=rgb[y:y+hh,x:x+ww]
                rec['target_pixels']=int((target&valid).sum());rec['coverage']=float(rec['target_pixels']/rec['valid_pixels'])
                image,soft,v,hard=pool_window(color,target,valid)
                reconstructed=float((soft.astype(np.float32)*v.astype(np.float32)).sum()/v.astype(np.float32).sum())
                error=abs(reconstructed-rec['coverage']);max_coverage_error=max(max_coverage_error,error)
                assert error<1e-6
                assert image.shape==(256,256,3) and image.dtype==np.uint8
                assert soft.shape==v.shape==hard.shape==(256,256) and hard.dtype==np.bool_
                cached=(cache/(rec['id']+'.npz')).resolve()
                np.savez_compressed(cached,image=image,soft=soft,valid=v,hard=hard)
                rec['cache']=str(cached);records.append(rec)
        if (source_number+1)%25==0:print(json.dumps({'sources_seen':source_number+1,'cache_tiles':len(records),'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
    for split in SPLITS:save_json(OUT/(split+'.json'),[r for r in records if r['split']==split])
    save_json(OUT/'excluded.json',excluded);save_json(OUT/'source_hashes.json',source_hashes)
    assert len({r['id'] for r in records})==len(records)
    for source_id in {r['source_id'] for r in records}:
        assert len({r['split'] for r in records if r['source_id']==source_id})==1
    summary={'protocol':'cofly_data_v1','status':'prepared_unsealed','created_utc':stamp(),'source_index_sha256':sha(INDEX),'script_sha256':sha(__file__),
       'source_counts':parent_counts,'counts':dict(Counter(r['split'] for r in records)),
       'positive_counts':dict(Counter(r['split'] for r in records if r['coverage']>0)),
       'exclusions':dict(Counter(r['exclusion_type'] for r in excluded)),'duplicate_explicit_sources_removed':removed,
       'nearest_cross_split_camera_distances':nearest,'pca_split':split_metadata,'minimum_valid_fraction':MIN_VALID,
       'max_cached_soft_coverage_reconstruction_error':max_coverage_error,'source_frame_shape':[720,1280],
       'window_shape':[512,512],'cache_image_shape':[256,256,3],
       'cache_dtypes':{'image':'uint8','soft':'float16','valid':'float16','hard':'bool'},
       'coverage_definition':'Exact target pixels in {1,2,3} divided by valid native pixels of the original 512 window; padded pixels excluded.',
       'scope':'Within-CoFly spatial method replication, not Solanum rostratum cross-region testing.',
       'license':{'dataset':'CC BY 4.0 per official Zenodo record 6697343','record_url':'https://zenodo.org/records/6697343','citation':'Krestenitis et al. (2022), CoFly-WeedDB: A UAV image dataset for weed detection and species identification. Data in Brief 45:108575.','code_license_distinction':'The upstream GitHub GPL-3.0 code license is separate from the dataset record license.'},
       'caveats':['One field and one acquisition; source positions, not pixels, determine spatial split.',
                  '20m PCA-axis camera separation does not prove non-overlapping image footprints.',
                  'Labels are coarse annotated weed regions, not independently measured true leaf/canopy cover.',
                  'Every frame contributes 5 accepted windows; its bottom-right 256x208 remainder is excluded by the predeclared valid>=0.25 rule.',
                  'No model predictions or test model metrics were computed.'],
       'runtime_seconds':time.monotonic()-start}
    save_json(OUT/'manifest_summary.json',summary)
    rows='\n'.join(f"| {s} | {parent_counts.get(s,0)} | {summary['counts'].get(s,0)} |" for s in SPLITS)
    distance_rows='\n'.join(f"- {r['split_a']} / {r['split_b']}：{r['distance_m']:.3f} m（{r['source_a']} / {r['source_b']}）" for r in nearest)
    notes=f'''# CoFly data v1 适配与空间分组

仅使用 201 张有显式 `labels_1d` 且无重复的来源图；目标为 mask ID 1、2、3。原图及其全部窗口属于同一 split。本版本用于 CoFly 数据内空间方法复现，不是刺萼龙葵跨地区测试。

## 固定的无标签位置分组

以 201 个相机经纬度的均值为局部原点，按 WGS84 曲率换算东西/南北米坐标，取第一 PCA 轴。轴方向固定为绝对值最大分量为正；仅用位置元数据的 55% 和 80% 线性分位作边界，两侧各剔除 10 m。分区在任何 RGB/mask 解码前确定，未使用目标比例、模型输出或官方随机 split 调整。

| 分区 | 父图数 | 缓存窗口数 |
|---|---:|---:|
{rows}

缓冲排除父图：{parent_counts.get('buffer_excluded',0)}。PCA 边界：{bounds.tolist()} m。

最近跨分区相机位置距离：

{distance_rows}

20 m 的轴向相机位置分离**不能证明像素足迹不重叠**。全部数据仍来自同一地块、同一次航飞。空间图见 `spatial_distribution.png` / PDF，完整来源分配见 `source_assignments.json`。

## 缓存与覆盖定义

- 1280×720 图像按 512 非重叠网格分割，边缘补零，valid 只覆盖真实画幅；只保留 valid_fraction ≥ 0.25。
- 每张保留 5 个窗口；右下角 256×208 的窗口有效比例 0.203125，按预定规则排除。因此这些窗口并非整幅图像的无损全覆盖评价。
- 512→256：先分别对 RGB×valid、目标×valid、valid 做 2×2 面积平均，再除以 valid 面积均值；无效处输出 0。hard 使用最近邻。
- NPZ：image uint8 RGB[256,256,3]；soft float16[256,256]；valid float16[256,256]；hard bool[256,256]。
- coverage 是原始 512 窗口中目标像素/真实有效像素的精确比值，不使用缩小后的 hard mask 推算。
- 每条清单含短 ASCII id、绝对 cache 路径、父图 source_id、scene_group、split、经纬度/米制/PCA 坐标和来源路径。
- 面积归一化 soft 回算 coverage 的最大误差为 {max_coverage_error:.3g}。未进行 GPU 训练或模型测试评价。

## 标注与许可

来源：[CoFly-WeedDB 官方 Zenodo 数据记录](https://zenodo.org/records/6697343)，数据许可 CC BY 4.0；应署名并引用 Krestenitis et al. (2022), Data in Brief 45:108575。上游 GitHub 的 GPL-3.0 是代码许可，不能替代数据记录许可。

mask 对应的是三类杂草的粗多边形标注区域；“覆盖率”在本版本中严格指标注区域占比，不宣称独立测量的真实叶片或冠层覆盖。165 张无显式 mask 图没有纳入；其中已知一个精确重复副本也没有进入。

封存由规范根 `seal_dataset.py --data-dir cofly_data_v1` 完成；封存后的 DATASET_LOCK.json 绑定三个清单与缓存哈希。未来任何数据修改须使用新版本。
'''
    (OUT/'README.md').write_text(notes,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['protocol','source_counts','counts','exclusions','nearest_cross_split_camera_distances','max_cached_soft_coverage_reconstruction_error','runtime_seconds']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
