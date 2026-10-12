"""Read-only completed-result audit, with direct-vs-nested geometry checks."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,math,sys
sys.dont_write_bytecode=True
import grid_origin_sensitivity as job
ROOT=Path(__file__).resolve().parent
np,sparse,geo,Image,versions=job.dependencies()
started=job.utc();pins=job.read(ROOT/'input_manifest.json')
for p,v in pins.items():assert job.sha(p)==v['sha256'],p
g=job.read(job.ST/'local_geometry/metadata.json');md=job.models()
lookup=[{x['id']:x for x in job.read(job.ST/('predict_'+m)/'map_index.json')} for m in md]
native=job.read(job.ST/'truth_local/truth_manifest.json')['records'][0]['native_reference']
mask=Image.open(job.archive_path(native['mask_source'])).convert('L')
affine=geo.rasterio.Affine(*g['source_headers'][0]['transform']);trans=geo.Transformer.from_crs(4326,32649,always_xy=True)
checks=[]
# Fixed ordinal sample: first, middle and last archived window; not outcome chosen.
for index in (0,len(g['windows'])//2,len(g['windows'])-1):
    w=g['windows'][index]
    with np.load(job.ST/'local_geometry'/w['file']) as z:valid=(z['raw_valid_by_native']>0).reshape(512,512)
    x,y=w['x'],w['y'];target=(np.asarray(mask.crop((x,y,x+512,y+512)))>0)&valid
    lattice=geo.projected_lattice(trans,affine,x,y,512,512,g['origin_utm_m'])
    with np.load(ROOT/'window_integrals'/(w['id']+'.npz')) as z:basekeys=z['keys_00'];basevals=z['values_00']
    probs=np.empty((65536,96))
    for mi,look in enumerate(lookup):
        with np.load(job.archive_path(look[w['id']]['path'])) as z:probs[:,mi*4:mi*4+4]=z['probabilities'].reshape(4,-1).T
    for m in (2,4):
        for sx,sy in job.SHIFTS:
            keys,vals=geo.aggregate_nested(basekeys-np.array([m*sx//2,m*sy//2]),basevals,m)
            direct=geo.overlap_native(lattice/m-np.array([sx/2,sy/2]),valid,target,support=2)
            ref=m*m*np.column_stack((direct['area'],direct['target'],direct['weight']@probs))
            assert np.array_equal(keys,direct['cells'])
            diff=float(np.max(np.abs(vals-ref)))
            assert diff<1e-8,(w['id'],m,sx,sy,diff)
            checks.append({'window_index':index,'window_id':w['id'],'M':m,'shift_x_m':m*sx/2,'shift_y_m':m*sy/2,'all_96_prediction_integrals_checked':True,'area_target_and_prediction_max_abs_error_m2':diff,'keys_exact':True})
mask.close();job.table(ROOT/'direct_vs_nested_geometry_verification.csv',checks)
rows=job.rows(ROOT/'results_all.csv');assert len(rows)==4608
identities=set()
for row in rows:
    key=tuple(row[k] for k in ('run_id','R','M','shift_x_fraction','shift_y_fraction','budget'))
    assert key not in identities;identities.add(key)
    ids=json.loads(row['selected_ids_json']);assert len(ids)==int(row['selected_units'])==len(set(ids))
    a=float(row['actual_area_m2']);total=float(row['domain_area_m2']);budget=float(row['budget']);t=float(row['captured_target_m2'])
    assert 0<=t<=a<=total*budget
    assert abs(float(row['capture_pct'])-100*t/float(row['domain_target_m2']))<1e-10
result={'started_utc':started,'completed_utc':job.utc(),'all_frozen_input_files_unchanged_after_run':True,'checked_input_files':len(pins),'complete_unique_result_rows':len(rows),'budget_and_metric_checks':True,
        'direct_vs_nested_grid_checks':len(checks),'fixed_window_ordinals':[0,len(g['windows'])//2,len(g['windows'])-1],'max_abs_direct_vs_nested_error_m2':max(x['area_target_and_prediction_max_abs_error_m2'] for x in checks),
        'explanation':'Direct projected-native-pixel integration at M2/M4 agrees with shifted aggregation from the native-exact M1 integrals. This checks the grid shift/index formula; both paths reuse the archived geometric intersection implementation.',
        'verification_script_sha256':job.sha(__file__)}
job.write(ROOT/'postrun_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
