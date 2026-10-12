"""Tiny artificial layout check only; never calls the production input loader."""
from pathlib import Path
import datetime
import json
import math
import runpy
import shutil

ROOT=Path(__file__).resolve().parent
m=runpy.run_path(str(ROOT/'plot_confirmation_spatial_example.py'))
out=ROOT/'provenance'/('confirmation_spatial_example_synthetic_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir()
units=[]
for iy in range(4):
    for ix in range(5):
        a=2.0 if (ix,iy)==(0,0) else 4.0
        t=a*((ix+2*iy)%6)/5
        units.append(dict(M='2',id=f'M2_E{ix}_N{iy}',ix=str(ix),iy=str(iy),area_m2=str(a),target_m2=str(t)))
D=math.fsum(float(x['area_m2']) for x in units);T=math.fsum(float(x['target_m2']) for x in units);lookup={u['id']:u for u in units}
cases=[]
for r,ids in zip((1,2,4,6),(['M2_E0_N0','M2_E1_N0'],['M2_E2_N1'],['M2_E3_N2'],['M2_E4_N3'])):
    a=math.fsum(float(lookup[i]['area_m2']) for i in ids);t=math.fsum(float(lookup[i]['target_m2']) for i in ids)
    cases.append(dict(run_id=m['MODEL'],M='2',budget='0.1',policy='prefix',R=str(r),selected_ids_json=json.dumps(ids),
                      actual_area_m2=str(a),captured_target_m2=str(t),domain_area_m2=str(D),domain_target_m2=str(T),capture_pct=str(100*t/T)))
geometry={'origin_utm_m':[100.,200.],'area_m2':D,'crs':'EPSG:32649'}
display=m['prepare_display'](units,cases,geometry)
assert display['extent_m']==[0,10,0,8] and display['units'][0]['partial']
checks=['four fixed saved cases','coordinate origin and nominal rectangle','partial cell criterion']
for label,invalid in [('missing R6',cases[:-1]),('unknown ID',[{**cases[0],'selected_ids_json':'["missing"]'},*cases[1:]])]:
    try:m['prepare_display'](units,invalid,geometry)
    except ValueError:checks.append(label+' rejected')
    else:raise AssertionError(label)
shutil.copyfile(ROOT/'plot_confirmation_spatial_example.py',out/'plot_source.py')
shutil.copyfile(__file__,out/'check_source.py')
shutil.copyfile(m['PLAN'],out/'display_plan.json')
qa=m['render'](display,out,synthetic=True)
assert qa['coordinate_rows']==80 and qa['minimum_visible_font_pt']>=8
m['write'](out/'checks.json',{'passed':True,'checks':checks+['80 coordinate rows from20 units across4 panels','174mm/180mm with8pt minimum'],
    'source_sha256':m['sha'](ROOT/'plot_confirmation_spatial_example.py'),'plan_sha256':m['PLAN_SHA'],
    'actual_inputs_loaded':False,'production_loader_called':False,'production_claim_created':False,'QA':qa})
print(out)
