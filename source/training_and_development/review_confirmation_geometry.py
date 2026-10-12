"""Independent geometry producer review using exact rectangle arithmetic, no real TIFFs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,datetime,json,math,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'confirmation_v1'))
import local_geometry as g
import core

def main():
    out=ROOT/'provenance'/('confirmation_geometry_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();checks=[]
    def ok(n,v=True):assert v,n;checks.append({'name':n,'passed':True})
    def reject(n,fn):
        try:fn()
        except (ValueError,AssertionError,OSError,KeyError):ok(n);return
        raise AssertionError('Accepted '+n)
    reject('absent_freeze_no_manifest_import',lambda:g.execute(out/'absent.json'))
    ok('no_scientific_import_before_gate','numpy' not in sys.modules and 'rasterio' not in sys.modules)
    old=ROOT/'provenance/local_geometry_before_guard_d32117b2.py'
    def fnmap(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    before=fnmap(old);after=fnmap(Path(g.__file__))
    ok('all_geometry_arithmetic_AST_unchanged',all(after[k]==v for k,v in before.items() if k!='execute'))
    geo=g.dependencies();np=geo.np;identity=SimpleNamespace(transform=lambda x,y:(x,y));comparisons=0
    for origin in (.75,-.25):
        affine=geo.rasterio.Affine(.5,0,origin,0,.5,origin);lattice=geo.projected_lattice(identity,affine,0,0,512,512,[0,0])
        valid=np.zeros((512,512),bool);pixels=[(0,0),(0,1),(3,3)]
        for y,x in pixels:valid[y,x]=True
        result=g.area_only_overlap(geo,lattice,valid);expected={};coeff={}
        for y,x in pixels:
            x0=origin+x*.5;y0=origin+y*.5
            for ix in range(math.floor(x0),math.floor(x0+.5)+1):
                for iy in range(math.floor(y0),math.floor(y0+.5)+1):
                    area=max(0.,min(x0+.5,ix+1)-max(x0,ix))*max(0.,min(y0+.5,iy+1)-max(y0,iy))
                    if area:
                        expected[ix,iy]=expected.get((ix,iy),0)+area;coeff[ix,iy,y*512+x]=area
        keys=[tuple(map(int,k)) for k in result['cells']];assert set(keys)==set(expected)
        native=result['native_weight'].tocoo();actual={(keys[r][0],keys[r][1],int(c)):float(v) for r,c,v in zip(native.row,native.col,native.data)}
        assert actual==coeff
        for j,key in enumerate(keys):assert result['area'][j]==expected[key];comparisons+=1
        support=result['weight'].tocoo();want={}
        for (x,y,col),value in coeff.items():
            coarse=(col//512//2)*256+(col%512//2);k=(x,y,coarse);want[k]=want.get(k,0)+value
        got={(keys[r][0],keys[r][1],int(c)):float(v) for r,c,v in zip(support.row,support.col,support.data)};assert got==want
        ok('exact_rectangle_native_and_support_'+str(origin));ok('partial_valid_area_'+str(origin),sum(result['area'])==.75)
        nested,_=g.nested_area_checks(geo,result['cells'],result['area'])
        for m,(parents,area) in nested.items():
            manual={}
            for (x,y),v in expected.items():k=(x//m,y//m);manual[k]=manual.get(k,0)+v
            assert {tuple(map(int,k)):float(v) for k,v in zip(parents,area)}==manual
        ok('all_M_negative_floor_nesting_'+str(origin))
        stats=g.all_edge_midpoint_error(geo,identity,affine,0,0,[0,0],lattice)
        ok('every_native_edge_midpoint_'+str(origin),stats['edge_midpoints_checked']==525312 and stats['maximum_error_m']==0)
    source={'source_id':'a','tif':'synthetic','tif_sha256':'0'*64,'width':520,'height':512,'transform':[1,0,0,0,1,0],'crs':'EPSG:4326','global_offset':[0,0],'valid_band':4}
    wins=[dict(id='left',source_id='a',x=0,y=0,global_x=0,global_y=0,native_size=512),dict(id='edge',source_id='a',x=512,y=0,global_x=512,global_y=0,native_size=512)]
    manifest={'split':'artificial','expected_count':2,'geometry_sources':[source],'geometry_windows':wins,'records':[{'id':w['id']} for w in wins]}
    g.validate_whitelist(manifest,2,'artificial');g.ownership_checks([source],wins);ok('source_clipped_edge_not_padded_ownership')
    reject('label_source_key_forbidden',lambda:g.validate_whitelist({**manifest,'geometry_sources':[{**source,'mask':'forbidden'}]},2,'artificial'))
    reject('label_window_key_forbidden',lambda:g.validate_whitelist({**manifest,'geometry_windows':[{**w,'target':0} for w in wins]},2,'artificial'))
    reject('floating_native_origin_forbidden',lambda:g.validate_whitelist({**manifest,'geometry_windows':[wins[0],{**wins[1],'x':512.0}]},2,'artificial'))
    reject('swapped_identity_forbidden',lambda:g.validate_whitelist({**manifest,'records':list(reversed(manifest['records']))},2,'artificial'))
    reject('overlapping_real_support_forbidden',lambda:g.ownership_checks([source],[wins[0],{**wins[1],'x':511,'global_x':511}]))
    ok('padded_nominal_corners_origin_defined',g.geometric_origin(geo,identity,[source],wins)==[0.,0.])
    reject('empty_alpha_cannot_drop_member',lambda:g.area_only_overlap(geo,lattice,np.zeros((512,512),bool)))
    fake=out/'fake';(fake/'stages/inputs_local').mkdir(parents=True)
    for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json','core.py','stages/inputs_local/input_manifest.json','stages/inputs_local/artifact_manifest.json'):core.write(fake/name,{'synthetic':1})
    grant=out/'grant.json';core.write(grant,{'synthetic':1})
    with patch.object(g,'HERE',fake):watch=g.capture_execution_bindings(grant)
    core.write(fake/'stages/inputs_local/input_manifest.json',{'synthetic':2})
    reject('validator_interval_manifest_drift_rejected',lambda:[core.bound_file(p,h) for p,h in watch.items()])
    author=ROOT/'provenance/confirmation_geometry_readiness_20260916_094332_028413/focused_checks.json';raw=core.read(author)
    ok('earlier18_author_geometry_checks_separately_bound',raw['passed'] and raw['implementation_sha256']==core.sha(old))
    core.write(out/'review.json',dict(passed=True,scope='confirmation_geometry_producer_independent_preparation',implementation_sha256=core.sha(g.__file__),reviewer_sha256=core.sha(__file__),checks=checks,check_count=len(checks),exact_rectangle_cell_comparisons=comparisons,
      unchanged_arithmetic_previous_source=core.binding(old),earlier_author_checks=core.binding(author),test_or_R6_accessed=False,actual_TIFFs_opened=0,model_or_solver_calls=0,
      source_bindings={str(Path(g.__file__).resolve()):core.sha(g.__file__),str(g.GEO_PATH):g.GEO_SHA}))
    print(json.dumps({'out':str(out),'passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
