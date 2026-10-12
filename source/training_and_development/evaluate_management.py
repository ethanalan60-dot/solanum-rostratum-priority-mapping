"""CPU-only fixed R x M development evaluation on completed native geometry/maps.

No model is loaded, no training or test/R6 inference is available. The 1 m
geometry weights are reused and larger management grids only sum integer children.
"""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import datetime
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'geo_dependencies'))
import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import milp, Bounds, LinearConstraint
from threadpoolctl import threadpool_limits
import geo_management as gm

R_VALUES=(1,2,4)
M_VALUES=(1,2,4)
BUDGETS=(.10,.20,.30,.50,1.)
RANDOM_SEED=20260916
RANDOM_ORDERS=1000


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp():return datetime.datetime.now().astimezone().isoformat()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def relative_error(actual,expected):
    if expected==0:return 0. if actual==0 else math.inf
    return abs(actual-expected)/abs(expected)


def safe_number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def cell_ids(M,cells):
    return np.array([f'rmv1_m{M}_x{int(ix):+07d}_y{int(iy):+07d}' for ix,iy in cells])


def ordered_prefix(area,target,order):
    A=math.fsum(area);T=math.fsum(target)
    assert A>0 and np.all(area>0)
    selected_A=np.r_[0.,np.cumsum(area[order],dtype=np.float64)]
    selected_T=np.r_[0.,np.cumsum(target[order],dtype=np.float64)]
    # The last prefix is exactly the same whole domain, independent of sum order.
    selected_A[-1]=A;selected_T[-1]=T
    assert np.all(np.diff(selected_A)>=0) and np.all(np.diff(selected_T)>=-1e-12)
    return {'order':order,'area':selected_A,'target':selected_T,'area_fraction':selected_A/A,
            'capture_fraction':selected_T/T if T>0 else np.full_like(selected_T,np.nan),
            'total_area':A,'total_target':T}


def step_auc(prefix,upper=1.):
    """Right-continuous feasible prefix capture, integrated without interpolation."""
    assert 0<upper<=1
    if prefix['total_target']==0:return None
    s,c=prefix['area_fraction'],prefix['capture_fraction']
    widths=np.maximum(0.,np.minimum(s[1:],upper)-np.maximum(s[:-1],0.))
    return float(np.dot(c[:-1],widths)/upper)


def prefix_budget(prefix,budget):
    limit=budget*prefix['total_area']
    k=int(np.searchsorted(prefix['area'],limit,side='right')-1)
    assert 0<=k<len(prefix['area']) and prefix['area'][k]<=limit
    if k+1<len(prefix['area']):assert prefix['area'][k+1]>limit
    return {'k':k,'budget':budget,'budget_area':limit,'actual_area':float(prefix['area'][k]),
            'actual_area_fraction':float(prefix['area_fraction'][k]),
            'captured_target':float(prefix['target'][k]),
            'capture_fraction':float(prefix['capture_fraction'][k]) if prefix['total_target'] else None,
            'missed_target':float(prefix['total_target']-prefix['target'][k]),
            'zero_annotation_area':float(prefix['area'][k]-prefix['target'][k]),
            'unused_budget_area':float(limit-prefix['area'][k])}


def solve_label_knapsack(area,target,ids,budget,time_limit=30.,relative_gap=1e-8):
    """Primary max-target binary knapsack. No minimum-area claim or secondary goal."""
    A,T=math.fsum(area),math.fsum(target);limit=budget*A
    if T==0:
        return {'status':'no_target','solver_status':None,'primal_target':0.,'dual_target_bound':0.,
                'primal_capture_fraction':None,'dual_capture_bound':None,'relative_gap':None,
                'optimality_proven':True,'selected_ids':[],'selected_area_feasibility_audit':0.,
                'minimum_selected_area_claim':False}
    indices=np.flatnonzero(target>0)
    cost=area[indices]/A;value=target[indices]/T
    options={'time_limit':float(time_limit),'mip_rel_gap':float(relative_gap),'presolve':True,
             'threads':2,'random_seed':RANDOM_SEED,'mip_feasibility_tolerance':1e-9,
             'primal_feasibility_tolerance':1e-9}
    started=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=milp(-value,integrality=np.ones(len(indices)),bounds=Bounds(0.,1.),
                    constraints=LinearConstraint(cost[None,:],-np.inf,budget),options=options)
    selected=np.array([],dtype=np.int64);max_integrality_error=None
    candidate_feasible=False;candidate_violation=None
    if result.x is not None and np.isfinite(result.x).all():
        max_integrality_error=float(np.max(np.abs(result.x-np.rint(result.x))))
        if max_integrality_error<=1e-6:
            selected=indices[np.rint(result.x).astype(bool)]
            selected_A=math.fsum(area[selected]);candidate_violation=max(0.,selected_A-limit)
            candidate_feasible=candidate_violation==0.
    if not candidate_feasible:selected=np.array([],dtype=np.int64)
    chosen_A=math.fsum(area[selected]);primal=math.fsum(target[selected]);capture=primal/T
    raw_dual=safe_number(getattr(result,'mip_dual_bound',None))
    raw_upper=-raw_dual if raw_dual is not None else None
    upper=min(1.,raw_upper) if raw_upper is not None else 1.
    # Resolve machine-roundoff inversion; raw solver bound is also retained.
    if upper<capture:
        assert capture-upper<=1e-10,'MILP dual below feasible solution'
        upper=capture
    gap=(upper-capture)/max(abs(capture),1e-30)
    proven=(result.status==0 and candidate_feasible and gap<=relative_gap+1e-10)
    return {'status':('optimal' if proven else 'solver_result_not_proven_optimal'),
            'solver_status':int(result.status),'solver_message':str(result.message),
            'optimality_proven':proven,'primal_target':primal,'dual_target_bound':upper*T,
            'primal_capture_fraction':capture,'dual_capture_bound':upper,
            'raw_solver_minimization_primal':safe_number(getattr(result,'fun',None)),
            'raw_solver_minimization_dual':raw_dual,'raw_solver_capture_upper':raw_upper,
            'raw_solver_relative_gap':safe_number(getattr(result,'mip_gap',None)),
            'relative_gap':gap,'absolute_target_gap':(upper-capture)*T,
            'dual_bound_source':'HiGHS plus known total-target bound' if raw_dual is not None else 'known total-target bound only',
            'selected_area_feasibility_audit':chosen_A,'budget_area':limit,
            'candidate_budget_violation':candidate_violation,'candidate_integrality_error':max_integrality_error,
            'candidate_feasible':candidate_feasible,'selected_ids':[str(ids[i]) for i in selected],
            'positive_target_variables':len(indices),'zero_target_variables_excluded':len(area)-len(indices),
            'minimum_selected_area_claim':False,'secondary_min_area_solved':False,
            'elapsed_seconds':time.time()-started,'options':options,
            'solver_warnings':[str(w.message) for w in caught]}


def run_toys():
    lattice=np.stack(np.meshgrid(np.arange(3,dtype=float),np.arange(3,dtype=float)),axis=-1)
    valid=np.array([[1,0],[1,0]],dtype=bool);target=np.array([[1,0],[0,0]],dtype=bool)
    toy=gm.overlap_native(lattice,valid,target,support=2)
    assert np.array_equal(toy['cells'],[[0,0],[0,1]])
    assert np.allclose(toy['area'],[1,1],rtol=0,atol=1e-12)
    assert np.allclose(toy['target'],[1,0],rtol=0,atol=1e-12)
    p=toy['weight']@np.array([.6])
    assert np.allclose(p,[.6,.6],rtol=0,atol=1e-12)
    parent,nested=gm.aggregate_nested(toy['cells'],np.c_[toy['area'],toy['target'],p],2)
    assert np.array_equal(parent,[[0,0]]) and np.allclose(nested,[[2.,1.,1.2]],rtol=0,atol=1e-12)
    forward=ordered_prefix(np.array([.4,.6]),np.array([.4,0.]),np.array([0,1]))
    reverse=ordered_prefix(np.array([.4,.6]),np.array([.4,0.]),np.array([1,0]))
    assert abs(step_auc(forward)-.6)<1e-12 and abs(step_auc(forward,.5)-.2)<1e-12
    assert step_auc(reverse)==0.
    equal=ordered_prefix(np.ones(7),np.ones(7),np.arange(7))
    assert abs(step_auc(equal)-6/14)<1e-12
    area=np.array([2.,3.,3.]);truth=np.array([1.9,2.7,2.7]);ids=np.array(['a','b','c'])
    label=ordered_prefix(area,truth,np.lexsort((ids,-truth/area)))
    heuristic=prefix_budget(label,.75)
    optimal=solve_label_knapsack(area,truth,ids,.75)
    assert abs(heuristic['captured_target']-4.6)<1e-12
    assert optimal['optimality_proven'] and abs(optimal['primal_target']-5.4)<1e-9
    return {'passed':True,'partial_valid_spatial_case':{'area':[1,1],'target':[1,0],'predicted':[.6,.6]},
            'nested_aggregation_passed':True,'step_auc_forward':step_auc(forward),
            'step_auc_half_normalized':step_auc(forward,.5),'step_auc_reverse':step_auc(reverse),
            'equal_density_seven_cells_auc':step_auc(equal),
            'non_equal_cost_label_prefix_target':heuristic['captured_target'],'non_equal_cost_milp':optimal}


def check_ownership(manifest,maps):
    assert len(manifest)==len({r['id'] for r in manifest})==357
    assert all(r['split']=='val' for r in manifest)
    rects=[]
    for r in manifest:
        source=maps['sources'][r['source_id']]
        w=min(512,source['width']-r['x']);h=min(512,source['height']-r['y'])
        assert w>0 and h>0
        rect=(r['global_x'],r['global_y'],r['global_x']+w,r['global_y']+h)
        for old in rects:
            assert min(rect[2],old[2])<=max(rect[0],old[0]) or min(rect[3],old[3])<=max(rect[1],old[1]),'Duplicate native ownership'
        rects.append(rect)
    return {'windows':len(rects),'source_clipped_native_rectangles_pairwise_disjoint':True}


def integrate(geometry,maps,manifest,status_path,status):
    geo_rows={r['id']:r for r in geometry['windows']};map_rows={r['id']:r for r in maps['windows']}
    expected={r['id'] for r in manifest};native={r['id']:r for r in manifest}
    assert len(geo_rows)==len(geometry['windows'])==357 and set(geo_rows)==expected
    assert len(map_rows)==len(maps['windows'])==357 and set(map_rows)==expected
    all_cells=[];all_values=[];audits=[];input_windows=[]
    for index,key in enumerate(sorted(expected),1):
        g,m,r=geo_rows[key],map_rows[key],native[key]
        assert m['split']=='val' and m['factors']==[1,2,4]
        assert g['source_id']==m['source_id']==r['source_id'] and g['x']==r['x'] and g['y']==r['y']
        assert m['native_window_xywh']==[r['x'],r['y'],512,512]
        assert m['manifest_sha256']==maps['manifest_sha256'] and m['checkpoint_sha256']==maps['checkpoint_sha256']
        assert m['native_pixels_per_output_pixel']==[2,2]
        tr=gm.rasterio.Affine(*maps['sources'][r['source_id']]['affine'])
        expected_affine=tr*gm.rasterio.Affine.translation(r['x'],r['y'])*gm.rasterio.Affine.scale(2,2)
        assert np.allclose(m['output_affine'],list(expected_affine)[:6],rtol=0,atol=1e-15)
        assert sha(g['file'])==g['sha256'] and sha(m['file'])==m['file_sha256']
        with np.load(g['file'],allow_pickle=False) as d:
            cells=d['cells'].copy();area=d['area'].copy();target=d['target'].copy()
            raw=d['raw_valid_by_support'].copy()
            shape=tuple(map(int,d['shape']))
            W=sparse.csr_matrix((d['weights'],d['indices'],d['indptr']),shape=shape)
        assert cells.shape==(len(area),2) and shape==(len(area),65536)
        assert len(np.unique(cells,axis=0))==len(cells) and np.issubdtype(cells.dtype,np.integer)
        assert np.isfinite(W.data).all() and (W.data>=0).all()
        assert np.isfinite(area).all() and np.all(area>0) and np.isfinite(target).all()
        assert np.all(target>=0) and np.all(target<=area+1e-12)
        assert raw.shape==(65536,) and np.isfinite(raw).all() and np.all(raw>=0)
        with np.load(m['file'],allow_pickle=False) as d:
            p=d['p'].copy();valid=d['valid'].copy()
        assert p.shape==(3,256,256) and p.dtype==np.float32
        assert valid.shape==(256,256) and np.isfinite(valid).all() and np.all((valid>=0)&(valid<=1))
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid or out-of-range probability'
        assert np.array_equal(raw>0,valid.ravel()>0),'Validity domain differs'
        assert float(valid.sum())*4==g['native_valid_pixels']
        row_mass=np.asarray(W.sum(axis=1)).ravel();column_mass=np.asarray(W.sum(axis=0)).ravel()
        assert np.allclose(row_mass,area,rtol=1e-9,atol=1e-14)
        assert np.max(np.abs(column_mass-raw))<=1e-9*max(math.fsum(raw),1e-30)
        P=W@p.reshape(3,-1).astype(np.float64).T
        independent=raw@p.reshape(3,-1).astype(np.float64).T
        rel=[relative_error(math.fsum(P[:,i]),float(independent[i])) for i in range(3)]
        assert max(rel)<=1e-9,('Probability integration error',key,rel)
        assert np.all(P>=0) and np.all(P<=area[:,None]+1e-12)
        assert relative_error(math.fsum(area),g['raw_area_m2'])<=1e-9
        assert relative_error(math.fsum(target),g['raw_target_m2'])<=1e-9
        all_cells.append(cells);all_values.append(np.c_[area,target,P])
        audits.append({'id':key,'probability_integral_relative_errors':rel,
                       'raw_probability_integrals':[float(x) for x in independent],
                       'max_support_abs_error':float(np.max(np.abs(column_mass-raw))),
                       'row_mass_max_abs_error':float(np.max(np.abs(row_mass-area)))})
        input_windows.append({'id':key,'geometry_file':g['file'],'geometry_sha256':g['sha256'],
                              'maps_file':m['file'],'maps_sha256':m['file_sha256']})
        if index%25==0 or index==357:
            status.update(stage='probability_integration',integrated_windows=index,updated_at=stamp())
            write_json(status_path,status);print(json.dumps({'integrated_windows':index,'total':357}),flush=True)
    cells,values=gm.aggregate_nested(np.concatenate(all_cells),np.concatenate(all_values),1)
    assert relative_error(math.fsum(values[:,0]),geometry['area_m2'])<=1e-9
    assert relative_error(math.fsum(values[:,1]),geometry['target_m2'])<=1e-9
    global_raw=[math.fsum(a['raw_probability_integrals'][i] for a in audits) for i in range(3)]
    global_errors=[relative_error(math.fsum(values[:,i+2]),global_raw[i]) for i in range(3)]
    assert max(global_errors)<=1e-9
    return cells,values,{'per_window':audits,'global_probability_integrals':global_raw,
                        'global_probability_relative_errors':global_errors,
                        'max_window_probability_relative_error':max(max(a['probability_integral_relative_errors']) for a in audits)},input_windows


def random_reference(M,area,target,ids,out):
    canonical=np.argsort(ids,kind='stable');n=len(area)
    rng=np.random.Generator(np.random.PCG64(RANDOM_SEED))
    all_s=np.zeros((RANDOM_ORDERS,n+1),dtype=np.float64)
    all_c=np.zeros_like(all_s);orders=np.zeros((RANDOM_ORDERS,n),dtype=np.int32)
    budget_rows=[];auc_rows=[]
    for repetition in range(RANDOM_ORDERS):
        order=canonical[rng.permutation(n)]
        prefix=ordered_prefix(area,target,order)
        all_s[repetition]=prefix['area_fraction'];all_c[repetition]=prefix['capture_fraction'];orders[repetition]=order
        auc_rows.append({'M':M,'repetition':repetition+1,'auc_0_1':step_auc(prefix),
                         'auc_0_0p5_normalized':step_auc(prefix,.5)})
        for b in BUDGETS:
            budget_rows.append({'M':M,'repetition':repetition+1,**prefix_budget(prefix,b)})
    path=out/f'random_full_curves_M{M}.npz'
    np.savez_compressed(path,area_fraction=all_s,capture_fraction=all_c,order_indices=orders,unit_ids=ids)
    summaries=[]
    for b in BUDGETS:
        selected=[r for r in budget_rows if r['budget']==b]
        row={'M':M,'method':'random_reference','R':None,'budget':b,'random_orders':RANDOM_ORDERS}
        for field in ('capture_fraction','captured_target','missed_target','actual_area','actual_area_fraction','zero_annotation_area','unused_budget_area','k'):
            values=np.array([r[field] for r in selected],dtype=np.float64)
            row[field+'_mean']=float(values.mean())
            row[field+'_q025']=float(np.quantile(values,.025,method='linear'))
            row[field+'_q975']=float(np.quantile(values,.975,method='linear'))
        summaries.append(row)
    auc_summary={}
    for field in ('auc_0_1','auc_0_0p5_normalized'):
        values=np.array([r[field] for r in auc_rows])
        auc_summary[field]={'mean':float(values.mean()),'q025':float(np.quantile(values,.025,method='linear')),
                            'q975':float(np.quantile(values,.975,method='linear'))}
    return {'seed':RANDOM_SEED,'orders':RANDOM_ORDERS,'rng':'PCG64 reset to the same seed separately for each M',
            'initial_order':'ASCII grid ID ascending','full_curves':str(path),'full_curves_sha256':sha(path),
            'curve_array_shape':list(all_s.shape),'auc':auc_summary,
            'quantile_interpretation':'random-order variation within the same fixed scene, not a confidence interval'},summaries,budget_rows,auc_rows


def evaluate_tables(base_cells,base_values,protocol,out,status_path,status):
    unit_rows=[];curve_rows=[];fixed_rows=[];random_rows=[];random_auc_rows=[];by_M={}
    total_base=[math.fsum(base_values[:,i]) for i in range(5)]
    for M in M_VALUES:
        cells,values=gm.aggregate_nested(base_cells,base_values,M)
        totals=[math.fsum(values[:,i]) for i in range(5)]
        nested_errors=[relative_error(a,b) for a,b in zip(totals,total_base)]
        assert max(nested_errors)<=1e-10
        area,target=values[:,0],values[:,1];predicted=values[:,2:]
        assert np.all(area>0) and np.all(target>=0) and np.all(target<=area+1e-12)
        assert np.all(predicted>=0) and np.all(predicted<=area[:,None]+1e-12)
        ids=cell_ids(M,cells)
        assert len(ids)==len(set(ids))
        for i,(ix,iy) in enumerate(cells):
            unit_rows.append({'M':M,'ix':int(ix),'iy':int(iy),'id':str(ids[i]),'area':float(area[i]),
                              'target':float(target[i]),'q_true':float(target[i]/area[i]),
                              **{f'predicted_R{r}':float(predicted[i,ri]) for ri,r in enumerate(R_VALUES)},
                              **{f'q_R{r}':float(predicted[i,ri]/area[i]) for ri,r in enumerate(R_VALUES)}})
        grid={'units':len(area),'positive_target_units':int(np.count_nonzero(target)),
              'total_area':totals[0],'total_target':totals[1],
              'total_predicted':{str(r):totals[ri+2] for ri,r in enumerate(R_VALUES)},
              'nested_relative_errors':nested_errors,'prefix_results':[],'label_milp':[]}
        score_specs=[('model_prefix',r,predicted[:,ri]/area) for ri,r in enumerate(R_VALUES)]
        score_specs.append(('label_prefix',None,target/area))
        for method,R,score in score_specs:
            order=np.lexsort((ids,-score));prefix=ordered_prefix(area,target,order)
            for k in range(len(area)+1):
                curve_rows.append({'M':M,'method':method,'R':R,'k':k,
                                   'last_added_id':str(ids[order[k-1]]) if k else '',
                                   'actual_area':float(prefix['area'][k]),
                                   'actual_area_fraction':float(prefix['area_fraction'][k]),
                                   'captured_target':float(prefix['target'][k]),
                                   'capture_fraction':float(prefix['capture_fraction'][k]),
                                   'missed_target':float(prefix['total_target']-prefix['target'][k]),
                                   'zero_annotation_area':float(prefix['area'][k]-prefix['target'][k])})
            cuts=[{'M':M,'method':method,'R':R,**prefix_budget(prefix,b)} for b in BUDGETS]
            fixed_rows.extend(cuts)
            grid['prefix_results'].append({'method':method,'R':R,'auc_0_1':step_auc(prefix),
                                            'auc_0_0p5_normalized':step_auc(prefix,.5),'fixed_budgets':cuts})
        for b in BUDGETS:
            status.update(stage='label_milp',M=M,budget=b,updated_at=stamp());write_json(status_path,status)
            result=solve_label_knapsack(area,target,ids,b,protocol['milp_time_limit_seconds'],protocol['milp_relative_gap'])
            grid['label_milp'].append({'budget':b,**result})
            fixed_rows.append({'M':M,'method':'label_milp','R':None,'budget':b,
                               'captured_target':result['primal_target'],'capture_fraction':result['primal_capture_fraction'],
                               'missed_target':totals[1]-result['primal_target'],
                               'milp_status':result['status'],'milp_solver_status':result['solver_status'],
                               'optimality_proven':result['optimality_proven'],
                               'primal_target':result['primal_target'],'dual_target_bound':result['dual_target_bound'],
                               'relative_gap':result['relative_gap'],
                               'selected_area_feasibility_audit':result['selected_area_feasibility_audit'],
                               'minimum_selected_area_claim':False})
            print(json.dumps({'M':M,'budget':b,'milp_status':result['status'],
                              'primal_capture':result['primal_capture_fraction'],'gap':result['relative_gap']}),flush=True)
        status.update(stage='random_reference',M=M,updated_at=stamp());write_json(status_path,status)
        random_summary,random_cuts,raw_trials,raw_auc=random_reference(M,area,target,ids,out)
        grid['random_reference']=random_summary;fixed_rows.extend(random_cuts)
        random_rows.extend(raw_trials);random_auc_rows.extend(raw_auc)
        by_M[str(M)]=grid
    write_csv(out/'unit_predictions.csv',unit_rows)
    write_csv(out/'decision_curves.csv',curve_rows)
    write_csv(out/'fixed_budgets.csv',fixed_rows)
    write_csv(out/'random_budget_replicates.csv',random_rows)
    write_csv(out/'random_auc_replicates.csv',random_auc_rows)
    # Intervals keep solver uncertainty explicit; no positive-granularity-gap assumption.
    gaps=[]
    for bi,b in enumerate(BUDGETS):
        finest=by_M['1']['label_milp'][bi]
        for M in M_VALUES:
            current=by_M[str(M)]['label_milp'][bi]
            gaps.append({'budget':b,'M':M,
                         'granularity_capture_gap_lower':finest['primal_capture_fraction']-current['dual_capture_bound'],
                         'granularity_capture_gap_upper':finest['dual_capture_bound']-current['primal_capture_fraction'],
                         'meaning':'finest-vs-M label-knapsack optimum interval, not model error'})
    return {'by_M':by_M,'granularity_gap_intervals':gaps,'unit_rows':len(unit_rows),
            'decision_curve_rows':len(curve_rows),'fixed_budget_rows':len(fixed_rows),
            'random_budget_replicates':len(random_rows),'random_auc_replicates':len(random_auc_rows)}


def make_report(summary,out):
    lines=['# R×M 统一管理网格开发评价','',
           '**完整验证域的几何积分与决策评价已完成。**这是固定检查点、同一验证场景上的描述性分析；未用新指标选模、未训练、未读取 test 或 R6。','',
           f"- 验证窗口：357；原生有效像素 {summary['domain']['native_valid_pixels']:,}，目标像素 {summary['domain']['native_target_pixels']:,}。",
           f"- UTM 平面有效面积：{summary['domain']['area_m2']:.9f} m²；目标标注面积：{summary['domain']['target_m2']:.9f} m²。",
           f"- 每窗 W@p 与独立原生支持积分的最大相对差：{summary['integration']['max_window_probability_relative_error']:.3g}（阈值 1e-9）。",
           '- 所有 R 共享完全相同的 A/T；2 m、4 m 由 1 m 单元按整数父索引求和。没有把旧近似平方米总量强行归一到新测度。','',
           '## 完整阶梯曲线 AUC','',
           '曲线取可支付的最大完整前缀；横轴为名义有效面积预算比例。AUC 用阶梯积分，半区间结果除以 0.5；不做梯形连线。','',
           '| M（m） | 单元数 | 方法 / R | AUC [0,1] | AUC [0,0.5] 标准化 |',
           '|---:|---:|---|---:|---:|']
    for M in M_VALUES:
        grid=summary['results']['by_M'][str(M)]
        for p in grid['prefix_results']:
            label=f"model R={p['R']}" if p['method']=='model_prefix' else 'label-informed prefix'
            lines.append(f"| {M} | {grid['units']} | {label} | {p['auc_0_1']:.6f} | {p['auc_0_0p5_normalized']:.6f} |")
        rand=grid['random_reference']['auc']
        lines.append(f"| {M} | {grid['units']} | random mean | {rand['auc_0_1']['mean']:.6f} | {rand['auc_0_0p5_normalized']['mean']:.6f} |")
    lines+=['','## 全部固定预算：模型目标面积捕获率','',
            '| M（m） | 预算 | R1 | R2 | R4 | label-prefix | MILP 下界 / 上界 | 随机均值（2.5%–97.5%分位） |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|']
    fixed=list(csv.DictReader((out/'fixed_budgets.csv').open(encoding='utf-8-sig',newline='')))
    for M in M_VALUES:
        grid=summary['results']['by_M'][str(M)]
        for bi,b in enumerate(BUDGETS):
            values=[grid['prefix_results'][ri]['fixed_budgets'][bi]['capture_fraction'] for ri in range(4)]
            opt=grid['label_milp'][bi]
            rand=next(r for r in fixed if r['method']=='random_reference' and int(r['M'])==M and float(r['budget'])==b)
            fields=' | '.join(f'{v:.2%}' for v in values)
            lines.append(f"| {M} | {b:.0%} | {fields} | {opt['primal_capture_fraction']:.2%} / {opt['dual_capture_bound']:.2%} | {float(rand['capture_fraction_mean']):.2%} ({float(rand['capture_fraction_q025']):.2%}–{float(rand['capture_fraction_q975']):.2%}) |")
    statuses=Counter(x['status'] for g in summary['results']['by_M'].values() for x in g['label_milp'])
    lines+=['','## 参考策略与解释边界','',
            '- label-informed prefix 按真值比例排序，只是标签辅助参考，非全局最优声明。',
            f'- 每个 M×预算分别求解一次 0/1 MILP，去掉 T=0 变量；状态计数：{dict(statuses)}。结果保存原始求解器 primal/dual/gap、原始面积上的可行性复核和选中 ID。',
            '- MILP 仅优化目标捕获；没有求二级最小面积，因此 feasibility_audit 面积只用于检查预算，不能宣称它是最省面积或最佳空间布局。',
            '- 每个 M 使用 PCG64 seed=20260916 的 1,000 次随机完整排序；实际前缀预算与模型相同。分位仅描述同一场景的随机排序变异，不是跨场景置信区间。',
            '- 概率在各 2×2 原生支撑内取常数，但 alpha/mask 保留原生空间位置。几何使用 UTM 原生角点直线四边形，是已记录数值验收下的平面近似，不代表无误差的田间测绘。',
            '- 全部管理单元只来自现有 357 个验证窗口有效域；未填补未预测边界条带。单位数量和三个 R 不增加独立田块数。',
            '- 当前目标是粗标注目标区域；捕获、遗漏和零标注复核面积不等于药剂、产量、机器作业或真实控制效果。','',
            '## 文件与复现','',
            '- unit_predictions.csv：每个 M 的 (ix,iy)、A/T、R1/2/4 预测面积及覆盖率。',
            '- decision_curves.csv：模型三个 R 与 label-prefix 的全部前缀点，包括零点。',
            '- fixed_budgets.csv：所有预定预算的模型、label-prefix、MILP 与随机汇总。',
            '- random_full_curves_M*.npz：1,000 次随机完整阶梯曲线和排列索引；CSV 保存每次固定预算与 AUC。',
            '- toy_checks.json、integration_checks.json、inputs.json：人工案例、全窗积分与来源哈希。',
            '- source/ 保存本次实际执行代码及冻结协议副本；metadata/source/manifest/map/geometry 文件 SHA256 已记录。',
            f"- 运行时间 {summary['elapsed_seconds']:.1f} 秒；CPU 最多 2 线程，SciPy {summary['versions']['scipy']}。",'']
    (out/'summary.md').write_text('\n'.join(lines),encoding='utf-8')


def main(args):
    if not 0<=args.wait_seconds<=30:raise ValueError('Geometry wait must be 0..30 seconds')
    geometry_dir=Path(args.geometry_dir).resolve();maps_dir=Path(args.maps_dir).resolve()
    out=ROOT/'precision_agriculture'/('RM_evaluation_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False);(out/'source').mkdir()
    started=time.time();status_path=out/'status.json'
    status={'state':'running','stage':'toy_checks','created_at':stamp(),'pid':os.getpid(),
            'output_directory':str(out),'threads_max':2,'test_accessed':False,'R6_accessed':False}
    write_json(status_path,status);print(json.dumps(status,ensure_ascii=True),flush=True)
    try:
        with threadpool_limits(limits=2):
            toys=run_toys();write_json(out/'toy_checks.json',toys)
            print(json.dumps({'toy_checks':'passed'}),flush=True)
            deadline=time.monotonic()+args.wait_seconds
            while True:
                geometry=read(geometry_dir/'metadata.json')
                if geometry.get('state')=='completed':break
                if time.monotonic()>=deadline:
                    status.update(state='waiting_for_geometry',stage='inputs_not_ready',geometry_state=geometry.get('state'),
                                  geometry_directory=str(geometry_dir),updated_at=stamp())
                    write_json(status_path,status);print(json.dumps(status,ensure_ascii=True),flush=True);return 3
                time.sleep(min(5,max(0,deadline-time.monotonic())))
            assert read(geometry_dir/'status.json')['state']=='completed'
            assert not geometry['subset_diagnostic'] and geometry['expected_windows']==357 and not geometry['test_accessed']
            assert geometry['native_valid_pixels']==91061049 and geometry['native_target_pixels']==885226
            maps=read(maps_dir/'metadata.json');map_status=read(maps_dir/'status.json')
            assert map_status['status']=='completed' and map_status['verification']['all_checks_passed']
            assert maps['schema']=='local_validation_probability_maps_v1' and maps['factors']==[1,2,4]
            assert not maps['test_accessed'] and not maps['R6_accessed'] and not maps['run_id'].startswith('smoke')
            model_config=read(maps['config'])
            assert model_config['stage']=='development' and model_config['n_val']==357 and not model_config.get('max_val',0)
            assert model_config['variant'] in ('soft_seg','hard_seg','consistent_seg')
            assert sha(maps['checkpoint'])==maps['checkpoint_sha256']
            assert sha(maps_dir/'metadata.json')==map_status['metadata_sha256']
            protocol_path=ROOT/'precision_agriculture/RM_development_protocol_v1.json'
            review_path=ROOT/'precision_agriculture/RM_geometric_protocol_review.md'
            protocol=read(protocol_path);manifest_path=ROOT/'local_data_v2/val.json';manifest=read(manifest_path)
            manifest_hash=sha(manifest_path)
            assert protocol['image_degradation_factors']==list(R_VALUES) and protocol['management_grid_m']==list(M_VALUES)
            assert protocol['budgets']==list(BUDGETS) and protocol['grid_crs']=='EPSG:32649'
            assert protocol['manifest_sha256']==geometry['manifest_sha256']==maps['manifest_sha256']==manifest_hash
            assert geometry['protocol_sha256']==sha(protocol_path) and geometry['crs']=='EPSG:32649' and geometry['base_grid_m']==1
            assert sha(ROOT/'geo_management.py')==geometry['script_sha256']==sha(geometry_dir/'geo_management_source.py')
            for sid,source in geometry['sources'].items():
                assert source['tif_sha256']==maps['sources'][sid]['tif_sha256_from_dataset_summary']
                assert source['mask_sha256']==maps['sources'][sid]['mask_sha256_from_dataset_summary']
                assert np.allclose(source['transform'][:6],maps['sources'][sid]['affine'],rtol=0,atol=1e-15)
            ownership=check_ownership(manifest,maps)
            input_paths=[Path(__file__).resolve(),ROOT/'geo_management.py',protocol_path,review_path,manifest_path,
                         geometry_dir/'metadata.json',geometry_dir/'status.json',geometry_dir/'geo_management_source.py',
                         maps_dir/'metadata.json',maps_dir/'status.json',maps_dir/'verification.json',Path(maps['config'])]
            inputs={'file_sha256':{str(p):sha(p) for p in input_paths},'geometry_directory':str(geometry_dir),
                    'maps_directory':str(maps_dir),'checkpoint_sha256':maps['checkpoint_sha256'],
                    'model_source_sha256':maps['model_source_sha256'],'map_export_source_sha256':maps['export_source_sha256'],
                    'sources':geometry['sources'],'ownership':ownership,'protocol':protocol,
                    'no_inference':True,'no_test_or_R6':True}
            for p in [Path(__file__).resolve(),ROOT/'geo_management.py',protocol_path,review_path]:
                shutil.copy2(p,out/'source'/p.name)
            cells,values,integration,input_windows=integrate(geometry,maps,manifest,status_path,status)
            inputs['windows']=input_windows;write_json(out/'inputs.json',inputs)
            write_json(out/'integration_checks.json',integration)
            results=evaluate_tables(cells,values,protocol,out,status_path,status)
            for p in input_paths:assert sha(p)==inputs['file_sha256'][str(p)],f'Input changed during evaluation: {p}'
            summary={'state':'completed','created_at':stamp(),'model_run':maps['run_id'],'checkpoint_epoch':maps['checkpoint_epoch'],
                     'domain':{k:geometry[k] for k in ('native_valid_pixels','native_target_pixels','area_m2','target_m2','origin_utm_m','crs')},
                     'results':results,'integration':{k:v for k,v in integration.items() if k!='per_window'},
                     'ownership':ownership,'source_sha256':inputs['file_sha256'],
                     'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
                                 'geometry_versions':geometry['versions']},
                     'threads_max':2,'auc_policy':'right-continuous complete-prefix step integral; half interval normalized by .5',
                     'milp_area_policy':'primary maximum capture only; selected area retained for feasibility audit, never minimum-area claim',
                     'random_quantiles':'same-scene random-order variation, not confidence intervals',
                     'test_accessed':False,'R6_accessed':False,'model_tuned':False,'elapsed_seconds':time.time()-started}
            write_json(out/'summary.json',summary);make_report(summary,out)
            products=[p for p in out.iterdir() if p.is_file() and p.name!='status.json']
            status.update(state='completed',stage='complete',completed_at=stamp(),elapsed_seconds=time.time()-started,
                          files_sha256={p.name:sha(p) for p in products},units_by_M={m:g['units'] for m,g in results['by_M'].items()},
                          max_probability_integration_relative_error=integration['max_window_probability_relative_error'])
            write_json(status_path,status);print(json.dumps(status,ensure_ascii=True),flush=True)
            return 0
    except BaseException as error:
        status.update(state='failed',failed_at=stamp(),error=repr(error),traceback=traceback.format_exc())
        write_json(status_path,status);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir',default=str(ROOT/'precision_agriculture/geometry_20260916_v1'))
    parser.add_argument('--maps-dir',default=str(ROOT/'runs/dev_v2_03_softseg_aug_s17/validation_maps_20260916_013931_638980'))
    parser.add_argument('--wait-seconds',type=float,default=0)
    sys.exit(main(parser.parse_args()))
