"""Independent synthetic review. Does not open any reserved input or saved prediction."""
from pathlib import Path
import ast,copy,datetime,hashlib,importlib.util,json,sys
import numpy as np
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parent
HOME=ROOT/'confirmation_v1'
SOURCE=HOME/'audit_prediction_geometry.py'
EXPECTED='7331845c41896f999b4088163ea3abd7bb7d079f42b132bb7ed73bb50e6636ce'
SELF=ROOT/'provenance/confirmation_prediction_geometry_checks_20260916_105017_314971/focused_checks.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    assert sha(SOURCE)==EXPECTED
    sys.path.insert(0,str(HOME))
    spec=importlib.util.spec_from_file_location('independent_audit_subject',SOURCE)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    checks=[]
    def yes(name,value):
        assert value,name
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError):checks.append(name);return
        raise AssertionError(name)

    # Independent scalar coordinate arithmetic, including row transitions and bottom-right.
    cols=[0,1,511,512,513,1023,1024,1025,261631,261632,262142,262143,513]
    rows=[0,0,0,1,1,1,2,2,2,2,1,1,0]
    weights=np.asarray([.25,.5,.75,1.,.125,.625,.375,.875,.25,.5,.75,1.,.875],np.float64)
    native=coo_matrix((weights,(rows,cols)),shape=(3,512*512)).tocsr()
    coarsecols=[]
    for j in cols:
        y,x=divmod(j,512)
        coarsecols.append((y//2)*256+x//2)
    coarse=coo_matrix((weights,(rows,coarsecols)),shape=(3,256*256)).tocsr()
    raw=np.zeros(512*512,np.float64);support=np.zeros(256*256,np.float64)
    for j,j2,w in zip(cols,coarsecols,weights):raw[j]+=w;support[j2]+=w
    z={'cells':np.asarray([[-5,-3],[0,1],[3,7]],np.int64),'area':np.asarray(native.sum(axis=1)).ravel(),
       'weights':coarse.data.copy(),'indices':coarse.indices.copy(),'indptr':coarse.indptr.copy(),'shape':np.asarray(coarse.shape,np.int64),
       'raw_valid_by_support':support,'native_weights':native.data.copy(),'native_indices':native.indices.copy(),
       'native_indptr':native.indptr.copy(),'native_shape':np.asarray(native.shape,np.int64),'raw_valid_by_native':raw}
    stats,valid=a.geometry_arrays(z)
    expectedvalid=np.zeros((256,256),np.float32)
    for j in set(cols):
        y,x=divmod(j,512);expectedvalid[y//2,x//2]+=.25
    yes('Independent divmod coordinate coarsening including row boundaries and split native pixel',stats['coefficient_error']==0)
    yes('Native row column and support area conserved independently',all(stats[k]==0 for k in ('row_error','native_column_error','support_area_error')))
    yes('Validity counts distinct native pixels not CSR entries',np.array_equal(valid,expectedvalid) and stats['native_valid_pixels']==len(set(cols)))
    yes('Total area matches independent Python sum',stats['area_m2']==sum(weights))
    for key in ('raw_valid_by_support','raw_valid_by_native'):
        for badvalue in (float('nan'),float('inf'),-1.):
            bad=copy.deepcopy(z);bad[key][123]=badvalue
            reject(f'{key} rejects {badvalue}',lambda bad=bad:a.geometry_arrays(bad))
    for key in ('area','weights','native_weights','raw_valid_by_support','raw_valid_by_native'):
        bad=copy.deepcopy(z);bad[key]=bad[key].astype(np.float32)
        reject(f'{key} requires float64',lambda bad=bad:a.geometry_arrays(bad))
    for key in ('indices','indptr','shape','native_indices','native_indptr','native_shape','cells'):
        bad=copy.deepcopy(z);bad[key]=bad[key].astype(np.float64)
        reject(f'{key} requires integer dtype',lambda bad=bad:a.geometry_arrays(bad))
    bad=copy.deepcopy(z);bad['weights'][0]+=.01
    reject('Wrong support coefficient rejected',lambda:a.geometry_arrays(bad))
    bad=copy.deepcopy(z);bad['native_indices'][1]=bad['native_indices'][0]
    reject('Noncanonical duplicate sparse column rejected',lambda:a.geometry_arrays(bad))
    bad=copy.deepcopy(z);bad['cells'][1]=bad['cells'][0]
    reject('Duplicate management coordinate rejected',lambda:a.geometry_arrays(bad))
    reject('Unexpected annotation array rejected',lambda:a.geometry_arrays(dict(z,target=np.zeros(3))))

    rr=np.asarray([1,2,4,6],np.int64);identifier='synthetic-only'
    ramp=np.arange(256*256,dtype=np.float32).reshape(256,256)/np.float32(256*256)
    p=np.stack([ramp*.5,ramp*.75,1-ramp*.5,np.full_like(ramp,.375)])
    v=np.zeros_like(ramp);v[:3,:5]=np.asarray([0,.25,.5,.75,1],np.float32)
    q=[sum(float(p[k,y,x])*float(v[y,x]) for y in range(3) for x in range(5))/sum(float(v[y,x]) for y in range(3) for x in range(5)) for k in range(4)]
    call=lambda pp=p,vv=v,qq=q,ii=np.asarray(identifier),r=rr,t=1e-6:a.compare_map(pp,vv,qq,rr,identifier,ii,r,t)
    yes('Map integral equals independent scalar Python loop',call()<1e-15)
    near=np.asarray(q)+5e-7
    yes('Local map scalar precision tolerance remains explicit',call(qq=near)<1e-6)
    reject('External tighter precision rejects same residual',lambda:call(qq=near,t=1e-7))
    reject('Map R integer dtype enforced',lambda:call(r=rr.astype(np.float64)))
    reject('Map R order enforced',lambda:call(r=rr[::-1]))
    reject('Map member enforced',lambda:call(ii=np.asarray('other')))
    reject('Map member scalar shape enforced',lambda:call(ii=np.asarray([identifier])))
    reject('Zero valid support rejected',lambda:call(vv=np.zeros_like(v)))
    for name,array in (('valid',v),('probability',p)):
        bad=array.copy();bad.flat[-1]=np.nan
        reject(f'Nonfinite {name} rejected also outside support',lambda bad=bad,name=name:call(vv=bad) if name=='valid' else call(pp=bad))
    reject('Map shape enforced',lambda:call(pp=p[:3]))

    model={'run_id':'synthetic_run','dataset':'local','encoder':'resnet18','variant':'soft_seg','seed':17,'best_epoch':9,'checkpoint_sha256':'c'*64,'config_sha256':'d'*64}
    ib={'path':'synthetic/input_manifest.json','sha256':'a'*64};ab={'path':'synthetic/artifact_manifest.json','sha256':'b'*64};freeze='f'*64
    meta=dict(model,member_ids=[identifier],R=[1,2,4,6],input_manifest=ib,freeze_sha256=freeze,truth_read_by_prediction=False,checkpoint_reselected=False,CPU_dtype='float32',batch_size=4,CPU_threads=2,scalar_rows=4,map_count=1)
    seal=dict(input_manifest=ib,input_artifact_manifest=ab,checkpoint={'sha256':model['checkpoint_sha256']},config={'sha256':model['config_sha256']},original_best_epoch=9,truth_read_by_prediction=False,checkpoint_reselected=False)
    a.validate_model_provenance(model,meta,seal,[identifier],ib,ab,freeze);checks.append('Exact synthetic prediction provenance accepted')
    for key,value in (('member_ids',[]),('R',[1,2,4]),('input_manifest',ab),('freeze_sha256','0'*64),('truth_read_by_prediction',True),('checkpoint_reselected',True),('CPU_dtype','float16'),('batch_size',8),('CPU_threads',4),('scalar_rows',3),('map_count',0),('best_epoch',8)):
        bad=copy.deepcopy(meta);bad[key]=value
        reject(f'Model metadata rejects changed {key}',lambda bad=bad:a.validate_model_provenance(model,bad,seal,[identifier],ib,ab,freeze))
    for key,value in (('input_manifest',ab),('input_artifact_manifest',ib),('checkpoint',{'sha256':'0'*64}),('config',{'sha256':'0'*64}),('original_best_epoch',8),('truth_read_by_prediction',True),('checkpoint_reselected',True)):
        bad=copy.deepcopy(seal);bad[key]=value
        reject(f'Start seal rejects changed {key}',lambda bad=bad:a.validate_model_provenance(model,meta,bad,[identifier],ib,ab,freeze))
    folder=ROOT/'provenance'/'synthetic_not_created'
    record=dict(id=identifier,input_cache_sha256='1'*64,source_id='fake_source',native_window_xywh=[1,2,512,512],valid_semantics='alpha')
    row=dict(record,path=str(folder/'maps'/(identifier+'.npz')),run_id=model['run_id'],dataset='local',R=[1,2,4,6],shape=[4,256,256],dtype='float32')
    a.validate_map_provenance(row,model,record,folder);checks.append('Exact synthetic map provenance accepted')
    for key,value in (('path',str(folder/'outside.npz')),('run_id','wrong'),('dataset','weeds'),('id','wrong'),('R',[1,2,4]),('shape',[4,128,128]),('dtype','float16'),('input_cache_sha256','0'*64),('source_id','wrong'),('native_window_xywh',[0,0,512,512]),('valid_semantics','all_ones')):
        bad=copy.deepcopy(row);bad[key]=value
        reject(f'Map metadata rejects changed {key}',lambda bad=bad:a.validate_map_provenance(bad,model,record,folder))

    # Source-level interface review, without calling any production stage or loading FREEZE/test inputs.
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name,helper in (('audit_predictions','validate_model_provenance'),('audit_predictions','validate_map_provenance')):
        yes(f'Production {name} calls checked {helper}',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==helper for n in ast.walk(funcs[name])))
    execution=ast.unparse(funcs['execute'])
    yes('Execution binds formal gate before attempt and before arithmetic',execution.index('execution_bindings(')<execution.index('core.attempt(')<execution.index('audit_predictions if'))
    yes('Execution rechecks watched identities and source lock before final receipt','for path, digest in watched.items()' in execution and execution.index('core.validate_source_lock()')<execution.index("core.write(out / 'review.json'"))
    queue=(HOME/'run_selection_queue.py').read_text(encoding='utf-8')
    source=SOURCE.read_text(encoding='utf-8')
    for key in ('confirmation_predictions_independent','confirmation_geometry_independent','all_predictions_sha256','metadata_sha256'):
        yes('Selection queue and auditor share '+key,key in source and key in queue)
    yes('Original root selfcheck is bound to exact reviewed source',json.loads(SELF.read_text(encoding='utf-8'))['implementation_sha256']==EXPECTED)
    yes('Reviewed source unchanged after independent tests',sha(SOURCE)==EXPECTED)
    out=ROOT/'provenance'/('confirmation_prediction_geometry_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    report={'passed':True,'scope':'preparation_synthetic_independent_review','implementation_sha256':EXPECTED,'reviewer_sha256':sha(__file__),
      'root_selfcheck':{'path':str(SELF),'sha256':sha(SELF)},'selection_queue':{'path':str(HOME/'run_selection_queue.py'),'sha256':sha(HOME/'run_selection_queue.py')},
      'checks':checks,'check_count':len(checks),'geometry_fixture':stats,'actual_reserved_arrays_read':0,'actual_model_forwards':0,'actual_solver_calls':0,'actual_R6_forward':0,'formal_freeze_created':False,
      'limits':['Synthetic and static preparation review; not an audit of any real confirmation outputs.','Full 37-model/338-window execution and formal production grant are deliberately not invoked.','Independent geometry arithmetic checks saved sparse conservation and support mapping; it is not a land survey or independent reprojection.']}
    dump(out/'review.json',report)
    (out/'review.md').write_text('# Independent preparation review: prediction and geometry audit\n\nPASS, '+str(len(checks))+' focused checks. No blocking issue remains in the reviewed source.\n\nThe native-column mapping was independently reconstructed with Python divmod(row, 512), including adjacent native rows, image corners and a native pixel split across management cells. Sparse coefficient/row/column conservation and validity fractions agreed exactly. Artificial metadata mutations covered model epoch, source identity, input bindings, precision, membership and phase. NaN, wrong dtype, noncanonical CSR, changed coefficients and unsupported map identities were rejected.\n\nBegin/end binding and the two consumer scopes were reviewed statically. No real confirmation data, model or solver was accessed; production remains gated by formal FREEZE and root authorization.\n',encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'out':str(out),'review_sha256':sha(out/'review.json')}))
if __name__=='__main__':main()
