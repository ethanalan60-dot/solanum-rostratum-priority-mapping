"""Synthetic array tests only; never accesses reserved datasets or learned R6 maps."""
from pathlib import Path
import datetime,json
import numpy as np
from scipy.sparse import coo_matrix
import core,audit_prediction_geometry as a
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    checks=[]
    def yes(name,ok):assert ok,name;checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError,KeyError):yes(name,True);return
        raise AssertionError(name)
    p=np.stack([np.full((256,256),x,np.float32) for x in (.1,.2,.3,.4)])
    v=np.ones((256,256),np.float32);v[0,0]=0
    values=[float(x) for x in p[:,0,1]]
    call=lambda pp=p,vv=v,ss=values,ii=np.asarray('toy'),rr=np.array([1,2,4,6],np.int64):a.compare_map(pp,vv,ss,(1,2,4,6),'toy',ii,rr)
    yes('full4R weighted scalar agrees',call()<1e-12)
    reject('wrong scalar rejected',lambda:call(ss=[.1,.2,.3,.5]))
    reject('wrong identity rejected',lambda:call(ii=np.asarray('wrong')))
    reject('wrong R order rejected',lambda:call(rr=np.array([1,4,2,6],np.int64)))
    reject('float64 map rejected',lambda:call(pp=p.astype(np.float64)))
    reject('empty support rejected',lambda:call(vv=np.zeros_like(v)))
    pn=p.copy();pn[0,0,0]=np.nan;reject('NaN even outside valid support rejected',lambda:call(pp=pn))
    # Four distant native positions, including row boundary and opposite corner;
    # three management cells and a split native pixel form a nontrivial CSR.
    nr=np.array([0,1,1,2,2]);nc=np.array([0,0,513,1024,262143]);data=np.array([.1,.9,1.,1.,1.])
    native=coo_matrix((data,(nr,nc)),shape=(3,262144)).tocsr()
    cr=np.array([0,0,0,256,65535]);support=coo_matrix((data,(nr,cr)),shape=(3,65536)).tocsr()
    raw=np.zeros(262144);np.add.at(raw,nc,data);coarse=np.zeros(65536);np.add.at(coarse,cr,data)
    z=dict(cells=np.array([[-1,-1],[0,0],[1,1]],dtype=np.int64),area=np.asarray(native.sum(axis=1)).ravel(),
      weights=support.data,indices=support.indices,indptr=support.indptr,shape=np.array(support.shape),raw_valid_by_support=coarse,
      native_weights=native.data,native_indices=native.indices,native_indptr=native.indptr,native_shape=np.array(native.shape),raw_valid_by_native=raw)
    stats,valid=a.geometry_arrays(z)
    yes('native conservation',stats['native_valid_pixels']==4 and stats['area_m2']==4 and stats['coefficient_error']==0)
    yes('two native pixels in same support retain exact valid fraction',valid[0,0]==.5 and valid[1,0]==.25 and valid[-1,-1]==.25)
    reject('unexpected truth key rejected',lambda:a.geometry_arrays(dict(z,target=np.zeros(3))))
    bad=dict(z,raw_valid_by_support=coarse.copy());bad['raw_valid_by_support'][0]+=.01
    reject('coarse area drift rejected',lambda:a.geometry_arrays(bad))
    bad=dict(z,weights=support.data.copy());bad['weights'][0]+=.01
    reject('coefficient drift rejected',lambda:a.geometry_arrays(bad))
    bad=dict(z,cells=np.zeros((3,2),dtype=np.int64));reject('duplicate management ID rejected',lambda:a.geometry_arrays(bad))
    reject('missing actual freeze blocks real stage',lambda:a.execution_bindings('geometry',HERE/'absent_review.json'))
    out=ROOT/'provenance'/('confirmation_prediction_geometry_checks_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    core.write(out/'focused_checks.json',{'passed':True,'checks':checks,'implementation_sha256':core.sha(HERE/'audit_prediction_geometry.py'),'checker_sha256':core.sha(__file__),
      'actual_reserved_inputs_read':0,'actual_model_or_solver_calls':0,'actual_R6_forward':0,'formal_freeze_created':False})
    print(json.dumps({'passed':True,'checks':len(checks),'out':str(out)}))
if __name__=='__main__':main()
