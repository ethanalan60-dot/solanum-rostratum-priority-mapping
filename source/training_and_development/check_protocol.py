"""Small checks of scientific invariants; no evaluation of sealed test scores."""
from pathlib import Path
import json
import numpy as np
import torch
from train_local import degrade,coverage,summarize,Model
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    train=json.loads((ROOT/'local_data_v2'/'train.json').read_text(encoding='utf-8'))
    val=json.loads((ROOT/'local_data_v2'/'val.json').read_text(encoding='utf-8'))
    for rows in [train,val]:assert len({r['id'] for r in rows})==len(rows)
    assert not ({r['id'] for r in train}&{r['id'] for r in val})
    for rows in [train,val]:
        for r in rows[::max(1,len(rows)//20)]:
            with np.load(r['cache']) as d:
                reconstructed=float((d['soft'].astype(float)*d['valid'].astype(float)).sum()/d['valid'].astype(float).sum())
                assert abs(reconstructed-r['coverage'])<2e-5,(r['id'],reconstructed,r['coverage'])
    x=torch.ones(1,3,32,32);v=torch.ones(1,1,32,32);v[:,:,:,17:]=0;x=x*v
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    max_error=0.
    for factor in [1,2,4,6]:
        rgb=degrade(x,factor,v)*std+mean
        err=float((rgb[v.expand_as(rgb)>0]-1).abs().max());max_error=max(max_error,err)
        assert err<1e-5,err
    for shape in ['hole','stripe','fractional','empty']:
        v=torch.ones(1,1,64,64)
        if shape=='hole':v[:,:,19:42,23:51]=0
        if shape=='stripe':v.zero_();v[:,:,:,31:33]=1
        if shape=='fractional':v[:,:,:,30:]=0;v[:,:,:,29]=.25
        if shape=='empty':v.zero_()
        x=torch.full((1,3,64,64),.7)*v.gt(0)
        noisy=torch.where(v>0,x,torch.rand_like(x))
        for factor in [1,2,4,6]:
            a=degrade(x,factor,v);b=degrade(noisy,factor,v)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert torch.allclose(a,b,atol=1e-5)
            restored=a*std+mean
            if v.gt(0).any():assert (restored[v.expand_as(restored)>0]-.7).abs().max()<1e-5
            if v.eq(0).any():assert restored[v.expand_as(restored)==0].abs().max()<1e-5
    m=Model(regression=False,pretrained=False).eval()
    with torch.inference_mode():out=m(torch.zeros(2,3,64,64),torch.ones(2,1,64,64))
    assert out.shape==(2,1,64,64)
    r=Model(regression=True,pretrained=False).eval()
    with torch.inference_mode():out=r(torch.zeros(2,3,64,64),torch.ones(2,1,64,64))
    assert out.shape==(2,)
    summary={'checks':'passed','train_locations':len(train),'validation_locations':len(val),'nodata_constant_color_max_error':max_error,'sealed_test_evaluated':False}
    (ROOT/'protocol_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))

if __name__=='__main__':main()
