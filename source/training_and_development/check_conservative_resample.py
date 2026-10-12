"""Scientific invariants and counterexample for noninteger area resampling."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from conservative_resample import area_resize, normalized_area_resize, overlap_weights

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(20260916)
    cases = []
    for height, width, target_h, target_w in [(512,512,256,256),(256,256,42,42),(37,53,23,17),(17,19,37,41),(5,7,1,1)]:
        for dtype in (torch.float32, torch.float64):
            value = torch.rand(2,3,height,width,generator=generator,dtype=dtype)
            valid = torch.rand(2,1,height,width,generator=generator,dtype=dtype)
            valid[valid < .3] = 0
            resized, weights = normalized_area_resize(value, valid, (target_h,target_w))
            expected = (value*valid).sum((-2,-1))/valid.sum((-2,-1))
            actual = (resized*weights).sum((-2,-1))/weights.sum((-2,-1))
            error = float((expected-actual).abs().max())
            valid_area_error = float((valid.mean((-2,-1))-weights.mean((-2,-1))).abs().max())
            tolerance = 2e-6 if dtype == torch.float32 else 1e-12
            assert error < tolerance and valid_area_error < tolerance
            constant, constant_weights = normalized_area_resize(torch.ones_like(value)*.37, valid, (target_h,target_w))
            expected_constant = torch.where(constant_weights > 0, torch.full_like(constant,.37), torch.zeros_like(constant))
            assert torch.allclose(constant, expected_constant, atol=tolerance, rtol=0)
            cases.append({"input":[height,width],"output":[target_h,target_w],"dtype":str(dtype),"coverage_error":error,"valid_area_error":valid_area_error})
    # Known counterexample: an impulse has mass 1/5 in a 5-cell domain.
    impulse = torch.tensor([0.,1.,0.,0.,0.],dtype=torch.float64).view(1,1,1,5)
    exact = area_resize(impulse,(1,3))
    adaptive = F.interpolate(impulse,size=(1,3),mode="area")
    assert abs(float(exact.mean())-.2) < 1e-12
    assert abs(float(adaptive.mean())-.2) > 1e-2
    x = torch.rand(1,1,8,10,generator=generator,dtype=torch.float64)
    assert torch.allclose(area_resize(x,(4,5)),F.avg_pool2d(x,2),atol=1e-12,rtol=0)
    empty, weights = normalized_area_resize(x,torch.zeros_like(x),(3,7))
    assert not empty.any() and not weights.any() and torch.isfinite(empty).all()
    # Arbitrary-grid averaging must remain differentiable for loss construction.
    gradient_input = x.clone().requires_grad_()
    area_resize(gradient_input,(3,7)).mean().backward()
    assert torch.allclose(gradient_input.grad,torch.ones_like(x)/x.numel(),atol=1e-12,rtol=0)
    fractional_half = torch.full((1,1,2,2),1e-4,dtype=torch.float16,requires_grad=True)
    half_value, _ = normalized_area_resize(fractional_half,fractional_half.detach(),(1,1))
    assert torch.allclose(half_value,fractional_half.float().mean().reshape(1,1,1,1),atol=1e-10,rtol=0)
    half_value.sum().backward()
    assert torch.isfinite(fractional_half.grad).all()
    invalid_nan = torch.tensor([float('nan'),.4,.4,.4]).view(1,1,2,2).requires_grad_()
    nan_valid = torch.tensor([0.,1.,1.,1.]).view(1,1,2,2)
    nan_output, _ = normalized_area_resize(invalid_nan,nan_valid,(1,1))
    assert torch.allclose(nan_output,torch.full_like(nan_output,.4),atol=1e-7,rtol=0)
    nan_output.sum().backward()
    assert torch.isfinite(invalid_nan.grad).all() and invalid_nan.grad.flatten()[0] == 0
    report={"status":"passed","test_dataset_accessed":False,"existing_runs_changed":False,
            "torch":torch.__version__,"cases":cases,
            "counterexample":{"input":impulse.flatten().tolist(),"true_mean":.2,
                              "overlap_output":exact.flatten().tolist(),"overlap_mean":float(exact.mean()),
                              "pytorch_area_output":adaptive.flatten().tolist(),"pytorch_area_mean":float(adaptive.mean())},
            "additional_checks":["float16 multiplication promoted before underflow", "invalid-domain NaN excluded before multiplication", "finite gradients for half and masked-NaN inputs"],
            "interpretation":"Cell-area conservation on a fixed footprint; not a claim about optical image formation."}
    (ROOT/'conservative_resample_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({"status":"passed","cases":len(cases),"counterexample":report['counterexample']}))


if __name__=='__main__':main()
