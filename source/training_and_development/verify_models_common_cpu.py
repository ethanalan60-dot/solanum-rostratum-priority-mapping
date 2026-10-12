"""Synthetic CPU integration checks only. Never reads dataset manifests or GPU."""
from pathlib import Path
import ast, gc, hashlib, json, time, traceback
from datetime import datetime, timezone
import torch
from torch.nn import functional as F
from models_common import build_model, DEFAULT_CHECKPOINTS, sha256_file, DropPath, ConvNeXtV2Tiny, load_local_strict

ROOT = Path(__file__).resolve().parent

def compare_official_dense_definition():
    """Check reviewed pinned official dense source without installing its helpers."""
    source_path = ROOT/'vendor'/'convnextv2'/'original'/'models'/'convnextv2.py'
    if not source_path.is_file():
        return {'status':'blocked_original_source_download'}
    source = source_path.read_text(encoding='utf-8')
    allowed = (ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef)
    if any(not isinstance(node,allowed) for node in ast.parse(source).body):
        raise RuntimeError('Unexpected top-level statement in original source; review needed')
    replacements = {
        'from timm.models.layers import trunc_normal_, DropPath':
        'from torch.nn.init import trunc_normal_\nfrom models_common import DropPath',
        'from .utils import LayerNorm, GRN':''}
    for old,new in replacements.items():
        assert source.count(old)==1
        source = source.replace(old,new)
    namespace = {'__name__':'reviewed_official_dense_reference','torch':torch,'nn':torch.nn,'F':F}
    utilities_path = source_path.parent/'utils.py'
    utility_tree = ast.parse(utilities_path.read_text(encoding='utf-8'))
    classes = [n for n in utility_tree.body if isinstance(n,ast.ClassDef) and n.name in ('LayerNorm','GRN')]
    assert {n.name for n in classes}=={'LayerNorm','GRN'}
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(utilities_path),'exec'),namespace)
    exec(compile(source,str(source_path),'exec'),namespace)
    reference = namespace['convnextv2_tiny']().cpu().eval()
    adapted = ConvNeXtV2Tiny().cpu().eval()
    checkpoint = DEFAULT_CHECKPOINTS['convnextv2_tiny']
    actual_pretrained = checkpoint.is_file()
    loading = load_local_strict(reference,checkpoint) if actual_pretrained else None
    adapted.load_state_dict(reference.state_dict(),strict=True)
    x = torch.randn(1,3,64,80,device='cpu')
    with torch.inference_mode():
        a,b = adapted(x),reference(x)
        max_error = float((a-b).abs().max())
        assert torch.equal(a,b)
    result = {'status':'passed','source_sha256':sha256_file(source_path),
              'official_state_keys_exactly_equal':set(reference.state_dict())==set(adapted.state_dict()),
              'classifier_output_max_abs_error':max_error,'classifier_shape':list(a.shape),
              'reference_weights':'official_pretrained' if actual_pretrained else 'random_same_state',
              'official_pretrained_load':loading,
              'original_utils_sha256':sha256_file(utilities_path),
              'boundary':'Original dense body and original dense LayerNorm/GRN definitions; only trunc_normal_/DropPath import helpers are substituted. Default drop_path=0, checkpoint replaces initialization.'}
    del reference,adapted,a,b,x
    gc.collect()
    return result

def check_model(name):
    path = DEFAULT_CHECKPOINTS[name]
    pretrained = path.is_file()
    model = build_model(name,pretrained=pretrained).cpu()
    result = {'encoder':name,'pretrained_requested':pretrained,
              'official_pretrained_alignment':model.pretrained_report,
              'pretrained_check_status':'passed' if pretrained else 'blocked_checkpoint_download',
              'parameters_total':sum(p.numel() for p in model.parameters()),
              'parameters_encoder':sum(p.numel() for p in model.encoder.parameters()),
              'parameters_decoder':sum(p.numel() for p in model.decoder.parameters()),
              'shapes':[]}
    expected_channels = (64,128,256,512) if name=='resnet18' else (96,192,384,768)
    model.eval()
    with torch.inference_mode():
        for height,width in [(256,256),(64,80),(42,42)]:
            x = torch.randn(1,3,height,width,device='cpu')
            stages = model.encoder(x)
            assert tuple(f.shape[1] for f in stages) == expected_channels
            if height==256 and width==256:
                assert [tuple(f.shape[-2:]) for f in stages] == [(64,64),(32,32),(16,16),(8,8)]
            output = model(x,torch.ones(1,1,height,width))
            assert tuple(output.shape)==(1,1,height,width)
            assert torch.isfinite(output).all()
            result['shapes'].append({'input':list(x.shape),'stages':[list(f.shape) for f in stages],
                                     'output':list(output.shape),'finite':True})
    model.train()
    x = torch.randn(2,3,64,80,device='cpu')
    logits = model(x)
    valid = torch.ones_like(logits)
    valid[:,:,:,0:5] = 0
    prediction = ((logits.sigmoid()*valid).sum((1,2,3))/valid.sum((1,2,3)))
    loss = F.mse_loss(prediction,torch.tensor([.2,.6]))
    loss.backward()
    grads = {n:p.grad for n,p in model.encoder.named_parameters() if p.requires_grad}
    missing = [n for n,g in grads.items() if g is None]
    nonfinite = [n for n,g in grads.items() if g is not None and not torch.isfinite(g).all()]
    nonzero = [n for n,g in grads.items() if g is not None and g.abs().sum().item()>0]
    assert not missing and not nonfinite and len(nonzero)==len(grads)
    first = next(iter(grads))
    result['coverage_prediction_backward'] = {'passed':True,'loss':float(loss.detach()),
        'encoder_gradient_tensors':len(grads),'nonzero_tensors':len(nonzero),
        'missing':missing,'nonfinite':nonfinite,'first_parameter':first,
        'first_parameter_gradient_l1':float(grads[first].abs().sum())}
    del model,grads,stages,logits,prediction,loss,x
    gc.collect()
    return result

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260916)
    report = {'started_at':datetime.now(timezone.utc).isoformat(),'device':'cpu',
              'torch_version':torch.__version__,'threads':2,'gpu_accessed':False,
              'datasets_or_test_accessed':False,'module_sha256':sha256_file(ROOT/'models_common.py'),
              'models':[],'status':'running'}
    destination = ROOT/'model_integration_cpu_checks.json'
    started = time.monotonic()
    try:
        # Per-sample stochastic depth is unchanged in eval and unbiased by rescaling.
        drop = DropPath(.25).eval()
        probe = torch.ones(8,3,4,4)
        assert torch.equal(drop(probe),probe)
        drop.train()
        output = drop(probe)
        assert set(output.unique().tolist()).issubset({0.0,float(torch.tensor(1/.75))})
        report['drop_path_check'] = 'passed_row_mask_and_eval_identity'
        report['official_dense_reference'] = compare_official_dense_definition()
        for name in ['resnet18','convnextv2_tiny']:
            report['models'].append(check_model(name))
            destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(name+' CPU checks passed',flush=True)
        report['status'] = 'passed_with_pretrained_download_block' if any(
            x['pretrained_check_status']!='passed' for x in report['models']) else 'passed'
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    report['elapsed_seconds'] = time.monotonic()-started
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['status'],flush=True)
    if report['status']=='failed':
        raise SystemExit(1)
