"""Independent ResNet18 / ConvNeXt V2-T encoders and a shared segmentation head.

Importing this module performs no download, cache/environment change, or GPU work.
Inputs are already-normalized BCHW RGB tensors. Output is one-channel logits.

ConvNeXt V2 dense definitions adapted from Meta's MIT-licensed official code:
https://github.com/facebookresearch/ConvNeXt-V2/tree/2553895753323c6fe0b2bf390683f5ea358a42b9
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Changes: local dense-only LayerNorm/GRN; torch trunc_normal_ and per-sample
stochastic depth instead of timm; stage feature extraction; no sparse imports;
strict local-checkpoint loading. Architecture/parameter names are preserved.
The original license is recorded in vendor/convnextv2. Official pretrained
weights separately use CC-BY-NC-4.0; code permission is not a weight license.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parent
CONVNEXT_COMMIT = '2553895753323c6fe0b2bf390683f5ea358a42b9'
CONVNEXT_URL = 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt'
RESNET_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
DEFAULT_CHECKPOINTS = {
    'resnet18': ROOT/'cache'/'torch'/'hub'/'checkpoints'/'resnet18-f37072fd.pth',
    'convnextv2_tiny': ROOT/'vendor'/'convnextv2'/'weights'/'convnextv2_tiny_1k_224_ema.pt',
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class DropPath(nn.Module):
    """Drop whole residual paths independently per sample (timm row semantics)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError('drop_prob must be in [0,1)')
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        random_tensor = x.new_empty((x.shape[0],)+(1,)*(x.ndim-1)).bernoulli_(keep)
        return x * random_tensor.div_(keep)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError(data_format)
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x-u).pow(2).mean(1, keepdim=True)
        x = (x-u)/torch.sqrt(s+self.eps)
        return self.weight[:,None,None]*x + self.bias[:,None,None]


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1,1,1,dim))
        self.beta = nn.Parameter(torch.zeros(1,1,1,dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        nx = gx/(gx.mean(dim=-1, keepdim=True)+1e-6)
        return self.gamma*(x*nx)+self.beta+x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.grn = GRN(4*dim)
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0,2,3,1)
        x = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(x)))))
        return residual + self.drop_path(x.permute(0,3,1,2))


class ConvNeXtV2Tiny(nn.Module):
    """Dense official Tiny classifier layout; all checkpoint keys retained."""
    channels = (96,192,384,768)
    strides = (4,8,16,32)

    def __init__(self, num_classes=1000, drop_path_rate=0.0):
        super().__init__()
        self.depths = (3,3,9,3)
        dims = self.channels
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3,dims[0],kernel_size=4,stride=4),
                          LayerNorm(dims[0],eps=1e-6,data_format='channels_first'))])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i],eps=1e-6,data_format='channels_first'),
                nn.Conv2d(dims[i],dims[i+1],kernel_size=2,stride=2)))
        rates = torch.linspace(0,drop_path_rate,sum(self.depths)).tolist()
        self.stages = nn.ModuleList()
        cursor = 0
        for dim,depth in zip(dims,self.depths):
            self.stages.append(nn.Sequential(*[
                ConvNeXtV2Block(dim,rates[cursor+j]) for j in range(depth)]))
            cursor += depth
        self.norm = nn.LayerNorm(dims[-1],eps=1e-6)
        self.head = nn.Linear(dims[-1],num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            nn.init.trunc_normal_(module.weight,std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_stages(self,x):
        features = []
        for down,stage in zip(self.downsample_layers,self.stages):
            x = stage(down(x))
            features.append(x)
        return tuple(features)

    def forward_features(self,x):
        return self.norm(self.forward_stages(x)[-1].mean((-2,-1)))

    def forward(self,x):
        return self.head(self.forward_features(x))


def load_local_strict(model, checkpoint, *, expected_sha256=None):
    """Load a trusted local state dict with weights_only; never silently skip keys."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'Local pretrained checkpoint missing: {path}')
    digest = sha256_file(path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(f'Checkpoint SHA256 mismatch for {path}')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict):
        raise TypeError('Expected tensor dictionary or a model state dictionary wrapper')
    state = payload['model'] if isinstance(payload.get('model'),dict) else payload
    if not state or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise TypeError('Checkpoint state must contain only named tensors')
    # No prefix stripping, shape coercion or partial loading. PyTorch explicitly
    # migrates old BatchNorm checkpoints without num_batches_tracked counters.
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state))
    unexpected = sorted(set(state)-set(expected))
    mismatched = {k:{'expected':list(expected[k].shape),'actual':list(state[k].shape)}
                  for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape}
    legacy_bn = {name+'.num_batches_tracked' for name,module in model.named_modules()
                 if isinstance(module,nn.modules.batchnorm._BatchNorm)}
    if set(missing)-legacy_bn or unexpected or mismatched:
        raise RuntimeError(f'Strict state alignment failed: missing={missing}; unexpected={unexpected}; shapes={mismatched}')
    result = model.load_state_dict(state,strict=True)
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size,
            'state_tensors':len(state),'missing_keys':list(result.missing_keys),
            'unexpected_keys':list(result.unexpected_keys),'strict':True,
            'source_missing_legacy_batchnorm_counters':missing,
            'compatibility_note':'Missing legacy BatchNorm counters are migrated by PyTorch strict loader; all trainable tensors must align.' if missing else None}


class FeatureEncoder(nn.Module):
    def __init__(self,name,pretrained=False,checkpoint=None,expected_sha256=None,drop_path_rate=0.0):
        super().__init__()
        self.name = name
        if name == 'resnet18':
            if drop_path_rate != 0:
                raise ValueError('ResNet18 has no drop_path_rate; use0 for matched default')
            self.body = resnet18(weights=None)
            self.channels = (64,128,256,512)
        elif name == 'convnextv2_tiny':
            self.body = ConvNeXtV2Tiny(drop_path_rate=drop_path_rate)
            self.channels = self.body.channels
        else:
            raise ValueError(f'Unknown encoder: {name}')
        self.strides = (4,8,16,32)
        if checkpoint is not None and not pretrained:
            raise ValueError('checkpoint was provided but pretrained=False')
        self.pretrained_report = None
        if pretrained:
            self.pretrained_report = load_local_strict(
                self.body,checkpoint or DEFAULT_CHECKPOINTS[name],expected_sha256=expected_sha256)
        if name == 'resnet18':
            self.body.fc = nn.Identity()
        else:
            # Strict classifier-state validation happens BEFORE discarding unused head.
            self.body.norm = nn.Identity()
            self.body.head = nn.Identity()

    def forward(self,x):
        if self.name == 'convnextv2_tiny':
            return self.body.forward_stages(x)
        b = self.body
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        features = []
        for stage in (b.layer1,b.layer2,b.layer3,b.layer4):
            x = stage(x)
            features.append(x)
        return tuple(features)


class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.ReLU(inplace=True))

    def forward(self,x):
        return self.layers(x)


class CommonDecoder(nn.Module):
    """Identical four-stage head design for both encoders, independent of loss."""
    def __init__(self,channels,width=64,head_bias=-2.5):
        super().__init__()
        if len(channels) != 4 or width % 8:
            raise ValueError('Four feature stages and width divisible by8 are required')
        self.projections = nn.ModuleList([nn.Conv2d(c,width,1) for c in channels])
        self.fuse = ConvBlock(width*4,width)
        self.head = nn.Conv2d(width,1,1)
        nn.init.constant_(self.head.bias,head_bias)

    def forward(self,features,output_size):
        if len(features) != 4:
            raise ValueError('Expected four encoder stage tensors')
        size = features[0].shape[-2:]
        lateral = [F.interpolate(p(x),size=size,mode='bilinear',align_corners=False)
                   for p,x in zip(self.projections,features)]
        logits = self.head(self.fuse(torch.cat(lateral,dim=1)))
        return F.interpolate(logits,size=output_size,mode='bilinear',align_corners=False)


class CommonSegmentationModel(nn.Module):
    """No internal normalization or valid-aware architecture; mask losses externally.

    The optional valid argument is accepted for compatibility with train_local's
    segmentation call signature, but does not alter convolutional features.
    Supply already-normalized RGB using the same external valid-aware pipeline.
    """
    def __init__(self,encoder='resnet18',pretrained=False,checkpoint=None,
                 expected_sha256=None,width=64,head_bias=-2.5,drop_path_rate=0.0):
        super().__init__()
        self.encoder = FeatureEncoder(encoder,pretrained,checkpoint,expected_sha256,drop_path_rate)
        self.decoder = CommonDecoder(self.encoder.channels,width,head_bias)

    @property
    def pretrained_report(self):
        return self.encoder.pretrained_report

    def forward(self,x,valid=None):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 32:
            raise ValueError('Expected BCHW RGB with each spatial dimension >=32')
        if valid is not None and (valid.shape[0] != x.shape[0] or valid.shape[-2:] != x.shape[-2:]):
            raise ValueError('valid and input spatial/batch dimensions must match')
        return self.decoder(self.encoder(x),x.shape[-2:])


def build_model(encoder='resnet18',**kwargs):
    return CommonSegmentationModel(encoder=encoder,**kwargs)
