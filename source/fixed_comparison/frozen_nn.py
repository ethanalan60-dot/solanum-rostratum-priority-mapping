"""Exact frozen inference functions; only R allowlist extends to 6. No data loader.
Always construct with pretrained=False; load the root-frozen selected checkpoint.
"""
import torch
from torch import nn
from torch.nn import functional as F
from frozen_resample import normalized_area_resize
from frozen_models import FeatureEncoder,build_model
PRETRAINED_SHA256={}
VALIDATION_R=(1,2,4,6)

def degrade_rgb(x, factor, valid):
    """Same-footprint valid-normalized RGB degradation then ImageNet normalization."""
    if factor not in VALIDATION_R:
        raise ValueError('Development RGB factors are restricted to 1,2,4')
    if factor != 1:
        small, small_valid = normalized_area_resize(x, valid, (x.shape[-2]//factor, x.shape[-1]//factor))
        numerator = F.interpolate(small*small_valid, size=x.shape[-2:], mode='bilinear', align_corners=False)
        denominator = F.interpolate(small_valid, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.where(denominator > 0, numerator/torch.where(denominator > 0, denominator, torch.ones_like(denominator)), torch.zeros_like(numerator))
    x = torch.where(valid > 0, x, torch.zeros_like(x))
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x-mean)/std

def coverage(probability, valid):
    denominator = valid.sum((1, 2, 3))
    numerator = (torch.where(valid > 0, probability, torch.zeros_like(probability))*valid).sum((1, 2, 3))
    return torch.where(denominator > 0, numerator/torch.where(denominator > 0, denominator, torch.ones_like(denominator)), torch.zeros_like(numerator))

def require_finite_forward(logits, model):
    # Sigmoid can hide +/-Inf logits; GradScaler does not protect BN buffers.
    values = [logits] + [buffer for buffer in model.buffers() if buffer.is_floating_point()]
    if not bool(torch.stack([torch.isfinite(value).all() for value in values]).all()):
        raise FloatingPointError('Nonfinite forward logits or model buffers; AMP skip is not applicable')

class CommonRegressionModel(nn.Module):
    """Direct window regression; intentionally no spatial decoder or inferred map."""
    def __init__(self, encoder, pretrained=True):
        super().__init__()
        self.encoder = FeatureEncoder(encoder, pretrained=pretrained,
                                      expected_sha256=PRETRAINED_SHA256[encoder] if pretrained else None)
        self.head = nn.Linear(self.encoder.channels[-1], 1)
        nn.init.constant_(self.head.bias, -2.5)

    @property
    def pretrained_report(self):
        return self.encoder.pretrained_report

    def forward(self, x, valid):
        features = self.encoder(x)[-1]
        # Features occupy the same nominal footprint; valid is an area weight,
        # not a claim that convolutional receptive fields exclude invalid pixels.
        _, weights = normalized_area_resize(valid, valid, features.shape[-2:])
        denominator = weights.sum((2, 3))
        pooled = (features.float()*weights).sum((2, 3))/denominator.clamp_min(torch.finfo(torch.float32).tiny)
        return self.head(pooled).squeeze(1)

def make_model(encoder, variant, pretrained=True):
    if variant == 'regression':
        return CommonRegressionModel(encoder, pretrained)
    return build_model(encoder, pretrained=pretrained,
                       expected_sha256=PRETRAINED_SHA256[encoder] if pretrained else None)
