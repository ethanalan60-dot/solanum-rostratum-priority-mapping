"""Hash-verified CPU reader for the recorded external_weeds_v1 checkpoints.

API: load_checkpoint(path, expected_sha256) -> payload.
These original files are NOT torch.load(weights_only=True) compatible because
their config contains TorchVersion metadata. This isolated reader maps that
one metadata class to plain str and allows only the five observed reconstruction
GLOBALS. Unknown classes, extension opcodes and STACK_GLOBAL are rejected.
No torch/global allowlist is modified, no model is constructed or inferred.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib, io, pickle, pickletools, re, types, zipfile

ALLOWED_GLOBALS=frozenset({
    ('collections','OrderedDict'),('torch','FloatStorage'),('torch','LongStorage'),
    ('torch._utils','_rebuild_tensor_v2'),('torch.torch_version','TorchVersion'),
})
REJECTED_OPCODES=frozenset({'STACK_GLOBAL','EXT1','EXT2','EXT4','INST','OBJ'})

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def inspect_checkpoint(path,expected_sha256):
    path=Path(path).resolve()
    if not isinstance(expected_sha256,str) or re.fullmatch(r'[0-9a-fA-F]{64}',expected_sha256) is None:
        raise ValueError('An exact externally verified SHA256 is mandatory')
    if sha256_file(path)!=expected_sha256.lower():raise ValueError('Checkpoint SHA256 mismatch')
    with zipfile.ZipFile(path) as archive:
        members=[info for info in archive.infolist() if info.filename.endswith('/data.pkl')]
        if len(members)!=1 or members[0].file_size>4*1024*1024:raise ValueError('Expected one bounded torch data.pkl')
        data=archive.read(members[0])
    symbols=set()
    for opcode,value,position in pickletools.genops(data):
        if opcode.name in REJECTED_OPCODES:raise pickle.UnpicklingError('Unsupported pickle opcode: '+opcode.name)
        if opcode.name=='GLOBAL':
            module,name=value.split(' ',1)
            if (module,name) not in ALLOWED_GLOBALS:raise pickle.UnpicklingError('Unknown GLOBAL rejected: '+value)
            symbols.add((module,name))
    return {'path':str(path),'sha256':expected_sha256.lower(),'pickle_globals':[list(x) for x in sorted(symbols)],
            'TorchVersion_to_plain_str':('torch.torch_version','TorchVersion') in symbols,
            'torch_native_weights_only_compatibility_claim':False,'map_location':'cpu',
            'shared_torch_or_pickle_state_modified':False}

class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        key=(module,name)
        if key not in ALLOWED_GLOBALS:raise pickle.UnpicklingError('Unknown GLOBAL rejected: '+module+'.'+name)
        if key==('torch.torch_version','TorchVersion'):return str
        if key==('collections','OrderedDict'):return OrderedDict
        import torch
        if key==('torch._utils','_rebuild_tensor_v2'):return torch._utils._rebuild_tensor_v2
        if key==('torch','FloatStorage'):return torch.FloatStorage
        if key==('torch','LongStorage'):return torch.LongStorage
        raise pickle.UnpicklingError('Unimplemented allowed GLOBAL')

def _restricted_load(stream,**kwargs):return _RestrictedUnpickler(stream,**kwargs).load()
def _restricted_loads(data,**kwargs):return _restricted_load(io.BytesIO(data),**kwargs)

def load_checkpoint(path,expected_sha256):
    """Return verified checkpoint metadata/tensors on CPU; never use a permissive loader."""
    inspection=inspect_checkpoint(path,expected_sha256)
    module=types.ModuleType('external_weeds_restricted_pickle')
    module.Unpickler=_RestrictedUnpickler;module.load=_restricted_load;module.loads=_restricted_loads
    import torch
    # Custom pickle_module and weights_only=True cannot be combined by PyTorch.
    # False here selects this explicit restricted Unpickler, not the default one.
    payload=torch.load(inspection['path'],map_location='cpu',pickle_module=module,weights_only=False)
    if sha256_file(inspection['path'])!=inspection['sha256']:raise ValueError('Checkpoint changed while loading')
    if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict):raise ValueError('Expected model-state checkpoint dictionary')
    if not all(isinstance(k,str) and isinstance(v,torch.Tensor) and v.device.type=='cpu' for k,v in payload['model'].items()):
        raise ValueError('Expected named CPU tensors only in model state')
    return payload
