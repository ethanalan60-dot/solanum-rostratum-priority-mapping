"""Fetch pinned official ConvNeXt V2 sources/weights into this G: project only."""
from pathlib import Path
import hashlib, json, os, time, traceback
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'vendor' / 'convnextv2'
COMMIT = '2553895753323c6fe0b2bf390683f5ea358a42b9'
BASE = f'https://raw.githubusercontent.com/facebookresearch/ConvNeXt-V2/{COMMIT}/'
WEIGHTS = 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt'

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(data):
    temporary = DEST / 'provenance.json.tmp'
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, DEST / 'provenance.json')

def fetch(url, path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        record.update(status='existing_file', bytes=path.stat().st_size, sha256=sha(path))
        return
    partial = path.with_suffix(path.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() and record['kind']=='weight' else 0
    headers = {'User-Agent': 'Codex-authorized-research-download/1.0', 'Accept-Encoding':'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    # Use a process-local direct connection; the system proxy is not modified.
    session = requests.Session()
    session.trust_env = False
    record['transport'] = 'requests direct HTTPS; certificate verification enabled'
    with session.get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        record['http_status'] = response.status_code
        record['final_url'] = response.url
        record['etag'] = response.headers.get('ETag')
        record['last_modified'] = response.headers.get('Last-Modified')
        record['content_type'] = response.headers.get('Content-Type')
        if offset and response.status_code == 206:
            assert response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
            mode = 'ab'
        else:
            mode, offset = 'wb', 0
        length = response.headers.get('Content-Length')
        record['content_encoding'] = response.headers.get('Content-Encoding')
        expected = offset + int(length) if length and not record['content_encoding'] else None
        record.update(status='downloading', downloaded_bytes=offset, expected_bytes=expected)
        with open(partial, mode) as f:
            last = time.monotonic()
            for chunk in response.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
                    record['downloaded_bytes'] += len(chunk)
                if time.monotonic()-last > 10:
                    record['updated_at'] = now()
                    save(state)
                    print(json.dumps({'path':str(path), 'bytes':record['downloaded_bytes'], 'total':expected}), flush=True)
                    last = time.monotonic()
        if expected is not None:
            assert partial.stat().st_size == expected
        os.replace(partial, path)
    record.update(status='complete', bytes=path.stat().st_size, sha256=sha(path), completed_at=now())

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    state = {'repository':'https://github.com/facebookresearch/ConvNeXt-V2', 'commit':COMMIT,
             'started_at':now(), 'code_license':'MIT', 'weight_license':'CC-BY-NC-4.0',
             'pretraining':'ImageNet-1K FCMAE followed by ImageNet-1K supervised fine-tuning',
             'hash_scope':'Locally computed SHA256 for integrity and repeatability; publisher checksum not independently supplied.',
             'artifacts':[], 'status':'running'}
    pairs = [(BASE+x, DEST/'original'/x, 'source') for x in
             ['models/convnextv2.py', 'models/utils.py', 'LICENSE', 'README.md', 'INSTALL.md']]
    pairs.append((WEIGHTS, DEST/'weights'/'convnextv2_tiny_1k_224_ema.pt', 'weight'))
    for url, path, kind in pairs:
        record = {'url':url, 'path':str(path.relative_to(ROOT)), 'kind':kind, 'status':'pending'}
        state['artifacts'].append(record)
        save(state)
        try:
            fetch(url, path, record)
        except Exception as error:
            record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            state['status'] = 'failed'
            save(state)
            print(record['traceback'], flush=True)
            raise
        save(state)
    state.update(status='complete', completed_at=now())
    save(state)
    print('COMPLETE', flush=True)
