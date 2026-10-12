"""Root acceptance of the reviewed preexecution interface/claim-only revision."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent;SEED=ROOT/'local_seed_stability_v1';PA=ROOT/'precision_agriculture'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
source=ROOT/'evaluate_local_seed_management.py';protocol=PA/'RM_local_seed_protocol_v1.json'
checks=ROOT/'provenance/local_seed_management_readiness_20260916_093300_444463/focused_checks.json'
assert sha(source)=='bac32dcaf83c1ecf0e7b7f6bb702c5ac1c3ff6bbea59a2839858b207582336c0'
assert sha(protocol)=='4b3bf3def97c97263329c13621aeb3b24ca928f4b421659608e24f3c62c4f125'
assert read(checks)['passed'] and read(checks)['implementation_sha256']==sha(source) and read(checks)['protocol_sha256']==sha(protocol)
review=SEED/'root_spatial_execution_review.json';old=read(review)
assert sha(review)=='153af9bafece4a561d147bda43b3b99e5b874e9f5541712972fff23e0aabcd40'
archive=ROOT/'provenance/seed_pipeline_preexecution_history_20260916_093149_768633/3_root_spatial_execution_review.json'
assert sha(archive)==sha(review)
old.update(created_at=datetime.datetime.now().astimezone().isoformat(),implementation_sha256=sha(source),protocol_sha256=sha(protocol),
           producer_readiness_path=str(checks),producer_readiness_sha256=sha(checks),supersedes_sha256=sha(archive),superseded_review_archive=str(archive))
old['root_static_review']+=['Reviewed final limited diff: selector receives a five-field child_inputs manifest with geometric area and hashes only; target totals remain in parent-only inputs.',
 'Claim serialization for parent and policy child moved inside failure handling after exclusive creation; partial claims remain and prevent retry. No scientific equations, thresholds, case order or solver settings changed.']
review.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'review_sha256':sha(review),'protocol_sha256':sha(protocol),'checks_sha256':sha(checks)}))
