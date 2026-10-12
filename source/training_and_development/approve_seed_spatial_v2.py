from pathlib import Path
import hashlib,json,datetime
root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
impl=root/'evaluate_local_seed_management_v2.py'
proto=root/'precision_agriculture/RM_local_seed_protocol_v2.json'
assert sha(impl)=='43ad50a4128ccec43e9d57f2d618f7e30b8084a029539c2de3232891bff04e66'
assert sha(proto)=='ea5f2504ae2bcebff603c691ccfa09c559f223e7fd3e37a0d0fcac8d04244fd8'
p=json.loads(proto.read_text(encoding='utf-8-sig'))
for path,digest in p['pinned_files_sha256'].items():assert sha(path)==digest,path
assert p['new_runs']==14 and len(p['declared_runs'])==14
grant={'passed':True,'execution_scope':'local_seed_management_v2_remaining14','implementation_sha256':sha(impl),'protocol_sha256':sha(proto),'test_or_R6_authorized':False,'root_reviewed_at':datetime.datetime.now().astimezone().isoformat(),'reviewer':'root','review_scope':['Read complete v1-to-v2 diff','Physical archive names shortened; logical origin and source hashes preserved','Only remaining original fourteen runs, first two results reused and third failed preparation retained','Original geometry integration, prefix, bound-consistency and label evaluation function bodies unchanged','Batch launch remains subject to final supervisor independent review'],'pinned_files_verified':len(p['pinned_files_sha256'])}
path=root/'local_seed_stability_v1/root_spatial_execution_review_v2.json'
with path.open('x',encoding='utf-8') as f:json.dump(grant,f,ensure_ascii=False,indent=2)
print(sha(path))

