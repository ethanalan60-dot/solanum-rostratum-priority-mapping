from pathlib import Path
import hashlib,json,datetime
root=Path(__file__).resolve().parent;home=root/'local_seed_stability_v1/postprocess_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
driver=root/'run_local_seed_postprocess_queue_v2.py';protocol=home/'protocol.json'
audit=root/'provenance/seed_postprocess_v2_independent_20260916_100014_257144/review.json'
assert sha(driver)=='be53702860cd35418c832dd77f5ff7b736566589373ae868c4a523b318f9bfa7'
assert sha(protocol)=='027543f10bb1b7b0dfd2abde250c88e0f075c4c5bb66bfb0f640d566ebe51713'
assert sha(audit)=='3084cbeafe5501328d5806a32f0c1778613d4dbe6eeca0fe0600bb62c97f54ec'
review={'passed':True,'execution_scope':'local_seed_postprocess_queue_v2_remaining14','implementation_sha256':sha(driver),'protocol_sha256':sha(protocol),'independent_review_path':str(audit),'independent_review_sha256':sha(audit),'test_or_R6_authorized':False,'automatic_retry_authorized':False,'reviewer':'root','reviewed_at':datetime.datetime.now().astimezone().isoformat(),'root_review_scope':['Read all v1-to-v2 exporter/spatial/supervisor diffs','Only original slots03..16; first two complete outputs retained and independently recomputed','Original third preparation failed on long source-copy path before execution claim or model forward','Short path and fixed-length archive names retain source/checkpoint/geometry/selection definitions','Reviewed 128 exporter checks, 46 supervisor checks and independent path/claim/provenance audit','No original claim reset; one fixed new version after diagnosed non-numerical preparation failure'],'limitations':['Future production numerical outputs require separate acceptance','No model/epoch or case selection authorized from resulting metrics']}
with (home/'root_execution_review.json').open('x',encoding='utf-8') as f:json.dump(review,f,ensure_ascii=False,indent=2)
print(sha(home/'root_execution_review.json'))

