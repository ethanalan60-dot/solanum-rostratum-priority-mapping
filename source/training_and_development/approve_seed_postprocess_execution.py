from pathlib import Path
import hashlib,json,datetime
root=Path(__file__).resolve().parent
home=root/'local_seed_stability_v1/postprocess_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
driver=root/'run_local_seed_postprocess_queue.py'
protocol=home/'protocol.json'
audit=root/'provenance/seed_postprocess_independent_20260916_093538_892753/review.json'
assert sha(driver)=='b1fedb740f9a420d9d1f7d7bbec8ebc2b870a38e2acdae82a00be85a183827c7'
assert sha(protocol)=='2cb9a9e9dbcf13cc0004110472869e0ac384f7617205e02386f2a695ed3e2974'
assert sha(audit)=='04401fc4df22f4dccf90cfd0a378f54852097291cab250250f1bc7a31745c2a9'
review={'passed':True,'execution_scope':'local_seed_postprocess_queue_v1_fixed16','implementation_sha256':sha(driver),'protocol_sha256':sha(protocol),'independent_review_path':str(audit),'independent_review_sha256':sha(audit),'test_or_R6_authorized':False,'automatic_retry_authorized':False,'reviewer':'root','reviewed_at':datetime.datetime.now().astimezone().isoformat(),'root_review_scope':['Read final complete supervisor source and independent 46-check receipt','Verified fixed sixteen declared completed-training eligibility and original epoch preservation','Reviewed permanent claims, first failure stop, owned-process cleanup and changing-source guards','CPU validation export then spatial evaluation only; no training changes or reserved confirmation'],'limitations':['Future real numerical outputs still require independent acceptance','No checkpoint or hyperparameter selection based on this validation postprocessing']}
with (home/'root_execution_review.json').open('x',encoding='utf-8') as f:json.dump(review,f,ensure_ascii=False,indent=2)
print(sha(home/'root_execution_review.json'))

