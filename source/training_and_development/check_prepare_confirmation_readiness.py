"""Small synthetic metadata guards for the draft assembler; no real model/data."""
from pathlib import Path
import copy,datetime,hashlib,json,shutil
import prepare_confirmation_readiness as a
ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'provenance'/('readiness_assembler_checks_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();checks=[]
    def ok(name,fn):fn();checks.append({'name':name,'passed':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError,FileNotFoundError):checks.append({'name':name,'passed':True});return
        raise AssertionError(name+' accepted')
    w=a.Watch();sample=out/'sample.json';a.dump(sample,{'passed':True,'implementation_sha256':'a'*64})
    ok('explicit_json_hash',lambda:w.read(sample,a.sha(sample)))
    reject('wrong_json_hash',lambda:w.read(sample,'f'*64))
    reject('reserved_manifest_blocked_before_open',lambda:w.read(ROOT/'local_data_v2/test_sealed.json'))
    reject('development_cache_even_hash_blocked',lambda:w.pin(ROOT/'local_data_v2/cache/mock.npz'))
    reject('prediction_stage_metadata_blocked',lambda:w.read(ROOT/'confirmation_v1/stages/predict_mock/metadata.json'))
    reject('outside_campaign_root',lambda:w.pin(ROOT.parent/'outside.json'))
    ok('nested_field_path_explicit',lambda:a.require(a.field({'x':[{'sha':'a'*64}]},['x',0,'sha'])=='a'*64,'field'))
    reject('empty_field_path',lambda:a.field({'x':'a'*64},[]))
    reject('field_not_sha',lambda:a.field({'x':'passed'},['x']))
    module=out/'module.py';module.write_text('# synthetic\n',encoding='utf-8');rv=out/'review.json';a.dump(rv,{'passed':True,'implementation_sha256':a.sha(module)})
    entry={'path':str(rv),'sha256':a.sha(rv),'source_bindings':[{'file':'module.py','sha256_field':['implementation_sha256']}]}
    ok('exact_review_binding',lambda:a.validate_review(a.Watch(),entry,out,{'module.py'}))
    reject('unlisted_source_binding',lambda:a.validate_review(a.Watch(),entry,out,{'other.py'}))
    bad=copy.deepcopy(entry);bad['source_bindings']*=2
    reject('duplicate_source_field',lambda:a.validate_review(a.Watch(),bad,out,{'module.py'}))
    module.write_text('# changed synthetic\n',encoding='utf-8')
    reject('source_changed_after_review',lambda:a.validate_review(a.Watch(),entry,out,{'module.py'}))
    a.dump(rv,{'passed':False,'implementation_sha256':a.sha(module)});bad={**entry,'sha256':a.sha(rv)}
    reject('failed_independent_review',lambda:a.validate_review(a.Watch(),bad,out,{'module.py'}))
    # Complete37 table shape fixture: pointers and hashes are artificial, never opened.
    home=out/'virtual';cp=home/'checkpoints.candidate.json';ap=home/'training_audit.json'
    ids=['run%02d'%i for i in range(37)];roster=[{'run_id':rid,'dataset':'local' if i<28 else 'weeds','encoder':'synthetic','variant':'hard_seg','seed':17,'lr':.0001} for i,rid in enumerate(ids)]
    spec={'path':str(cp),'sha256':'c'*64};index={'path':str(out/'index.json'),'sha256':'i'*64};full={'path':str(out/'full.json'),'sha256':'f'*64}
    rows=[];proofs=[]
    for row in roster:
        rr={**row,'best_epoch':1,'original_selection_audited':True}
        for p in a.PREFIXES:rr[p+'_path']=str(ap if p=='training_audit' else home/row['run_id']/(p+'.json'));rr[p+'_sha256']='a'*64
        rows.append(rr);proofs.append({'run_id':row['run_id'],'best_epoch':1,'checkpoint_sha256':'a'*64,'status_sha256':'a'*64})
    objects={str(cp):{'complete':True,'frozen':False,'all37_training_audited':True,'models':rows},str(home/'artifact_manifest.json'):{'files_sha256':{'checkpoints.candidate.json':'c'*64,'training_audit.json':'a'*64}},
        str(home/'status.json'):{'state':'completed','mode':'full','audited':37,'pending':0,'training_audit_sha256':'a'*64},
        str(ap):{'passed':True,'mode':'full','all37_complete':True,'audited_count':37,'pending':[],'epochs_verified':888,'scales_verified':2664,'optimizer_events_verified':86136,'test_or_R6_accessed':False,'no_model_or_solver_calls':True,
                 'implementation_sha256':'b'*64,'protocol_sha256':'p'*64,'prerequisites':{'all16_numerical_accepted':True,'full24_summary_accepted':True,'evidence':{'seed_numerical_index':index,'full24_independent_review':full}},'runs':proofs}}
    class Fake:
        def pin(self,path,expected=None):return Path(path)
        def read(self,path,expected=None):return copy.deepcopy(objects[str(path)])
    ok('complete37_candidate_metadata',lambda:a.full_candidate(Fake(),spec,roster,index,full))
    for key,value in [('complete',False),('frozen',True),('all37_training_audited',False)]:
        old=objects[str(cp)][key];objects[str(cp)][key]=value
        reject('candidate_'+key,lambda:a.full_candidate(Fake(),spec,roster,index,full));objects[str(cp)][key]=old
    for key,value in [('mode','partial'),('audited_count',33),('epochs_verified',792),('optimizer_events_verified',0),('test_or_R6_accessed',True)]:
        old=objects[str(ap)][key];objects[str(ap)][key]=value
        reject('training_audit_'+key,lambda:a.full_candidate(Fake(),spec,roster,index,full));objects[str(ap)][key]=old
    reject('different_full16_index',lambda:a.full_candidate(Fake(),spec,roster,{**index,'sha256':'x'*64},full))
    reject('different_full24_review',lambda:a.full_candidate(Fake(),spec,roster,index,{**full,'sha256':'x'*64}))
    objects[str(cp)]['models'][-1]=copy.deepcopy(rows[0])
    reject('candidate_duplicate_replaces_missing_model',lambda:a.full_candidate(Fake(),spec,roster,index,full));objects[str(cp)]['models']=rows
    # Incomplete index fails without looking up any reviewed artifacts.
    class BadIndex:
        def read(self,*args):return {'all16_complete':False,'passed_for_registered_models':True,'accepted_count':10}
    reject('partial_numerical_index_is_not_complete',lambda:a.complete_index(BadIndex(),index,ids[:16]))
    class BadReview:
        def read(self,*args):return {'passed':True,'scope':'preparation_only'}
    reject('preparation_is_not_actual_full24_output_review',lambda:a.full24_review(BadReview(),full,ids[:24]))
    # Official files and reserved paths cannot be selected as output targets: CLI
    # has no output/promotion parameter and writes only a new provenance directory.
    import ast
    tree=ast.parse(Path(a.__file__).read_text(encoding='utf-8'));imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|{x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names}
    ok('stdlib_without_model_array_process_modules',lambda:a.require(not imports&{'torch','numpy','scipy','subprocess','pickle'},'Disallowed import'))
    source=Path(a.__file__).read_text(encoding='utf-8')
    ok('no_formal_FREEZE_or_checkpoint_writer',lambda:a.require("dump(CONF/" not in source and "--output" not in source and "SOURCE_LOCK.candidate.json" in source,'Official output path'))
    for file in (Path(__file__),Path(a.__file__),a.REGISTRY):shutil.copy2(file,out/file.name)
    report={'passed':True,'checks':checks,'count':len(checks),'assembler_sha256':a.sha(a.__file__),'registry_sha256':a.sha(a.REGISTRY),'checker_sha256':a.sha(__file__),
            'actual_candidate_or_reserved_data_read':False,'model_or_solver_or_subprocess':False,'actual_full16_or_full24_review_read':False,'official_files_written':False}
    a.dump(out/'focused_checks.json',report);print(json.dumps({'out':str(out),'passed':True,'checks':len(checks),'sha256':a.sha(out/'focused_checks.json')}))
if __name__=='__main__':main()
