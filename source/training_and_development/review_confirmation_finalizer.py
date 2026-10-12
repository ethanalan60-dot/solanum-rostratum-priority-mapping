"""Independent sandbox-only checks of metadata finalization; never promotes real files."""
from pathlib import Path
import contextlib, datetime, hashlib, importlib.util, io, json, shutil, sys, types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
TARGET=ROOT/'finalize_confirmation_metadata.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def bind(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}

def main():
    start=sha(TARGET)
    out=ROOT/'provenance'/('confirmation_finalizer_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir();shutil.copy2(TARGET,out/'finalizer_reviewed.py');shutil.copy2(__file__,out/'review_source.py')
    spec=importlib.util.spec_from_file_location('finalizer_for_independent_mock',TARGET)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);core=module.core
    actual_validate_freeze=core.validate_freeze;actual_validate_source=core.validate_source_lock
    results=[];case_records=[]

    def fixture(name):
        home=out/'fixtures'/name;conf=home/'confirmation_v1';folder=home/'provenance'/'assembly'
        conf.mkdir(parents=True);folder.mkdir(parents=True)
        for file in ('producer.py','check_readiness_inventory.py'):(conf/file).write_text('# Synthetic only\n',encoding='utf-8')
        (conf/'INTERFACE.md').write_text('Synthetic interface only.\n',encoding='utf-8')
        roster=[]
        for dataset,n,seg in [('local',28,24),('weeds',9,6)]:
            for i in range(n):roster.append({'run_id':f'{dataset}_{i:02d}','dataset':dataset,'variant':'soft_seg' if i<seg else 'regression','seed':17,'encoder':'dummy','lr':0.0001})
        contract={'version':'confirmation_v1','R':[1,2,4,6],'intermediate_sizes':[256,128,64,42],'output_size':256,'CPU_threads':2,'batch_size':4,'dtype':'float32','roster':roster,'val_R6':False,'norm_diagnostics_included':False,'development_plans':[]}
        write(conf/'contract.json',contract)
        for file in ('extraction.json','external_arithmetic_extraction.json','scalar_baselines.json','checkpoint_candidate_protocol.json'):write(conf/file,{'synthetic':True})
        names={p.name for p in conf.glob('*.py')}|{p.name for p in conf.glob('*.md')}|{'contract.json','extraction.json','external_arithmetic_extraction.json','scalar_baselines.json','checkpoint_candidate_protocol.json'}
        lock={'version':'confirmation_source_v1','files':{n:sha(conf/n) for n in names},'external_dependency_sha256':{},'root_output_auditor_sha256':{}}
        write(conf/'SOURCE_LOCK.json',{'version':'old_synthetic','files':{}});write(conf/'checkpoints.json',{'frozen':False,'models':[]})
        candidate=home/'candidate.json';write(candidate,{'frozen':False,'complete':True,'all37_training_audited':True,'models':roster})
        write(folder/'SOURCE_LOCK.candidate.json',lock);write(folder/'readiness_evidence.draft.json',{'explicit_final_sources':{'synthetic':True}})
        report={'complete_for_root_review':True,'integrity_checks_passed':True,'pending':0,'failed':0,'counts':{'production_py':1,'reviewed_production_py':1,'reviewed_output_auditors':3,'accepted_seed_numerical_reviews':16},'source_inputs_sha256':{str(candidate):sha(candidate),str(conf/'contract.json'):sha(conf/'contract.json'),str(conf/'producer.py'):sha(conf/'producer.py')},'candidate_metadata':{'candidate':bind(candidate)}}
        write(folder/'assembly_report.json',report)
        write(folder/'artifact_manifest.json',{'files_sha256':{n:sha(folder/n) for n in ('assembly_report.json','SOURCE_LOCK.candidate.json','readiness_evidence.draft.json')}})
        return types.SimpleNamespace(home=home,conf=conf,folder=folder,candidate=candidate,report=report,commands=[],checkpoint_validations=0,freeze_gate_calls=0,old={n:sha(conf/n) for n in ('SOURCE_LOCK.json','checkpoints.json')})

    def reseal(f):
        write(f.folder/'artifact_manifest.json',{'files_sha256':{n:sha(f.folder/n) for n in ('assembly_report.json','SOURCE_LOCK.candidate.json','readiness_evidence.draft.json')}})

    def execute(f,kind='valid',args_mutator=None):
        argv=[str(TARGET),'--assembly-directory',str(f.folder),'--assembly-sha256',sha(f.folder/'assembly_report.json'),'--assembly-manifest-sha256',sha(f.folder/'artifact_manifest.json'),'--execute']
        if args_mutator:args_mutator(argv)
        def validate_entry(row):
            # Training-file verification is covered separately by the real inventory and builder.
            f.checkpoint_validations+=1
            if kind=='checkpoint_failure':raise ValueError('synthetic checkpoint verification failure')
            return row
        def freeze_gate():
            f.freeze_gate_calls+=1
            return actual_validate_freeze(f.conf)
        def child(command,**kwargs):
            f.commands.append(command)
            expected=[sys.executable,'-X','utf8','-B','-s',str(f.conf/'check_readiness_inventory.py'),'--evidence',str(f.folder/'readiness_evidence.draft.json'),'--require-ready']
            assert command==expected and kwargs['cwd']==f.home,'Unexpected subprocess command'
            assert not (f.conf/'FREEZE.json').exists(),'Child ran after FREEZE'
            p=f.home/'provenance'/'inventory';p.mkdir()
            inputs={'source_lock':bind(f.conf/'SOURCE_LOCK.json'),'checkpoint_table':bind(f.conf/'checkpoints.json'),'contract':bind(f.conf/'contract.json'),'evidence':bind(f.folder/'readiness_evidence.draft.json')}
            if kind=='foreign_inventory':
                other=f.home/'foreign_evidence.json';write(other,{'not_the_assembly':True});inputs['evidence']=bind(other)
            inv={'passed':kind!='inventory_failed','counts':{'checkpoint_sources_accepted':36 if kind=='count36' else 37,'new_seed_numerical_reviews_accepted':15 if kind=='count15' else 16},'inputs':inputs,'scope':'wrong_scope' if kind=='inventory_wrong_scope' else 'read_only_pre_freeze_inventory_not_execution_authorization','real_freeze_exists':kind=='inventory_already_frozen'}
            write(p/'inventory.json',inv)
            if kind=='inventory_missing_evidence':
                inv['inputs'].pop('evidence');write(p/'inventory.json',inv)
            if kind=='source_drift':(f.conf/'producer.py').write_text('# Drift after mocked inventory\n',encoding='utf-8')
            if kind=='contract_drift':write(f.conf/'contract.json',{'bad':True})
            if kind=='candidate_drift':write(f.candidate,{'changed':True})
            if kind=='evidence_drift':write(f.folder/'readiness_evidence.draft.json',{'changed':True})
            message={'out':str(p),'sha256':sha(p/'inventory.json')}
            kwargs['stdout'].write(json.dumps(message)+'\n')
            if kind=='two_messages':kwargs['stdout'].write(json.dumps(message)+'\n')
            return types.SimpleNamespace(returncode=2 if kind=='nonzero_child' else 0)
        real_dump=json.dump;real_copy=shutil.copy2;real_mkdir=Path.mkdir
        def dump(obj,stream,*a,**k):
            if kind=='claim_dump_failure' and Path(stream.name).name=='METADATA_PROMOTION_CLAIM.json':raise OSError('synthetic claim serialization failure')
            return real_dump(obj,stream,*a,**k)
        def copy(src,dst,*a,**k):
            if kind=='source_copy_failure' and Path(dst).name=='promote_source.py':raise OSError('synthetic source copy failure')
            return real_copy(src,dst,*a,**k)
        def mkdir(path,*a,**k):
            if kind=='out_mkdir_failure' and Path(path).name.startswith('confirmation_promotion_'):raise OSError('synthetic out mkdir failure')
            return real_mkdir(path,*a,**k)
        stream=io.StringIO();failure=None
        with patch.object(module,'ROOT',f.home),patch.object(module,'CONF',f.conf),patch.object(sys,'argv',argv),patch.object(core,'validate_checkpoint_entry',validate_entry),patch.object(core,'validate_source_lock',lambda home=None:actual_validate_source(f.conf)),patch.object(core,'validate_freeze',freeze_gate),patch.object(module.subprocess,'run',child),patch.object(json,'dump',dump),patch.object(shutil,'copy2',copy),patch.object(Path,'mkdir',mkdir),contextlib.redirect_stdout(stream):
            try:module.main()
            except BaseException as e:failure=repr(e)
        record={'name':f.home.name,'injection':kind,'error':failure,'claim_exists':(f.home/'confirmation_runtime/METADATA_PROMOTION_CLAIM.json').exists(),'freeze_exists':(f.conf/'FREEZE.json').exists(),'commands':f.commands,'checkpoint_validation_calls':f.checkpoint_validations,'freeze_gate_calls':f.freeze_gate_calls,'failure_receipts':[str(p.relative_to(f.home)) for p in list(f.home.rglob('failure.json'))+list(f.home.rglob('promotion_failure_*.json'))],'stdout':stream.getvalue()}
        case_records.append(record);return record

    def yes(name,condition):
        if not condition:raise AssertionError(name)
        results.append({'name':name,'passed':True})
    f=fixture('valid');r=execute(f)
    yes('valid_metadata_only_lifecycle',r['error'] is None and r['freeze_exists'] and len(r['commands'])==1)
    yes('valid_original_metadata_archived',all(sha(next((f.home/'provenance').glob('confirmation_promotion_*/previous_'+n)))==h for n,h in f.old.items()))
    yes('37_checkpoint_checks_before_and_after_freeze',r['checkpoint_validation_calls']==74 and r['freeze_gate_calls']==1)
    yes('valid_result_denies_started_predictions',read(next((f.home/'provenance').glob('confirmation_promotion_*/result.json')))['predictions_started'] is False)
    for kind in ('foreign_inventory','inventory_missing_evidence','inventory_failed','count36','count15','inventory_wrong_scope','inventory_already_frozen','nonzero_child','two_messages','source_drift','contract_drift','candidate_drift','evidence_drift','checkpoint_failure','claim_dump_failure','source_copy_failure','out_mkdir_failure'):
        f=fixture(kind);r=execute(f,kind)
        yes(kind+'_rejected_without_freeze',r['error'] is not None and not r['freeze_exists'])
        yes(kind+'_permanent_claim_and_receipt',r['claim_exists'] and bool(r['failure_receipts']))
        previous=len(r['commands']);second=execute(f,'valid')
        yes(kind+'_no_automatic_retry',second['error'] is not None and len(f.commands)==previous)
    for kind in ('bad_report_sha','bad_manifest_sha','missing_required_artifact','manifest_escape','incomplete_report','wrong_candidate_count','candidate_already_frozen','missing_source_file','existing_freeze','existing_claim'):
        f=fixture(kind);mutator=None
        if kind=='bad_report_sha':mutator=lambda a:a.__setitem__(a.index('--assembly-sha256')+1,'0'*64)
        elif kind=='bad_manifest_sha':mutator=lambda a:a.__setitem__(a.index('--assembly-manifest-sha256')+1,'0'*64)
        elif kind=='missing_required_artifact':
            m=read(f.folder/'artifact_manifest.json');m['files_sha256'].pop('readiness_evidence.draft.json');write(f.folder/'artifact_manifest.json',m)
        elif kind=='manifest_escape':
            other=f.folder.parent/'escape.json';write(other,{'outside_assembly':True});m=read(f.folder/'artifact_manifest.json');m['files_sha256']['../escape.json']=sha(other);write(f.folder/'artifact_manifest.json',m)
        elif kind=='incomplete_report':
            report=read(f.folder/'assembly_report.json');report['pending']=1;write(f.folder/'assembly_report.json',report);reseal(f)
        elif kind in ('wrong_candidate_count','candidate_already_frozen'):
            table=read(f.candidate)
            if kind=='wrong_candidate_count':table['models'].pop()
            else:table['frozen']=True
            write(f.candidate,table);report=read(f.folder/'assembly_report.json');report['candidate_metadata']['candidate']=bind(f.candidate);report['source_inputs_sha256'][str(f.candidate)]=sha(f.candidate);write(f.folder/'assembly_report.json',report);reseal(f)
        elif kind=='missing_source_file':
            lock=read(f.folder/'SOURCE_LOCK.candidate.json');lock['files'].pop('INTERFACE.md');write(f.folder/'SOURCE_LOCK.candidate.json',lock);reseal(f)
        elif kind=='existing_freeze':write(f.conf/'FREEZE.json',{'existing':True})
        elif kind=='existing_claim':
            (f.home/'confirmation_runtime').mkdir();write(f.home/'confirmation_runtime/METADATA_PROMOTION_CLAIM.json',{'prior':True})
        r=execute(f,args_mutator=mutator)
        yes(kind+'_rejects_before_subprocess',r['error'] is not None and not r['commands'])
        yes(kind+'_official_metadata_unchanged',all(sha(f.conf/n)==h for n,h in f.old.items()))
    yes('reviewed_source_unchanged',sha(TARGET)==start)
    # Static safety boundary: only one run() site and no data/model framework imports.
    import ast
    tree=ast.parse(TARGET.read_text(encoding='utf-8-sig'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess']
    yes('single_subprocess_run_callsite',len(calls)==1 and calls[0].func.attr=='run')
    yes('no_scientific_or_tensor_imports',not any((isinstance(n,ast.Import) and any(a.name.split('.')[0] in {'torch','numpy','scipy','rasterio','PIL'} for a in n.names)) or (isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] in {'torch','numpy','scipy','rasterio','PIL'}) for n in ast.walk(tree)))
    report={'passed':True,'scope':'independent_sandbox_finalizer_source_review_not_actual_promotion','implementation_sha256':start,'implementation_path':str(TARGET),'reviewer_sha256':sha(__file__),'checks':results,'check_count':len(results),'case_records':case_records,'actual_official_files_modified':False,'actual_subprocess_calls':0,'actual_test_or_model_access':False,'limitations':['Checkpoint payload validators and inventory subprocess were replaced with isolated metadata fixtures; full37 training integrity is bound to the separately reviewed builder/inventory.','All FREEZE-like files in this report are synthetic and reside under this provenance fixture directory.']}
    write(out/'review.json',report)
    write(out/'README.md.json',{'review':'Synthetic lifecycle and gate checks only. No official metadata promotion or reserved dataset/model execution.'})
    print(json.dumps({'out':str(out),'passed':True,'checks':len(results),'review_sha256':sha(out/'review.json'),'implementation_sha256':start},ensure_ascii=False))

if __name__=='__main__':main()
