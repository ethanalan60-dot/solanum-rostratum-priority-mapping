"""Isolated fake-filesystem lifecycle checks; no Torch/checkpoint/real NPZ use."""
from pathlib import Path
from contextlib import ExitStack, redirect_stdout, nullcontext
from unittest.mock import patch
import copy, io, json

def run_checks(x, out):
    checks=[];fixture_roots=[];plan=copy.deepcopy(x.read(x.SEED/'plan.json'));entry=copy.deepcopy(plan['runs'][2])
    real_source=Path(x.__file__)
    def fail(name,fn,error_type=Exception):
        try:fn()
        except error_type as error:checks.append(name);return error
        raise AssertionError('Unexpected acceptance: '+name)
    def fixture(label):
        # Preserve isolated fixtures on G using a short path; nested production-like
        # filenames can exceed Windows MAX_PATH under the long report directory.
        root=Path('./temporary_checks/local_seed_export_lifecycle_v2')/out.name/str(len(fixture_roots)+1)
        fixture_roots.append({'label':label,'root':str(root)})
        x.write(out/'lifecycle_fixture_roots.json',fixture_roots)
        seed=root/'local_seed_stability_v1';output=seed/'vx2'
        run=root/'runs'/entry['run_id'];run.mkdir(parents=True)
        cfg={**entry,'batch_size':4};state={'status':'completed'};original={'epoch':1}
        records=[{'id':f'FAKE{i:04d}'} for i in range(357)]
        def put(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value)+'\n',encoding='utf-8')
        for name in x.SOURCE_NAMES:put(root/name,{'synthetic_source':name})
        put(root/'export_common_validation.py',{'synthetic_source':'legacy'})
        for name in x.PINS:put(seed/name,{'synthetic_source':name})
        put(seed/'plan.json',plan)
        for name,value in [('config.json',cfg),('status.json',state),('best_validation.json',original),('best.pt',{'not_a_real_checkpoint':True})]:put(run/name,value)
        for name in ('val.json','DATASET_LOCK.json','manifest_summary.json'):put(root/'local_data_v2'/name,{'synthetic':name})
        receipt=seed/'completion_records'/(entry['run_id']+'.json');put(receipt,{'synthetic_completion':True})
        training_review=seed/'root_execution_review.json';put(training_review,{'synthetic_training_review':True})
        put(seed/'QUEUE_CLAIM.json',{'review_path':str(training_review),'review_sha256':x.sha(training_review)})
        checker=root/'check_local_seed_export_v2.py';put(checker,{'synthetic_checker':True})
        lifecycle_checker=root/'check_local_seed_export_lifecycle_v2.py';put(lifecycle_checker,{'synthetic_lifecycle_checker':True})
        protocol=seed/'export_v2'/'protocol.json';put(protocol,{'focuschecker_sha256':x.sha(checker),'lifecycle_checker_sha256':x.sha(lifecycle_checker)})
        hashes={str(p.resolve()):x.sha(p) for p in root.rglob('*') if p.is_file()}
        calls=[]
        def validate(run_id,*,load_model=False):
            assert run_id==entry['run_id']
            if load_model:raise AssertionError('Real/checkpoint loading forbidden by lifecycle fixture')
            return run,cfg,state,records,original,run/'best.pt',None,dict(hashes)
        def context():
            stack=ExitStack()
            for name,value in [('ROOT',root),('SEED',seed),('OUTPUT',output),('PROTOCOL',protocol),('validate_run',validate)]:
                stack.enter_context(patch.object(x,name,value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return stack
        def review(folder):
            focus=root/'review'/('focus_'+label+'.json')
            put(focus,{'passed':True,'exporter_sha256':x.sha(real_source),'export_protocol_sha256':x.sha(protocol),'checker_sha256':x.sha(checker),'lifecycle_checker_sha256':x.sha(lifecycle_checker)})
            grant=root/'review'/('grant_'+label+'.json')
            data={'passed':True,'execution_scope':'local_seed_validation_export_v2','run_id':entry['run_id'],
                'test_or_R6_authorized':False,'exporter_sha256':x.sha(real_source),'export_protocol_sha256':x.sha(protocol),
                'source_lock_sha256':x.PINS['SOURCE_LOCK.json'],'plan_sha256':x.PINS['plan.json'],
                'pre_execution_seal_sha256':x.sha(folder/'pre_execution_seal.json'),'focuschecks_path':str(focus),'focuschecks_sha256':x.sha(focus)}
            put(grant,data)
            return grant,data
        return dict(root=root,seed=seed,output=output,run=run,context=context,review=review,put=put,calls=calls,
                    cfg=cfg,state=state,records=records,original=original,hashes=hashes,protocol=protocol)

    f=fixture('archive_and_gate')
    with f['context']():
        prepared=x.prepare(entry['run_id']);inputs=x.read(prepared/'pre_execution_inputs.json')
        seal=x.read(prepared/'pre_execution_seal.json');assert seal['inference_started'] is False
        for relative,digest in seal['file_sha256'].items():assert x.sha(prepared/relative)==digest
        checks.append('mock prepare pre-execution seal hashes all archived files')
        copies=inputs['source_copy_bindings'];assert len(copies)==19 and set(copies)==set(inputs['source_copies'])
        assert len({v['archive_path'] for v in copies.values()})==19
        for name,binding in copies.items():
            assert x.source_key(binding['source_path'])==name
            assert Path(binding['archive_path']).is_relative_to(prepared/'source')
            assert x.sha(binding['source_path'])==x.sha(binding['archive_path'])==binding['sha256']
        checks.append('mock prepare unique19 original-to-archive maps and exact bytes')
        grant,good=f['review'](prepared)
        def prohibited_load():f['calls'].append('CPU_LOAD');raise AssertionError('Must reject before CPU dependency/model load')
        with patch.object(x,'dependencies',prohibited_load):
            mutations={'passed':False,'execution_scope':'other','run_id':'unknown','test_or_R6_authorized':True,
                'exporter_sha256':'0'*64,'export_protocol_sha256':'0'*64,'source_lock_sha256':'0'*64,
                'plan_sha256':'0'*64,'pre_execution_seal_sha256':'0'*64,'focuschecks_sha256':'0'*64}
            for key,value in mutations.items():
                bad={**good,key:value};f['put'](grant,bad)
                fail('execution gate before CPU load: '+key,lambda:x.execute(prepared,grant),ValueError)
                assert not f['calls'] and not (prepared/'EXECUTION_CLAIM.json').exists()
            f['put'](grant,good)
            accepted=x.check_execution_review(prepared,grant)
            assert accepted[str(grant)]==x.sha(grant);checks.append('mock good review exactly binds prepared seal and focuschecks')
            archived=Path(next(iter(copies.values()))['archive_path']);original=archived.read_bytes();archived.write_bytes(b'tampered')
            fail('prepared archived source tamper rejected before CPU load',lambda:x.execute(prepared,grant),ValueError)
            archived.write_bytes(original);assert not f['calls'] and not (prepared/'EXECUTION_CLAIM.json').exists()
        def initialization_failure():f['calls'].append('mock_init');raise RuntimeError('MOCK initialization after claim')
        with patch.object(x,'dependencies',initialization_failure):
            error=fail('postclaim initialization preserves failure',lambda:x.execute(prepared,grant),RuntimeError)
            assert str(error)=='MOCK initialization after claim'
            assert (prepared/'EXECUTION_CLAIM.json').exists() and x.read(prepared/'status.json')['status']=='failed'
            assert 'MOCK initialization' in x.read(prepared/'failure.json')['error']
            fail('failed execution cannot restart',lambda:x.execute(prepared,grant),ValueError)
            assert f['calls']==['mock_init']

    f=fixture('prepare_mkdir_failure')
    with f['context']():
        original_mkdir=Path.mkdir
        def fail_attempt_mkdir(path,*args,**kwargs):
            if path.parent==x.output_parent(entry['run_id']) and path.name.startswith('a20'):
                raise OSError('MOCK attempt directory inaccessible')
            return original_mkdir(path,*args,**kwargs)
        with patch.object(Path,'mkdir',fail_attempt_mkdir):
            error=fail('claimed prepare mkdir failure remains primary',lambda:x.prepare(entry['run_id']),OSError)
        assert str(error)=='MOCK attempt directory inaccessible'
        claim=x.claim_path(entry['run_id']);assert claim.exists()
        receipts=list((f['seed']/'export_v2'/'failure_receipts').rglob('failure.json'))
        assert len(receipts)==1 and 'MOCK attempt directory' in x.read(receipts[0])['error']
        assert x.read(receipts[0])['claim_retained'] is True;checks.append('inaccessible attempt has independent fallback failure receipt')
        secondaries=list((f['seed']/'export_v2'/'failure_receipts').rglob('finalization_failure.json'))
        assert len(secondaries)==1 and 'MOCK attempt directory' in x.read(secondaries[0])['primary_error']
        fail('mkdir failed prepared attempt cannot restart',lambda:x.prepare(entry['run_id']),FileExistsError)

    f=fixture('prepare_copy_failure')
    with f['context']():
        with patch.object(x.shutil,'copy2',side_effect=OSError('MOCK source archive copy failure')):
            error=fail('source-copy failure remains primary',lambda:x.prepare(entry['run_id']),OSError)
        claim=x.read(x.claim_path(entry['run_id']));folder=Path(claim['output'])
        assert x.read(folder/'status.json')['status']=='failed' and 'MOCK source archive' in x.read(folder/'failure.json')['error']
        assert (folder/'artifact_manifest.json').exists();checks.append('source-copy failure has failed status and retained manifest')
        fail('source-copy failed attempt cannot restart',lambda:x.prepare(entry['run_id']),FileExistsError)

    f=fixture('primary_and_finalization_failure')
    with f['context']():
        folder=x.prepare(entry['run_id']);grant,_=f['review'](folder)
        with patch.object(x,'dependencies',side_effect=RuntimeError('MOCK primary initialization error')), \
             patch.object(x,'artifact_manifest',side_effect=OSError('MOCK secondary manifest error')):
            error=fail('manifest failure cannot mask primary execution error',lambda:x.execute(folder,grant),RuntimeError)
        assert str(error)=='MOCK primary initialization error'
        assert 'MOCK primary' in x.read(folder/'failure.json')['error']
        secondary=x.read(folder/'finalization_failure.json')
        assert 'MOCK secondary' in secondary['error'] and 'MOCK primary' in secondary['primary_error']
        assert x.read(folder/'status.json')['status']=='failed';checks.append('primary and finalization failures both retained independently')

    f=fixture('successful_mock_then_manifest_failure')
    with f['context']():
        folder=x.prepare(entry['run_id']);grant,_=f['review'](folder)
        class FakeTorch:
            @staticmethod
            def set_num_threads(value):assert value==2
            @staticmethod
            def set_num_interop_threads(value):assert value==1
            @staticmethod
            def inference_mode():return nullcontext()
        def fake_validate(run_id,*,load_model=False):
            assert run_id==entry['run_id'] and load_model is True
            return f['run'],f['cfg'],f['state'],f['records'],f['original'],f['run']/'best.pt',object(),dict(f['hashes'])
        def fake_predictions(out,*args):
            x.write(out/'metadata.json',{'synthetic_only':True})
            return {},{'synthetic_only':True}
        with patch.object(x,'dependencies',return_value=None),patch.object(x,'torch',FakeTorch,create=True), \
             patch.object(x,'validate_run',fake_validate),patch.object(x,'export_predictions',fake_predictions), \
             patch.object(x,'artifact_manifest',side_effect=OSError('MOCK final manifest failure')):
            error=fail('finalization-only error prevents completed acceptance',lambda:x.execute(folder,grant),OSError)
        assert str(error)=='MOCK final manifest failure'
        assert x.read(folder/'status.json')['status']=='failed'
        assert x.read(folder/'finalization_failure.json')['primary_error'] is None
        checks.append('mock completed body downgraded to failed if artifact manifest cannot be written')
    f=fixture('prepare_claim_serialization_failure')
    with f['context']():
        with patch.object(x.json,'dump',side_effect=RuntimeError('MOCK prepare claim serialization failure')):
            fail('prepare claim serialization failure remains primary',lambda:x.prepare(entry['run_id']),RuntimeError)
        assert x.claim_path(entry['run_id']).exists()
        receipts=list((f['seed']/'export_v2'/'failure_receipts').rglob('failure.json'))
        assert len(receipts)==1 and 'claim serialization' in x.read(receipts[0])['error']
        checks.append('prepare claim serialization gets independent failure receipt')
        fail('failed prepare serialization never restarts',lambda:x.prepare(entry['run_id']),FileExistsError)

    f=fixture('execute_claim_serialization_failure')
    with f['context']():
        folder=x.prepare(entry['run_id']);grant,_=f['review'](folder)
        with patch.object(x.json,'dump',side_effect=RuntimeError('MOCK execution claim serialization failure')):
            fail('execute claim serialization failure remains primary',lambda:x.execute(folder,grant),RuntimeError)
        assert (folder/'EXECUTION_CLAIM.json').exists() and x.read(folder/'status.json')['status']=='failed'
        assert 'claim serialization' in x.read(folder/'failure.json')['error']
        checks.append('execute claim serialization retained before scientific dependencies')
        fail('failed execute serialization never restarts',lambda:x.execute(folder,grant),ValueError)
    return checks
