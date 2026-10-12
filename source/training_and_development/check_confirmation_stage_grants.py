"""Synthetic-only complete grant-schema checks; no production source entrypoint."""
from pathlib import Path
from unittest.mock import patch
import argparse,ast,contextlib,copy,datetime,io,json,sys,traceback
import prepare_confirmation_stage_grants as g

ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_grant_checks_%Y%m%d_%H%M%S_%f');out.mkdir();root=out/'fixture';home=root/'confirmation_v1';home.mkdir(parents=True);(root/'provenance').mkdir()
    checks=[];saved={};opened=[]
    def yes(name,value=True):assert value,name;checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError,AssertionError):yes(name);return
        raise AssertionError('Accepted '+name)
    def pin(p):return {'path':str(p.resolve()),'sha256':g.digest(p)}
    def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);g.dump(p,obj);return pin(p)
    def stage(name,files):
        p=home/'stages'/name;p.mkdir(parents=True);write(p/'status.json',{'state':'completed'})
        for n,v in files.items():
            q=p/n;q.parent.mkdir(parents=True,exist_ok=True)
            if n.endswith('.json'):g.dump(q,v)
            else:q.write_text(str(v),encoding='ascii')
        write(p/'artifact_manifest.json',{'files':{q.relative_to(p).as_posix():g.digest(q) for q in p.rglob('*') if q.is_file()}});return p
    def approved(stage,evidence,name):
        _,draft,_=g.build(stage,evidence,root);yes(stage+'_draft_is_not_authorization',draft['passed'] is False and draft['draft'] is True)
        return write(home/'synthetic_reviews'/name,{**draft,'passed':True,'draft':False,'synthetic_root_action':True})
    def build(name,evidence):
        m,draft,commands=g.build(name,evidence,root);yes(name+'_schema_built',draft['scope']==g.PROGRAM[name][1] and not draft['passed']);m.finish();saved[name]={'draft':draft,'commands':commands};return draft
    try:
        yes('stdlib_import_only',not any(x in sys.modules for x in ('numpy','torch','scipy')))
        reject('no_freeze_no_draft',lambda:g.prepare('prediction_queue',{},root));yes('no_output_when_freeze_absent',not list((root/'provenance').iterdir()))
        models=[]
        for e in ('resnet18','convnextv2_tiny'):
            for lr in (.0001,.0003):
                for variant in ('regression','hard_seg','soft_seg'):
                    for seed in ((17,) if variant=='regression' else (17,29,43)):
                        models.append(dict(run_id=f'L{len(models):02}',dataset='local',encoder=e,lr=lr,variant=variant,seed=seed,best_epoch=7,checkpoint_sha256='c'*64))
        for variant in ('regression','hard_seg','soft_seg'):
            for seed in (17,29,43):models.append(dict(run_id=f'W{len(models):02}',dataset='weeds',encoder='resnet18',lr=.0001,variant=variant,seed=seed,best_epoch=4,checkpoint_sha256='d'*64))
        for script,_ in g.PROGRAM.values():
            p=(home/script).resolve();p.write_text('# artificial source identity; never executed\n',encoding='ascii')
        (home/'external_arithmetic.py').write_text('# synthetic arithmetic identity only\n',encoding='ascii')
        c=dict(version='confirmation_v1',R=g.RS,intermediate_sizes=[256,128,64,42],output_size=256,CPU_threads=2,batch_size=4,dtype='float32',val_R6=False,norm_diagnostics_included=False,roster=models,datasets={'local':{'count':338},'weeds':{'count':26}},development_plans=[])
        write(home/'contract.json',c);write(home/'checkpoints.json',{'frozen':True,'models':models});write(home/'scalar_baselines.json',{'synthetic':True})
        write(home/'SOURCE_LOCK.json',{'version':'confirmation_source_v1','files':{p.name:g.digest(p) for p in home.glob('*.py')}})
        write(home/'FREEZE.json',dict(passed=True,frozen=True,scope='confirmation_v1',source_lock_sha256=g.digest(home/'SOURCE_LOCK.json'),contract_sha256=g.digest(home/'contract.json'),checkpoints_sha256=g.digest(home/'checkpoints.json'),R=g.RS,run_ids=[x['run_id'] for x in models],truth_after_all_predictions_and_selections=True))
        fsha=g.digest(home/'FREEZE.json')
        # Trace consumption. Any tensor/array/CSV-value or truth-manifest read is a test failure.
        original_open=Path.open
        def guarded(path,*args,**kwargs):
            mode=args[0] if args else kwargs.get('mode','r')
            if 'r' in mode:
                require=path.suffix.lower() not in ('.npz','.pt','.tif','.png','.csv') and path.name!='truth_manifest.json'
                assert require,'Prohibited payload open: '+str(path);opened.append(str(path))
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            build('prediction_queue',{});result=g.prepare('prediction_queue',{},root)
            yes('real_prepare_function_only_writes_artificial_draft',json.loads((result/'root_review.draft.json').read_text())['passed'] is False)
        original_contract=(home/'contract.json').read_bytes();original_dump=g.dump
        def drift_after_commands(path,value):
            original_dump(path,value)
            if Path(path).name=='commands.json':original_dump(home/'contract.json',{'drift':'ARTIFICIAL ONLY'})
        with patch.object(g,'dump',side_effect=drift_after_commands):
            reject('postwrite_metadata_drift_preserves_failed_draft',lambda:g.prepare('prediction_queue',{},root))
        failed=[p for p in (root/'provenance').iterdir() if (p/'failure.json').exists()]
        yes('failed_draft_has_no_passed_preparation',len(failed)==1 and json.loads((failed[0]/'preparation.json').read_text())['preparation_passed'] is False and json.loads((failed[0]/'root_review.draft.json').read_text())['passed'] is False)
        (home/'contract.json').write_bytes(original_contract)
        evidence_path=home/'synthetic_evidence.json';evidence_pin=write(evidence_path,{})
        with patch.object(Path,'open',guarded):
            prepared=g.prepare('prediction_queue',{},root,evidence_file=evidence_pin)
            yes('evidence_file_pin_retained',str(evidence_path.resolve()) in json.loads((prepared/'preparation.json').read_text())['watched_metadata_sha256'])
        (home/'claims').mkdir();write(home/'claims/prediction_queue.json',{'synthetic':True});reject('existing_prediction_claim_refused',lambda:g.build('prediction_queue',{},root));(home/'claims/prediction_queue.json').unlink()
        for ds in ('local','weeds'):stage('inputs_'+ds,{'input_manifest.json':{'dataset':ds,'synthetic':True},'cache/NOT_REAL.npz':'DO NOT OPEN'})
        meta=g.Metadata(root);meta.stage('inputs_local');extra=home/'stages/inputs_local/EXTRA.txt';extra.write_text('SYNTHETIC EXTRA')
        reject('postvalidation_stage_inventory_drift_rejected',meta.finish);extra.unlink()
        build('local_geometry',{})
        stage('local_geometry',{'metadata.json':dict(geometry_kind='native_alpha_only',split='test_sealed',complete_declared_domain=True,expected_windows=338,windows=[{'id':str(i)} for i in range(338)],annotation_arrays_opened=0,mask_paths_consumed=False),'area_units.csv':'DO NOT READ VALUES'})
        build('audit_geometry',{});entries=[]
        for model in models:
            p=stage('predict_'+model['run_id'],{'metadata.json':{'synthetic':True},'map_index.json':[],'predicted_scalars.csv':'DO NOT READ VALUES','maps/NOT_REAL.npz':'DO NOT OPEN'})
            n=c['datasets'][model['dataset']]['count'];entries.append(dict(run_id=model['run_id'],dataset=model['dataset'],variant=model['variant'],stage=str(p),scalar_rows=4*n,map_count=0 if model['variant']=='regression' else n,**{k:pin(p/(k+('.csv' if k=='predicted_scalars' else '.json'))) for k in ('artifact_manifest','metadata','map_index','predicted_scalars')}))
        stage('predictions_seal',{'all_predictions.json':dict(freeze_sha256=fsha,run_ids=[x['run_id'] for x in models],models=entries,all_declared_models_completed=True,scalar_rows=38792,map_count=8268)})
        predsha=g.digest(home/'stages/predictions_seal/all_predictions.json');geosha=g.digest(home/'stages/local_geometry/metadata.json')
        build('audit_predictions',{})
        stage('audit_predictions',{'review.json':dict(passed=True,scope='confirmation_predictions_independent',implementation_sha256=g.digest(home/'audit_prediction_geometry.py'),freeze_sha256=fsha,all_predictions_sha256=predsha)})
        stage('audit_geometry',{'review.json':dict(passed=True,scope='confirmation_geometry_independent',implementation_sha256=g.digest(home/'audit_prediction_geometry.py'),freeze_sha256=fsha,metadata_sha256=geosha)})
        pr=pin(home/'stages/audit_predictions/review.json');gr=pin(home/'stages/audit_geometry/review.json');ev={'predictions_review':pr,'geometry_review':gr}
        with patch.object(Path,'open',guarded):
            build('select_local',ev);build('select_external',ev)
            lr=approved('select_local',ev,'lr.json');er=approved('select_external',ev,'er.json');build('selection_queue',{**ev,'local_review':lr,'external_review':er})
        bad=copy.deepcopy(ev);bad['predictions_review']['sha256']='0'*64;reject('incorrect_audit_hash_rejected',lambda:g.build('select_local',bad,root))
        rejected=write(home/'synthetic_reviews/pred.json',dict(passed=False,scope='confirmation_predictions_independent'));bad={**ev,'predictions_review':rejected};reject('unaccepted_or_noncanonical_prediction_audit_rejected',lambda:g.build('select_external',bad,root))
        local=[x for x in models if x['dataset']=='local' and x['variant']!='regression'];seg=[x for x in models if x['variant']!='regression'];localpop=[];audited=[]
        for model in seg:
            name='local_select_%02d'%(local.index(model)+1) if model['dataset']=='local' else 'select_'+model['run_id'];count=120 if model['dataset']=='local' else 3120
            p=stage(name,{'selection_seal.json':dict(run_id=model['run_id'],freeze_sha256=fsha,all_predictions_sha256=predsha,truth_used_for_choices=False,case_count=count,checkpoint_sha256=model['checkpoint_sha256'])})
            pinseal=pin(p/'selection_seal.json');audited.append(dict(run_id=model['run_id'],case_count=count,selection_seal_sha256=pinseal['sha256']))
            if model['dataset']=='local':localpop.append(dict(run_id=model['run_id'],case_count=count,selection_seal=pinseal,artifact_manifest=pin(p/'artifact_manifest.json')))
        stage('local_selections_seal',{'all_local_selections.json':dict(models=localpop,freeze_sha256=fsha,all_predictions_sha256=predsha,truth_used_for_choices=False,P_solver_cases=1440,policy_choices=2880)})
        with patch.object(Path,'open',guarded):build('audit_selections',ev)
        stage('audit_all_selections',{'review.json':dict(passed=True,scope='confirmation_all_selections_independent',implementation_sha256=g.digest(home/'audit_selections.py'),freeze_sha256=fsha,all_predictions_sha256=predsha,truth_used_for_choices=False,complete_case_inventory_verified=True,models=audited)})
        sr=pin(home/'stages/audit_all_selections/review.json')
        with patch.object(Path,'open',guarded):build('accept_choices',{'selections_review':sr});choice=approved('accept_choices',{'selections_review':sr},'choices_accepted.json')
        # Execute only the consumer's single stdlib validator function on artificial files.
        tree=ast.parse((ROOT/'confirmation_v1/pipeline.py').read_text(encoding='utf-8-sig'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_choices')
        ns={'HERE':home,'read':lambda p:json.loads(Path(p).read_text()),'sha':g.digest,'require':g.require,'bound_file':lambda p,h:g.require(g.digest(p)==h,'bad test pin')}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<one frozen stdlib consumer>', 'exec'),ns)
        yes('all30_draft_fields_match_pipeline_validate_choices',len(ns['validate_choices'](choice['path'])['models'])==30)
        for ds in ('local','weeds'):stage('truth_'+ds,{'truth_manifest.json':{'DO_NOT_PARSE_TRUTH':True},'cache/NOT_REAL.npz':'DO NOT OPEN'})
        reject('no_label_free_grant_after_truth_stage',lambda:g.build('select_local',ev,root))
        with patch.object(Path,'open',guarded):
            for name in ('local_outcomes','external_outcomes','scalar_outcomes'):build(name,{'choice_review':choice})
        for name in g.OUTCOMES:stage(name,{'summary.json':{'synthetic':True},'some_values.csv':'DO NOT READ','some_array.npz':'DO NOT READ'})
        il=write(home/'synthetic_reviews/local_impl.json',dict(passed=True,implementation_sha256=g.digest(root/'review_confirmation_local_outputs.py')));ie=write(home/'synthetic_reviews/external_impl.json',dict(passed=True,implementation_sha256=g.digest(root/'review_confirmation_external_scalar.py')))
        with patch.object(Path,'open',guarded):
            build('audit_local_outcomes',{'choice_review':choice,'geometry_review':gr,'implementation_review':il});build('audit_external_scalar',{'choice_review':choice,'implementation_review':ie})
        report_ev={}
        for name,key in zip(g.OUTCOMES,('scalar_review','local_review','external_review')):report_ev[key]=write(home/'synthetic_reviews'/(name+'_audit.json'),dict(passed=True,stage=name,freeze_sha256=fsha,audited_artifact_manifest_sha256=g.digest(home/'stages'/name/'artifact_manifest.json')))
        with patch.object(Path,'open',guarded):build('report_tables',report_ev)
        yes('all15_stage_interfaces_covered',set(saved)==set(g.STAGES))
        parser_sources={}
        for name in ('prediction_queue','local_geometry','selection_queue'):
            source=ROOT/'confirmation_v1'/g.PROGRAM[name][0];tree=ast.parse(source.read_text(encoding='utf-8-sig'))
            main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');prefix=[]
            for statement in main.body:
                prefix.append(statement)
                if isinstance(statement,ast.Assign) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute) and statement.value.func.attr=='parse_args':break
            else:raise AssertionError('Expected simple real argparse prefix: '+name)
            require_calls={'ArgumentParser','add_argument','add_mutually_exclusive_group','parse_args'}
            assert all(isinstance(n.func,ast.Attribute) and n.func.attr in require_calls for s in prefix for n in ast.walk(s) if isinstance(n,ast.Call))
            code=compile(ast.Module(body=prefix,type_ignores=[]),str(source)+':parser_only','exec')
            argv=saved[name]['commands']['commands'][0]['argv'];science_called=[]
            ns={'argparse':argparse,'__doc__':'Parser-only synthetic test'}
            with patch.object(sys,'argv',[str(source),*argv[6:]]):exec(code,ns)
            parsed=ns[prefix[-1].targets[0].id];yes(name+'_real_argparse_accepts_generated_command',parsed.execute is True and parsed.root_review=='{ROOT_APPROVED_REVIEW}')
            try:
                with patch.object(sys,'argv',[str(source),*[x for x in argv[6:] if x!='--execute']]),contextlib.redirect_stderr(io.StringIO()):exec(code,ns)
            except SystemExit as error:yes(name+'_original_missing_execute_is_rejected',error.code==2)
            else:raise AssertionError('Expected required --execute failure')
            parser_sources[str(source)]={'sha256':g.digest(source),'executed_scope':'AST parser statements through parse_args only; no module import/validate/execute'}
        yes('no_commands_executed',all(x['manual_execution_only'] for v in saved.values() for x in v['commands']['commands']))
        m=g.Metadata(root);m.pin(home/'contract.json');original=(home/'contract.json').read_bytes();write(home/'contract.json',{'changed':True});reject('begin_end_source_drift_rejected',m.finish);(home/'contract.json').write_bytes(original)
        reject('path_escape_rejected',lambda:g.beneath(home/'stages','../../outside'))
        reject('payload_pin_rejected',lambda:g.Metadata(root).pin(home/'stages/inputs_local/cache/NOT_REAL.npz'))
        reject('unrequested_evidence_key_refused',lambda:g.build('prediction_queue',{'automatic_retry':True},root))
        yes('no_scientific_modules_loaded',not any(x in sys.modules for x in ('numpy','torch','scipy')))
        tree=ast.parse(Path(g.__file__).read_text(encoding='utf-8'))
        yes('generator_has_no_process_launch_or_dynamic_import',not any(isinstance(x,ast.Name) and x.id in ('subprocess','exec','eval','__import__') for x in ast.walk(tree)))
        g.dump(out/'synthetic_grant_matrix.json',saved)
        receipt=dict(passed=True,implementation_sha256=g.digest(g.__file__),checker_sha256=g.digest(__file__),check_count=len(checks),checks=checks,stage_count=len(saved),actual_freeze_created=False,actual_stage_calls=0,payload_reads=0,model_calls=0,solver_calls=0,only_synthetic_root=str(root),metadata_open_count_under_guard=len(opened),real_parser_sources=parser_sources)
        g.dump(out/'focused_checks.json',receipt);print(json.dumps({'passed':True,'path':str(out/'focused_checks.json'),'sha256':g.digest(out/'focused_checks.json'),'checks':len(checks)},ensure_ascii=False))
    except BaseException as error:g.dump(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'checks':checks});raise
if __name__=='__main__':main()
