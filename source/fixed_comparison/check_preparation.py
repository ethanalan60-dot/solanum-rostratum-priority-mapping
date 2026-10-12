"""Synthetic/metadata readiness only. Never accesses real test members or model forwards."""
from pathlib import Path
import ast,contextlib,copy,datetime,hashlib,inspect,json,os,sys,traceback
from unittest.mock import patch
import core
OWN_PY=('core.py','data_io.py','pipeline.py','check_preparation.py','frozen_nn.py','frozen_models.py','frozen_resample.py','safe_checkpoint.py')

def main():
    home=Path(__file__).resolve().parent
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=home/'checks'/stamp;out.mkdir(parents=True)
    core.configure_cpu(out)
    # Short G-owned mock roots avoid Windows MAX_PATH; artifacts record their exact locations.
    mockroot=Path('./temporary_checks/confirmation_checks')/stamp;mockroot.mkdir(parents=True)
    checks=[]
    def good(name,condition):
        if not condition:raise AssertionError(name)
        checks.append({'name':name,'passed':True})
    def reject(name,func):
        try:func()
        except (ValueError,FileNotFoundError,FileExistsError,RuntimeError):good(name,True)
        else:raise AssertionError('Did not reject: '+name)
    try:
        source_before={name:core.sha(home/name) for name in OWN_PY}
        for name in OWN_PY:
            p=home/name;compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
        good('all_source_compiles',True)
        contract=core.read(home/'contract.json');core.validate_contract(contract)
        good('fixed_37_models_30_spatial',len(contract['roster'])==37 and sum(x['variant']!='regression' for x in contract['roster'])==30)
        good('fixed_counts_38792_scalars_8268_npz',contract['expected']['scalar_rows']==28*338*4+9*26*4 and contract['expected']['npz_files']==24*338+6*26)
        good('R6_actual_intermediate42',contract['intermediate_sizes']==[256//r for r in core.R_VALUES])
        good('checkpoint_table_remains_placeholder',core.read(home/'checkpoints.json')['frozen'] is False and all(x['checkpoint_sha256'] is None for x in core.read(home/'checkpoints.json')['models']))
        ex=core.read(home/'extraction.json');original=Path(ex['parent_train_source']).read_text(encoding='utf-8-sig');new=(home/'frozen_nn.py').read_text()
        oldnodes={n.name:ast.get_source_segment(original,n) for n in ast.parse(original).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        newnodes={n.name:ast.get_source_segment(new,n) for n in ast.parse(new).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name,digest in ex['exact_AST_functions'].items():good('exact_original_AST_'+name,newnodes[name]==oldnodes[name] and hashlib.sha256(oldnodes[name].encode()).hexdigest()==digest)
        for name,entry in ex['copies'].items():good('byte_exact_copy_'+name,core.sha(home/name)==entry['original_sha256']==core.sha(entry['original_path']))
        for obj in contract['development_plans']:core.bound_file(obj['path'],obj['sha256'])
        good('plan_roster_sources_bound',True)
        # Metadata-only root freeze lifecycle using synthetic checkpoint placeholders.
        froot=mockroot/'freeze_fixture';froot.mkdir()
        core.write(froot/'contract.json',contract)
        core.write(froot/'checkpoints.json',{'frozen':True,'models':copy.deepcopy(contract['roster'])})
        (froot/'stub.py').write_text('# synthetic source\n')
        core.write(froot/'SOURCE_LOCK.json',{'version':'confirmation_source_v1','files':{'stub.py':core.sha(froot/'stub.py')}})
        freeze={'passed':True,'scope':'confirmation_v1','frozen':True,'source_lock_sha256':core.sha(froot/'SOURCE_LOCK.json'),'contract_sha256':core.sha(froot/'contract.json'),'checkpoints_sha256':core.sha(froot/'checkpoints.json'),'R':[1,2,4,6],'run_ids':[x['run_id'] for x in contract['roster']],'truth_after_all_predictions_and_selections':True}
        core.write(froot/'FREEZE.json',freeze)
        core.validate_freeze(froot,check_models=False);good('synthetic_freeze_metadata_contract',True)
        for key,value in [('passed',False),('scope','development'),('R',[1,2,4]),('truth_after_all_predictions_and_selections',False),('contract_sha256','0'*64)]:
            bad={**freeze,key:value};core.write(froot/'FREEZE.json',bad)
            reject('freeze_rejects_'+key,lambda:core.validate_freeze(froot,check_models=False))
        core.write(froot/'FREEZE.json',freeze);(froot/'stub.py').write_text('# changed source\n')
        reject('freeze_rejects_source_tamper',lambda:core.validate_freeze(froot,check_models=False))
        # Earliest strict development best: ties retain the first epoch; no tensors are loaded.
        croot=mockroot/'checkpoint_fixture';croot.mkdir();entry={**contract['roster'][0],'best_epoch':1,'original_selection_audited':True}
        history=[{'epoch':i,'selection_balanced_mae':.1 if i in (1,2) else .2} for i in range(1,25)]
        config={**{k:entry[k] for k in ('run_id','encoder','variant','seed','lr')},'epochs':24}
        for key,obj in [('config',config),('status',{'status':'completed','executed_epochs':24,'best_epoch':1,'test_accessed':False}),('best_validation',history[0]),('completion_receipt',{'mock':True}),('training_audit',{'mock':True})]:
            path=croot/(key+'.json');core.write(path,obj);entry[key+'_path']=str(path);entry[key+'_sha256']=core.sha(path)
        path=croot/'history.jsonl';path.write_text('\n'.join(json.dumps(x) for x in history)+'\n');entry['history_path']=str(path);entry['history_sha256']=core.sha(path)
        path=croot/'best.pt';path.write_bytes(b'synthetic no Torch payload');entry['checkpoint_path']=str(path);entry['checkpoint_sha256']=core.sha(path)
        core.validate_checkpoint_entry(entry);good('original_earliest_tied_best_accepts',True)
        reject('later_tied_epoch_rejected',lambda:core.validate_checkpoint_entry({**entry,'best_epoch':2}))
        reject('checkpoint_byte_mismatch_rejected',lambda:core.validate_checkpoint_entry({**entry,'checkpoint_sha256':'0'*64}))
        reject('unaudited_selection_rejected',lambda:core.validate_checkpoint_entry({**entry,'original_selection_audited':False}))
        # Missing root freeze must reject before any test path, checkpoint import, or arrays.
        import data_io,pipeline
        good('metadata_modules_do_not_import_torch','torch' not in sys.modules)
        data_io.validate_native_window(22016,4608,22320,14677)
        good('nominal512_edge_window_with304_remaining_columns_supported',True)
        data_io.validate_native_window(99,99,100,100)
        good('padding_support_origin_inside_preserves_nominal_size',True)
        reject('window_origin_outside_source_rejected',lambda:data_io.validate_native_window(100,0,100,100))
        reject('negative_window_origin_rejected',lambda:data_io.validate_native_window(-1,0,100,100))
        opened=[];realread=core.read
        def guarded_read(path):
            path=Path(path)
            if 'test' in path.name.lower():raise AssertionError('Forbidden real test metadata read')
            opened.append(str(path));return realread(path)
        with patch.object(core,'read',guarded_read):
            reject('no_freeze_rejects_input_stage',lambda:data_io.prepare_inputs('local'))
            reject('no_freeze_rejects_prediction_stage',lambda:pipeline.export_run(contract['roster'][0]['run_id']))
            reject('no_freeze_rejects_truth_stage',lambda:data_io.materialize_truth('weeds','nonexistent'))
        good('no_real_stage_or_claim_created',not(home/'stages').exists() and not(home/'claims').exists())
        # One-attempt lifecycle tests with purely synthetic files; never touch real stage roots.
        def fixture(name):
            p=mockroot/name;p.mkdir();core.write(p/'FREEZE.json',{'mock':True})
            for f in ('SOURCE_LOCK.json','contract.json','checkpoints.json'):core.write(p/f,{'mock':True})
            return p
        p=fixture('success')
        with core.attempt('alpha',p) as stage:core.write(stage/'payload.json',{'synthetic':True})
        core.completed_stage(stage);good('one_attempt_completion_and_manifest',True)
        before=core.sha(stage/'payload.json');reject('existing_stage_no_overwrite',lambda:core.attempt('alpha',p).__enter__());good('existing_payload_unchanged',before==core.sha(stage/'payload.json'))
        p=fixture('body_failure')
        try:
            with core.attempt('alpha',p):raise RuntimeError('synthetic dependency initialization failure')
        except RuntimeError:pass
        good('postclaim_initialization_failure_preserved',core.read(p/'stages/alpha/status.json')['state']=='failed' and (p/'claims/alpha.json').exists())
        reject('failed_attempt_permanent',lambda:core.attempt('alpha',p).__enter__())
        p=fixture('claim_dump_failure')
        with patch.object(core.json,'dump',side_effect=RuntimeError('synthetic claim dump failure')):
            reject('claim_dump_failure_propagates',lambda:core.attempt('alpha',p).__enter__())
        good('claim_dump_failure_recorded',core.read(p/'stages/alpha/failure.json')['state']=='failed' and (p/'claims/alpha.json').exists())
        p=fixture('mkdir_failure');realmkdir=Path.mkdir
        def badmkdir(path,*a,**kw):
            if path==p/'stages/alpha':raise RuntimeError('synthetic unavailable output')
            return realmkdir(path,*a,**kw)
        with patch.object(Path,'mkdir',badmkdir):reject('mkdir_failure_propagates',lambda:core.attempt('alpha',p).__enter__())
        good('unwritable_output_has_independent_fallback',bool(list((p/'failure_receipts').glob('*.json'))))
        p=fixture('finalization_failure');realwrite=core.write
        def badwrite(path,value):
            if Path(path).name=='artifact_manifest.json':raise RuntimeError('synthetic manifest failure')
            return realwrite(path,value)
        with patch.object(core,'write',badwrite):
            try:
                with core.attempt('alpha',p):pass
            except RuntimeError:pass
        good('manifest_failure_downgrades_success',core.read(p/'stages/alpha/status.json')['state']=='failed')
        # The post-selection gate is strict even when no reference decoding is attempted.
        reject('missing_global_predictions_rejected',pipeline.validate_global_predictions)
        reject('missing_global_choices_rejected',lambda:pipeline.validate_choices(mockroot/'missing_review.json'))
        # Isolated metadata contract for all37 prediction entries. Artifact checking is mocked;
        # malformed population must fail independently of any payload/model computations.
        groot=fixture('global_population');core.write(groot/'contract.json',contract)
        gdir=groot/'stages/predictions_seal';gdir.mkdir(parents=True)
        entries=[]
        for row in contract['roster']:
            path=groot/'stages'/('predict_'+row['run_id']);n=contract['datasets'][row['dataset']]['count']
            entries.append({**{k:row[k] for k in ('run_id','dataset','variant')},'stage':str(path),'scalar_rows':n*4,'map_count':0 if row['variant']=='regression' else n,**{k:{'path':str(path/(k+('.csv' if k=='predicted_scalars' else '.json'))),'sha256':'0'*64} for k in ('artifact_manifest','metadata','map_index','predicted_scalars')}})
        global_seal={'freeze_sha256':core.sha(groot/'FREEZE.json'),'run_ids':[x['run_id'] for x in contract['roster']],'all_declared_models_completed':True,'scalar_rows':38792,'map_count':8268,'models':entries}
        with patch.object(pipeline,'HERE',groot),patch.object(pipeline,'completed_stage',return_value={}),patch.object(pipeline,'bound_file',return_value=None):
            core.write(gdir/'all_predictions.json',global_seal);pipeline.validate_global_predictions();good('all37_global_population_contract_accepts',True)
            for name,value in [('empty',[]),('missing',entries[:-1]),('duplicate',entries[:-1]+[entries[0]])]:
                core.write(gdir/'all_predictions.json',{**global_seal,'models':value});reject('global_model_entries_'+name, pipeline.validate_global_predictions)
            bad=copy.deepcopy(global_seal);bad['models'][0]['scalar_rows']-=1;core.write(gdir/'all_predictions.json',bad);reject('global_permodel_counts_rejected',pipeline.validate_global_predictions)
            bad=copy.deepcopy(global_seal);bad['models'][0]['stage']=str(groot/'wrong');core.write(gdir/'all_predictions.json',bad);reject('global_wrong_stage_rejected',pipeline.validate_global_predictions)
        good('prediction_API_has_no_reference_argument',list(inspect.signature(pipeline.prediction_arrays).parameters)==['model','x','valid','variant','factor'])
        # Pure array arithmetic; no model creation, pretrained load, or forward call.
        import numpy as np
        import torch
        import frozen_nn
        torch.set_num_threads(2)
        image=np.arange(256*256*3,dtype=np.uint32).reshape(256,256,3).astype(np.uint8);valid=np.ones((256,256),np.float16);valid[:20,:20]=0;valid[20:40,:20]=.25
        x,v=data_io.local_input(image,valid)
        good('local_cache_uint8_conversion_and_fractional_valid',x.dtype==np.float32 and np.array_equal(x,image.transpose(2,0,1).astype(np.float32)/np.float32(255)) and np.array_equal(v,valid.astype(np.float32)))
        reject('local_float_RGB_rejected',lambda:data_io.local_input(image.astype(np.float32),valid))
        reject('empty_valid_rejected',lambda:data_io.local_input(image,np.zeros((256,256),np.float32)))
        for factor in (1,2,4,6):
            tensor=torch.full((1,3,256,256),.37);weights=torch.from_numpy(v)[None,None];altered=tensor.clone();altered[:,:,0:20,0:20]=float('nan')
            a=frozen_nn.degrade_rgb(tensor,factor,weights);b=frozen_nn.degrade_rgb(altered,factor,weights)
            good('synthetic_invalid_RGB_invariance_R'+str(factor),torch.isfinite(b).all().item() and torch.equal(a,b))
        rng=np.random.default_rng(20260916);bands=[rng.integers(0,65536,(600,600),dtype=np.uint16) for _ in range(3)]
        rgb,wv=data_io.weeds_input(bands)
        good('weeds_allvalid_and_float32_no_extra255',rgb.dtype==np.float32 and np.array_equal(wv,np.ones((256,256),np.float32)) and .45<float(rgb.mean())<.55)
        label=rng.integers(0,6,(600,600),dtype=np.uint8);truth=data_io.weeds_truth(label)
        good('pooled_weed_gt1_crop1_bg0',np.array_equal(truth['native_class_counts'],[int((label==0).sum()),int((label==1).sum()),int((label>1).sum())]))
        maxerr=float(np.max(np.abs(truth['soft3'].mean((1,2),dtype=np.float64)-truth['native_class_counts']/360000)))
        good('native600_to256_class_conservation',maxerr<1e-7)
        source={'count':1,'manifest_path':'synthetic','manifest_sha256':'a'*64};manifest={'dataset':'local','expected_count':1,'source_manifest':{'path':'synthetic','sha256':'a'*64},'records':[{'id':'synthetic'}]}
        data_io.validate_input_rows(manifest,source,'local');bad=copy.deepcopy(manifest);bad['records'][0]['coverage']=.9
        reject('prediction_whitelist_rejects_coverage',lambda:data_io.validate_input_rows(bad,source,'local'))
        bad['records'][0].pop('coverage');bad['records'][0]['some_new_truth_field']=.9
        reject('prediction_whitelist_rejects_unknown_field',lambda:data_io.validate_input_rows(bad,source,'local'))
        good('source_unchanged_during_checks',source_before=={name:core.sha(home/name) for name in OWN_PY})
        report={'passed':True,'checks':checks,'check_count':len(checks),'created_at':core.now(),'source_sha256':source_before,'contract_sha256':core.sha(home/'contract.json'),'placeholder_checkpoints_sha256':core.sha(home/'checkpoints.json'),'mock_root':str(mockroot),'max_synthetic_class_fraction_error':maxerr,'actual_test_manifest_or_arrays_read':False,'real_model_forward':False,'real_R6_forward':False,'GPU':False,'test_scores_or_checkpoint_reselection':False,'scope':'Pure synthetic arrays and isolated lifecycle controls; no real confirmation stage or formal freeze.'}
        core.write(out/'checks.json',report)
        import shutil
        (out/'source').mkdir()
        for p in [home/n for n in OWN_PY]+[home/'contract.json',home/'checkpoints.json']:shutil.copyfile(p,out/'source'/p.name)
        core.write(out/'artifact_manifest.json',core.stage_manifest(out));print(json.dumps({'passed':True,'checks':len(checks),'output':str(out),'sha256':core.sha(out/'checks.json')}))
    except BaseException as error:
        core.write(out/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'passed_checks':checks,'mock_root':str(mockroot),'test_accessed':False});raise

if __name__=='__main__':main()
