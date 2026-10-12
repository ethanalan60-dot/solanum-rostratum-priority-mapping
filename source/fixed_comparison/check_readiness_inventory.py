"""Read-only preparation inventory. Never creates FREEZE or opens reserved datasets.

Checks source and existing development metadata/binary hashes only. Missing final
evidence is pending, not inferred from queue completion. No tensor loader exists.
"""
from pathlib import Path
import argparse, ast, datetime, json, re, sys
import core

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRODUCTION=(
    'core.py','data_io.py','pipeline.py','run_prediction_queue.py','run_selection_queue.py',
    'frozen_nn.py','frozen_models.py','frozen_resample.py','safe_checkpoint.py',
    'local_geometry.py','local_management.py','local_outcomes.py',
    'external_arithmetic.py','external_spatial.py','scalar_outcomes.py',
    'prepare_scalar_baselines.py',
)
STATIC_RECORDS=('contract.json','extraction.json','external_arithmetic_extraction.json','scalar_baselines.json')
PREFIXES=('checkpoint','config','status','history','best_validation','completion_receipt','training_audit')
COUNTS={'maps':357,'scalar_pairs':1071,'fixed_prefix_rows':45,'P_policy_rows':45,'auc_rows':9}


def metadata_path(path):
    """No dataset, reserved-manifest or prediction-array reads through this tool."""
    path=Path(path).resolve()
    core.require(path.is_relative_to(ROOT),'Evidence must be under campaign ROOT')
    rel=path.relative_to(ROOT)
    forbidden={'local_data_v2','weedsgalore_data_v1','external_data','stages','validation_exports','vx2'}
    core.require(not (set(rel.parts)&forbidden),'Dataset/output content is outside metadata review scope')
    core.require('test' not in path.name.lower() and 'r6' not in path.name.lower(),'Reserved filename rejected')
    core.require(path.suffix.lower() in {'.json','.jsonl','.py','.md','.txt','.pt'},'Unsupported metadata/binary source')
    return path


def bound_metadata(path,digest):
    return core.bound_file(metadata_path(path),digest)


def lookup_model(audit,run_id):
    core.require(audit.get('passed') is True,'Training audit did not pass')
    candidates=[x for key in ('models','runs') for x in audit.get(key,[]) if x.get('run_id')==run_id]
    # Both aliases may be present in the same receipt; they must agree exactly.
    core.require(candidates and all(x==candidates[0] for x in candidates),'Missing/ambiguous audited run')
    return candidates[0]


def validate_audit_binding(audit,entry):
    row=lookup_model(audit,entry['run_id'])
    for key in ('best_epoch','checkpoint_sha256','status_sha256'):
        core.require(row.get(key)==entry[key],'Audited selected model differs: '+key)
    return row


def numeric_record(audit,run_id):
    row=lookup_model(audit,run_id)
    core.require(audit.get('test_or_R6_accessed') is False and row.get('test_or_R6_accessed') is False,'Wrong numerical-audit scope')
    for key,count in COUNTS.items():core.require(row.get(key)==count,'Incomplete numerical audit: '+key)
    core.require(row.get('no_new_model_forward') is True and row.get('no_new_solver') is True,'Expected artifact-only numerical audit')
    core.require(isinstance(row.get('verified_files'),int) and row['verified_files']>0,'No artifact verification recorded')
    return row


def static_constants(path):
    """Read literal source pins without importing geometry, SciPy, torch or helpers."""
    values={}
    for node in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            for target in node.targets:
                if isinstance(target,ast.Name):values[target.id]=value
    return values


def self_checks():
    """Synthetic receipt/schema checks only; no development or reserved arrays."""
    results=[]
    def yes(name,condition):
        core.require(condition,name);results.append({'name':name,'passed':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):yes(name,True);return
        raise AssertionError(name)
    entry={'run_id':'synthetic','best_epoch':4,'checkpoint_sha256':'1'*64,'status_sha256':'2'*64}
    audit={'passed':True,'runs':[dict(entry)]}
    yes('one_training_audit_matches',validate_audit_binding(audit,entry)==entry)
    reject('failed_training_audit_rejected',lambda:validate_audit_binding({**audit,'passed':False},entry))
    reject('other_run_rejected',lambda:validate_audit_binding(audit,{**entry,'run_id':'other'}))
    reject('wrong_checkpoint_rejected',lambda:validate_audit_binding(audit,{**entry,'checkpoint_sha256':'3'*64}))
    reject('wrong_best_epoch_rejected',lambda:validate_audit_binding(audit,{**entry,'best_epoch':5}))
    yes('matching_machine_aliases_allowed',lookup_model({**audit,'models':[dict(entry)]},'synthetic')==entry)
    reject('conflicting_machine_aliases_rejected',lambda:lookup_model({**audit,'models':[{**entry,'best_epoch':9}]},'synthetic'))
    record={'run_id':'synthetic',**COUNTS,'no_new_model_forward':True,'no_new_solver':True,'test_or_R6_accessed':False,'verified_files':615}
    numeric={'passed':True,'test_or_R6_accessed':False,'models':[record]}
    yes('complete_numerical_record_matches',numeric_record(numeric,'synthetic')==record)
    for field in COUNTS:
        reject('incomplete_'+field+'_rejected',lambda field=field:numeric_record({**numeric,'models':[{**record,field:0}]},'synthetic'))
    reject('reserved_review_scope_rejected',lambda:numeric_record({**numeric,'test_or_R6_accessed':True},'synthetic'))
    reject('dataset_manifest_path_rejected',lambda:metadata_path(ROOT/'local_data_v2/test_sealed.json'))
    reject('cache_array_path_rejected',lambda:metadata_path(ROOT/'weedsgalore_data_v1/cache/frame.npz'))
    reject('prediction_stage_path_rejected',lambda:metadata_path(HERE/'stages/predict_dummy/metadata.json'))
    return results


def examine(evidence_path=None):
    checks=[];notes=[]
    def add(name,state,detail=None):checks.append({'name':name,'state':state,'detail':detail})
    def verify(name,fn,missing='pending'):
        try:detail=fn();add(name,'passed',detail);return detail
        except (FileNotFoundError,KeyError,TypeError) as e:add(name,missing,repr(e))
        except (ValueError,AssertionError) as e:add(name,'failed',repr(e))
        return None
    contract=core.read(HERE/'contract.json');core.validate_contract(contract)
    roster=contract['roster'];table=core.read(HERE/'checkpoints.json')
    expected=[x['run_id'] for x in roster]
    core.require([x['run_id'] for x in table['models']]==expected,'Checkpoint population/order differs')
    seeds=core.read(ROOT/'local_seed_stability_v1/plan.json')['runs'];seed_ids=[x['run_id'] for x in seeds]
    core.require(len(seed_ids)==16 and len(set(seed_ids))==16,'Original seed plan differs')
    core.require([x['run_id'] for x in roster if x['seed'] in (29,43) and x['dataset']=='local']==seed_ids,'Fixed16 roster differs')
    local_seg=[x for x in roster if x['dataset']=='local' and x['variant']!='regression']
    core.require(len(local_seg)==24,'Expected all24 local segmentation fits')
    add('complete_declared_population','passed',{'models':37,'local_segmentation':24,'new_local_seeds':16})

    lock=core.read(HERE/'SOURCE_LOCK.json')
    required_production=sorted(set(PRODUCTION)|{p.name for p in HERE.glob('*.py') if not p.name.startswith('check_')})
    required=sorted(set(PRODUCTION+STATIC_RECORDS)|{p.name for p in HERE.glob('*.py')})
    missing=[x for x in required if x not in lock['files']]
    add('final_source_lock_complete','pending' if missing else 'passed',{'required':required,'missing':missing})
    verify('current_registered_source_hashes',lambda:bool(core.validate_source_lock()))
    pins={**static_constants(HERE/'local_management.py')['PINS'],
        'run_local_seed_postprocess_queue_v2.py':static_constants(HERE/'run_prediction_queue.py')['OWNER_SHA']}
    core.require(pins['geo_management.py']==static_constants(HERE/'local_geometry.py')['GEO_SHA'],'Geometry dependency pins differ')
    verify('all_external_source_dependencies',lambda:[str(core.bound_file(ROOT/n,h)) for n,h in pins.items()])
    # Validate declarations, never follow dataset paths or hash their files here.
    for ds,count in (('local',338),('weeds',26)):
        declared=contract['datasets'][ds]
        core.require(declared['count']==count,'Dataset support count differs')
        core.require(re.fullmatch('[0-9a-f]{64}',declared['manifest_sha256']) is not None,'Missing reserved-source declaration')
    add('reserved_source_declarations_only','passed','Dataset paths and hashes declared; no dataset files opened or rehashed.')

    evidence=core.read(metadata_path(evidence_path)) if evidence_path else None
    if evidence is not None:core.require(evidence['version']=='confirmation_readiness_evidence_v1','Unknown evidence schema')
    else:add('readiness_evidence','pending','Provide explicitly reviewed evidence; no auto-discovery or presumed acceptance.')
    reviewed_sources=set()
    for number,row in enumerate((evidence or {}).get('module_reviews',[])):
        def module_review(row=row):
            path=bound_metadata(row['path'],row['sha256']);review=core.read(path)
            core.require(review.get('passed') is True,'Component review not passed')
            for item in row['source_bindings']:
                name=item['file'];core.require(name in required,'Unknown component source')
                value=review
                for token in item['sha256_field']:value=value[token]
                core.require(value==core.sha(HERE/name),'Reviewed component SHA differs: '+name)
                reviewed_sources.add(name)
            return str(path)
        verify('component_review_'+str(number+1),module_review)
    missing_reviews=sorted(set(required_production)-reviewed_sources)
    add('all_production_sources_independently_reviewed','pending' if missing_reviews else 'passed',missing_reviews)

    model_rows=[]
    for expected_row,entry in zip(roster,table['models']):
        rid=entry['run_id'];core.require(all(entry[k]==v for k,v in expected_row.items()),'Checkpoint identity differs')
        status_path=metadata_path(Path(entry['run_directory'])/'status.json')
        status=core.read(status_path) if status_path.exists() else {}
        row={'run_id':rid,'declared_dataset':entry['dataset'],'variant':entry['variant'],'seed':entry['seed'],
             'observed_training_state':status.get('status'),'observed_executed_epochs':status.get('executed_epochs')}
        missing_fields=[p+'_'+s for p in PREFIXES for s in ('path','sha256') if not entry.get(p+'_'+s)]
        if missing_fields or not entry.get('original_selection_audited'):
            row['readiness']='pending';row['missing']=missing_fields+([] if entry.get('original_selection_audited') else ['original_selection_audited'])
            add('checkpoint_'+rid,'pending',row['missing'])
        else:
            def inspect(entry=entry):
                for p in PREFIXES:bound_metadata(entry[p+'_path'],entry[p+'_sha256'])
                core.validate_checkpoint_entry(entry)
                validate_audit_binding(core.read(entry['training_audit_path']),entry)
                return {'best_epoch':entry['best_epoch'],'checkpoint_sha256':entry['checkpoint_sha256'],'no_model_load':True}
            result=verify('checkpoint_'+rid,inspect)
            row['readiness']='passed' if result is not None else 'not_accepted'
        model_rows.append(row)

    numeric=(evidence or {}).get('seed_numerical_reviews',[])
    core.require(len({x['run_id'] for x in numeric})==len(numeric),'Duplicate numerical evidence entry')
    core.require(set(x['run_id'] for x in numeric)<=set(seed_ids),'Unplanned numerical evidence entry')
    numeric_by={x['run_id']:x for x in numeric};numeric_pass=[]
    for rid in seed_ids:
        if rid not in numeric_by:add('numerical_'+rid,'pending','Missing independent numerical receipt');continue
        def inspect_numeric(rid=rid):
            source=numeric_by[rid];path=bound_metadata(source['path'],source['sha256']);audit=core.read(path)
            row=numeric_record(audit,rid);receipt_path=bound_metadata(row['receipt'],row['receipt_sha256']);receipt=core.read(receipt_path)
            core.require(receipt['run_id']==rid and receipt['original_epoch_preserved'] is True and receipt['all_declared_cases_retained'] is True and receipt['test_or_R6_accessed'] is False,'Postprocess receipt identity/scope differs')
            # These are existing development outputs; only their small status/hash manifests are read.
            export=Path(receipt['export_directory']);spatial=Path(receipt['spatial_directory'])
            for folder,name,key in ((export,'status.json','export_status_sha256'),(export,'artifact_manifest.json','export_manifest_sha256'),(spatial,'status.json','spatial_status_sha256'),(spatial,'summary.json','spatial_summary_sha256')):
                core.require(folder.resolve().is_relative_to(ROOT),'Development output escaped ROOT')
                core.bound_file(folder/name,receipt[key])
            core.require(core.read(export/'status.json')['status']=='completed' and core.read(spatial/'status.json')['state']=='completed','Postprocess not completed')
            return {'review':core.binding(path),'receipt':core.binding(receipt_path),'run_id':rid}
        result=verify('numerical_'+rid,inspect_numeric)
        if result:numeric_pass.append(result)
    # Bind the immutable source/authorization chain of new seed training, never dynamic queue status.
    seedhome=ROOT/'local_seed_stability_v1'
    def seed_authorization():
        claim=core.read(seedhome/'QUEUE_CLAIM.json');grant_path=core.bound_file(claim['review_path'],claim['review_sha256']);grant=core.read(grant_path)
        core.require(grant['passed'] is True and grant['execution_scope']=='local_seed_stability_v1_fixed16' and grant['test_or_R6_authorized'] is False,'Seed grant differs')
        for key,file in (('source_lock_sha256','SOURCE_LOCK.json'),('plan_sha256','plan.json'),('protocol_sha256','protocol.json'),('input_lock_sha256','INPUT_LOCK.json')):
            digest=core.sha(seedhome/file);core.require(claim['binding'][key]==grant[key]==digest,'Seed source authorization differs: '+key)
        return {'claim':core.binding(seedhome/'QUEUE_CLAIM.json'),'grant':core.binding(grant_path)}
    verify('seed_training_authorization',seed_authorization)
    add('checkpoint_table_finalized','passed' if table.get('frozen') is True else 'pending','Checkpoints may be finalized before root creates formal FREEZE; this tool does not change them.')
    return {'passed':all(x['state']=='passed' for x in checks),'scope':'read_only_pre_freeze_inventory_not_execution_authorization',
        'checks':checks,'counts':{'models':len(model_rows),'completed_training_statuses':sum(x['observed_training_state']=='completed' and x['observed_executed_epochs']==24 for x in model_rows),
        'new_seed_training_statuses_completed':sum(x['run_id'] in seed_ids and x['observed_training_state']=='completed' and x['observed_executed_epochs']==24 for x in model_rows),
        'checkpoint_sources_accepted':sum(x['readiness']=='passed' for x in model_rows),'new_seed_numerical_reviews_accepted':len(numeric_pass)},
        'models':model_rows,'numerical_receipts':numeric_pass,'inputs':{'source_lock':core.binding(HERE/'SOURCE_LOCK.json'),'contract':core.binding(HERE/'contract.json'),'checkpoint_table':core.binding(HERE/'checkpoints.json'),'evidence':core.binding(evidence_path) if evidence_path else None},
        'checker':core.binding(__file__),'real_freeze_exists':(HERE/'FREEZE.json').exists(),'no_reserved_dataset_file_opened':True,'no_tensor_load_or_inference':True,'no_solver_or_subprocess':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path);parser.add_argument('--require-ready',action='store_true')
    args=parser.parse_args();controlled=self_checks();report=examine(args.evidence);report['synthetic_checks']=controlled
    out=ROOT/'provenance'/('confirmation_readiness_inventory_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    (out/'checker.py').write_bytes(Path(__file__).read_bytes());core.write(out/'inventory.json',report)
    print(json.dumps({'out':str(out),'passed':report['passed'],'counts':report['counts'],'pending':sum(x['state']=='pending' for x in report['checks']),'failed':sum(x['state']=='failed' for x in report['checks']),'sha256':core.sha(out/'inventory.json')},ensure_ascii=False))
    if args.require_ready and not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
