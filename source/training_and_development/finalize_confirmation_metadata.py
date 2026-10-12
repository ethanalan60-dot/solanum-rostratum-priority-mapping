"""Root metadata promotion only: archive old metadata, require readiness, then freeze once."""
from pathlib import Path
import argparse,datetime,json,shutil,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parent;CONF=ROOT/'confirmation_v1';sys.path.insert(0,str(CONF))
import core

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assembly-directory',type=Path,required=True);p.add_argument('--assembly-sha256',required=True);p.add_argument('--assembly-manifest-sha256',required=True);p.add_argument('--execute',action='store_true',required=True);a=p.parse_args()
    folder=a.assembly_directory.resolve();core.require(folder.is_relative_to(ROOT/'provenance'),'Assembly outside provenance')
    watched={str(Path(__file__).resolve()):core.sha(__file__)}
    for name,digest in [('assembly_report.json',a.assembly_sha256),('artifact_manifest.json',a.assembly_manifest_sha256)]:
        path=core.bound_file(folder/name,digest);watched[str(path)]=digest
    report_path=core.bound_file(folder/'assembly_report.json',a.assembly_sha256);report=core.read(report_path)
    core.require(report['complete_for_root_review'] is True and report['integrity_checks_passed'] is True and report['pending']==report['failed']==0,'Complete assembly required')
    core.require(report['counts']['production_py']==report['counts']['reviewed_production_py'] and report['counts']['reviewed_output_auditors']==3 and report['counts']['accepted_seed_numerical_reviews']==16,'Incomplete source/numerical scope')
    files=core.read(folder/'artifact_manifest.json')['files_sha256'];core.require({'assembly_report.json','SOURCE_LOCK.candidate.json','readiness_evidence.draft.json'}<=set(files),'Missing assembly artifacts')
    core.require(files['assembly_report.json']==a.assembly_sha256,'Report not bound to explicit manifest')
    for name,digest in files.items():
        path=core.resolve_under(folder,name);core.bound_file(path,digest);watched[str(path)]=digest
    for path,digest in report['source_inputs_sha256'].items():core.bound_file(path,digest);watched[str(Path(path).resolve())]=digest
    candidate=report['candidate_metadata']['candidate'];core.bound_file(candidate['path'],candidate['sha256']);table=core.read(candidate['path'])
    core.require(table['frozen'] is False and table['complete'] is True and table['all37_training_audited'] is True and len(table['models'])==37,'Unfrozen full37 candidate required')
    lock=core.read(folder/'SOURCE_LOCK.candidate.json');evidence=folder/'readiness_evidence.draft.json'
    expected={p.name for p in CONF.glob('*.py')}|{p.name for p in CONF.glob('*.md')}|{'contract.json','extraction.json','external_arithmetic_extraction.json','scalar_baselines.json','checkpoint_candidate_protocol.json'}
    core.require(set(lock['files'])==expected,'Candidate lock misses/adds source files')
    for name,digest in lock['files'].items():core.bound_file(CONF/name,digest)
    for name,digest in lock['external_dependency_sha256'].items():core.bound_file(ROOT/name,digest)
    for name,digest in lock['root_output_auditor_sha256'].items():core.bound_file(ROOT/name,digest)
    core.require(not (CONF/'FREEZE.json').exists(),'Existing FREEZE must never be overwritten')
    core.require(not core.read(CONF/'checkpoints.json')['frozen'],'Official checkpoint table already promoted')
    old_official={str(CONF/name):core.sha(CONF/name) for name in ('SOURCE_LOCK.json','checkpoints.json')}
    for path,digest in {**watched,**old_official}.items():core.bound_file(path,digest)
    control=ROOT/'confirmation_runtime';control.mkdir(exist_ok=True);claim=control/'METADATA_PROMOTION_CLAIM.json'
    out=ROOT/'provenance'/('confirmation_promotion_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    stream=claim.open('x',encoding='utf-8')
    try:
        with stream:json.dump({'created_at':core.now(),'out':str(out),'assembly':core.binding(report_path),'assembly_manifest':core.binding(folder/'artifact_manifest.json'),'source':core.binding(__file__)},stream,ensure_ascii=False,indent=2)
        out.mkdir();shutil.copy2(__file__,out/'promote_source.py')
        for path,digest in {**watched,**old_official}.items():core.bound_file(path,digest)
        for name in ('SOURCE_LOCK.json','checkpoints.json'):shutil.copy2(CONF/name,out/('previous_'+name))
        table.update(frozen=True,version='confirmation_checkpoint_table_v1',candidate_origin=candidate,promoted_at=core.now(),root_metadata_promotion=True)
        core.write(CONF/'checkpoints.json',table);core.write(CONF/'SOURCE_LOCK.json',lock)
        promotion={'scope':'root_explicit_confirmation_metadata_promotion','created_at':core.now(),'assembly':core.binding(report_path),'candidate':candidate,'evidence':core.binding(evidence),
          'official_checkpoints':core.binding(CONF/'checkpoints.json'),'official_source_lock':core.binding(CONF/'SOURCE_LOCK.json'),
          'previous_metadata':{name:core.binding(out/('previous_'+name)) for name in ('SOURCE_LOCK.json','checkpoints.json')},'model_or_solver_calls':0,'reserved_inputs_opened':False}
        core.write(out/'promotion.json',promotion)
        expected_inventory={'source_lock':promotion['official_source_lock'],'checkpoint_table':promotion['official_checkpoints'],'contract':core.binding(CONF/'contract.json'),'evidence':core.binding(evidence)}
        command=[sys.executable,'-X','utf8','-B','-s',str(CONF/'check_readiness_inventory.py'),'--evidence',str(evidence),'--require-ready']
        with (out/'inventory.stdout.log').open('x',encoding='utf-8') as so,(out/'inventory.stderr.log').open('x',encoding='utf-8') as se:
            result=subprocess.run(command,cwd=ROOT,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        core.require(result.returncode==0,'Readiness inventory failed; metadata preserved, no FREEZE')
        messages=[json.loads(line) for line in (out/'inventory.stdout.log').read_text(encoding='utf-8-sig').splitlines() if line.strip()];core.require(len(messages)==1,'Unexpected inventory command output')
        inventory_path=Path(messages[0]['out'])/'inventory.json';core.bound_file(inventory_path,messages[0]['sha256']);inventory=core.read(inventory_path)
        core.require(inventory['passed'] is True and inventory['counts']['checkpoint_sources_accepted']==37 and inventory['counts']['new_seed_numerical_reviews_accepted']==16,'Final inventory incomplete')
        core.require(inventory['inputs']==expected_inventory,'Inventory did not validate this exact promoted metadata')
        core.require(inventory['scope']=='read_only_pre_freeze_inventory_not_execution_authorization' and inventory['real_freeze_exists'] is False,'Wrong inventory phase')
        for item in inventory['inputs'].values():
            if item:core.bound_file(item['path'],item['sha256'])
        contract=core.read(CONF/'contract.json');core.validate_contract(contract)
        for path,digest in watched.items():core.bound_file(path,digest)
        for item in expected_inventory.values():core.bound_file(item['path'],item['sha256'])
        core.validate_source_lock()
        for row in table['models']:core.validate_checkpoint_entry(row)
        freeze={'passed':True,'scope':'confirmation_v1','frozen':True,'created_at':core.now(),'source_lock_sha256':core.sha(CONF/'SOURCE_LOCK.json'),
          'contract_sha256':core.sha(CONF/'contract.json'),'checkpoints_sha256':core.sha(CONF/'checkpoints.json'),'R':[1,2,4,6],
          'run_ids':[x['run_id'] for x in table['models']],'truth_after_all_predictions_and_selections':True,'root_promotion':core.binding(out/'promotion.json'),'readiness_inventory':core.binding(inventory_path),
          'development_completion':core.read(evidence)['explicit_final_sources'],'no_additional_outcome_driven_tuning':True,
          'interpretation':'Fixed confirmation after iterative development; same local scene and one external field, not prospective registration or independent-field replication.'}
        with (CONF/'FREEZE.json').open('x',encoding='utf-8') as f:json.dump(freeze,f,ensure_ascii=False,indent=2,allow_nan=False)
        core.validate_freeze();core.write(out/'result.json',{'passed':True,'formal_freeze':core.binding(CONF/'FREEZE.json'),'inventory':core.binding(inventory_path),'predictions_started':False,'reserved_arrays_opened':False})
        core.write(out/'artifact_manifest.json',core.stage_manifest(out));print(json.dumps({'passed':True,'out':str(out),'FREEZE':core.binding(CONF/'FREEZE.json'),'reserved_arrays_opened':False}))
    except BaseException as error:
        failure={'error':repr(error),'traceback':traceback.format_exc(),'official_metadata_preserved_for_review':True,'FREEZE_exists':(CONF/'FREEZE.json').exists(),'no_automatic_retry':True,'claim':str(claim)}
        try:out.mkdir(exist_ok=True);core.write(out/'failure.json',failure)
        except BaseException as secondary:
            failure['recording_error']=repr(secondary);core.write(control/('promotion_failure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'),failure)
        raise
if __name__=='__main__':main()
