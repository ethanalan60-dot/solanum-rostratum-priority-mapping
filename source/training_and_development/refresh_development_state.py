"""Snapshot development evidence and observe confirmation phases through JSON metadata only.

No checkpoint, image, prediction array, truth cache or CSV value is opened by the
confirmation observer. A launched/failed prediction is not a successful forward.
"""
from pathlib import Path
import datetime,hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).resolve().read_bytes()).hexdigest()
def save(p,x):
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def rel(p):return os.path.relpath(p,PA).replace('\\','/')
def declared_artifacts(folder):
    """Verify only manifest-declared products; do not sweep Word temp files."""
    folder=Path(folder).resolve();manifest=folder/'artifact_manifest.json';obj=read(manifest)
    files=obj['files_sha256'];assert isinstance(files,dict) and files
    paths=[manifest]
    for name,digest in files.items():
        path=(folder/name.replace('\\','/')).resolve()
        assert path.is_relative_to(folder) and path!=manifest,'Invalid artifact path'
        assert sha(path)==digest,path;paths.append(path)
    for item in obj.get('external_outputs',[]):
        path=Path(item['path']).resolve();assert path.is_relative_to(ROOT.resolve())
        assert sha(path)==item['sha256'],path;paths.append(path)
    return paths

def accepted_manuscript_artifacts():
    """Bind the accepted development deliverables, not submission approval."""
    integration=PA/'integration_v3_20260916_115525_862700'
    supplement=PA/'integration_v3_1_20260916_115845_582633'
    figures=PA/'full24_publication_figures_20260916_115434_899378'
    visual=ROOT/'provenance/full24_visual_review_20260916_115636_317234/visual_QA.json'
    delivery_path=ROOT/'provenance/manuscript_v3_delivery_20260916_120755_705782/delivery.json'
    content=ROOT/'provenance/manuscript_v3_1_independent_content_20260916_120827_934522/review.json'
    assert sha(visual)=='6c83278c31fe263168abc1a9d0efcfbe1f7320ff2745c7b529c2fab445a2cbb3'
    assert sha(content)=='6c5399e59945df84d3240e921cd6172d53aeb964090f3f1477d231a15757174c'
    extra=read(supplement/'integration_supplement.json');qa=read(visual);review=read(content);delivery=read(delivery_path)
    assert extra['passed'] and qa['passed'] and review['passed'] and delivery['passed']
    assert qa['all_seven_pages_manually_inspected'] and not qa['unresolved_blocking_findings']
    assert sha(figures/'artifact_manifest.json')==qa['artifact_manifest_sha256']==extra['accepted_figure_manifest']['sha256']
    assert not delivery['submission_ready'] and delivery['new_output_immutable_after_manifest']
    assert delivery['source_markdown_sha256']==extra['manuscript']['sha256']==review['manuscript_sha256']
    word=Path(delivery['output_directory']).resolve();wordqa=read(word/'visual_qa.json')
    assert wordqa['passed'] and wordqa['accepted_for_development_content_review'] and not wordqa['submission_ready']
    assert delivery['main_pages']==wordqa['main_page_count']==20 and delivery['supplement_pages']==wordqa['supplement_page_count']==4
    paths=[]
    for folder in (integration,supplement,figures,word):paths.extend(declared_artifacts(folder))
    for field in ('manuscript','prior_manuscript_preserved','prior_integration_report','prior_author_self_review','visual_QA','accepted_figure_manifest','figure_specs','equation_specs','source'):
        item=extra[field];path=Path(item['path']);assert sha(path)==item['sha256'];paths.append(path)
    for name,digest in delivery['delivery_pins'].items():
        path=Path(name);assert sha(path)==digest;paths.append(path)
    chinese=PA/'农业价值与投稿定位_完整开发结果版.md'
    paths.extend([delivery_path,delivery_path.with_suffix('.md'),content,content.with_suffix('.md'),content.parent/'recomputed_local_summary.json',chinese])
    return paths,dict(integration=integration,supplement=supplement,figures=figures,visual=visual,content=content,delivery_path=delivery_path,delivery=delivery,word=word,extra=extra,chinese=chinese)

def validate_report_acceptance(pointer,review,review_sha256,freeze_sha256,report_folder,manifest,manifest_sha256,summary_sha256):
    """Only an exact root-selected independent review closes the report phase."""
    assert pointer['sha256']==review_sha256,'Report acceptance review hash mismatch'
    assert review.get('passed') is True and review.get('scope')=='confirmation_report_tables_independent'
    assert review.get('stage')=='report_tables' and review.get('freeze_sha256')==freeze_sha256
    assert Path(review['output_directory']).resolve()==Path(report_folder).resolve()
    assert review.get('audited_artifact_manifest_sha256')==manifest_sha256
    assert review.get('summary_sha256')==summary_sha256==manifest['files'].get('summary.json')
    return True

def current_processing(stages,prediction_queue,selection_queue,all_predictions,final):
    """Workflow priority, not process-liveness inference; completed work is historical."""
    def stage_group(names,waiting):
        pending=[n for n in names if not stages[n].get('completed_metadata_verified') or stages[n].get('independent_review_passed') is False]
        failed=[n for n in pending if stages[n].get('state')=='failed' or stages[n].get('independent_review_passed') is False]
        active=failed or pending
        return {'kind':'stage_metadata' if active else waiting,'state':'failed' if failed else 'in_progress' if active else 'awaiting_next_stage',
          'stages':active,'owner_pid':None,'child_pid':None,'error':{n:stages[n].get('error') or 'Independent review metadata not accepted' for n in failed},'queue_source':None}
    if final:return {'kind':'confirmation_complete','state':'completed','stages':[],'owner_pid':None,'child_pid':None,'error':None,'queue_source':None}
    groups=[(('report_tables',),'report_output_independent_acceptance'),
      (('scalar_outcomes','local_outcomes','external_outcomes','audit_local_outcomes'),'remaining_outcomes_or_report'),
      (('truth_local','truth_weeds'),'remaining_truth_or_outcomes'),
      (('audit_all_selections',),'root_choice_acceptance_or_truth')]
    for names,waiting in groups:
        present=[n for n in names if n in stages]
        if present:return stage_group(present,waiting)
    if selection_queue is not None:
        state=selection_queue.get('state','unknown');stage=selection_queue.get('stage') or ''
        return {'kind':'selection_queue' if state!='completed' else 'selection_audit_pending','state':state if state!='completed' else 'awaiting_next_stage',
          'stages':[stage] if stage and state!='completed' else [],'owner_pid':selection_queue.get('owner_pid') if state!='completed' else None,
          'child_pid':selection_queue.get('child_pid') if state=='running' else None,'error':selection_queue.get('error') if state=='failed' else None,'queue_source':'selection'}
    if all_predictions:
        active=[n for n in ('audit_predictions','local_geometry','audit_geometry') if n in stages and not stages[n].get('completed_metadata_verified')]
        return stage_group(active,'selection_queue_not_started')
    q=prediction_queue or {};state=q.get('state','not_started');stage=q.get('stage') or ''
    return {'kind':'prediction_queue','state':state,'stages':[stage] if stage else [],'owner_pid':q.get('owner_pid'),
      'child_pid':q.get('child_pid') if state=='running' else None,'error':q.get('error') if state=='failed' else None,'queue_source':'prediction'}

def derive_confirmation_phase(freeze,launch,queue,stages,selection_queue=None,selection_expected=None):
    """Pure status interpretation; None means actual partial access is unknown."""
    completed={name for name,item in stages.items() if item.get('completed_metadata_verified') is True}
    inputs=[name for name in ('inputs_local','inputs_weeds') if name in completed]
    prediction_names=[name for name in stages if name.startswith('predict_')]
    expected_predictions={'predict_'+x for x in (freeze or {}).get('run_ids',[])}
    predictions=[name for name in prediction_names if name in completed and name in expected_predictions]
    truth_names=[name for name in ('truth_local','truth_weeds') if name in stages]
    truth_complete=[name for name in truth_names if name in completed]
    current=(queue or {}).get('stage') or ''
    inference_attempted=bool(prediction_names or current.startswith('predict_'))
    known_pre_forward_failure=bool(prediction_names) and all(stages[name].get('failed_before_prediction_seal') is True for name in prediction_names) and not predictions and (not current.startswith('predict_') or current in prediction_names)
    input_attempted=bool(inputs or inference_attempted or any(name in stages for name in ('inputs_local','inputs_weeds')) or current.startswith('inputs_'))
    all_predictions='predictions_seal' in completed and len(predictions)==37
    selection_expected=selection_expected or {'local':[],'external':[]}
    local_selections=[n for n in selection_expected['local'] if n in completed]
    external_selections=[n for n in selection_expected['external'] if n in completed]
    selection_audit_completed='audit_all_selections' in completed
    selection_audit_accepted=selection_audit_completed and stages.get('audit_all_selections',{}).get('independent_review_passed') is True
    outcomes=[name for name in ('scalar_outcomes','local_outcomes','external_outcomes') if name in completed]
    report_completed='report_tables' in completed
    report_accepted=report_completed and stages.get('report_tables',{}).get('independent_review_root_accepted') is True
    final=all_predictions and len(truth_complete)==2 and len(outcomes)==3 and report_accepted
    processing=current_processing(stages,queue,selection_queue,all_predictions,final)
    if final:phase='confirmation_outcomes_accepted_and_reported'
    elif report_completed:phase='confirmation_report_awaiting_independent_acceptance'
    elif 'report_tables' in stages:phase='confirmation_report_'+('failed' if processing['state']=='failed' else 'in_progress')
    elif truth_names or outcomes:phase='confirmation_truth_or_outcomes_in_progress'
    elif any(n in stages for n in ('scalar_outcomes','local_outcomes','external_outcomes')):phase='confirmation_outcomes_in_progress'
    elif selection_audit_accepted:phase='confirmation_selections_audited_awaiting_root_acceptance_or_truth'
    elif 'audit_all_selections' in stages:phase='confirmation_selection_audit_'+('failed' if processing['state']=='failed' else 'in_progress')
    elif selection_queue is not None:phase='confirmation_selection_queue_'+str(selection_queue.get('state','unknown'))
    elif all_predictions:phase='all_predictions_completed_awaiting_or_running_spatial_stages'
    elif (queue or {}).get('state')=='failed':phase='confirmation_prediction_queue_failed'
    elif inference_attempted:phase='confirmation_prediction_attempt_in_progress'
    elif input_attempted:phase='confirmation_input_preparation_in_progress'
    elif launch:phase='confirmation_prediction_queue_launched'
    elif freeze:phase='confirmation_frozen_not_launched'
    else:phase='development_or_confirmation_preparation'
    return dict(phase=phase,formal_freeze_present=bool(freeze),prediction_queue_launch_recorded=bool(launch),prediction_queue_state=(queue or {}).get('state','not_started'),
      observed_owner_pid=(queue or {}).get('owner_pid',(launch or {}).get('pid')),process_liveness_checked=False,
      prepared_input_datasets=len(inputs),expected_input_datasets=2,prediction_attempt_started=inference_attempted,
      prediction_forward_started=True if predictions else False if known_pre_forward_failure or not inference_attempted else None,
      completed_prediction_models=len(predictions),expected_prediction_models=37,all_prediction_seal_completed=all_predictions,
      predictions_independent_accepted=stages.get('audit_predictions',{}).get('independent_review_passed') is True,
      completed_local_selection_models=len(local_selections),expected_local_selection_models=24,
      completed_external_selection_models=len(external_selections),expected_external_selection_models=6,
      local_selection_seal_completed='local_selections_seal' in completed,
      selection_queue_state=(selection_queue or {}).get('state','not_started'),selection_queue_current_stage=(selection_queue or {}).get('stage') or None,
      selection_queue_current_step=(selection_queue or {}).get('step'),selection_queue_expected_steps=31,
      selection_queue_error=(selection_queue or {}).get('error') if (selection_queue or {}).get('state')=='failed' else None,
      selection_queue_reported_completed_steps=len((selection_queue or {}).get('completed',[])),
      selection_audit_completed=selection_audit_completed,selection_audit_independently_accepted=selection_audit_accepted,
      current_processing=processing,
      local_geometry_completed='local_geometry' in completed,local_geometry_independent_accepted=stages.get('audit_geometry',{}).get('independent_review_passed') is True,
      test_accessed=True if inputs or predictions else None if input_attempted else False,
      test_access_definition='Reserved input image/valid preparation or completed prediction; this is not label access or scored results.',
      R6_accessed=True if predictions else False if known_pre_forward_failure or not inference_attempted else None,
      R6_access_definition='True only after a full four-R prediction stage completes; false for a documented pre-forward GPU guard failure or no attempt; otherwise unknown for partial/failed attempts.',
      test_labels_accessed=True if truth_complete else None if truth_names else False,
      truth_arrays_accessed=True if truth_complete else None if truth_names else False,
      test_labels_accessed_semantics='Truth-array materialization/decoding only; legacy metadata may contain reference scalars.',
      truth_materialization_attempted=bool(truth_names),
      completed_truth_datasets=len(truth_complete),completed_outcome_stages=outcomes,
      confirmation_outcome_report_completed=report_completed,confirmation_outcome_report_independently_accepted=report_accepted,final_confirmation_complete=final,
      final_confirmation_definition='All37 sealed predictions, both truth stages, all three outcomes and report tables completed; exact passed report audit must also be selected by the root report_output_acceptance pointer and match the freeze, report manifest and summary hashes. This is not publication readiness.',
      status_evidence_limit='Read-only JSON status/artifact inventories. No independent payload-integrity validation, prediction scoring or process-liveness inspection.')

def observe_confirmation(home,snapshot):
    """Copy exact observed JSON bytes into an immutable snapshot, never live pointers."""
    archive=snapshot/'confirmation_metadata';archive.mkdir();observed={};records=[]
    def capture(path):
        path=Path(path)
        if not path.exists():return None
        assert path.suffix=='.json';raw=path.read_bytes();obj=json.loads(raw.decode('utf-8-sig'));h=hashlib.sha256(raw).hexdigest()
        dest=archive/f'{len(records):03d}_{path.name}';dest.write_bytes(raw)
        records.append({'source_path':str(path.resolve()),'sha256':h,'snapshot_path':str(dest.resolve())});observed[str(path.resolve())]=(obj,h)
        return obj
    freeze=capture(home/'FREEZE.json')
    if freeze:
        assert freeze.get('passed') is True and freeze.get('frozen') is True and freeze['scope']=='confirmation_v1'
        assert freeze['R']==[1,2,4,6] and len(freeze['run_ids'])==37
        for name,key in [('SOURCE_LOCK.json','source_lock_sha256'),('contract.json','contract_sha256'),('checkpoints.json','checkpoints_sha256')]:
            capture(home/name);assert observed[str((home/name).resolve())][1]==freeze[key]
    launch=capture(home.parent/'confirmation_runtime/prediction_launch.json')
    if launch and launch.get('root_review'):
        pin=launch['root_review'];capture(Path(pin['path']));assert observed[str(Path(pin['path']).resolve())][1]==pin['sha256']
    original_queue=capture(home/'stages/prediction_queue/queue_status.json');queue=original_queue;stages={}
    recovery_claim=capture(home.parent/'confirmation_runtime/PREDICTION_RECOVERY_V1_CLAIM.json')
    recovery_queue=None
    if recovery_claim:
        recovered=Path(recovery_claim['output']).resolve()
        assert recovered.is_relative_to((home.parent/'provenance').resolve()),'Recovery metadata path escaped provenance'
        recovery_queue=capture(recovered/'queue_status.json')
        capture(recovered/'supersession.json')
        queue=recovery_queue or {'state':'initializing','owner_pid':None,'completed':[],'models':37,'steps':38}
    selection_queue=capture(home/'stages/selection_queue/queue_status.json')
    selection_expected={'local':[],'external':[]}
    if freeze:
        contract=observed[str((home/'contract.json').resolve())][0]
        local=[x for x in contract['roster'] if x['dataset']=='local' and x['variant']!='regression']
        external=[x for x in contract['roster'] if x['dataset']=='weeds' and x['variant']!='regression']
        assert len(local)==24 and len(external)==6
        selection_expected={'local':[f'local_select_{i:02d}' for i in range(1,25)],'external':['select_'+x['run_id'] for x in external]}
    selection_names=selection_expected['local']+['local_selections_seal']+selection_expected['external']
    if selection_queue is not None:
        assert freeze and selection_queue['models']==30 and selection_queue['steps']==31
        assert selection_queue['freeze']['sha256']==observed[str((home/'FREEZE.json').resolve())][1]
        assert Path(selection_queue['freeze']['path']).resolve()==(home/'FREEZE.json').resolve()
        pin=selection_queue['root_review'];grant=capture(Path(pin['path']))
        assert observed[str(Path(pin['path']).resolve())][1]==pin['sha256']
        assert grant.get('passed') is True and grant['scope']=='confirmation_selection_queue' and grant['freeze_sha256']==selection_queue['freeze']['sha256']
        assert grant['run_ids']==[x['run_id'] for x in contract['roster'] if x['variant']!='regression']
        assert grant['implementation_sha256']==observed[str((home/'SOURCE_LOCK.json').resolve())][0]['files']['run_selection_queue.py']
        assert selection_queue.get('stage') in selection_names+[None]
    if (home/'stages').exists():
        for folder in sorted((home/'stages').iterdir()):
            if not folder.is_dir():continue
            status=capture(folder/'status.json');manifest=capture(folder/'artifact_manifest.json')
            if status is None:continue
            complete=status.get('state')=='completed' and manifest is not None
            if complete:
                # Status is atomically written before the final manifest. In-progress
                # transition snapshots remain unverified rather than claiming completion.
                complete=manifest.get('files',{}).get('status.json')==observed[str((folder/'status.json').resolve())][1]
            stages[folder.name]={'state':status.get('state'),'completed_metadata_verified':bool(complete),'artifact_manifest_present':manifest is not None,
              'at':status.get('at'),'error':status.get('error')}
            audit_scopes={'audit_geometry':'confirmation_geometry_independent','audit_predictions':'confirmation_predictions_independent','audit_all_selections':'confirmation_all_selections_independent'}
            if folder.name in audit_scopes and complete:
                audit=capture(folder/'review.json')
                assert manifest['files']['review.json']==observed[str((folder/'review.json').resolve())][1]
                stages[folder.name]['independent_review_passed']=audit.get('passed') is True and audit.get('scope')==audit_scopes[folder.name] and audit.get('freeze_sha256')==observed[str((home/'FREEZE.json').resolve())][1]
            if folder.name.startswith('predict_') and status.get('state')=='failed':
                failure=capture(folder/'failure.json')
                stages[folder.name]['failed_before_prediction_seal']=bool(failure and failure.get('error')=="ValueError('GPU must be hidden')" and "require(not torch.cuda.is_available(),'GPU must be hidden')" in failure.get('traceback','') and not (folder/'prediction_input_seal.json').exists())
    if selection_queue is not None:
        done=selection_queue.get('completed',[])
        assert [x['stage'] for x in done]==selection_names[:len(done)] and len(done)<=31
        for item in done:
            folder=home/'stages'/item['stage'];pin=item['artifact_manifest']
            assert stages.get(item['stage'],{}).get('completed_metadata_verified') is True
            assert Path(pin['path']).resolve()==(folder/'artifact_manifest.json').resolve()
            assert observed[str((folder/'artifact_manifest.json').resolve())][1]==pin['sha256']
    if stages.get('audit_all_selections',{}).get('independent_review_passed'):
        audit=observed[str((home/'stages/audit_all_selections/review.json').resolve())][0]
        assert [x['run_id'] for x in audit['models']]==[x['run_id'] for x in contract['roster'] if x['variant']!='regression']
        assert audit['complete_case_inventory_verified'] is True and audit['truth_used_for_choices'] is False
        assert audit['all_predictions_sha256']==observed[str((home/'stages/predictions_seal/artifact_manifest.json').resolve())][0]['files']['all_predictions.json']
    report_pointer=capture(home.parent/'confirmation_runtime/report_output_acceptance.json')
    if report_pointer is not None:
        report_folder=home/'stages/report_tables'
        assert stages.get('report_tables',{}).get('completed_metadata_verified') is True,'Report acceptance requires completed report metadata'
        review_path=Path(report_pointer['path']);assert review_path.is_absolute(),'Acceptance review path must be absolute'
        review=capture(review_path);assert review is not None
        capture(report_folder/'summary.json')
        manifest,manifest_hash=observed[str((report_folder/'artifact_manifest.json').resolve())]
        accepted=validate_report_acceptance(report_pointer,review,observed[str(review_path.resolve())][1],
          observed[str((home/'FREEZE.json').resolve())][1],report_folder,manifest,manifest_hash,
          observed[str((report_folder/'summary.json').resolve())][1])
        stages['report_tables']['independent_review_root_accepted']=accepted
        stages['report_tables']['root_acceptance_pointer']=report_pointer
    phase=derive_confirmation_phase(freeze,launch,queue,stages,selection_queue,selection_expected)
    phase.update(observed_at=datetime.datetime.now().astimezone().isoformat(),freeze_sha256=observed.get(str((home/'FREEZE.json').resolve()),(None,None))[1],
      stage_statuses=stages,queue_status_snapshot=queue,launch_snapshot=launch,metadata_snapshots=records,
      active_queue_kind='explicit_manual_prediction_recovery_v1' if recovery_claim else 'original_prediction_queue',
      active_queue_kind_role='Legacy prediction queue identity; use current_processing for the current activity.',
      original_prediction_queue_snapshot=original_queue,recovery_claim_snapshot=recovery_claim,recovery_queue_snapshot=recovery_queue,
      selection_queue_snapshot=selection_queue,
      report_output_acceptance_snapshot=report_pointer)
    save(snapshot/'confirmation_observation.json',phase)
    return phase,freeze,records

def main():
    index=read(PA/'latest_results.json')
    for row in index['files']:assert sha(PA/row['path'])==row['sha256'],row['path']
    norm=ROOT/'provenance/local_norm_training_independent_20260916_084156_513453'
    spatial=ROOT/'provenance/external_spatial_independent_20260916_084430_561134'
    scale=ROOT/'provenance/scale_contrasts_independent_20260916_070030_408261'
    assert sha(norm/'review.json')=='0442668300af8e57b0042351555cce25157b912b5e9eae25d1e3f14543a83ed5'
    assert sha(spatial/'audit.json')=='a277c7c38538fb2c2d7f152b348e5e546beb12f51c8bdf0450497bb546556d49'
    assert sha(scale/'review.json')=='a30e848d8e320cbd01f771fffd38c814f855770189bf3704fd0c1c1a52a79161'
    for p in (norm/'review.json',spatial/'audit.json',scale/'review.json'):assert read(p)['passed']
    seed=ROOT/'local_seed_stability_v1';sstatus=read(seed/'queue_status.json') if (seed/'queue_status.json').exists() else {'state':'prepared_not_started','completed':[]}
    post=seed/'postprocess_v1';poststatus=read(post/'queue_status.json') if (post/'queue_status.json').exists() else None
    post2=seed/'postprocess_v2';post2status=read(post2/'queue_status.json') if (post2/'queue_status.json').exists() else None
    nstatus=read(ROOT/'local_norm_regression_v1/queue_status.json');assert nstatus['state']=='completed' and len(nstatus['completed'])==4
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');now=datetime.datetime.now().astimezone().isoformat()
    snapshot=ROOT/'provenance/state_snapshots'/stamp;snapshot.mkdir(parents=True)
    old=[ROOT/'CAMPAIGN_STATUS.md',ROOT/'NEXT_DEVELOPMENT_PLAN.md',ROOT/'CHANGELOG.md',ROOT/'progress.md',PA/'README.md',PA/'latest_results.json',PA/'latest_index_verification.json']
    for p in old:
        if p.exists():shutil.copy2(p,snapshot/p.name)
    shutil.copy2(__file__,snapshot/'refresh_development_state.py')
    phase,freeze,confirmation_snapshots=observe_confirmation(ROOT/'confirmation_v1',snapshot)
    save(snapshot/'manifest.json',{'created_local':now,'files':{p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*') if p.is_file()}})
    additions=[]
    folders=[norm,spatial,scale,PA/'scale_contrasts_20260916_065523_428384',ROOT/'external_weeds_v1/spatial_v1/reports/full6_20260916_083942_697081']
    for folder in folders:additions.extend(p for p in folder.rglob('*') if p.is_file())
    result=ROOT/'external_weeds_v1/spatial_v1/results_20260916_065643_813785'
    additions.extend(p for p in result.iterdir() if p.is_file() and p.suffix in ('.json','.md'))
    additions += [ROOT/'external_weeds_v1/spatial_v1/root_completed_acceptance.json',PA/'PA_declarations_preparation.md']
    additions += [ROOT/'local_norm_regression_v1'/n for n in ['SOURCE_LOCK.json','serial_plan.json','RESEARCH_PROTOCOL.json','root_production_review.json','root_production_launch.json','queue_status.json']]
    additions += [seed/n for n in ['SOURCE_LOCK.json','INPUT_LOCK.json','protocol.json','plan.json','run_seed_queue.py']]
    word=PA/'manuscript_word_20260916_v2_1c';word_qa=read(word/'visual_qa.json')
    assert word_qa['accepted_for_development_content_review'] and not word_qa['submission_ready']
    assert sha(word/'PA_manuscript_development_v2.docx')==word_qa['docx_sha256']
    for folder in [word,PA/'local_publication_figures_20260916_092355_358608',PA/'external_publication_figure_20260916_090533_178532',ROOT/'provenance/integrated_v2_author_self_review_20260916_091748_328542',ROOT/'provenance/local_seed_export_independent_20260916_092142_029248',seed/'export_focuschecks/20260916_092027_457382']:
        additions.extend(p for p in folder.rglob('*') if p.is_file() and not any(x in p.parts for x in ('temp','mpl_config','__pycache__')))
    additions.extend(PA.glob('PA_manuscript_integrated_development_v2*'))
    additions += [seed/n for n in ('root_execution_review.json','root_launch.json','QUEUE_CLAIM.json','export_protocol.json','EXPORT_INTERFACE.md')]
    additions += [ROOT/'provenance/confirmation_readiness_memo.md',ROOT/'provenance/confirmation_readiness_sources.json']
    if poststatus and poststatus['state'] in ('failed','completed'):
        additions.extend(p for p in post.rglob('*') if p.is_file())
        for receipt in (post/'run_records').glob('*.completed.json'):
            record=read(receipt)
            for directory in ('export_directory','spatial_directory'):
                additions.extend(p for p in Path(record[directory]).iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md'))
        for folder in [ROOT/'provenance/seed_postprocess_independent_20260916_093538_892753',ROOT/'provenance/local_seed_management_readiness_20260916_093300_444463']:
            additions.extend(p for p in folder.rglob('*') if p.is_file())
        additions += [seed/'root_spatial_execution_review.json',ROOT/'precision_agriculture/RM_local_seed_protocol_v1.json']
    if post2status:
        additions += [post2/name for name in ('protocol.json','recovery.json','root_execution_review.json','root_launch.json','QUEUE_CLAIM.json','PREPARATION_HANDOFF.md')]
        additions += [seed/'root_spatial_execution_review_v2.json',seed/'export_v2/protocol.json',PA/'RM_local_seed_protocol_v2.json']
        additions += [ROOT/name for name in ('export_local_seed_validation_v2.py','evaluate_local_seed_management_v2.py','run_local_seed_postprocess_queue_v2.py')]
        for folder in [ROOT/'provenance/seed_postprocess_v2_independent_20260916_100014_257144',ROOT/'provenance/seed_export_v2_independent_20260916_095705_201592',ROOT/'provenance/completed_seed_outputs_independent_20260916_095939_048943']:
            additions.extend(p for p in folder.rglob('*') if p.is_file())
    seed_index=ROOT/'provenance/seed_numerical_acceptance_index_20260916_103832_972904/index.json'
    numerical_queue=read(ROOT/'seed_numeric_review_v1/queue_status.json')
    if numerical_queue.get('full16_index'):
        item=numerical_queue['full16_index'];seed_index=Path(item['path']);assert sha(seed_index)==item['sha256']
    seed_accepted=read(seed_index)
    registered_ids={x['run_id'] for x in seed_accepted['new_models']}
    observed_accepted=len(registered_ids)
    for item in numerical_queue.get('completed',[]):
        path=Path(item['review']);assert sha(path)==item['review_sha256'];review=read(path)
        assert review['passed'] and sum(x['run_id']==item['run_id'] for x in review['models'])==1
        additions.append(path)
        if item['run_id'] not in registered_ids:registered_ids.add(item['run_id']);observed_accepted+=1
    additions.append(seed_index)
    for entry in seed_accepted['new_models']:
        for key in ('completion_receipt','numerical_audit'):
            p=Path(entry[key]['path']);assert sha(p)==entry[key]['sha256'];additions.append(p)
    closed_preparations=['seed_summary_independent_20260916_102709_016960','confirmation_queue_independent_20260916_102429_879432',
       'confirmation_selection_queue_independent_20260916_103409_402787','confirmation_local_outcomes_independent_20260916_103225_191542']
    for name in closed_preparations:
        additions.extend(p for p in (ROOT/'provenance'/name).rglob('*') if p.is_file())
    closed_reviews=[
      'confirmation_prediction_geometry_independent_20260916_105603_580937/review.json',
      'confirmation_report_independent_20260916_105448_774935/review.json',
      'checkpoint_builder_independent_20260916_111423_552325/review.json',
      'confirmation_geometry_independent_20260916_112001_715969/review.json',
      'confirmation_selection_auditor_independent_20260916_112036_221638/review.json',
      'confirmation_ext_scalar_independent_preparation_20260916_112225_851701/review.json',
      'confirmation_local_auditor_independent_20260916_112644_158893/review.json',
      'full24_reviewer_root_source_acceptance_20260916_1126.json',
      'full24_reviewer_preparation_20260916_111543_479211/review.json',
      'full24_reviewer_checks_20260916_111302_906777/focused_checks.json',
      'repro_preparation_independent_20260916_105414_593159/review.json']
    additions.extend(ROOT/'provenance'/name for name in closed_reviews)
    framework=PA/'allocation_framework_20260916_103409_851025'
    assert read(framework/'visual_qa.json')['accepted_for_manuscript_integration']
    additions.extend(p for p in framework.iterdir() if p.is_file())
    additions.extend(p for p in PA.glob('agronomic_literature_*') if p.is_file())
    additions.extend(p for p in (PA/'agronomic_literature_sources_20260916_100740_663417').iterdir() if p.is_file())
    full24=PA/'seed_summary_v1_20260916_114615_738298'
    full24_review=ROOT/'provenance/full24_seed_summary_independent_20260916_114628_210412/review.json'
    findings=PA/'full24_key_findings_20260916_114642_017861'
    figures=PA/'full24_publication_figures_20260916_114642_116802'
    full24_summary=read(full24/'summary.json');full24_audit=read(full24_review)
    assert full24_summary['models']==24 and full24_audit['passed'] and full24_audit['tables_audited']==25
    assert full24_audit['summary_sha256']==sha(full24/'summary.json') and full24_audit['audited_artifact_manifest_sha256']==sha(full24/'artifact_manifest.json')
    assert len(seed_accepted['new_models'])==observed_accepted==16
    figure_specs=read(figures/'figure_specs.json')
    for folder in (full24,findings,figures):
        additions.extend(p for p in folder.rglob('*') if p.is_file() and not any(x in p.parts for x in ('temp','mpl_config','__pycache__')))
    additions.extend([full24_review,PA/'FINAL_INTEGRATION_CHECKLIST.md'])
    accepted_paths,manuscript=accepted_manuscript_artifacts();additions.extend(accepted_paths)
    old_figures=figures;figures=manuscript['figures'];figure_specs=read(figures/'figure_specs.json')
    latest_word=manuscript['word'];delivery=manuscript['delivery'];extra=manuscript['extra']
    repro_plan=ROOT/'INTERNAL_REPRO_FINAL_MANIFEST_UPDATE_PLAN.md';additions.append(repro_plan)
    promotion=ROOT/'provenance/confirmation_promotion_20260916_114922_983479/result.json'
    promoted=read(promotion);assert promoted['passed']
    assert freeze and phase['freeze_sha256']==promoted['formal_freeze']['sha256']=='24f8d1605c58df6bd6169b8434a451975abca0c3f8176e9c55c252abf8f3e732'
    additions.append(promotion)
    for item in [promoted['inventory'],freeze['root_promotion'],freeze['readiness_inventory'],*freeze['development_completion'].values()]:
        p=Path(item['path']);assert sha(p)==item['sha256'];additions.append(p)
    assert read(Path(promoted['inventory']['path']))['passed']
    # Snapshot mutable runtime metadata; never index a live status file as immutable.
    additions.extend(p for p in snapshot.rglob('*') if p.is_file())
    table={x['path']:x for x in index['files']}
    for p in additions:table[rel(p)]={'path':rel(p),'sha256':sha(p)}
    current_pointers={
      'manuscript_docx':rel(latest_word/'PA_manuscript_development_v3_1.docx'),
      'primary_result_directory':rel(full24),
      'primary_result_directory_role':'Accepted complete24 local development summary; confirmation results remain separate and incomplete.',
      'current_training_status':{'state':'completed','role':'Finished development training references; no active training queue.',
        'common12_reference':'../common_queue_status.json','seed16_reference':'../local_seed_stability_v1/queue_status.json',
        'external9_reference':'../external_weeds_v1/current_queue.json','norm4_reference':'../local_norm_regression_v1/queue_status.json'},
      'current_common_report_directory':rel(full24),'current_common_results_draft':rel(Path(extra['manuscript']['path'])),
      'integrated_manuscript_markdown':rel(Path(extra['manuscript']['path'])),
      'integrated_manuscript_word':rel(latest_word/'PA_manuscript_development_v3_1.docx'),
      'integrated_manuscript_visual_review':rel(latest_word/'visual_qa.json'),'integrated_manuscript_pages':20,
      'local_full24_figures':rel(figures)}
    history=index.setdefault('historical_pointers',[])
    for key,value in current_pointers.items():
        if key in index and index[key]!=value:
            history.append({'key':key,'value':index[key],'superseded_local':now,'prior_index_snapshot':rel(snapshot/'latest_results.json')})
    index.update(current_pointers)
    processing=phase['current_processing'];queue_source=processing['queue_source']
    current_queue_path=(ROOT/'confirmation_v1/stages/selection_queue/queue_status.json') if queue_source=='selection' else (
      Path(phase['recovery_claim_snapshot']['output'])/'queue_status.json' if phase['recovery_claim_snapshot'] else ROOT/'confirmation_v1/stages/prediction_queue/queue_status.json') if queue_source=='prediction' else None
    active_record=next((x for x in confirmation_snapshots if current_queue_path and Path(x['source_path'])==current_queue_path.resolve()),None)
    index.update(schema_version=8,updated_local=now,
      scope='Full24 local development completed and audited; formal confirmation phase observed from JSON metadata. Submission package incomplete.',
      weedsgalore_spatial_executed=True,weedsgalore_spatial_accepted=True,weedsgalore_spatial_result=rel(result),weedsgalore_spatial_independent_review=rel(spatial/'audit.json'),
      weedsgalore_spatial_root_acceptance='../external_weeds_v1/spatial_v1/root_completed_acceptance.json',
      local_norm_training_completed=4,local_norm_independent_review=rel(norm/'review.json'),local_norm_queue_snapshot=nstatus,
      local_scale_contrasts='scale_contrasts_20260916_065523_428384',local_scale_contrasts_review=rel(scale/'review.json'),
      local_seed_extension_protocol='../local_seed_stability_v1/protocol.json',local_seed_extension_status_snapshot=sstatus,
      local_seed_postprocess_v1_status_snapshot=poststatus,local_seed_postprocess_v1_numerically_accepted=True,
      local_seed_postprocess_v1_accepted_completed_models=2,local_seed_postprocess_v2_status_snapshot=post2status,
      integrated_manuscript_pdf=rel(latest_word/'PA_manuscript_development_v3_1.pdf'),
      integrated_supplement_word=rel(latest_word/'PA_supplementary_figures_v3_1.docx'),integrated_supplement_pdf=rel(latest_word/'PA_supplementary_figures_v3_1.pdf'),integrated_supplement_pages=4,
      integrated_manuscript_figure_specs=rel(Path(extra['figure_specs']['path'])),integrated_manuscript_equation_specs=rel(Path(extra['equation_specs']['path'])),
      integrated_manuscript_integration_report=rel(manuscript['integration']/'integration_report.json'),integrated_manuscript_integration_supplement=rel(manuscript['supplement']/'integration_supplement.json'),
      integrated_manuscript_content_review=rel(manuscript['content']),integrated_manuscript_delivery=rel(manuscript['delivery_path']),
      manuscript_submission_ready=False,chinese_agricultural_positioning=rel(manuscript['chinese']),internal_repro_final_manifest_update_plan=rel(repro_plan),
      word_content_version='Accepted v3.1 full24 development manuscript:20 main pages+4 supplement pages. Confirmation results not yet integrated; not submission-ready.',allocation_framework=rel(framework),
      local_seed_numerical_acceptance_index=rel(seed_index),local_seed_numerically_accepted=observed_accepted,seed_numerical_queue_snapshot=numerical_queue,
      local_full24_summary=rel(full24),local_full24_independent_review=rel(full24_review),local_full24_models=24,
      local_full24_table_rows=full24_summary['table_rows'],local_full24_recorded_flags=full24_summary['recorded_flags'],
      local_full24_key_findings=rel(findings),local_full24_figures=rel(figures),
      local_full24_figure_status={'generated_figures':len(figure_specs['figures']),'visual_review_passed':True,'visual_review':rel(manuscript['visual']),
        'visual_review_sha256':sha(manuscript['visual']),'artifact_manifest_sha256':sha(figures/'artifact_manifest.json'),'historical_first_version':rel(old_figures)},
      confirmation_promotion=rel(promotion),confirmation_readiness=rel(Path(promoted['inventory']['path'])),
      confirmation_checkpoint_candidate=rel(Path(freeze['development_completion']['checkpoint_candidate']['path'])),
      confirmation_observation=phase,confirmation_observation_snapshot=rel(snapshot/'confirmation_observation.json'),
      current_confirmation_processing=processing,
      active_confirmation_queue={**processing,
        'completed_prediction_models':phase['completed_prediction_models'],'expected_prediction_models':37,'process_liveness_checked':False,
        'completed_local_selection_models':phase['completed_local_selection_models'],'completed_external_selection_models':phase['completed_external_selection_models'],
        'selection_queue_step':phase['selection_queue_current_step'],'selection_queue_steps':31,
        'queue_metadata_snapshot':active_record,'observation_snapshot':rel(snapshot/'confirmation_observation.json')},
      next_spatial_step='Follow observed frozen confirmation phase and root-approved grants. Preserve failures; no automatic retry, checkpoint reselection or outcome-driven tuning.',
      test_accessed=phase['test_accessed'],R6_accessed=phase['R6_accessed'],test_labels_accessed=phase['test_labels_accessed'],
      truth_arrays_accessed=phase['truth_arrays_accessed'],test_labels_accessed_semantics=phase['test_labels_accessed_semantics'],
      confirmation_outcome_report_completed=phase['confirmation_outcome_report_completed'],
      confirmation_outcome_report_independently_accepted=phase['confirmation_outcome_report_independently_accepted'],
      final_confirmation_complete=phase['final_confirmation_complete'],files=list(table.values()))
    save(PA/'latest_results.json',index)
    match=all(sha(PA/r['path'])==r['sha256'] for r in index['files']);assert match,'Indexed source changed during refresh'
    save(PA/'latest_index_verification.json',{'passed':True,'updated_local':now,'indexed_files':len(table),'all_previous_indexed_hashes_unchanged':True,
      'all_indexed_hashes_match':match,'latest_results_sha256':sha(PA/'latest_results.json'),'local_norm_completed':4,'external_spatial_accepted':True,
      'seed_extension_status':sstatus['state'],'local_seed_numerically_accepted':observed_accepted,'local_full24_models':24,
      'confirmation_phase':phase['phase'],'test_accessed':phase['test_accessed'],'R6_accessed':phase['R6_accessed'],
      'test_labels_accessed':phase['test_labels_accessed'],'final_confirmation_complete':phase['final_confirmation_complete'],
      'confirmation_observer_payload_reads':0,'confirmation_observation_snapshot':rel(snapshot/'confirmation_observation.json')})
    label=lambda value: '未知（不能以部分尝试推定成功）' if value is None else ('是' if value else '否')
    error=processing['error'] or '当前阶段无错误元数据'
    current=f'''# 刺萼龙葵研究进度

更新：{now}。目标期刊 **Precision Agriculture**。当前阶段 **{phase['phase']}**。此为JSON元数据快照，不是进程存活探测。

## 完整开发证据

新增16组训练{sstatus['state']}，完成{len(sstatus.get('completed',[]))}/16，原best导出/空间评价数值接受{observed_accepted}/16。前2组后处理v1、剩14组v2及所有历史失败保留；旧训练/后处理/数值队列不重启。

完整本地24分割模型的25表经独立复算：2160预算行、216完整前缀AUC、72 CPU标量、1080同seed硬软预算配对和108 AUC配对。目录：precision_agriculture/{full24.name}/；审核：{full24_review.relative_to(ROOT).as_posix()}；完整关键结果：precision_agriculture/{findings.name}/report.md。seed17/29/43的sample SD描述同一场景训练变异，不是田块CI。旧seed17尺度方向不得当作完整24模型方向。

完整数值标记：{json.dumps(full24_summary['recorded_flags'],ensure_ascii=False)}。异常数、实际降级数、fallback与保留资格标记不同，不称独立最优性证明。修正版七图在precision_agriculture/{figures.name}/，已通过逐页实际视觉审核：{manuscript['visual'].relative_to(ROOT).as_posix()}。旧七图版本仍保留。

原本地12训练/8空间、WeedsGalore9训练/6空间和norm4诊断均已独立核验，全部历史路径仍留在latest_results.json。

## 确认阶段观察

正式FREEZE：{phase['freeze_sha256']}。全37原检查点及R1/2/4/6于开发完成后固定，promotion/readiness通过。它不是对先前探索开发的事前注册。

- 当前处理：**{processing['kind']} / {processing['state']}**；阶段{processing['stages']}；元数据owner PID {processing['owner_pid']}、child PID {processing['child_pid']}，未检查存活。原预测失败及人工恢复属于历史，不作为当前错误。
- 预测历史：{phase['active_queue_kind']} / {phase['prediction_queue_state']}，原owner PID {phase['observed_owner_pid']}（历史身份）。
- image/valid输入准备：**{phase['prepared_input_datasets']}/2**。
- 预测阶段已尝试：{label(phase['prediction_attempt_started'])}；完整预测：**{phase['completed_prediction_models']}/37**；全局预测seal完成：{label(phase['all_prediction_seal_completed'])}；独立预测审核接受：{label(phase['predictions_independent_accepted'])}。
- 本地选区完成：**{phase['completed_local_selection_models']}/24**；外部选区完成：**{phase['completed_external_selection_models']}/6**；本地population seal：{label(phase['local_selection_seal_completed'])}。31步选区队列状态{phase['selection_queue_state']}，当前step={phase['selection_queue_current_step']}、stage={phase['selection_queue_current_stage']}，队列记录完成{phase['selection_queue_reported_completed_steps']}/31。
- 全选区独立审核stage完成：{label(phase['selection_audit_completed'])}；精确来源/全30模型审核接受：{label(phase['selection_audit_independently_accepted'])}。队列完成不等于全选区接受，也不自动授权truth。
- 实际前向开始状态：{label(phase['prediction_forward_started'])}；R6访问：{label(phase['R6_accessed'])}。已证实GPU可见性guard在prediction seal前失败时不写成已前向；其余部分/失败任务没有证据则记未知。
- 保留输入访问：{label(phase['test_accessed'])}；真值数组物化/解码：{label(phase['truth_arrays_accessed'])}；truth完成{phase['completed_truth_datasets']}/2。test_labels_accessed仅表示真值数组物化/解码；旧清单可能包含参考标量，不表示从未读取任何标签相关元数据字节。
- 完成outcomes：{phase['completed_outcome_stages']}；规范表已生成：{label(phase['confirmation_outcome_report_completed'])}；规范表独审并由root接受：{label(phase['confirmation_outcome_report_independently_accepted'])}；完整确认结果并规范报告：**{label(phase['final_confirmation_complete'])}**。规范表仅生成而未接受时不记确认完成；确认完成也不等于投稿准备完成。
- 本地有效域几何完成：{label(phase['local_geometry_completed'])}；对应独立几何审核接受：{label(phase['local_geometry_independent_accepted'])}。
- 错误元数据：`{error}`。

失败必须保留原stage、claim及队列，仅执行root明确审核的有限恢复，不自动重试或改冻结科学源。全部37预测和30选区封存、独立审核与root接受后才能物化真值。确认结果不得回流调参。

## 解释与稿件边界

本地单场景，开发985train/357val；开发有效域1457.4861587858427m²、粗目标14.168571381451866m²。338保留窗的确认域与目标分母须由自己的已接受几何/native真值提供，不沿用开发分母。R是模拟信息退化，M是完整选区网格，不等于新增航高或喷雾设备尺度。

WeedsGalore为单田块多日期；开发104train/26val，确认26保留帧。外部按帧预算与FAE不等于唯一地面面积，不与本地全域平方米捕获水平直接比较。crop圈入不是损伤；捕获/选区面积不是药效、节药、产量或利润。

当前已验收开发稿：precision_agriculture/{Path(extra['manuscript']['path']).name}；Word/PDF为precision_agriculture/{latest_word.name}/，主文20页、补充4页，5表、5个可编辑方程、主图4张/补充图4张。内容独审：{manuscript['content'].relative_to(ROOT).as_posix()}；交付与全页视觉验收：{manuscript['delivery_path'].relative_to(ROOT).as_posix()}。中文农业定位：precision_agriculture/{manuscript['chinese'].name}。

此稿包含完整24模型开发结果，尚未整合确认性结果，不称投稿终稿或作者最终批准稿。历史v2.2/18页v2.1 Word、原v3及旧图原样保留；替换前的入口记录在historical_pointers。作者资料/已纠正联系方式/3基金继承已核版，不虚构贡献、利益冲突或最终认可。内部复现包后续更新仅有计划{repro_plan.name}，本轮未生成最终包或压缩包。

旧状态与本次观察快照：{snapshot.relative_to(ROOT).as_posix()}。活动阶段JSON复制到快照，不把可变文件作为永久证据指针。发表目标未完成；自动跟进由root维护。本刷新未打开确认图像、预测数组、标签、CSV数值或模型，不调用模型/solver、不改CONF或队列。
'''
    (ROOT/'CAMPAIGN_STATUS.md').write_text(current,encoding='utf-8');(ROOT/'progress.md').write_text(current,encoding='utf-8')
    (ROOT/'NEXT_DEVELOPMENT_PLAN.md').write_text(f'''# 后续工作

更新：{now}。先读CAMPAIGN_STATUS.md、precision_agriculture/latest_results.json和真实队列。当前 **{phase['phase']}**。

1. 新增16组及全部数值审核已完成，全24开发25表已通过独立审核；不得重跑训练/旧空间/旧数值队列或按结果扩大搜索。
2. 正式确认已经FREEZE。当前处理{processing['kind']}/{processing['state']}，阶段{processing['stages']}；完整预测{phase['completed_prediction_models']}/37，本地选区{phase['completed_local_selection_models']}/24、外部{phase['completed_external_selection_models']}/6，选区队列step={phase['selection_queue_current_step']}/31。已结束预测/恢复队列只作历史。保留失败，不自动重试、不改冻结CONF。
3. 固定37预测与seal、有效域几何、独立预测/几何审核、全部30选区封存与接受，之后才truth和三个outcomes。部分R6不冒称成功，原MAE检查点不重选，确认不回流调参。
4. 全部确认输出独立复核后再规范汇总。规范表生成后还须独立输出审核通过，root以confirmation_runtime/report_output_acceptance.json精确绑定review路径/SHA；观察器核freeze、规范表manifest及summary哈希后才记完整确认。当前完整确认={phase['final_confirmation_complete']}，规范表生成={phase['confirmation_outcome_report_completed']}、最终接受={phase['confirmation_outcome_report_independently_accepted']}。保留两个数据集、开发/确认域及NA分母的区别。
5. v3.1开发稿及20页主文/4页补充Word/PDF、修正版七图已通过内容/视觉验收，当前入口见precision_agriculture/README.md。下一稿仅整合独审接受的确认性结果，保留预算依赖的全部正负方向；补齐作者待确认事实、许可、复现包和投稿前审查。内部包目前只有{repro_plan.name}，不把开发稿当作终稿。

快照：{snapshot.relative_to(ROOT).as_posix()}。E环境不升级，执行器控制CPU/GPU资源；本刷新只观察JSON，不控制队列。自动跟进由root维护。
''',encoding='utf-8')
    (PA/'README.md').write_text(f'''# Precision Agriculture 工作区

更新：{now}。完整24模型开发已接受；当前确认阶段 **{phase['phase']}**，仍非投稿终稿。

## 当前入口

- [完整24开发25表]({rel(full24)}/report.md)、[独立数值审核]({rel(full24_review)})、[关键结果]({rel(findings)}/report.md)
- [当前v3.1英文正文]({rel(Path(extra['manuscript']['path']))})、[主文Word（20页）]({rel(latest_word/'PA_manuscript_development_v3_1.docx')})、[主文PDF]({rel(latest_word/'PA_manuscript_development_v3_1.pdf')})
- [补充Word（4页）]({rel(latest_word/'PA_supplementary_figures_v3_1.docx')})、[补充PDF]({rel(latest_word/'PA_supplementary_figures_v3_1.pdf')})、[内容独审]({rel(manuscript['content'])})、[交付/视觉验收]({rel(manuscript['delivery_path'])})
- [中文农业价值与投稿定位]({rel(manuscript['chinese'])})、[修正版七图]({rel(figures)}/figure_specs.json)、[七图视觉验收]({rel(manuscript['visual'])})
- [新增16最终数值索引]({rel(seed_index)})、[37固定检查点候选]({rel(Path(freeze['development_completion']['checkpoint_candidate']['path']))})
- [正式promotion]({rel(promotion)})、[全量readiness]({rel(Path(promoted['inventory']['path']))})、[本次状态快照]({rel(snapshot/'confirmation_observation.json')})
- [全部历史证据索引](latest_results.json)、[进度](../CAMPAIGN_STATUS.md)、[后续计划](../NEXT_DEVELOPMENT_PLAN.md)、[整合清单](FINAL_INTEGRATION_CHECKLIST.md)

当前处理为{processing['kind']} / {processing['state']}，阶段{processing['stages']}（owner PID {processing['owner_pid']}，未检查存活）；输入完成{phase['prepared_input_datasets']}/2、完整预测{phase['completed_prediction_models']}/37、本地选区{phase['completed_local_selection_models']}/24、外部选区{phase['completed_external_selection_models']}/6；全选区独审接受={phase['selection_audit_independently_accepted']}，truth完成{phase['completed_truth_datasets']}/2；R6访问={phase['R6_accessed']}，完整确认={phase['final_confirmation_complete']}。null表示证据不足，不等于false。test_labels_accessed/真值数组访问仅表示物化或解码，旧清单可能包含参考标量。旧预测队列只作历史；只按批准的有限恢复前进，不删除失败。

## 历史文稿与结果

- [原seed17八模型报告](common_management_reports/full8_20260916_050927_560091/full_common_management_report_with_figures.md)和[72 AUC](common_prefix_auc_full8_20260916_052212_839933/report.md)，属于全24的历史子集。
- [外部六模型开发空间报告](../external_weeds_v1/spatial_v1/reports/full6_20260916_083942_697081/report.md)、[其独审](../provenance/external_spatial_independent_20260916_084430_561134/review.md)、[norm4诊断](../provenance/local_norm_training_independent_20260916_084156_513453/review.md)。
- [历史v2.2正文](PA_manuscript_integrated_development_v2_2.md)、[18页v2.1 Word](manuscript_word_20260916_v2_1c/PA_manuscript_development_v2.docx)。旧文件不覆盖；最新v3.1开发稿已通过当前范围审核，确认性结果仍待整合。

作者/3基金继承已核版，不虚构CRediT、利益冲突、许可或作者认可。单场景/单外部田块，seed SD不是田块CI；粗目标标注不是植株/冠层，选区面积不是节药/药效，crop圈入不是损伤。
''',encoding='utf-8')
    with (ROOT/'CHANGELOG.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n## {now} 已验收v3.1交付与活动确认状态刷新\n\n当前入口切换到全24开发25表、v3.1正文/规格/整合记录、修正版七图115434及115636视觉验收、v3_1b主文20页和补充4页Word/PDF、120755交付及120827内容独审，新增中文农业定位。新交付文件仅按不可变manifest和明确收据列举，旧指针写入historical_pointers；旧结果/稿件保留。确认阶段{phase["phase"]}，当前处理{processing["kind"]}/{processing["state"]}，本地/外部选区{phase["completed_local_selection_models"]}/24、{phase["completed_external_selection_models"]}/6，全选区独审接受{phase["selection_audit_independently_accepted"]}，输入{phase["prepared_input_datasets"]}/2，完整预测{phase["completed_prediction_models"]}/37。test/R6/truth数组按证据分别为{phase["test_accessed"]}/{phase["R6_accessed"]}/{phase["truth_arrays_accessed"]}，unknown不记false；truth数组标志不排除旧元数据中的参考标量。快照{snapshot.relative_to(ROOT).as_posix()}。冻结CONF/队列未改，确认payload读取0；复现最终包仅补计划未打包。\n')
    print(json.dumps({'indexed_files':len(table),'snapshot':str(snapshot),'seed_numerically_accepted':observed_accepted,'full24_models':24,
      'confirmation_phase':phase['phase'],'prepared_inputs':phase['prepared_input_datasets'],'completed_prediction_models':phase['completed_prediction_models'],
      'current_processing':processing,'completed_local_selection_models':phase['completed_local_selection_models'],
      'completed_external_selection_models':phase['completed_external_selection_models'],'selection_audit_accepted':phase['selection_audit_independently_accepted'],
      'test_accessed':phase['test_accessed'],'R6_accessed':phase['R6_accessed'],'test_labels_accessed':phase['test_labels_accessed'],
      'final_confirmation_complete':phase['final_confirmation_complete'],'updated_local':now},ensure_ascii=False))
if __name__=='__main__':main()
