"""Standard-library, metadata-only DRAFT grants. Never freezes, approves or executes.

Payload bytes (arrays, tensors, images and CSV values) are not opened. Completed
stage metadata/inventory is checked; producers and independent audits retain
responsibility for payload byte validation. Every consumed metadata/source pin
is captured before parsing and checked again before and after draft writing.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,re,sys

ROOT=Path(__file__).resolve().parent
PYTHON=__import__('sys').executable
RS=[1,2,4,6]
OUTCOMES=('scalar_outcomes','local_outcomes','external_outcomes')
STAGES=('prediction_queue','local_geometry','audit_predictions','audit_geometry','select_local','select_external','selection_queue','audit_selections','accept_choices','local_outcomes','external_outcomes','scalar_outcomes','audit_local_outcomes','audit_external_scalar','report_tables')
PROGRAM={
 'prediction_queue':('run_prediction_queue.py','confirmation_prediction_queue'),
 'local_geometry':('local_geometry.py','confirmation_local_geometry'),
 'audit_predictions':('audit_prediction_geometry.py','confirmation_prediction_geometry_audit'),
 'audit_geometry':('audit_prediction_geometry.py','confirmation_prediction_geometry_audit'),
 'select_local':('local_management.py','confirmation_local_management'),
 'select_external':('external_spatial.py','confirmation_external_selection_execution'),
 'selection_queue':('run_selection_queue.py','confirmation_selection_queue'),
 'audit_selections':('audit_selections.py','confirmation_all_selections_audit_execution'),
 'accept_choices':('pipeline.py','confirmation_all_label_free_selections'),
 'local_outcomes':('local_outcomes.py','confirmation_local_outcomes'),
 'external_outcomes':('external_spatial.py','confirmation_external_evaluation_execution'),
 'scalar_outcomes':('scalar_outcomes.py','confirmation_scalar_outcomes'),
 'audit_local_outcomes':('../review_confirmation_local_outputs.py','confirmation_local_outputs_independent'),
 'audit_external_scalar':('../review_confirmation_external_scalar.py','confirmation_external_scalar_independent'),
 'report_tables':('report_tables.py','confirmation_report_tables')}
EVIDENCE={
 'prediction_queue':set(),'local_geometry':set(),'audit_predictions':set(),'audit_geometry':set(),
 'select_local':{'predictions_review','geometry_review'},'select_external':{'predictions_review','geometry_review'},
 'selection_queue':{'predictions_review','geometry_review','local_review','external_review'},
 'audit_selections':{'predictions_review','geometry_review'},'accept_choices':{'selections_review'},
 'local_outcomes':{'choice_review'},'external_outcomes':{'choice_review'},'scalar_outcomes':{'choice_review'},
 'audit_local_outcomes':{'choice_review','geometry_review','implementation_review'},
 'audit_external_scalar':{'choice_review','implementation_review'},
 'report_tables':{'scalar_review','local_review','external_review'}}

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def beneath(root,name):
    p=(Path(root)/name).resolve();require(p.is_relative_to(Path(root).resolve()),'Path escapes metadata stage');return p

class Metadata:
    def __init__(self,root):self.root=Path(root).resolve();self.home=self.root/'confirmation_v1';self.watched={};self.inventories={};self.source_lock={}
    def pin(self,path,expected=None):
        p=Path(path).resolve();require(p.suffix.lower() in ('.json','.py','.md','.txt','.toml','.yaml','.yml'), 'Payload or nonmetadata file refused: '+str(p))
        h=digest(p);require(expected is None or expected==h,'Hash mismatch: '+str(p));require(str(p) not in self.watched or self.watched[str(p)]==h,'Source changed: '+str(p));self.watched[str(p)]=h
        return {'path':str(p),'sha256':h}
    def read(self,path,expected=None):
        pin=self.pin(path,expected);require(Path(path).suffix.lower()=='.json','Only JSON metadata parsed');return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
    def binding_read(self,item):
        require(isinstance(item,dict) and set(item)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'Evidence requires explicit path and SHA256')
        return self.read(item['path'],item['sha256'])
    def finish(self):
        for p,h in self.watched.items():require(digest(p)==h,'Consumed source drift: '+p)
        for folder,names in self.inventories.items():require(self.inventory(Path(folder))==names,'Stage file inventory changed')
    @staticmethod
    def inventory(folder):return {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    def stage(self,name):
        f=beneath(self.home/'stages',name);manifest=self.read(f/'artifact_manifest.json');require(isinstance(manifest['files'],dict),'Bad artifact inventory')
        for rel,h in manifest['files'].items():
            p=beneath(f,rel);require(p.is_file() and re.fullmatch('[0-9a-f]{64}',h) is not None,'Missing artifact or invalid digest')
        require(self.inventory(f)==set(manifest['files']),'Incomplete/extraneous stage files')
        require(self.read(f/'status.json',manifest['files']['status.json'])['state']=='completed','Incomplete stage')
        self.inventories[str(f)]=set(manifest['files']);return f,manifest
    def stage_json(self,name,filename):
        f,m=self.stage(name);return self.read(beneath(f,filename),m['files'][filename])
    def fresh(self,names):
        for name in names:require(not (self.home/'stages'/name).exists() and not (self.home/'claims'/(name+'.json')).exists(),'Existing claim/stage cannot be retried: '+name)
    def freeze(self):
        # First read: a real root freeze must already exist. No dataset access.
        freeze=self.read(self.home/'FREEZE.json')
        require(freeze.get('passed') is True and freeze.get('frozen') is True and freeze.get('scope')=='confirmation_v1','Formal root FREEZE required')
        lock=self.read(self.home/'SOURCE_LOCK.json',freeze['source_lock_sha256']);require(lock['version']=='confirmation_source_v1','Wrong lock')
        for rel,h in lock['files'].items():self.pin(beneath(self.home,rel),h)
        self.source_lock=lock['files']
        c=self.read(self.home/'contract.json',freeze['contract_sha256']);tab=self.read(self.home/'checkpoints.json',freeze['checkpoints_sha256']);models=tab['models']
        require(c['version']=='confirmation_v1' and c['R']==RS and c['intermediate_sizes']==[256,128,64,42] and c['output_size']==256,'Wrong confirmation support')
        require(c['CPU_threads']==2 and c['batch_size']==4 and c['dtype']=='float32' and c['val_R6'] is False and c['norm_diagnostics_included'] is False,'Wrong frozen resources/scope')
        require(tab.get('frozen') is True and len(models)==len(c['roster'])==37 and len({x['run_id'] for x in models})==37,'Wrong final model population')
        for model,row in zip(models,c['roster']):
            require(all(model[k]==v for k,v in row.items()),'Checkpoint/roster order mismatch');require(re.fullmatch('[A-Za-z0-9_-]+',model['run_id']) is not None,'Unsafe run ID')
        for ds,n,s in [('local',28,24),('weeds',9,6)]:
            own=[x for x in models if x['dataset']==ds];require(len(own)==n and sum(x['variant']!='regression' for x in own)==s,'Dataset/segmentation inventory')
        require(c['datasets']['local']['count']==338 and c['datasets']['weeds']['count']==26,'Dataset counts changed')
        require(freeze['R']==RS and freeze['run_ids']==[x['run_id'] for x in models] and freeze['truth_after_all_predictions_and_selections'] is True,'Freeze population/phase differs')
        for item in c['development_plans']:self.binding_read(item)
        self.finish();self.contract=c;self.models=models;self.freeze_sha=self.pin(self.home/'FREEZE.json')['sha256'];return c,models
    def predictions(self):
        s=self.stage_json('predictions_seal','all_predictions.json');models=self.models
        require(s['freeze_sha256']==self.freeze_sha and s['run_ids']==[x['run_id'] for x in models] and [x['run_id'] for x in s['models']]==s['run_ids'],'Prediction seal identity')
        require(s['all_declared_models_completed'] is True and s['scalar_rows']==38792 and s['map_count']==8268,'Incomplete prediction totals')
        for row,model in zip(s['models'],models):
            f,m=self.stage('predict_'+model['run_id']);require(Path(row['stage']).resolve()==f,'Prediction stage path')
            n=338 if model['dataset']=='local' else 26
            require(row['dataset']==model['dataset'] and row['variant']==model['variant'] and row['scalar_rows']==4*n and row['map_count']==(0 if model['variant']=='regression' else n),'Per-model prediction counts')
            for key in ('artifact_manifest','metadata','map_index','predicted_scalars'):
                suffix='.csv' if key=='predicted_scalars' else '.json';p=f/(key+suffix)
                require(Path(row[key]['path']).resolve()==p,'Prediction artifact path')
                expected=self.pin(p)['sha256'] if key=='artifact_manifest' else m['files'][p.name]
                require(row[key]['sha256']==expected,'Prediction artifact seal mismatch')
        return s
    def geometry(self):
        g=self.stage_json('local_geometry','metadata.json')
        require(g['geometry_kind']=='native_alpha_only' and g['split']=='test_sealed' and g['complete_declared_domain'] is True and g['expected_windows']==338 and len(g['windows'])==338,'Geometry population/scope')
        require(g['annotation_arrays_opened']==0 and g['mask_paths_consumed'] is False,'Geometry truth boundary')
        return g
    def audit(self,item,scope,**bindings):
        a=self.binding_read(item);require(a.get('passed') is True and not a.get('draft',False) and a.get('scope')==scope,'Qualified independent audit missing')
        for key,value in bindings.items():require(a.get(key)==value,'Independent audit identity: '+key)
        return a
    def completed_audit(self,item,mode,**bindings):
        name='audit_'+mode;f,manifest=self.stage(name)
        require(Path(item['path']).resolve()==f/'review.json' and item['sha256']==manifest['files']['review.json'],'Audit must be the exact completed auditor output')
        program='audit_selections.py' if mode=='all_selections' else 'audit_prediction_geometry.py'
        scope='confirmation_all_selections_independent' if mode=='all_selections' else 'confirmation_'+mode+'_independent'
        return self.audit(item,scope,implementation_sha256=self.pin(self.home/program)['sha256'],**bindings)
    def choice_population(self,review=None):
        local=[x for x in self.models if x['dataset']=='local' and x['variant']!='regression'];seg=[x for x in self.models if x['variant']!='regression']
        population=self.stage_json('local_selections_seal','all_local_selections.json')
        require([x['run_id'] for x in population['models']]==[x['run_id'] for x in local] and population['freeze_sha256']==self.freeze_sha and population['all_predictions_sha256']==self.pred_sha and population['truth_used_for_choices'] is False,'Local population binding')
        require(population['P_solver_cases']==1440 and population['policy_choices']==2880,'Local population case totals')
        entries=[]
        for model in seg:
            name=f"local_select_{[x['run_id'] for x in local].index(model['run_id'])+1:02d}" if model['dataset']=='local' else 'select_'+model['run_id']
            seal=self.stage_json(name,'selection_seal.json');pin=self.pin(self.home/'stages'/name/'selection_seal.json');count=120 if model['dataset']=='local' else 3120
            require(seal['run_id']==model['run_id'] and seal['freeze_sha256']==self.freeze_sha and seal['all_predictions_sha256']==self.pred_sha and seal['truth_used_for_choices'] is False and seal['case_count']==count,'Selection scope/population')
            if model['dataset']=='local':
                item=population['models'][local.index(model)];require(item['selection_seal']==pin and seal['checkpoint_sha256']==model['checkpoint_sha256'],'Local selection model binding')
            entries.append({'run_id':model['run_id'],'selection_seal':pin,'case_count':count})
        if review is not None:
            a=self.audit(review,'confirmation_all_label_free_selections',freeze_sha256=self.freeze_sha,all_predictions_sha256=self.pred_sha,truth_used_for_choices=False,complete_case_inventory_verified=True)
            require([x['run_id'] for x in a['models']]==[x['run_id'] for x in entries],'All30 accepted order')
            for actual,expected in zip(a['models'],entries):
                for k,v in expected.items():require(actual[k]==v,'Accepted selection differs: '+k)
                independent=self.completed_audit(actual['independent_review'],'all_selections',freeze_sha256=self.freeze_sha,all_predictions_sha256=self.pred_sha,truth_used_for_choices=False,complete_case_inventory_verified=True)
                require([x['run_id'] for x in independent['models']]==[x['run_id'] for x in entries],'Independent audit is not the complete all30 population')
                matching=[x for x in independent.get('models',[independent]) if x.get('run_id')==expected['run_id']]
                require(len(matching)==1 and matching[0]['selection_seal_sha256']==expected['selection_seal']['sha256'],'Independent selection seal binding')
        return entries

def build(stage,evidence,root=ROOT):
    require(stage in STAGES,'Unknown stage');require(isinstance(evidence,dict) and set(evidence)==EVIDENCE[stage],'Evidence keys differ; expected '+str(sorted(EVIDENCE[stage])))
    m=Metadata(root);m.pin(__file__);m.freeze();c,models=m.contract,m.models;home=m.home
    implementation,scope=PROGRAM[stage];program=(home/implementation).resolve();source=m.pin(program)
    if program.is_relative_to(home):require(m.source_lock.get(program.relative_to(home).as_posix())==source['sha256'],'Implementation absent from formal source freeze')
    else:
        a=m.binding_read(evidence['implementation_review']);require(a.get('passed') is True and a.get('implementation_sha256')==source['sha256'],'External auditor source independent review required')
    draft=dict(passed=False,draft=True,root_approval_required=True,scope=scope,implementation_sha256=source['sha256'],freeze_sha256=m.freeze_sha)
    args=[];targets=[];local=[x for x in models if x['dataset']=='local' and x['variant']!='regression'];external=[x for x in models if x['dataset']=='weeds' and x['variant']!='regression'];seg=[x for x in models if x['variant']!='regression']
    def cmd(script,*values):return [PYTHON,'-X','utf8','-B','-s',str(script),*map(str,values)]
    grant_token='{ROOT_APPROVED_REVIEW}'
    def pred():
        m.predictions();m.pred_sha=m.pin(home/'stages/predictions_seal/all_predictions.json')['sha256']
    def independent_inputs():
        pred();m.geometry()
        m.completed_audit(evidence['predictions_review'],'predictions',freeze_sha256=m.freeze_sha,all_predictions_sha256=m.pred_sha)
        m.completed_audit(evidence['geometry_review'],'geometry',freeze_sha256=m.freeze_sha,metadata_sha256=m.pin(home/'stages/local_geometry/metadata.json')['sha256'])
    if stage=='prediction_queue':
        targets=['prediction_queue','inputs_local','inputs_weeds',*['predict_'+x['run_id'] for x in models],'predictions_seal']
        draft.update(run_ids=[x['run_id'] for x in models],R=RS,test_inference_authorized=True,truth_materialization_authorized=False,automatic_retry_authorized=False)
        args=[cmd(program,'--execute','--root-review',grant_token)]
    elif stage=='local_geometry':
        m.stage('inputs_local');draft['input_manifest_sha256']=m.pin(home/'stages/inputs_local/input_manifest.json')['sha256'];targets=['local_geometry'];args=[cmd(program,'--execute','--root-review',grant_token)]
    elif stage in ('audit_predictions','audit_geometry'):
        mode=stage.removeprefix('audit_');target='predictions_seal' if mode=='predictions' else 'local_geometry'
        pred() if mode=='predictions' else m.geometry()
        for ds in (('local','weeds') if mode=='predictions' else ('local',)):m.stage('inputs_'+ds)
        draft.update(mode=mode,target_artifact_manifest_sha256=m.pin(home/'stages'/target/'artifact_manifest.json')['sha256'],annotation_access_authorized=False,model_forward_authorized=False)
        targets=['audit_'+mode];args=[cmd(program,'--mode',mode,'--root-review',grant_token)]
    elif stage in ('select_local','select_external','selection_queue','audit_selections'):
        require(not any((home/'stages'/('truth_'+ds)).exists() for ds in ('local','weeds')),'Label-free stage refused after truth materialization')
        independent_inputs();draft.update(predictions_review=evidence['predictions_review'],geometry_review=evidence['geometry_review'])
        if stage=='select_local':
            draft.update(run_ids=[x['run_id'] for x in local],case_count_per_model=120,all_predictions_sha256=m.pred_sha,geometry_metadata_sha256=m.pin(home/'stages/local_geometry/metadata.json')['sha256'])
            targets=[f'local_select_{i:02d}' for i in range(1,25)]+['local_selections_seal'];args=[cmd(program,'--root-review',grant_token,'--select-run',x['run_id']) for x in local]+[cmd(program,'--root-review',grant_token,'--seal-population')]
        elif stage=='select_external':
            draft['arithmetic_sha256']=m.pin(home/'external_arithmetic.py')['sha256'];targets=['select_'+x['run_id'] for x in external];args=[cmd(program,'select','--run-id',x['run_id'],'--root-review',grant_token) for x in external]
        elif stage=='selection_queue':
            lr=m.audit(evidence['local_review'],'confirmation_local_management',freeze_sha256=m.freeze_sha,implementation_sha256=m.pin(home/'local_management.py')['sha256'],all_predictions_sha256=m.pred_sha,geometry_metadata_sha256=m.pin(home/'stages/local_geometry/metadata.json')['sha256'],case_count_per_model=120,run_ids=[x['run_id'] for x in local])
            er=m.audit(evidence['external_review'],'confirmation_external_selection_execution',freeze_sha256=m.freeze_sha,implementation_sha256=m.pin(home/'external_spatial.py')['sha256'],arithmetic_sha256=m.pin(home/'external_arithmetic.py')['sha256'])
            draft.update(local_review=evidence['local_review'],external_review=evidence['external_review'],run_ids=[x['run_id'] for x in seg],R=RS,label_free_selection_authorized=True,truth_materialization_authorized=False,automatic_retry_authorized=False)
            targets=['selection_queue',*['local_select_%02d'%i for i in range(1,25)],'local_selections_seal',*['select_'+x['run_id'] for x in external]];args=[cmd(program,'--execute','--root-review',grant_token)]
        else:
            m.choice_population();draft.update(run_ids=[x['run_id'] for x in seg],truth_access_authorized=False,new_model_or_solver_authorized=False)
            targets=['audit_all_selections'];args=[cmd(program,'--execute','--root-review',grant_token)]
    elif stage=='accept_choices':
        pred();entries=m.choice_population();audit=m.completed_audit(evidence['selections_review'],'all_selections',freeze_sha256=m.freeze_sha,all_predictions_sha256=m.pred_sha,truth_used_for_choices=False,complete_case_inventory_verified=True)
        require([x['run_id'] for x in audit['models']]==[x['run_id'] for x in entries],'Independent all30 inventory differs')
        for item,record in zip(entries,audit['models']):
            require(record['selection_seal_sha256']==item['selection_seal']['sha256'] and record['case_count']==item['case_count'],'Independent all30 seal/count differs');item['independent_review']=evidence['selections_review']
        draft.update(all_predictions_sha256=m.pred_sha,truth_used_for_choices=False,complete_case_inventory_verified=True,models=entries)
        # Separate explicit commands, no runner or automatic chaining.
        targets=['truth_local','truth_weeds'];args=[cmd(program,'materialize-truth','--dataset',ds,'--choice-review',grant_token) for ds in ('local','weeds')]
    elif stage in ('local_outcomes','external_outcomes','scalar_outcomes','audit_local_outcomes','audit_external_scalar'):
        pred();m.choice_population(evidence['choice_review']);draft['choice_review_sha256']=evidence['choice_review']['sha256'];choice=evidence['choice_review']['path']
        datasets=('local',) if stage in ('local_outcomes','audit_local_outcomes') else ('weeds',) if stage=='external_outcomes' else ('local','weeds')
        for ds in datasets:m.stage('truth_'+ds)
        if stage=='local_outcomes':
            m.geometry();draft.update(local_population_sha256=m.pin(home/'stages/local_selections_seal/all_local_selections.json')['sha256'],geometry_metadata_sha256=m.pin(home/'stages/local_geometry/metadata.json')['sha256'],local_run_ids=[x['run_id'] for x in local],label_case_count=15,random_orders_per_M=1000);args=[cmd(program,'--root-review',grant_token,'--choice-review',choice)];targets=[stage]
        elif stage=='external_outcomes':
            draft['arithmetic_sha256']=m.pin(home/'external_arithmetic.py')['sha256'];args=[cmd(program,'evaluate','--choice-review',choice,'--root-review',grant_token)];targets=[stage]
        elif stage=='scalar_outcomes':
            draft['baselines_sha256']=m.pin(home/'scalar_baselines.json')['sha256'];args=[cmd(program,'--choice-review',choice,'--root-review',grant_token)];targets=[stage]
        elif stage=='audit_local_outcomes':
            m.geometry();m.stage('local_outcomes');m.completed_audit(evidence['geometry_review'],'geometry',freeze_sha256=m.freeze_sha,metadata_sha256=m.pin(home/'stages/local_geometry/metadata.json')['sha256'])
            draft.update(target_artifact_manifest_sha256=m.pin(home/'stages/local_outcomes/artifact_manifest.json')['sha256'],geometry_review=evidence['geometry_review'],local_run_ids=[x['run_id'] for x in local],model_forward_authorized=False,solver_authorized=False,native_mask_decode_authorized=False);args=[cmd(program,'--root-review',grant_token,'--choice-review',choice)];targets=['audit_local_outcomes']
        else:
            for name in OUTCOMES:m.stage(name)
            draft.update(all_predictions_sha256=m.pred_sha,outcome_artifact_sha256={name:m.pin(home/'stages'/name/'artifact_manifest.json')['sha256'] for name in OUTCOMES});args=[cmd(program,'--execute','--root-review',grant_token,'--choice-review',choice)]
    else:
        entries=[]
        for name,key in zip(OUTCOMES,('scalar_review','local_review','external_review')):
            f,_=m.stage(name);pin=m.pin(f/'artifact_manifest.json');a=m.binding_read(evidence[key]);require(a.get('passed') is True and a.get('stage')==name and a['audited_artifact_manifest_sha256']==pin['sha256'] and a['freeze_sha256']==m.freeze_sha,'Outcome audit not bound to completed producer')
            entries.append({'stage':name,'artifact_manifest_sha256':pin['sha256'],'summary_sha256':m.pin(f/'summary.json')['sha256'],'independent_review':evidence[key]})
        draft.update(all_three_outcomes_accepted=True,stages=entries);args=[cmd(program,'--root-review',grant_token)];targets=['report_tables']
    m.fresh(targets);m.finish()
    return m,draft,{'stage':stage,'commands':[{'ordinal':i+1,'argv':x,'manual_execution_only':True} for i,x in enumerate(args)],'root_approved_review_placeholder':grant_token,'no_runner_created':True,'no_automatic_retry':True,'downstream_stage_names':targets}

def prepare(stage,evidence,root=ROOT,evidence_file=None):
    m,draft,commands=build(stage,evidence,root);m.pin(__file__);m.finish()
    if evidence_file is not None:m.binding_read(evidence_file);require(m.read(evidence_file['path'])==evidence,'Evidence file differs from parsed arguments')
    out=m.root/'provenance'/datetime.datetime.now().strftime('confirmation_grant_draft_%Y%m%d_%H%M%S_%f');out.mkdir(parents=False,exist_ok=False)
    try:
        dump(out/'evidence.json',evidence);dump(out/'root_review.draft.json',draft);dump(out/'commands.json',commands)
        dump(out/'preparation.json',{'preparation_passed':True,'stage':stage,'draft_not_authorization':True,'generator_sha256':digest(__file__),'freeze_sha256':m.freeze_sha,'watched_metadata_sha256':m.watched,'inventories':{p:sorted(v) for p,v in m.inventories.items()},'payload_bytes_opened':0,'models_loaded':0,'stage_calls':0,'metadata_only_stage_completeness':True,'complete_payload_hash_validation_deferred_to_original_stage_and_bound_independent_audits':True})
        m.finish();dump(out/'artifact_manifest.json',{'files':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
        return out
    except BaseException as error:
        failure={'preparation_passed':False,'error':repr(error),'draft_not_authorization':True}
        dump(out/'failure.json',failure);dump(out/'preparation.json',failure);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=STAGES);p.add_argument('--evidence',required=True,help='Inline JSON evidence object or @JSON-file; every evidence item is path+sha256.')
    a=p.parse_args();argument=a.evidence
    item=None
    if argument.startswith('@'):
        meta=Metadata(ROOT);evidence=meta.read(Path(argument[1:]));item=meta.pin(Path(argument[1:]));meta.finish()
    else:evidence=json.loads(argument)
    print(prepare(a.stage,evidence,evidence_file=item))
if __name__=='__main__':main()
