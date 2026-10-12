"""Metadata-only checkpoint candidates. Never loads tensors, reserved data or FREEZE.

Recomputes all saved development epochs/events independently of trainer summarize.
Partial mode is preparation evidence. Full mode remains an unfrozen candidate and
requires all fixed37 fits, fixed16 numerical receipts and accepted full24 summary.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, math, os, random, shutil, struct, sys, traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROTOCOL=HERE/'checkpoint_candidate_protocol.json'
OUTPUT=ROOT/'checkpoint_candidates_v1'
IDENTITY=('run_id','encoder','variant','seed','lr')
CORE_FILES={'checkpoint':'best.pt','config':'config.json','status':'status.json',
            'history':'metrics.jsonl','best_validation':'best_validation.json'}

def require(ok,message):
    if not ok:raise ValueError(message)
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def close(a,b,label='arithmetic'):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(a,b,rel_tol=1e-10,abs_tol=2e-12),label+': '+repr((a,b)))
def mean(v):return math.fsum(v)/len(v)
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def under(root,path):
    path=Path(path).resolve();require(path.is_relative_to(Path(root).resolve()),'Path outside allowed root');return path
def safe_relative(root,name):
    require(not Path(name).is_absolute(),'Absolute path in relative manifest');return under(root,Path(root)/name)

class Ledger:
    """Hash bytes once, verify all again at end. Arrays are never deserialized.

    Dataset metadata is limited to explicit train/val/lock/summary names. NPZ raw
    bytes are allowed only after registration by train/val manifest member IDs.
    Reserved filenames are rejected even for a hash-only request.
    """
    def __init__(self):self.files={};self.cache_allowed=set()
    def path(self,path):
        p=under(ROOT,path)
        require('test' not in p.name.lower() and 'r6' not in p.name.lower(),'Reserved filename forbidden')
        if p.suffix.lower()=='.npz':require(str(p) in self.cache_allowed,'Array not a registered development member')
        rel=p.relative_to(ROOT)
        if rel.parts[0] in ('local_data_v2','weedsgalore_data_v1') and p.suffix.lower()!='.npz':
            require(p.name in ('train.json','val.json','DATASET_LOCK.json','manifest_summary.json'),'Unapproved dataset metadata')
        return p
    def digest(self,path,expected=None):
        p=self.path(path);key=str(p)
        value=self.files.get(key)
        if value is None:value=sha(p);self.files[key]=value
        if expected is not None:require(value==expected,'Input SHA mismatch: '+key)
        return value
    def read(self,path,expected=None):
        p=self.path(path);require(p.suffix.lower() in ('.json','.jsonl'),'Only JSON metadata can be parsed')
        data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
        if str(p) in self.files:require(self.files[str(p)]==digest,'Source changed: '+str(p))
        self.files[str(p)]=digest
        if expected is not None:require(digest==expected,'Metadata SHA mismatch: '+str(p))
        text=data.decode('utf-8-sig')
        return [json.loads(x) for x in text.splitlines() if x.strip()] if p.suffix=='.jsonl' else json.loads(text)
    def binding(self,path):return {'path':str(self.path(path)),'sha256':self.digest(path)}
    def finish(self):
        for path,digest in self.files.items():require(sha(path)==digest,'Input changed during audit: '+path)

def metric(samples):
    require(bool(samples),'Empty validation samples')
    p=[x['prediction'] for x in samples];t=[x['truth'] for x in samples]
    require(all(math.isfinite(v) and 0<=v<=1 for v in p+t),'Invalid scalar probability/truth')
    e=[a-b for a,b in zip(p,t)]
    return {'n':len(p),'mae':mean([abs(x) for x in e]),'bias':mean(e),
            'mean_prediction':mean(p),'mean_target':mean(t)}

def check_metric(saved,computed):
    require(saved['n']==computed['n'],'Validation count mismatch')
    for k in computed:
        if k!='n':close(saved[k],computed[k],'Saved sample metric '+k)

def check_history(history,best,status,cfg,val):
    require([x['epoch'] for x in history]==list(range(1,25)),'Expected unique ordered24 epochs')
    external=cfg.get('version')=='external_weeds_v1';n=104 if external else 985
    selection='selection_mae' if external else 'selection_balanced_mae'
    rng=random.Random(cfg['seed']+1000);rows=[];scales=[]
    for h in history:
        require(set(h['scales'])=={'1','2','4'},'Validation scales changed')
        values=[]
        for r in (1,2,4):
            s=h['scales'][str(r)]
            if external:
                samples=s['samples'];require([x['id'] for x in samples]==[x['id'] for x in val],'Saved val IDs/order differ')
                require([x['acquisition_date'] for x in samples]==[x['acquisition_date'] for x in val],'Saved dates differ')
                require([x['truth'] for x in samples]==[f32(x['coverage']) for x in val],'Original CPU float32 truth differs')
                computed=metric(samples);check_metric(s,computed)
                dates=sorted({x['acquisition_date'] for x in val});require(set(s['by_date'])==set(dates),'Date groups differ')
                for date in dates:check_metric(s['by_date'][date],metric([x for x in samples if x['acquisition_date']==date]))
                score=computed['mae']
            else:
                require(s['n']==357 and s['n_positive']==27,'Local validation population/strata differ')
                for k in ('mae','mae_zero','mae_positive','balanced_mae'):require(math.isfinite(s[k]) and 0<=s[k]<=1,'Invalid local '+k)
                require(math.isfinite(s['bias']) and abs(s['bias'])<=s['mae']+2e-12,'Invalid local bias')
                close(s['mae'],(330*s['mae_zero']+27*s['mae_positive'])/357,'Local weighted strata MAE')
                score=mean([s['mae_zero'],s['mae_positive']]);close(score,s['balanced_mae'],'Local balanced MAE')
                if cfg['variant']!='regression':
                    for k in ('target_iou','dice'):require(math.isfinite(s[k]) and 0<=s[k]<=1,'Invalid segmentation diagnostic')
            values.append(score);scales.append({'run_id':cfg['run_id'],'epoch':h['epoch'],'R':r,'selection_component':score,'mae':s['mae'],'bias':s['bias']})
        score=mean(values);close(score,h[selection],'Across-R original selection')
        factors={'1':0,'2':0,'4':0}
        for start in range(0,n,4):factors[str(rng.choice((1,2,4)))]+=min(4,n-start)
        require(factors==h['train_factor_sample_counts'],'Fixed augmentation RNG/counts differ')
        close(h['lr'],cfg['lr']*(1+math.cos(math.pi*(h['epoch']-1)/24))/2,'Cosine LR')
        require(math.isfinite(h['train_loss']) and h['train_loss']>=0,'Nonfinite/negative epoch loss')
        rows.append({'run_id':cfg['run_id'],'epoch':h['epoch'],'selection_metric':selection,'selection_value':score,
                     'attempts':h['optimizer_update_attempts'],'actual':h['optimizer_updates'],'skips':h['optimizer_skipped_overflows'],
                     'train_loss':h['train_loss'],'lr':h['lr'],'amp_scale_end':h['amp_scale_end']})
    earliest=min(history,key=lambda x:x[selection])
    require(best==earliest and best['epoch']==status['best_epoch'],'Earliest strict best/original epoch mismatch')
    close(best[selection],status['best_validation_mae' if external else 'best_validation_balanced_mae'],'Status best score')
    return rows,scales

def check_events(events,history,status,variant,external):
    groups=13 if external else 124;expected=groups*24
    require(len(events)==expected,'Incomplete optimizer event count')
    actual=skips=consecutive=0;previous=65536.
    for i,e in enumerate(events):
        group=i%groups+1;samples=8 if external or group<groups else 1
        require(e['attempt']==i+1 and e['epoch']==i//groups+1 and e['group']==group,'Optimizer event sequence mismatch')
        require(e['group_samples']==samples,'Tail/effective group size differs')
        den=e['group_denominator'];require(math.isfinite(den) and den>0,'Invalid group denominator')
        if variant=='regression':close(den,samples,'Regression denominator')
        elif external:close(den,samples*65536,'All-valid external denominator')
        else:require(den<=samples*65536,'Local valid denominator exceeds full support')
        require(e['amp_enabled'] is True,'AMP policy differs');close(e['scale_before'],previous,'AMP scale continuity')
        require(math.isfinite(e['group_loss']) and e['group_loss']>=0,'Invalid group loss')
        if e['optimizer_step_executed'] is True:
            require(e['all_unscaled_gradients_finite'] is True and e['skipped'] is False and e['status']=='updated','Invalid actual-step event')
            require(math.isfinite(e['unscaled_gradient_norm']) and e['unscaled_gradient_norm']>=0,'Invalid gradient norm')
            require(e['scale_after']>=e['scale_before'],'Updated step decreased AMP scale');actual+=1;consecutive=0
        else:
            require(e['optimizer_step_executed'] is False and e['all_unscaled_gradients_finite'] is False and e['skipped'] is True and e['status']=='overflow_skipped','Invalid overflow event')
            require(e['scale_after']<e['scale_before'],'Overflow did not back off');skips+=1;consecutive+=1
        require(math.isfinite(e['scale_after']) and e['scale_after']>=1 and consecutive<8,'Original overflow guard exceeded')
        require((e['actual_updates_total'],e['skipped_updates_total'],e['consecutive_overflows'])==(actual,skips,consecutive),'Cumulative optimizer counts differ')
        previous=e['scale_after']
    for h in history:
        batch=events[(h['epoch']-1)*groups:h['epoch']*groups]
        a=sum(e['optimizer_step_executed'] for e in batch);s=sum(e['skipped'] for e in batch)
        require(a>0 and (h['optimizer_update_attempts'],h['optimizer_updates'],h['optimizer_skipped_overflows'])==(groups,a,s),'Epoch optimizer counts differ')
        close(h['train_loss'],math.fsum(e['group_loss']*e['group_denominator'] for e in batch)/math.fsum(e['group_denominator'] for e in batch),'Weighted epoch train loss')
        close(h['amp_scale_end'],batch[-1]['scale_after'],'Epoch AMP scale')
        require(h['consecutive_overflows']==batch[-1]['consecutive_overflows'],'Epoch overflow count')
    require((status['optimizer_attempts_total'],status['optimizer_updates_total'],status['optimizer_skips_total'])==(expected,actual,skips),'Final optimizer counts differ')
    close(status['amp_scale'],previous,'Final AMP scale')
    return {'attempted_updates':expected,'actual_updates':actual,'skipped_updates':skips,'events_verified':len(events)}

def load_development(ledger):
    result={}
    for ds,dirname,ntrain,nval in [('local','local_data_v2',985,357),('weeds','weedsgalore_data_v1',104,26)]:
        folder=ROOT/dirname;lock=ledger.read(folder/'DATASET_LOCK.json');train=ledger.read(folder/'train.json',lock['manifest_sha256']['train.json']);val=ledger.read(folder/'val.json',lock['manifest_sha256']['val.json'])
        require(len(train)==ntrain and len(val)==nval,'Wrong development population')
        ids=[x['id'] for x in train+val];require(len(set(ids))==ntrain+nval,'Development member overlap/duplicates')
        caches={}
        for split,records in [('train',train),('val',val)]:
            for record in records:
                require(record['split']==split,'Wrong development split member')
                p=under(folder/'cache',record['cache']);require(p.suffix=='.npz','Unexpected cache type')
                ledger.cache_allowed.add(str(p));caches[str(p)]=record['cache_sha256']
        for p,h in caches.items():ledger.digest(p,h)
        result[ds]={'train':train,'val':val,'cache_sha256':caches,
                    'manifest_sha256':{name:ledger.digest(folder/name) for name in ('train.json','val.json','DATASET_LOCK.json')}}
    require(sum(x['coverage']>0 for x in result['local']['val'])==27,'Local positive stratum changed')
    return result

def check_files(ledger,folder,files):
    for name,digest in files.items():ledger.digest(safe_relative(folder,name),digest)

def prepare_sources(ledger,protocol,roster,data):
    for binding in protocol['fixed_inputs']:ledger.digest(ROOT/binding['relative_path'],binding['sha256'])
    common=ledger.read(ROOT/'common_queue_plan_v2.json');seedhome=ROOT/'local_seed_stability_v1';exhome=ROOT/'external_weeds_v1'
    seed=ledger.read(seedhome/'plan.json');sl=ledger.read(seedhome/'SOURCE_LOCK.json');il=ledger.read(seedhome/'INPUT_LOCK.json')
    ex=ledger.read(exhome/'serial_plan.json');el=ledger.read(exhome/'SOURCE_LOCK.json')
    require([x['run_id'] for x in roster if x['dataset']=='local' and x['seed']==17]==[x['run_id'] for x in common['runs']],'Common12 roster changed')
    require([x['run_id'] for x in roster if x['dataset']=='local' and x['seed']!=17]==[x['run_id'] for x in seed['runs']],'Seed16 roster changed')
    require([x['run_id'] for x in roster if x['dataset']=='weeds']==[x['run_id'] for x in ex['runs']],'External9 roster changed')
    check_files(ledger,ROOT,common['source_sha256']);check_files(ledger,seedhome,sl['source_sha256']);check_files(ledger,exhome,el['source_files_sha256'])
    for d in (il['external_inputs_sha256'],el['external_inputs_sha256']):
        for p,h in d.items():ledger.digest(p,h)
    require(common['manifest_sha256']==data['local']['manifest_sha256'],'Original local manifests changed')
    seedbinding={k:ledger.digest(seedhome/v) for k,v in {'source_lock_sha256':'SOURCE_LOCK.json','plan_sha256':'plan.json','protocol_sha256':'protocol.json','input_lock_sha256':'INPUT_LOCK.json'}.items()}
    claim=ledger.read(seedhome/'QUEUE_CLAIM.json');review=ledger.read(claim['review_path'],claim['review_sha256'])
    require(claim['binding']==seedbinding,'Training claim binding changed')
    require(review.get('passed') is True and review.get('execution_scope')=='local_seed_stability_v1_fixed16' and review.get('test_or_R6_authorized') is False,'Wrong seed training authorization')
    for k,v in seedbinding.items():require(review[k]==v,'Training authorization mismatch '+k)
    require(review['run_count']==16 and review['epochs_per_run']==24,'Training authorization population')
    extbindings={'source_lock_sha256':ledger.digest(exhome/'SOURCE_LOCK.json'),'source_files_sha256':el['source_files_sha256'],
        'external_inputs_sha256':el['external_inputs_sha256'],'development_cache_sha256':data['weeds']['cache_sha256'],
        'plan_sha256':ledger.digest(exhome/'serial_plan.json'),'protocol_sha256':ledger.digest(exhome/'RESEARCH_PROTOCOL.json'),
        'pretrained_sha256':protocol['resnet18_pretrained_sha256']}
    prior={}
    for item in protocol['legacy_local_audits']:
        audit=ledger.read(ROOT/item['path'],item['sha256']);require(audit.get('all_checks_passed') is True,'Legacy checkpoint audit failed')
        for row in audit.get('runs',[audit]):
            require(row.get('all_checks_passed',True) is True,'Legacy per-run audit failed')
            require(row['run_id'] not in prior,'Duplicate legacy run evidence')
            prior[row['run_id']]={'row':row,'audit':ledger.binding(ROOT/item['path'])}
    require(set(prior)=={x['run_id'] for x in common['runs']},'Legacy audit coverage must be exactly12')
    ea=protocol['external_training_audit'];external=ledger.read(ROOT/ea['path'],ea['sha256'])
    require(external.get('passed') is True,'External training audit failed')
    for k in ('source_lock_sha256','plan_sha256','protocol_sha256'):require(external[k]==extbindings[k],'Prior external audit source mismatch')
    require(len(external['runs'])==9 and {x['run_id'] for x in external['runs']}=={x['run_id'] for x in ex['runs']},'External audit population')
    queue=ledger.read(exhome/'queue_records/20260916_053741_352995/status.json')
    require(queue['state']=='completed' and [x['run_id'] for x in queue['results']]==[x['run_id'] for x in ex['runs']],'External queue incomplete')
    return {'common':common,'seed':seed,'external':ex,'external_lock':el,'seedbinding':seedbinding,'externalbindings':extbindings,
            'legacy':prior,'external_audit':external,'external_audit_binding':ledger.binding(ROOT/ea['path']),
            'external_queue':{x['run_id']:x for x in queue['results']}}

def audit_one(ledger,row,data,sources,out,index):
    rid=row['run_id'];folder=under(ROOT,row['run_directory']);external=row['dataset']=='weeds';newseed=not external and row['seed']!=17
    cfg=ledger.read(folder/'config.json');status=ledger.read(folder/'status.json');best=ledger.read(folder/'best_validation.json')
    history=ledger.read(folder/'metrics.jsonl');events=ledger.read(folder/'optimizer_events.jsonl')
    require(status['status']=='completed' and status['executed_epochs']==24,'Incomplete fit')
    for k in IDENTITY:require(cfg[k]==row[k],'Model identity mismatch '+k)
    settings=sources['external']['settings'] if external else sources['seed']['settings']
    for k,v in settings.items():
        if k=='device':require(cfg[k]=='cuda:0','Original training device differs')
        elif k=='lr':continue
        else:require(cfg[k]==v,'Original training config differs '+k)
    for x in (cfg,status):
        require(x.get('test_accessed') is False and x.get('R6_accessed',False) is False,'Reserved data used in original fit')
    require(cfg['decision_metrics_used_for_selection'] is False and cfg['fixed_output_grid']==256,'Selection/support changed')
    expected=data[row['dataset']]
    require(cfg['n_train']==len(expected['train']) and cfg['n_val']==len(expected['val']),'Subset config')
    for split in ('train','val'):require(cfg['selected_'+split+'_ids']==[x['id'] for x in expected[split]],'Selected development IDs/order differ')
    pre=cfg['pretrained_report'];require(pre['strict'] is True and not pre['missing_keys'] and not pre['unexpected_keys'],'Non-strict pretrained initialization')
    ledger.digest(pre['path'],pre['sha256'])
    epochs,scales=check_history(history,best,status,cfg,expected['val']);counts=check_events(events,history,status,row['variant'],external)
    cp=ledger.digest(folder/'best.pt')
    binding={prefix+'_path':str((folder/name).resolve()) for prefix,name in CORE_FILES.items()}
    binding.update({prefix+'_sha256':ledger.digest(folder/name) for prefix,name in CORE_FILES.items()})
    prior=None;completion=None
    if external:
        require(cfg['input_bindings']==sources['externalbindings']==ledger.read(folder/'input_hashes.json'),'External input chain differs')
        completion=ledger.read(folder/'completion_manifest.json');require(completion['state']=='completed' and completion['input_bindings']==sources['externalbindings'],'External completion chain')
        required=set(CORE_FILES.values())|{'optimizer_events.jsonl','input_hashes.json','constant_baselines.json','independent_review.json','source/SOURCE_LOCK.json'}
        require(required<=set(completion['files_sha256']),'External completion missing files');check_files(ledger,folder,completion['files_sha256'])
        check_files(ledger,folder/'source',sources['external_lock']['source_files_sha256']);ledger.digest(folder/'source/SOURCE_LOCK.json',sources['externalbindings']['source_lock_sha256'])
        auth=ledger.read(folder/'independent_review.json');original=ledger.read(auth['path'],auth['sha256'])
        require(original==auth['receipt'] and original['passed'] is True and original['execution_scope']=='external_weeds_v1_gpu' and original['test_or_R6_authorized'] is False,'External GPU authorization differs')
        for k in ('source_lock_sha256','plan_sha256','protocol_sha256'):require(original[k]==sources['externalbindings'][k],'External authorization hash')
        prior=next(x for x in sources['external_audit']['runs'] if x['run_id']==rid)
        for k in ('checkpoint','config','status','best_validation'):require(prior[k+'_sha256']==binding[k+'_sha256'],'Prior external identity '+k)
        for k,name in [('metrics_sha256','metrics.jsonl'),('optimizer_events_sha256','optimizer_events.jsonl'),('completion_manifest_sha256','completion_manifest.json')]:ledger.digest(folder/name,prior[k])
        require(prior['best_epoch']==best['epoch'] and status['checkpoint_sha256']==cp,'External selected checkpoint identity')
        q=sources['external_queue'][rid]
        require(q['checkpoint_sha256']==cp and q['status_sha256']==binding['status_sha256'] and q['best_epoch']==best['epoch'],'External queue identity')
        completion_path=folder/'completion_manifest.json';prior_binding=sources['external_audit_binding']
    else:
        require(cfg['manifest_sha256']==expected['manifest_sha256'],'Local config/data binding')
        fixed={k:sources['common']['source_sha256'][k] for k in ('train_common.py','models_common.py','conservative_resample.py')}
        require(cfg['source_sha256']==fixed,'Local trainer source differs');check_files(ledger,folder/'source',fixed)
        for name in ('train.json','val.json'):ledger.digest(folder/'source'/name,expected['manifest_sha256'][name])
        require(not any(cfg.get(x) for x in ('smoke','diagnostic','diagnostic_epochs','max_train','max_val','no_pretrained')),'Diagnostic/subset original config')
        require(cfg['validation_R']==[1,2,4] and cfg['effective_batch_size']==8,'Local fixed validation budget')
        if newseed:
            completion_path=ROOT/'local_seed_stability_v1/completion_records'/(rid+'.json');completion=ledger.read(completion_path)
            require(completion['run_id']==rid and completion['binding']==sources['seedbinding'] and completion['checkpoint_sha256']==cp and completion['best_epoch']==best['epoch'],'Seed producer receipt binding')
            close(completion['best_validation_balanced_mae'],best['selection_balanced_mae'],'Seed receipt selected metric')
            require(set(CORE_FILES.values())|{'optimizer_events.jsonl'}<=set(completion['files_sha256']),'Seed receipt missing core files')
            check_files(ledger,folder,completion['files_sha256']);prior_binding=None
        else:
            prior=sources['legacy'][rid]['row'];prior_binding=sources['legacy'][rid]['audit']
            require(prior['checkpoint_sha256']==cp and prior['checkpoint_epoch']==best['epoch'],'Legacy audited checkpoint differs')
            md=Path(prior.get('directory',prior.get('export_directory')))/'metadata.json';meta=ledger.read(md,prior['metadata_sha256'])
            require(meta['checkpoint_sha256']==cp and meta['checkpoint_epoch']==best['epoch'],'Legacy export checkpoint bridge')
            require(meta['config_sha256']==binding['config_sha256'],'Legacy export config bridge')
            for name in CORE_FILES.values():
                path=folder/name;require(meta['input_hashes'].get(str(path))==ledger.digest(path),'Legacy export original-file bridge '+name)
            # This new receipt is explicitly a metadata audit product, not a fabricated historical producer receipt.
            completion_path=out/'receipts'/('local%02d.json'%index)
            write(completion_path,{'kind':'new_metadata_completion_binding_not_historical_producer','created_at':now(),
                'run_id':rid,'state':'completed','best_epoch':best['epoch'],'checkpoint_sha256':cp,
                'files_sha256':{name:ledger.digest(folder/name) for name in list(CORE_FILES.values())+['optimizer_events.jsonl']},
                'prior_checkpoint_audit':prior_binding,'tensor_payload_reopened':False})
    result={**row,**binding,'best_epoch':best['epoch'],'original_selection_audited':True,
            'completion_receipt_path':str(completion_path.resolve()),'completion_receipt_sha256':sha(completion_path)}
    event_best=events[best['epoch']*(13 if external else 124)-1]
    record={'run_id':rid,'dataset':row['dataset'],'encoder':row['encoder'],'variant':row['variant'],'seed':row['seed'],'lr':row['lr'],
        'best_epoch':best['epoch'],'checkpoint_sha256':cp,'status_sha256':binding['status_sha256'],
        'config_sha256':binding['config_sha256'],'history_sha256':binding['history_sha256'],
        'best_validation_sha256':binding['best_validation_sha256'],**counts,
        'selected_epoch_counts':{k:event_best[k] for k in ('attempt','actual_updates_total','skipped_updates_total')},
        'completion_receipt_sha256':result['completion_receipt_sha256'],'prior_independent_checkpoint_binding':prior_binding,
        'selection_value':best['selection_mae' if external else 'selection_balanced_mae'],
        'local_sample_level_metrics_recomputed':False,'external_saved_sample_metrics_recomputed':external,
        'tensor_payload_loaded':False,'test_or_R6_accessed':False}
    return result,record,epochs,scales

def full_prerequisites(ledger,evidence,roster,audited):
    require(evidence['version']=='confirmation_checkpoint_candidate_evidence_v1','Wrong full evidence schema')
    seedids=[x['run_id'] for x in roster if x['dataset']=='local' and x['seed']!=17]
    localids=[x['run_id'] for x in roster if x['dataset']=='local' and x['variant']!='regression']
    def load(key):
        b=evidence[key];return ledger.read(b['path'],b['sha256'])
    index=load('seed_numerical_index')
    require(index.get('all16_complete') is True and index['accepted_count']==16 and index['passed_for_registered_models'] is True,'All16 numerical acceptance required')
    refs=index['readiness_seed_numerical_reviews'];require(len(refs)==16 and {x['run_id'] for x in refs}==set(seedids),'Numerical index exact16 mismatch')
    for ref in refs:
        review=ledger.read(ref['path'],ref['sha256']);require(review.get('passed') is True and review.get('test_or_R6_accessed') is False,'Numerical review not passed')
        rr=[x for x in review['models'] if x['run_id']==ref['run_id']];require(len(rr)==1,'Numerical review run missing/ambiguous');rr=rr[0]
        for k,n in {'maps':357,'scalar_pairs':1071,'management_units':2183,'prefix_curve_points':6558,'fixed_prefix_rows':45,'P_policy_rows':45,'auc_rows':9}.items():require(rr[k]==n,'Incomplete numerical audit '+k)
        require(rr['no_new_model_forward'] is True and rr['no_new_solver'] is True and rr['test_or_R6_accessed'] is False,'Numerical review wrong scope')
        receipt=ledger.read(rr['receipt'],rr['receipt_sha256']);require(receipt['run_id']==ref['run_id'] and receipt['original_epoch_preserved'] is True and receipt['all_declared_cases_retained'] is True and receipt['test_or_R6_accessed'] is False,'Numerical receipt identity')
        for prefix,file,key in [('export','status.json','export_status_sha256'),('export','artifact_manifest.json','export_manifest_sha256'),('spatial','status.json','spatial_status_sha256'),('spatial','summary.json','spatial_summary_sha256')]:ledger.digest(Path(receipt[prefix+'_directory'])/file,receipt[key])
        export_status=ledger.read(Path(receipt['export_directory'])/'status.json');spatial_status=ledger.read(Path(receipt['spatial_directory'])/'status.json')
        require(export_status['status']=='completed' and spatial_status['state']=='completed','Numerical output not completed')
        metadata=ledger.read(Path(receipt['export_directory'])/'metadata.json')
        require(metadata['checkpoint_sha256']==audited[ref['run_id']]['checkpoint_sha256'] and metadata['checkpoint_epoch']==audited[ref['run_id']]['best_epoch'],'Numerical result uses another checkpoint')
        # metadata must itself be covered by the verified artifact manifest, not an unbound sibling.
        manifest=ledger.read(Path(receipt['export_directory'])/'artifact_manifest.json')
        hashes=manifest.get('files_sha256',manifest.get('file_sha256',manifest.get('files')))
        require(isinstance(hashes,dict) and hashes.get('metadata.json')==ledger.digest(Path(receipt['export_directory'])/'metadata.json'),'Export metadata not sealed')
    summary=load('full24_summary');review=load('full24_independent_review')
    summary_path=Path(evidence['full24_summary']['path']);folder=summary_path.parent
    require(review.get('passed') is True and review.get('scope')=='full24_seed_summary_independent','Full24 independent review not passed')
    require(len(review['run_ids'])==24 and set(review['run_ids'])==set(localids),'Full24 review population')
    require(review['summary_sha256']==ledger.digest(summary_path),'Full24 summary identity')
    manifest=ledger.read(folder/'artifact_manifest.json',review['audited_artifact_manifest_sha256']);status=ledger.read(folder/'status.json')
    require(status.get('state',status.get('status'))=='completed','Full24 summary incomplete')
    require(status['artifact_manifest_sha256']==review['audited_artifact_manifest_sha256'],'Full24 status/manifest differs')
    check_files(ledger,folder,manifest['file_sha256'])
    require(summary['models']==24 and summary['seeds']==[17,29,43],'Full24 summary dimensions')
    models=ledger.read(folder/'models.json');ids=[x['run_id'] if isinstance(x,dict) else x for x in models]
    require(len(ids)==24 and set(ids)==set(localids),'Full24 models identity')
    by_id={x['run_id']:x for x in roster}
    for model in models:
        require(isinstance(model,dict),'Full24 models must include checkpoint identity')
        rid=model['run_id'];require(model['checkpoint_sha256']==audited[rid]['checkpoint_sha256'] and model['checkpoint_epoch']==audited[rid]['best_epoch'],'Full24 selected checkpoint differs')
        for k in IDENTITY:require(model[k]==by_id[rid][k],'Full24 config identity differs '+k)
    for k,n in {'all2160_model_policy_budgets':2160,'all216_model_prefix_auc':216,'all72_cpu_scalar_by_R':72,
                'all12_hard_soft_model_pairs':12,'all1080_hard_soft_budget_pairs':1080,'all108_hard_soft_auc_pairs':108,'all36_hard_soft_cpu_scalar_pairs':36}.items():require(summary['table_rows'][k]==n,'Full24 table counts differ')
    require(summary['test_or_R6_accessed'] is False and summary['new_model_calls']==0 and summary['new_solver_calls']==0,'Wrong full24 summary scope')
    ledger.digest(ROOT/'seed_stability_summary_v1.py',status['source_sha256'])
    require(manifest['file_sha256']['summary.json']==ledger.digest(summary_path) and manifest['file_sha256']['models.json']==ledger.digest(folder/'models.json'),'Full24 tables not bound')
    return {'all16_numerical_accepted':True,'full24_summary_accepted':True,'evidence':evidence}

def save_table(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def execute(mode,evidence_path=None,root_review=None):
    ledger=Ledger();protocol=ledger.read(PROTOCOL);ledger.digest(__file__)
    require(protocol['version']=='confirmation_checkpoint_candidate_protocol_v1','Wrong builder protocol')
    contract=ledger.read(HERE/'contract.json',protocol['contract_sha256']);roster=contract['roster']
    require(len(roster)==37 and len({x['run_id'] for x in roster})==37,'Wrong fixed population')
    require(sum(x['dataset']=='local' for x in roster)==28 and sum(x['dataset']=='weeds' for x in roster)==9,'Wrong dataset populations')
    evidence=None
    if mode=='full':
        require(evidence_path and root_review,'Full candidate requires explicit evidence and root review')
        evidence=ledger.read(evidence_path);review=ledger.read(root_review)
        require(review.get('passed') is True and review.get('scope')=='confirmation_checkpoint_candidate_full_metadata_only','Wrong root candidate authorization')
        for k,v in {'implementation_sha256':ledger.digest(__file__),'protocol_sha256':ledger.digest(PROTOCOL),'contract_sha256':ledger.digest(HERE/'contract.json'),'evidence_sha256':ledger.digest(evidence_path)}.items():require(review[k]==v,'Candidate authorization hash '+k)
        require(review.get('test_or_model_access_authorized') is False,'Metadata-only authorization required')
    # Snapshot status before any audit; pending models are not polled or opportunistically included later.
    ready=[];pending=[]
    for row in roster:
        p=under(ROOT,Path(row['run_directory'])/'status.json')
        status=json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
        if status.get('status')=='completed':ready.append(row)
        else:pending.append({'run_id':row['run_id'],'observed_state':status.get('status','not_started')})
    if mode=='full':require(not pending,'All37 fits must be completed before full candidate claim')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=OUTPUT/stamp
    OUTPUT.mkdir(exist_ok=True);require(not out.exists(),'Candidate directory immutable')
    claim=OUTPUT/('full_claim.json' if mode=='full' else 'partial_'+stamp+'.claim.json')
    stream=claim.open('x',encoding='utf-8');primary=None
    try:
        with stream:json.dump({'created_at':now(),'mode':mode,'out':str(out),'implementation_sha256':ledger.digest(__file__),'protocol_sha256':ledger.digest(PROTOCOL)},stream)
        out.mkdir();(out/'source').mkdir();(out/'receipts').mkdir()
        shutil.copy2(__file__,out/'source/builder.py');shutil.copy2(PROTOCOL,out/'source/protocol.json')
        write(out/'status.json',{'state':'running','mode':mode,'created_at':now()})
        data=load_development(ledger);sources=prepare_sources(ledger,protocol,roster,data)
        entries=[];records=[];epochs=[];scales=[]
        for number,row in enumerate(ready,1):
            print(json.dumps({'run_id':row['run_id'],'audited_before':len(records),'snapshot_completed':len(ready)}),flush=True)
            entry,record,er,sr=audit_one(ledger,row,data,sources,out,number)
            entries.append(entry);records.append(record);epochs.extend(er);scales.extend(sr)
        prereq={'all16_numerical_accepted':False,'full24_summary_accepted':False,'not_evaluated_in_partial_mode':True}
        if mode=='full':prereq=full_prerequisites(ledger,evidence,roster,{x['run_id']:x for x in records})
        ledger.finish()
        audit={'version':'confirmation_checkpoint_training_metadata_audit_v1','passed':True,'mode':mode,'scope':'all saved development epochs and optimizer events; checkpoint byte identity; no tensor deserialization',
               'created_at':now(),'expected_population':37,'audited_count':len(records),'all37_complete':len(records)==37,
               'runs':records,'pending':pending,'epochs_verified':len(epochs),'scales_verified':len(scales),'optimizer_events_verified':sum(x['events_verified'] for x in records),
               'prerequisites':prereq,'implementation_sha256':ledger.digest(__file__),'protocol_sha256':ledger.digest(PROTOCOL),
               'source_inputs_sha256':ledger.files,'no_model_or_solver_calls':True,'test_or_R6_accessed':False,
               'boundaries':['Original local histories contain only aggregate per-scale/stratum metrics, not per-sample vectors; only aggregate identities are independently recomputed here.',
                             'External24-epoch saved sample scalars/date metrics are recomputed against permitted development metadata float32 truths.',
                             'Checkpoint bytes are SHA-bound to original producer/prior independent audit; tensor payload is not reloaded.',
                             'A partial passed audit covers only listed completed models, not pending fits or full confirmation readiness.']}
        write(out/'training_audit.json',audit);ad=sha(out/'training_audit.json')
        for entry in entries:entry.update(training_audit_path=str(out/'training_audit.json'),training_audit_sha256=ad)
        lookup={x['run_id']:x for x in entries};table={'version':'confirmation_checkpoint_candidates_v1','frozen':False,'complete':mode=='full','all37_training_audited':len(entries)==37,
            'models':[lookup.get(x['run_id'],{**x,'candidate_state':'pending'}) for x in roster],
            'requires_root_independent_review_and_explicit_promotion':True,'official_checkpoints_or_FREEZE_written':False}
        write(out/'checkpoints.candidate.json',table);save_table(out/'all_epochs.csv',epochs);save_table(out/'all_epoch_scales.csv',scales);save_table(out/'all_completed_models.csv',[{k:x[k] for k in ('run_id','dataset','encoder','variant','seed','lr','best_epoch','selection_value','attempted_updates','actual_updates','skipped_updates','checkpoint_sha256')} for x in records])
        write(out/'status.json',{'state':'completed','mode':mode,'audited':len(records),'pending':len(pending),'full_candidate':mode=='full','frozen':False,'training_audit_sha256':ad,'completed_at':now()})
    except BaseException as error:
        primary=error
        try:
            out.mkdir(exist_ok=True);write(out/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'no_retry_claim_preserved':True,'at':now()});write(out/'status.json',{'state':'failed','error':repr(error),'frozen':False})
        except BaseException as second:
            write(OUTPUT/('failure_'+stamp+'.json'),{'error':repr(error),'recording_error':repr(second),'claim':str(claim),'out':str(out)})
        raise
    finally:
        try:write(out/'artifact_manifest.json',{'files_sha256':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}})
        except BaseException as secondary:
            try:write(out/'status.json',{'state':'failed','error':repr(primary),'finalization_error':repr(secondary),'frozen':False})
            except BaseException:pass
            write(OUTPUT/('finalization_failure_'+stamp+'.json'),{'error':repr(primary),'finalization_error':repr(secondary),'claim':str(claim),'out':str(out)})
            if primary is None:raise
    print(json.dumps({'out':str(out),'training_audit_sha256':sha(out/'training_audit.json'),'candidate_sha256':sha(out/'checkpoints.candidate.json'),'audited':len(records),'pending':len(pending),'frozen':False}),flush=True)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-partial',action='store_true');mode.add_argument('--prepare-full',action='store_true')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--evidence');ap.add_argument('--root-review');args=ap.parse_args()
    require(not args.prepare_full or args.execute,'Full candidate requires --execute and root metadata-only review')
    require(not args.prepare_partial or (not args.execute and not args.evidence and not args.root_review),'Partial mode cannot consume or imply full execution grants')
    execute('full' if args.prepare_full else 'partial',args.evidence,args.root_review)
if __name__=='__main__':main()
