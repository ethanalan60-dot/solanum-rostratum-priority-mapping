"""Assemble source/evidence DRAFTS only. No official promotion, FREEZE or data execution.

Every review is an explicitly registered path/hash/field tuple. Future outputs are
accepted only via explicit CLI paths plus SHA256. No latest-run discovery exists.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,re,shutil,traceback

ROOT=Path(__file__).resolve().parent
CONF=ROOT/'confirmation_v1'
REGISTRY=ROOT/'confirmation_readiness_registry_v1.json'
REGISTRY_SHA='459d1ad791a42768b8f5e5a51be94e4cf82250e7fe93be58d49da95976f42f0e'
STATIC=('contract.json','extraction.json','external_arithmetic_extraction.json','scalar_baselines.json','checkpoint_candidate_protocol.json')
FORBIDDEN={'local_data_v2','weedsgalore_data_v1','external_data','stages','validation_exports','vx2'}
PREFIXES=('checkpoint','config','status','history','best_validation','completion_receipt','training_audit')

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.datetime.now().astimezone().isoformat()
def field(obj,tokens):
    require(isinstance(tokens,list) and tokens,'Explicit source field path required')
    for token in tokens:obj=obj[token]
    require(isinstance(obj,str) and re.fullmatch('[a-f0-9]{64}',obj),'Source field is not a SHA256')
    return obj
def under(root,path):
    p=Path(path).resolve();require(p.is_relative_to(Path(root).resolve()),'Path escaped source ROOT');return p

class Watch:
    def __init__(self):self.files={}
    def path(self,path):
        p=under(ROOT,path);rel=p.relative_to(ROOT)
        require(not set(rel.parts)&FORBIDDEN,'Dataset or prediction content is forbidden in assembler')
        require('test' not in p.name.lower() and 'r6' not in p.name.lower(),'Reserved filename forbidden')
        require(p.suffix.lower() in ('.json','.jsonl','.py','.md','.txt','.csv','.pt','.pth'),'Unsupported source type')
        return p
    def pin(self,path,expected=None):
        p=self.path(path);h=sha(p)
        if expected is not None:require(h==expected,'SHA mismatch: '+str(p))
        if str(p) in self.files:require(self.files[str(p)]==h,'Source changed during assembly')
        self.files[str(p)]=h;return p
    def read(self,path,expected=None):
        p=self.path(path);require(p.suffix=='.json','Only JSON metadata is parsed')
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        if expected is not None:require(h==expected,'JSON SHA mismatch: '+str(p))
        if str(p) in self.files:require(self.files[str(p)]==h,'JSON changed during assembly')
        self.files[str(p)]=h;return json.loads(raw.decode('utf-8-sig'))
    def binding(self,path):
        p=self.pin(path);return {'path':str(p),'sha256':self.files[str(p)]}
    def finish(self):
        for p,h in self.files.items():require(sha(p)==h,'Input drift at finalization: '+p)

def validate_review(watch,entry,home,allowed):
    review=watch.read(entry['path'],entry['sha256']);require(review.get('passed') is True,'Independent review did not pass')
    require(bool(entry['source_bindings']),'Review has no explicit source bindings')
    names=[]
    for item in entry['source_bindings']:
        name=item['file'];require(name in allowed and Path(name).name==name,'Unknown reviewed component')
        watch.pin(home/name,field(review,item['sha256_field']));names.append(name)
    require(len(names)==len(set(names)),'Duplicate component in one review')
    return names

def complete_index(watch,spec,seed_ids):
    index=watch.read(spec['path'],spec['sha256'])
    require(index.get('passed_for_registered_models') is True and index.get('all16_complete') is True and index.get('accepted_count')==16,'Complete16 numerical index required')
    require(index.get('test_or_R6_accessed') is False,'Numerical index reserved-data scope')
    rows=index['readiness_seed_numerical_reviews'];require(len(rows)==16 and {x['run_id'] for x in rows}==set(seed_ids),'Exact16 numerical population required')
    for entry in rows:
        review=watch.read(entry['path'],entry['sha256']);require(review.get('passed') is True and review.get('test_or_R6_accessed') is False,'Numerical review did not pass')
        found=[x for x in review['models'] if x['run_id']==entry['run_id']];require(len(found)==1,'Numerical reviewed model missing/ambiguous');model=found[0]
        for key,value in {'maps':357,'scalar_pairs':1071,'management_units':2183,'prefix_curve_points':6558,'fixed_prefix_rows':45,'P_policy_rows':45,'auc_rows':9}.items():require(model.get(key)==value,'Incomplete numerical counts')
        require(model.get('no_new_model_forward') is True and model.get('no_new_solver') is True and model.get('test_or_R6_accessed') is False,'Numerical receipt scope')
        receipt=watch.read(model['receipt'],model['receipt_sha256']);require(receipt['run_id']==entry['run_id'] and receipt['original_epoch_preserved'] is True and receipt['all_declared_cases_retained'] is True,'Numerical receipt run/epoch scope')
    return rows

def full24_review(watch,spec,local_ids):
    review=watch.read(spec['path'],spec['sha256'])
    require(review.get('passed') is True and review.get('scope')=='full24_seed_summary_independent','Full24 review did not pass')
    require(len(review['run_ids'])==24 and set(review['run_ids'])==set(local_ids),'Full24 exact population')
    require(review.get('test_or_R6_accessed') is False and review.get('new_model_or_solver_calls')==0,'Full24 review scope')
    folder=under(ROOT/'precision_agriculture',review['output_directory'])
    summary=watch.read(folder/'summary.json',review['summary_sha256'])
    manifest=watch.read(folder/'artifact_manifest.json',review['audited_artifact_manifest_sha256']);status=watch.read(folder/'status.json')
    require(status['state']=='completed' and status['artifact_manifest_sha256']==review['audited_artifact_manifest_sha256'],'Full24 status/manifest binding')
    require(manifest['file_sha256']['summary.json']==review['summary_sha256'],'Full24 summary not manifested')
    require(summary['models']==24 and summary['seeds']==[17,29,43],'Full24 dimensions')
    return review

def full_candidate(watch,spec,roster,index_spec,review_spec):
    candidate_path=watch.pin(spec['path'],spec['sha256']);table=watch.read(candidate_path)
    require(table.get('complete') is True and table.get('frozen') is False and table.get('all37_training_audited') is True,'Require complete unfrozen37 candidate')
    require(len(table['models'])==37,'Candidate row count')
    require([{k:x[k] for k in row} for x,row in zip(table['models'],roster)]==roster,'Candidate population/config/order mismatch')
    folder=candidate_path.parent;manifest=watch.read(folder/'artifact_manifest.json');status=watch.read(folder/'status.json')
    require(status['state']=='completed' and status['mode']=='full' and status['audited']==37 and status['pending']==0,'Candidate is not completed full mode')
    require(manifest['files_sha256'][candidate_path.name]==spec['sha256'],'Candidate file not manifested')
    audits={(x['training_audit_path'],x['training_audit_sha256']) for x in table['models']};require(len(audits)==1,'Candidate requires one aggregate37 audit')
    ap,ah=next(iter(audits));audit=watch.read(ap,ah)
    require(Path(ap).resolve()==(folder/'training_audit.json').resolve() and manifest['files_sha256']['training_audit.json']==ah and status['training_audit_sha256']==ah,'Candidate training audit chain')
    require(audit.get('passed') is True and audit['mode']=='full' and audit['all37_complete'] is True and audit['audited_count']==37 and not audit['pending'],'Training audit not all37')
    require(audit['epochs_verified']==888 and audit['scales_verified']==2664 and audit['optimizer_events_verified']==86136,'Full37 audit dimensions')
    require(audit.get('test_or_R6_accessed') is False and audit.get('no_model_or_solver_calls') is True,'Training audit scope')
    watch.pin(CONF/'build_checkpoint_candidates.py',audit['implementation_sha256']);watch.pin(CONF/'checkpoint_candidate_protocol.json',audit['protocol_sha256'])
    accepted=audit['prerequisites'];require(accepted['all16_numerical_accepted'] is True and accepted['full24_summary_accepted'] is True,'Candidate lacks prerequisite acceptance')
    for key,expected in [('seed_numerical_index',index_spec),('full24_independent_review',review_spec)]:
        actual=accepted['evidence'][key]
        require(expected and Path(actual['path']).resolve()==Path(expected['path']).resolve() and actual['sha256']==expected['sha256'],'Candidate/evidence source mismatch '+key)
    audited={x['run_id']:x for x in audit['runs']};require(len(audit['runs'])==37 and len(audited)==37,'Training audit duplicate/missing model')
    for row in table['models']:
        require(row['original_selection_audited'] is True,'Original best not audited')
        for prefix in PREFIXES:watch.pin(row[prefix+'_path'],row[prefix+'_sha256'])
        proof=audited[row['run_id']]
        for key in ('best_epoch','checkpoint_sha256','status_sha256'):require(proof[key]==row[key],'Training audit selected checkpoint mismatch')
    return {'candidate':spec,'training_audit':{'path':str(Path(ap).resolve()),'sha256':ah},'models':37,'frozen':False}

def build(args):
    watch=Watch();watch.pin(__file__);registry=watch.read(REGISTRY,REGISTRY_SHA)
    require(registry['version']=='confirmation_readiness_registry_v1','Wrong registry')
    contract=watch.read(CONF/'contract.json');roster=contract['roster'];require(len(roster)==37 and len({r['run_id'] for r in roster})==37,'Wrong fixed37 roster')
    seed_ids=[r['run_id'] for r in roster if r['dataset']=='local' and r['seed']!=17]
    local_ids=[r['run_id'] for r in roster if r['dataset']=='local' and r['variant']!='regression'];require(len(seed_ids)==16 and len(local_ids)==24,'Fixed local populations')
    supplement={'module_reviews':[],'output_auditor_reviews':[]}
    if args.supplement:
        supplement=watch.read(args.supplement,args.supplement_sha256);require(supplement['version']=='confirmation_readiness_supplement_v1','Wrong explicit supplement schema')
    # This enumerates source filenames only; it never discovers result runs or selects by date.
    python_names={p.name for p in CONF.glob('*.py')};required_production={n for n in python_names if not n.startswith('check_')}
    documentation_names={p.name for p in CONF.glob('*.md')}
    required_files=python_names|set(STATIC)|documentation_names
    source_hashes={name:watch.binding(CONF/name)['sha256'] for name in sorted(required_files)}
    checks=[];accepted=[];covered=set();extra_covered=set();accepted_outputs=[]
    def item(name,state,detail=None):checks.append({'name':name,'state':state,'detail':detail})
    def review_entry(entry,home,allowed,destination,coverage):
        try:names=validate_review(watch,entry,home,allowed)
        except FileNotFoundError as error:item('review:'+entry['path'],'pending',repr(error));return
        except (ValueError,KeyError,TypeError) as error:item('review:'+entry['path'],'failed',repr(error));return
        coverage.update(names);destination.append(entry);item('review:'+entry['path'],'passed',names)
    for entry in registry['module_reviews']+supplement.get('module_reviews',[]):review_entry(entry,CONF,required_files,accepted,covered)
    for name in sorted(required_production-covered):item('independent_module:'+name,'pending',registry.get('pending_at_registration',{}).get(name,'Explicit current-source independent review missing'))
    dependencies={}
    for name,h in registry['external_dependency_pins'].items():
        watch.pin(ROOT/name,h);dependencies[name]=h;item('dependency:'+name,'passed')
    # Check literal consumer pins by AST, without importing spatial/model code.
    def literals(path):
        values={}
        for node in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body:
            if isinstance(node,ast.Assign):
                try:value=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                for target in node.targets:
                    if isinstance(target,ast.Name):values[target.id]=value
        return values
    lp=literals(CONF/'local_management.py');qp=literals(CONF/'run_prediction_queue.py');gp=literals(CONF/'local_geometry.py')
    require({**lp['PINS'],'run_local_seed_postprocess_queue_v2.py':qp['OWNER_SHA']}==dependencies,'Consumer dependency pins changed')
    require(gp['GEO_SHA']==dependencies['geo_management.py'],'Geometry source pin differs')
    for record in registry.get('supplementary_records',[]):
        review=watch.read(record['review_path'],record['review_sha256']);require(review['passed'] is True,'Supplementary record review failed')
        watch.pin(CONF/record['file'],field(review,record['sha256_field']));item('supplementary_record:'+record['file'],'passed')
    index_spec={'path':str(args.seed_index),'sha256':args.seed_index_sha256} if args.seed_index else None
    full_spec={'path':str(args.full24_review),'sha256':args.full24_review_sha256} if args.full24_review else None
    candidate_spec={'path':str(args.checkpoint_candidate),'sha256':args.checkpoint_candidate_sha256} if args.checkpoint_candidate else None
    output_names=set(registry['required_root_output_auditors']);output_sources={}
    for name in sorted(output_names):output_sources[name]=watch.binding(ROOT/name)['sha256']
    for entry in supplement.get('output_auditor_reviews',[]):review_entry(entry,ROOT,output_names,accepted_outputs,extra_covered)
    # The explicitly supplied completed full24 independent receipt already pins
    # this metadata reviewer. No extra self-referential audit/approval is needed.
    if full_spec:
        entry={**full_spec,'source_bindings':[{'file':'review_full24_seed_results.py','sha256_field':['reviewer_sha256']}]}
        review_entry(entry,ROOT,output_names,accepted_outputs,extra_covered)
    for name in sorted(output_names-extra_covered):item('output_auditor_review:'+name,'pending','Source bytes pinned; explicit final independent preparation acceptance not provided')
    numerical=[];candidate=None
    def optional(name,spec,fn):
        if not spec:item(name,'pending','Explicit path and SHA256 not supplied');return None
        try:value=fn();item(name,'passed');return value
        except (ValueError,KeyError,TypeError,FileNotFoundError) as error:item(name,'failed',repr(error));return None
    numerical=optional('complete16_numerical_index',index_spec,lambda:complete_index(watch,index_spec,seed_ids)) or []
    optional('full24_independent_output_review',full_spec,lambda:full24_review(watch,full_spec,local_ids))
    candidate=optional('complete37_checkpoint_candidate',candidate_spec,lambda:full_candidate(watch,candidate_spec,roster,index_spec,full_spec))
    watch.finish()
    evidence={'version':'confirmation_readiness_evidence_v1','module_reviews':accepted,'seed_numerical_reviews':numerical,
              'explicit_final_sources':{'checkpoint_candidate':candidate_spec,'seed_numerical_index':index_spec,'full24_independent_review':full_spec},
              'root_output_auditor_reviews':accepted_outputs,'scope':'draft_only_not_execution_authorization'}
    source_lock={'version':'confirmation_source_v1','created_at':now(),'files':source_hashes,'external_dependency_sha256':dependencies,
                 'root_output_auditor_sha256':output_sources,'assembly_scope':'candidate_only_no_automatic_promotion'}
    report={'scope':'confirmation_readiness_draft_assembly','integrity_checks_passed':not any(x['state']=='failed' for x in checks),
            'complete_for_root_review':all(x['state']=='passed' for x in checks),'checks':checks,
            'counts':{'source_files':len(source_hashes),'documentation_md':len(documentation_names),'production_py':len(required_production),'reviewed_production_py':len(required_production&covered),
                      'accepted_module_reviews':len(accepted),'required_output_auditors':len(output_names),'reviewed_output_auditors':len(extra_covered),'accepted_seed_numerical_reviews':len(numerical)},
            'candidate_metadata':candidate,'pending':sum(x['state']=='pending' for x in checks),'failed':sum(x['state']=='failed' for x in checks),
            'source_inputs_sha256':watch.files,'created_at':now(),'no_freeze_or_official_file_write':True,'no_tensor_or_reserved_data_load':True,
            'execution_authorized':False,'source_registry_sha256':REGISTRY_SHA}
    return source_lock,evidence,report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('supplement','checkpoint-candidate','seed-index','full24-review'):
        ap.add_argument('--'+name,type=Path);ap.add_argument('--'+name+'-sha256')
    ap.add_argument('--require-complete',action='store_true',help='Write draft, then exit2 if pending/failed; never promotes files.')
    args=ap.parse_args()
    for name in ('supplement','checkpoint_candidate','seed_index','full24_review'):
        value=getattr(args,name);digest=getattr(args,name+'_sha256')
        require((value is None)==(digest is None),'Path and SHA256 must both be explicit: '+name)
        if digest:require(re.fullmatch('[a-f0-9]{64}',digest) is not None,'Invalid SHA256')
    out=ROOT/'provenance'/('confirmation_readiness_draft_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    try:
        shutil.copy2(__file__,out/'prepare_confirmation_readiness.py');shutil.copy2(REGISTRY,out/'registry.json')
        lock,evidence,report=build(args)
        dump(out/'SOURCE_LOCK.candidate.json',lock);dump(out/'readiness_evidence.draft.json',evidence);dump(out/'assembly_report.json',report)
        dump(out/'artifact_manifest.json',{'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}})
        print(json.dumps({'out':str(out),'complete_for_root_review':report['complete_for_root_review'],'pending':report['pending'],'failed':report['failed'],'counts':report['counts'],'assembly_sha256':sha(out/'assembly_report.json')}))
        if args.require_complete and not report['complete_for_root_review']:return 2
    except BaseException as error:
        dump(out/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'official_files_changed':False});raise
if __name__=='__main__':raise SystemExit(main())
