"""Synthetic arithmetic, full prerequisite and lifecycle checks. No datasets/tensors."""
from pathlib import Path
import copy, datetime, importlib.util, json, math, random, shutil, sys
import build_checkpoint_candidates as b

def fixture():
    val=[{'id':'v%02d'%i,'acquisition_date':'d%d'%(i%4),'coverage':.2} for i in range(26)]
    cfg={'run_id':'synthetic','seed':17,'lr':1e-4,'variant':'hard_seg','version':'external_weeds_v1'}
    samples=[{'id':x['id'],'acquisition_date':x['acquisition_date'],'prediction':.3,'truth':b.f32(x['coverage'])} for x in val]
    measured={**b.metric(samples),'samples':samples,'by_date':{d:b.metric([x for x in samples if x['acquisition_date']==d]) for d in ('d0','d1','d2','d3')}}
    history=[];events=[];rng=random.Random(1017)
    for epoch in range(1,25):
        factors={'1':0,'2':0,'4':0}
        for _ in range(26):factors[str(rng.choice((1,2,4)))]+=4
        history.append({'epoch':epoch,'selection_mae':measured['mae'],'scales':{str(r):copy.deepcopy(measured) for r in (1,2,4)},'train_factor_sample_counts':factors,
            'lr':1e-4*(1+math.cos(math.pi*(epoch-1)/24))/2,'train_loss':.3,'optimizer_update_attempts':13,'optimizer_updates':13,'optimizer_skipped_overflows':0,'amp_scale_end':65536.,'consecutive_overflows':0})
        for group in range(1,14):
            attempt=(epoch-1)*13+group
            events.append({'epoch':epoch,'group':group,'attempt':attempt,'group_samples':8,'group_denominator':8*65536.,'group_loss':.3,
                'amp_enabled':True,'scale_before':65536.,'scale_after':65536.,'optimizer_step_executed':True,'all_unscaled_gradients_finite':True,
                'unscaled_gradient_norm':1.,'skipped':False,'status':'updated','actual_updates_total':attempt,'skipped_updates_total':0,'consecutive_overflows':0})
    status={'best_epoch':1,'best_validation_mae':measured['mae'],'optimizer_attempts_total':312,'optimizer_updates_total':312,'optimizer_skips_total':0,'amp_scale':65536.}
    return history,events,status,cfg,val

def main():
    out=b.ROOT/'provenance'/('checkpoint_candidate_checks_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    results=[]
    def ok(name,fn):fn();results.append({'name':name,'passed':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError,OSError):results.append({'name':name,'passed':True});return
        raise AssertionError(name+' was accepted')
    h,e,s,c,v=fixture()
    ok('all24_external_saved_sample_and_date_metrics',lambda:b.check_history(h,h[0],s,c,v))
    ok('all312_events',lambda:b.check_events(e,h,s,'hard_seg',True))
    reject('earliest_tie_cannot_choose_epoch2',lambda:b.check_history(h,h[1],{**s,'best_epoch':2},c,v))
    reject('missing_epoch',lambda:b.check_history(h[:-1],h[0],s,c,v))
    def corrupt_h(key,value):
        hh=copy.deepcopy(h);hh[5][key]=value;return b.check_history(hh,hh[0],s,c,v)
    for name,key,value in [('wrong_augmentation','train_factor_sample_counts',{'1':104,'2':0,'4':0}),('wrong_selection','selection_mae',.8),('wrong_lr','lr',.8)]:reject(name,lambda key=key,value=value:corrupt_h(key,value))
    def sample_corruption(kind):
        hh=copy.deepcopy(h)
        if kind=='truth':hh[0]['scales']['1']['samples'][0]['truth']=.5
        elif kind=='date':hh[0]['scales']['1']['samples'][0]['acquisition_date']='other'
        elif kind=='id':hh[0]['scales']['1']['samples'][0]['id']='other'
        elif kind=='nan':hh[0]['scales']['1']['samples'][0]['prediction']=float('nan')
        else:hh[0]['scales']['1']['by_date']['d0']['mae']=.5
        return b.check_history(hh,hh[0],s,c,v)
    for kind in ('truth','date','id','nan','date_metric'):reject('sample_'+kind,lambda kind=kind:sample_corruption(kind))
    def corrupt_e(key,value):
        ee=copy.deepcopy(e);ee[9][key]=value;return b.check_events(ee,h,s,'hard_seg',True)
    for key,value in [('attempt',11),('epoch',2),('group',9),('group_samples',4),('group_denominator',1),('skipped',True),('actual_updates_total',9),('scale_before',32768),('unscaled_gradient_norm',float('nan')),('consecutive_overflows',1)]:reject('event_'+key,lambda key=key,value=value:corrupt_e(key,value))
    reject('missing_event',lambda:b.check_events(e[:-1],h,s,'hard_seg',True))
    reject('final_count_mismatch',lambda:b.check_events(e,h,{**s,'optimizer_updates_total':311},'hard_seg',True))
    # One explicitly skipped event, with all cumulative/epoch accounting recomputed.
    ee=copy.deepcopy(e);hh=copy.deepcopy(h);ss=copy.deepcopy(s)
    for i,event in enumerate(ee):
        if i>=9:
            event['actual_updates_total']-=1;event['skipped_updates_total']=1;event['scale_after']=32768.
            if i>9:event['scale_before']=32768.
    ee[9].update(optimizer_step_executed=False,all_unscaled_gradients_finite=False,skipped=True,status='overflow_skipped',consecutive_overflows=1)
    hh[0].update(optimizer_updates=12,optimizer_skipped_overflows=1)
    for row in hh:row['amp_scale_end']=32768.
    ss.update(optimizer_updates_total=311,optimizer_skips_total=1,amp_scale=32768.)
    ok('valid_overflow_skip_counts_and_scale',lambda:b.check_events(ee,hh,ss,'hard_seg',True))
    ledger=b.Ledger()
    reject('reserved_manifest_even_hash_rejected',lambda:ledger.path(b.ROOT/'local_data_v2/test_sealed.json'))
    reject('unregistered_npz_rejected',lambda:ledger.path(b.ROOT/'local_data_v2/cache/unregistered.npz'))
    reject('outside_root_rejected',lambda:ledger.path(b.ROOT.parent/'outside.json'))
    reject('relative_manifest_traversal',lambda:b.safe_relative(out,'../../escape.json'))
    # Full gate object fixture: no real file IO through this synthetic Ledger.
    rows=[{'run_id':'old%d'%i,'dataset':'local','seed':17,'variant':'hard_seg','encoder':'resnet18','lr':1e-4} for i in range(8)]+[{'run_id':'new%d'%i,'dataset':'local','seed':29 if i<8 else 43,'variant':'soft_seg','encoder':'resnet18','lr':1e-4} for i in range(16)]
    evidence={'version':'confirmation_checkpoint_candidate_evidence_v1','seed_numerical_index':{'path':'index.json','sha256':'i'},'full24_summary':{'path':'summary/summary.json','sha256':'s'},'full24_independent_review':{'path':'review.json','sha256':'r'}}
    objects={};digests={};audited={};refs=[]
    for row in rows[8:]:
        rid=row['run_id'];refs.append({'run_id':rid,'path':rid+'/review.json','sha256':'a'})
        rr={'run_id':rid,'maps':357,'scalar_pairs':1071,'management_units':2183,'prefix_curve_points':6558,'fixed_prefix_rows':45,'P_policy_rows':45,'auc_rows':9,'no_new_model_forward':True,'no_new_solver':True,'test_or_R6_accessed':False,'receipt':rid+'/receipt.json','receipt_sha256':'b'}
        objects[rid+'/review.json']={'passed':True,'test_or_R6_accessed':False,'models':[rr]}
        objects[rid+'/receipt.json']={'run_id':rid,'original_epoch_preserved':True,'all_declared_cases_retained':True,'test_or_R6_accessed':False,'export_directory':rid+'/export','spatial_directory':rid+'/spatial','export_status_sha256':'c','export_manifest_sha256':'d','spatial_status_sha256':'e','spatial_summary_sha256':'f'}
        objects[rid+'/export/metadata.json']={'checkpoint_sha256':'cp','checkpoint_epoch':4}
        objects[rid+'/export/artifact_manifest.json']={'file_sha256':{'metadata.json':'m'}}
        objects[rid+'/export/status.json']={'status':'completed'};objects[rid+'/spatial/status.json']={'state':'completed'}
        digests[rid+'/export/metadata.json']='m';audited[rid]={'checkpoint_sha256':'cp','best_epoch':4}
    objects['index.json']={'all16_complete':True,'accepted_count':16,'passed_for_registered_models':True,'readiness_seed_numerical_reviews':refs}
    objects['summary/summary.json']={'models':24,'seeds':[17,29,43],'table_rows':{'all2160_model_policy_budgets':2160,'all216_model_prefix_auc':216,'all72_cpu_scalar_by_R':72,'all12_hard_soft_model_pairs':12,'all1080_hard_soft_budget_pairs':1080,'all108_hard_soft_auc_pairs':108,'all36_hard_soft_cpu_scalar_pairs':36},'test_or_R6_accessed':False,'new_model_calls':0,'new_solver_calls':0}
    objects['summary/models.json']=[{**x,'checkpoint_sha256':'cp','checkpoint_epoch':4} for x in rows]
    for row in rows:audited[row['run_id']]={'checkpoint_sha256':'cp','best_epoch':4}
    objects['review.json']={'passed':True,'scope':'full24_seed_summary_independent','run_ids':[x['run_id'] for x in rows],'summary_sha256':'s','audited_artifact_manifest_sha256':'am'}
    objects['summary/artifact_manifest.json']={'file_sha256':{'summary.json':'s','models.json':'models'}}
    objects['summary/status.json']={'state':'completed','artifact_manifest_sha256':'am','source_sha256':'source'}
    digests.update({'summary/summary.json':'s','summary/models.json':'models'})
    class Fake:
        def read(self,path,expected=None):return copy.deepcopy(objects[Path(path).as_posix()])
        def digest(self,path,expected=None):return digests.get(Path(path).as_posix(),expected or 'hash')
    original_under=b.under
    # Only the synthetic relative summary fixture needs this root bypass; no IO.
    b.under=lambda root,path:Path(path)
    try:
        ok('full16_and_full24_gate_valid_fixture',lambda:b.full_prerequisites(Fake(),evidence,rows,audited))
        for key,value in [('all16_complete',False),('accepted_count',10)]:
            prior=objects['index.json'][key];objects['index.json'][key]=value
            reject('full_gate_'+key,lambda:b.full_prerequisites(Fake(),evidence,rows,audited));objects['index.json'][key]=prior
        objects['review.json']['passed']=False
        reject('full24_failed_review',lambda:b.full_prerequisites(Fake(),evidence,rows,audited));objects['review.json']['passed']=True
        objects['new0/export/metadata.json']['checkpoint_sha256']='wrong'
        reject('numerical_other_checkpoint',lambda:b.full_prerequisites(Fake(),evidence,rows,audited));objects['new0/export/metadata.json']['checkpoint_sha256']='cp'
        objects['new0/export/artifact_manifest.json']['file_sha256']['metadata.json']='wrong'
        reject('unsealed_export_metadata',lambda:b.full_prerequisites(Fake(),evidence,rows,audited))
        objects['new0/export/artifact_manifest.json']['file_sha256']['metadata.json']='m'
        objects['summary/models.json'][0]['checkpoint_sha256']='wrong'
        reject('full24_other_checkpoint',lambda:b.full_prerequisites(Fake(),evidence,rows,audited));objects['summary/models.json'][0]['checkpoint_sha256']='cp'
        objects['summary/summary.json']['table_rows']['all2160_model_policy_budgets']=720
        reject('old8_summary_not_full24',lambda:b.full_prerequisites(Fake(),evidence,rows,audited));objects['summary/summary.json']['table_rows']['all2160_model_policy_budgets']=2160
    finally:b.under=original_under
    # Source safety and fixed no-promotion contract are static checks, not production execution.
    import ast
    tree=ast.parse(Path(b.__file__).read_text(encoding='utf-8'))
    imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|{x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names}
    ok('stdlib_only_no_tensor_arrays_process_solver_imports',lambda:b.require(not imports&{'torch','numpy','scipy','subprocess','pickle'},'Disallowed import'))
    text=Path(b.__file__).read_text(encoding='utf-8')
    ok('candidate_frozen_false_only',lambda:b.require("'frozen':True" not in text and "'frozen':False" in text,'Frozen candidate'))
    ok('separate_output_from_official_confirmation',lambda:b.require(b.OUTPUT.parent==b.ROOT and b.OUTPUT!=b.HERE,'Wrong output root'))
    # Entire execute lifecycle uses artificial statuses and stubbed computation.
    # The injected failure precedes all production data/checkpoint/source access.
    from unittest.mock import patch
    fakehome=out/'mock';fakehome.mkdir();fakehere=fakehome/'confirmation';fakehere.mkdir()
    fakeroster=[]
    for i in range(37):
        folder=fakehome/('r%02d'%i);folder.mkdir();b.write(folder/'status.json',{'status':'completed'})
        fakeroster.append({'run_id':'r%02d'%i,'dataset':'local' if i<28 else 'weeds','run_directory':str(folder)})
    fakeprotocol=fakehere/'protocol.json';b.write(fakeprotocol,{'fixture':True})
    class LifeLedger:
        def __init__(self):self.files={}
        def digest(self,path,expected=None):return 'hash'
        def read(self,path,expected=None):
            name=Path(path).name
            if name=='protocol.json':return {'version':'confirmation_checkpoint_candidate_protocol_v1','contract_sha256':'hash'}
            if name=='contract.json':return {'roster':fakeroster}
            if name=='review.json':return {'passed':True,'scope':'confirmation_checkpoint_candidate_full_metadata_only','implementation_sha256':'hash','protocol_sha256':'hash','contract_sha256':'hash','evidence_sha256':'hash','test_or_model_access_authorized':False}
            if name=='evidence.json':return {'fixture':True}
            raise AssertionError('Unexpected fixture read '+str(path))
    def primary_failure(*args):raise RuntimeError('synthetic initialization failure')
    original_write=b.write
    def finalization_failure(path,value):
        if Path(path).name=='artifact_manifest.json':raise OSError('synthetic manifest failure')
        original_write(path,value)
    def run_failure(output,manifest_bad=False,dump_bad=False):
        with patch.multiple(b,ROOT=fakehome,HERE=fakehere,PROTOCOL=fakeprotocol,OUTPUT=output,Ledger=LifeLedger,load_development=lambda ledger:{},prepare_sources=primary_failure):
            with patch.object(b,'write',side_effect=finalization_failure if manifest_bad else original_write):
                if dump_bad:
                    with patch.object(b.json,'dump',side_effect=OSError('synthetic claim serialization failure')):
                        try:b.execute('full','evidence.json','review.json')
                        except OSError as error:b.require(str(error)=='synthetic claim serialization failure','Primary serialization failure masked')
                        else:raise AssertionError('Failure not raised')
                else:
                    try:b.execute('full','evidence.json','review.json')
                    except RuntimeError as error:b.require(str(error)=='synthetic initialization failure','Primary initialization failure masked')
                    else:raise AssertionError('Failure not raised')
            require_claim=(output/'full_claim.json').exists();b.require(require_claim,'Claim missing after failure')
            failures=list(output.glob('*/failure.json'));b.require(len(failures)==1,'Failure receipt not preserved')
            b.require(json.loads(failures[0].read_text(encoding='utf-8'))['no_retry_claim_preserved'] is True,'Wrong failure semantics')
            reject('permanent_claim_rejects_'+output.name,lambda:b.execute('full','evidence.json','review.json'))
    ok('postclaim_initialization_failure_preserved',lambda:run_failure(fakehome/'o1'))
    ok('manifest_failure_retains_original_exception',lambda:run_failure(fakehome/'o2',manifest_bad=True))
    ok('claim_serialization_failure_preserved',lambda:run_failure(fakehome/'o3',dump_bad=True))
    ok('no_mock_official_checkpoint_or_freeze_write',lambda:b.require(not list(fakehome.rglob('FREEZE.json')) and not list(fakehome.rglob('checkpoints.json')),'Official file created'))
    for p in (Path(__file__),Path(b.__file__),b.PROTOCOL):shutil.copy2(p,out/p.name)
    report={'passed':True,'checks':results,'count':len(results),'builder_sha256':b.sha(b.__file__),'protocol_sha256':b.sha(b.PROTOCOL),'checker_sha256':b.sha(__file__),
            'actual_test_or_R6_accessed':False,'new_model_or_solver_calls':0,'actual_checkpoint_or_cache_opened':False}
    b.write(out/'focused_checks.json',report);print(json.dumps({'out':str(out),'passed':True,'checks':len(results),'sha256':b.sha(out/'focused_checks.json')}))
if __name__=='__main__':main()
