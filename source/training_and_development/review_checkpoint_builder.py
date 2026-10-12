"""Root independent synthetic review of checkpoint metadata arithmetic; no real fits."""
from pathlib import Path
import ast,copy,datetime,importlib.util,json,math,random,sys
ROOT=Path(__file__).resolve().parent;CONF=ROOT/'confirmation_v1';sys.path.insert(0,str(CONF))
import build_checkpoint_candidates as b

def main():
    out=ROOT/'provenance'/('checkpoint_builder_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();checks=[]
    def ok(name,fn):fn();checks.append({'name':name,'passed':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError,OSError,AssertionError):checks.append({'name':name,'passed':True});return
        raise AssertionError('Accepted '+name)
    def yes(v):assert v
    ok('no_array_tensor_solver_import',lambda:yes(not any(x in sys.modules for x in ('numpy','torch','scipy'))))
    cfg={'seed':71,'lr':.0003,'run_id':'manual_local','variant':'soft_seg'};rng=random.Random(1071);hist=[];events=[];actual=skip=0;scale=65536.
    for epoch in range(1,25):
        scores={};base=.02+abs(epoch-9)*.003
        if epoch==13:base=.02
        for r in (1,2,4):
            z=base+r*.001;p=base+.2+r*.002
            scores[str(r)]={'n':357,'n_positive':27,'mae_zero':z,'mae_positive':p,'balanced_mae':(z+p)/2,'mae':(330*z+27*p)/357,'bias':.001,'target_iou':.12,'dice':.2}
        factors={'1':0,'2':0,'4':0}
        for start in range(0,985,4):factors[str(rng.choice((1,2,4)))]+=min(4,985-start)
        batch=[]
        for group in range(1,125):
            n=1 if group==124 else 8;attempt=(epoch-1)*124+group;skipped=attempt in (13,1200);before=scale
            if skipped:skip+=1;scale/=2
            else:actual+=1
            batch.append(dict(attempt=attempt,epoch=epoch,group=group,group_samples=n,group_denominator=n*(31000.+group),group_loss=.01+group*.0001,
              amp_enabled=True,scale_before=before,scale_after=scale,optimizer_step_executed=not skipped,all_unscaled_gradients_finite=not skipped,
              skipped=skipped,status='overflow_skipped' if skipped else 'updated',unscaled_gradient_norm=None if skipped else .7,
              actual_updates_total=actual,skipped_updates_total=skip,consecutive_overflows=1 if skipped else 0))
        events.extend(batch)
        hist.append(dict(epoch=epoch,scales=scores,selection_balanced_mae=sum(v['balanced_mae'] for v in scores.values())/3,
          train_factor_sample_counts=factors,lr=cfg['lr']*(1+math.cos(math.pi*(epoch-1)/24))/2,
          train_loss=sum(x['group_loss']*x['group_denominator'] for x in batch)/sum(x['group_denominator'] for x in batch),
          optimizer_update_attempts=124,optimizer_updates=sum(not x['skipped'] for x in batch),optimizer_skipped_overflows=sum(x['skipped'] for x in batch),amp_scale_end=scale,consecutive_overflows=batch[-1]['consecutive_overflows']))
    best=hist[8];status=dict(best_epoch=9,best_validation_balanced_mae=best['selection_balanced_mae'],optimizer_attempts_total=2976,optimizer_updates_total=actual,optimizer_skips_total=skip,amp_scale=scale)
    ok('local24_threeR_stratum_cosine_and_nontie_earliest_tie',lambda:b.check_history(hist,best,status,cfg,[]))
    ok('2976_events_two_overflows_tail_partial_valid_weighted_loss',lambda:b.check_events(events,hist,status,'soft_seg',False))
    reject('later_equal_minimum_refused',lambda:b.check_history(hist,hist[12],{**status,'best_epoch':13},cfg,[]))
    for key,value in [('n',358),('n_positive',28),('mae',.5),('mae_zero',float('nan')),('bias',.9),('balanced_mae',.9),('dice',1.1)]:
        def change(key=key,value=value):
            h=copy.deepcopy(hist);h[5]['scales']['2'][key]=value;b.check_history(h,best,status,cfg,[])
        reject('local_'+key,change)
    for key,value in [('group_samples',8),('group_denominator',70000.),('actual_updates_total',1),('scale_before',32.),('scale_after',.5),('consecutive_overflows',8)]:
        def change(key=key,value=value):
            e=copy.deepcopy(events);e[123][key]=value;b.check_events(e,hist,status,'soft_seg',False)
        reject('tail_event_'+key,change)
    for key,value in [('train_loss',.8),('optimizer_updates',124),('amp_scale_end',1.)]:
        def change(key=key,value=value):
            h=copy.deepcopy(hist);h[0][key]=value;b.check_events(events,h,status,'soft_seg',False)
        reject('epoch_'+key,change)
    # Nonuniform externally saved scalar/date fixture, independently calculate all4 metric fields.
    samples=[{'prediction':p,'truth':t} for p,t in [(.1,.4),(.6,.2),(.9,.9),(.01,0.)]]
    m=b.metric(samples);expected={'n':4,'mae':.1775,'bias':.0275,'mean_prediction':.4025,'mean_target':.375}
    ok('independent_sample_metric_values',lambda:b.check_metric(m,expected))
    reject('empty_external_samples',lambda:b.metric([]));reject('nonfinite_external_samples',lambda:b.metric([{'prediction':float('nan'),'truth':0}]))
    ledger=b.Ledger()
    for name in ('test_sealed.json','R6_results.json','test.npz'):
        reject('reserved_filename_'+name,lambda name=name:ledger.path(ROOT/name))
    reject('unregistered_array_forbidden',lambda:ledger.path(ROOT/'local_data_v2/cache/unlisted.npz'))
    reject('unapproved_metadata_forbidden',lambda:ledger.path(ROOT/'local_data_v2/something.json'))
    p=out/'metadata.json';b.write(p,{'value':1});ledger.read(p);b.write(p,{'value':2});reject('begin_end_byte_drift',ledger.finish)
    source=Path(b.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    ok('no_tensor_array_loading_code',lambda:yes(not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('load','load_state_dict','milp','fit','predict') for n in calls)))
    ok('official_checkpoints_FREEZE_never_written',lambda:yes("write(HERE/'FREEZE.json'" not in source and "write(HERE/'checkpoints.json'" not in source))
    # Preserve exact author full-prerequisite and claim tests as separate, not falsely called independent.
    author=ROOT/'provenance/checkpoint_candidate_checks_20260916_105852_710672/focused_checks.json';receipt=json.loads(author.read_text(encoding='utf-8'))
    ok('author47_checks_bound',lambda:yes(receipt['passed'] is True and len(receipt['checks'])==47))
    result=dict(passed=True,scope='checkpoint_candidate_builder_independent_preparation',implementation_sha256=b.sha(b.__file__),protocol_sha256=b.sha(b.PROTOCOL),reviewer_sha256=b.sha(__file__),checks=checks,check_count=len(checks),author_full_gate_checks={'path':str(author),'sha256':b.sha(author)},test_or_R6_accessed=False,model_or_solver_calls=0,
      source_bindings={str(Path(b.__file__).resolve()):b.sha(b.__file__),str(b.PROTOCOL.resolve()):b.sha(b.PROTOCOL)},
      limits=['This is a preparation review using manual fixtures and source inspection, not a full37 actual training audit.','Full prerequisite lifecycle cases have separate producer47 evidence.'])
    b.write(out/'review.json',result);print(json.dumps({'out':str(out),'passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
