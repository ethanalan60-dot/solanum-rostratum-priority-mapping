"""Synthetic/metadata checks only. Never constructs a model or opens real NPZ."""
from pathlib import Path
import ast, copy, datetime, importlib.util, json, math, os, shutil, sys, traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'local_seed_stability_v1'/'export_focuschecks'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/Path(__file__).name)
    checks=[]
    try:
        exporter=ROOT/'export_local_seed_validation.py'
        shutil.copy2(exporter,out/exporter.name)
        shutil.copy2(ROOT/'local_seed_stability_v1/export_protocol.json',out/'export_protocol.json')
        source=exporter.read_text(encoding='utf-8');tree=ast.parse(source)
        spec=importlib.util.spec_from_file_location('seed_export_check_target',exporter)
        x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
        def passed(name):checks.append(name)
        def rejects(name,call):
            try:call()
            except (ValueError,FileNotFoundError,FileExistsError,KeyError):passed(name);return
            raise AssertionError('Accepted invalid fixture: '+name)
        assert 'torch' not in sys.modules and 'numpy' not in sys.modules
        passed('module import uses standard library only')
        plan,old,hashes=x.validate_static_sources();passed('fixed source-lock plan protocol and legacy source bytes')
        frozen_start=dict(hashes)
        assert len(plan['runs'])==16
        for row in plan['runs']:assert x.declared_entry(plan,row['run_id'])==row
        passed('all16 declared metadata identities')
        for rid in ('dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17','../bad','dev_seed_v1_lr1e4_resnet18_regression_L256_s29','unknown'):
            rejects('reject undeclared '+rid,lambda rid=rid:x.declared_entry(plan,rid))
        entry=plan['runs'][0]
        cfg={**x.SETTINGS,**{k:entry[k] for k in ('run_id','encoder','variant','lr','seed')},'device':'cuda:0',
             'stage':'development','test_accessed':False,'decision_metrics_used_for_selection':False,
             'n_train':985,'n_val':357,'fixed_output_grid':256,'validation_R':[1,2,4],'torch':'2.5.0'}
        state={'status':'completed','stage':'development','test_accessed':False,'executed_epochs':24,
               'optimizer_attempts_total':2976,'optimizer_updates_total':2976,'optimizer_skips_total':0,
               'best_epoch':1,'best_validation_balanced_mae':.2}
        x.validate_config(entry,cfg,state);passed('valid full synthetic config')
        mutations={'run_id':'unknown','encoder':'other','variant':'regression','lr':.1,'seed':17,'epochs':23,
            'batch_size':8,'accumulation_steps':1,'weight_decay':.01,'label_grid':128,'device':'cpu','threads':4,
            'gpu_memory_fraction':.5,'step_pause':0,'scale_augmentation':False,'amp':False,'sampler':'natural',
            'freeze_batchnorm':True,'stage':'smoke','test_accessed':True,'R6_accessed':True,
            'decision_metrics_used_for_selection':True,'n_train':984,'n_val':356,'fixed_output_grid':128,
            'validation_R':[1,2,4,6],'torch':{'version':'unexpected'}}
        mutations.update({k:1 for k in ('smoke','diagnostic','diagnostic_epochs','max_train','max_val','no_pretrained')})
        for key,value in mutations.items():
            bad=copy.deepcopy(cfg);bad[key]=value
            rejects('config rejects '+key,lambda bad=bad:x.validate_config(entry,bad,state))
        for key,value in {'status':'running','stage':'diagnostic','test_accessed':True,'R6_accessed':True,'executed_epochs':23,
                          'optimizer_attempts_total':2975,'optimizer_updates_total':0,'optimizer_skips_total':1}.items():
            bad=copy.deepcopy(state);bad[key]=value
            rejects('status rejects '+key,lambda bad=bad:x.validate_config(entry,cfg,bad))
        scale={'n':357,'n_positive':27,'mae':.1,'bias':0.,'mae_zero':.1,'mae_positive':.3,'balanced_mae':.2}
        history=[{'epoch':e,'scales':{str(r):dict(scale) for r in (1,2,4)},'selection_balanced_mae':.2,
                  'optimizer_update_attempts':124,'optimizer_updates':124,'optimizer_skipped_overflows':0} for e in range(1,25)]
        assert x.validate_history(history,history[0],state)['epoch']==1;passed('earliest exact tie original MAE')
        rejects('reject later exact tied checkpoint',lambda:x.validate_history(history,history[-1],{**state,'best_epoch':24}))
        for label,mutate in [
            ('23epochs',lambda h:h.pop()),('repeatedepoch',lambda h:h[-1].update(epoch=23)),
            ('R6',lambda h:h[2]['scales'].update({'6':dict(scale)})),
            ('wrongN',lambda h:h[2]['scales']['1'].update(n=356)),
            ('wrongPositiveN',lambda h:h[2]['scales']['1'].update(n_positive=26)),
            ('invalidbalance',lambda h:h[2]['scales']['1'].update(balanced_mae=.3)),
            ('wrongscore',lambda h:h[2].update(selection_balanced_mae=.3)),
            ('nonfinitebias',lambda h:h[2]['scales']['1'].update(bias=float('nan'))),
            ('stepcount',lambda h:h[2].update(optimizer_update_attempts=123)),
            ('zeroactual',lambda h:h[2].update(optimizer_updates=0,optimizer_skipped_overflows=124))]:
            bad=copy.deepcopy(history);mutate(bad)
            rejects('history rejects '+label,lambda bad=bad:x.validate_history(bad,history[0],state))
        fake=out/'synthetic_run';fake.mkdir()
        names=['config.json','status.json','metrics.jsonl','best_validation.json','best.pt','source/train.json','source/val.json']+[f'source/{n}' for n in x.SOURCE_NAMES]
        for name in names:
            p=fake/name;p.parent.mkdir(exist_ok=True);p.write_text('synthetic '+name,encoding='utf-8')
        receipt={'run_id':entry['run_id'],'binding':x.binding(),'best_epoch':1,'best_validation_balanced_mae':.2,
                 'files_sha256':{n:x.sha(fake/n) for n in names},'checkpoint_sha256':x.sha(fake/'best.pt')}
        x.validate_completion(fake,receipt,entry,state,{});passed('synthetic completion full ledger')
        for label,change in [('run',lambda r:r.update(run_id='other')),('binding',lambda r:r['binding'].update(plan_sha256='0'*64)),
            ('epoch',lambda r:r.update(best_epoch=2)),('score',lambda r:r.update(best_validation_balanced_mae=.3)),
            ('checkpoint',lambda r:r.update(checkpoint_sha256='0'*64)),('missingfile',lambda r:r['files_sha256'].pop('best.pt')),
            ('tamper',lambda r:r['files_sha256'].update({'best.pt':'0'*64}))]:
            bad=copy.deepcopy(receipt);change(bad)
            rejects('completion rejects '+label,lambda bad=bad:x.validate_completion(fake,bad,entry,state,{}))
        (fake/'extra.txt').write_text('unexpected');rejects('completion rejects unbound added file',lambda:x.validate_completion(fake,receipt,entry,state,{}))
        for path in ('../x','/absolute','C:/x','a//b','a/../b'):
            rejects('reject unsafe path '+path,lambda path=path:x.confined(fake,path))
        import numpy as np
        probability=np.stack([np.full((256,256),v,np.float32) for v in (.125,.5,.875)])
        valid=np.ones((256,256),np.float32);valid[::2,:]=.25;valid[:,::7]=0
        q=x.save_probability_map(out/'synthetic_map.npz',probability,valid)
        assert np.array_equal(q,np.array([.125,.5,.875]));passed('fractional valid support preserves constant predictions')
        rejects('never overwrite map',lambda:x.save_probability_map(out/'synthetic_map.npz',probability,valid))
        for label,p,v in [('wrongshape',probability[:2],valid),('wrongdtype',probability.astype(np.float64),valid),
            ('nan',probability*np.nan,valid),('outofrange',probability+1,valid),('emptyvalid',probability,np.zeros_like(valid)),
            ('invalidvalid',probability,valid+1)]:
            rejects('map rejects '+label,lambda p=p,v=v,label=label:x.save_probability_map(out/(label+'.npz'),p,v))
        # Guard inspection without importing Torch, loading checkpoint or invoking forward.
        loads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load']
        assert len(loads)==1 and {k.arg:ast.literal_eval(k.value) for k in loads[0].keywords}=={'map_location':'cpu','weights_only':True}
        passed('single checkpoint load explicitly CPU weights_only')
        assert 'torch' not in sys.modules;passed('no Torch import/model construction/forward in checks')
        missing=[r['run_id'] for r in plan['runs'] if not (x.SEED/'completion_records'/(r['run_id']+'.json')).exists()]
        if missing:rejects('real declared incomplete run has no completion receipt',lambda:x.validate_run(missing[0]))
        import check_local_seed_export_lifecycle as lifecycle
        shutil.copy2(lifecycle.__file__,out/Path(lifecycle.__file__).name)
        checks.extend(lifecycle.run_checks(x,out))
        assert 'torch' not in sys.modules
        x.recheck(frozen_start)
        result={'passed':True,'created_utc':x.now(),'checks':checks,'check_count':len(checks),
            'exporter_sha256':x.sha(exporter),'export_protocol_sha256':x.sha(x.PROTOCOL),'checker_sha256':x.sha(__file__),
            'lifecycle_checker_sha256':x.sha(lifecycle.__file__),
            'training_binding':x.binding(),'source_start_end_sha256':frozen_start,
            'real_cache_arrays_opened':False,'real_checkpoint_loaded':False,'torch_imported':False,
            'forward':False,'GPU_used':False,'test_accessed':False,'R6_accessed':False,
            'scope':'Source contract, synthetic configuration/history/completion rejection and NumPy-only map integration; not a production export acceptance.'}
        x.write(out/'focuschecks.json',result)
        print(json.dumps({'passed':True,'checks':len(checks),'path':str(out/'focuschecks.json'),'sha256':x.sha(out/'focuschecks.json')},ensure_ascii=False))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'completed_checks':checks},indent=2)+'\n',encoding='utf-8');raise

if __name__=='__main__':main()
