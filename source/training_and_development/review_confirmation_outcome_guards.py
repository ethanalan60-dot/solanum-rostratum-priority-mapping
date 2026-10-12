"""Independent final identity-guard checks on synthetic metadata; no test arrays or model."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,math,os,shutil,sys,traceback
ROOT=Path(__file__).resolve().parent;HOME=ROOT/'confirmation_v1';sys.path.insert(0,str(HOME))
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import core,external_spatial as ext,external_arithmetic as a

def main():
    out=ROOT/'provenance'/('confirmation_outcome_guards_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    original_review=ROOT/'provenance/confirmation_outcomes_independent_20260916_100026_174523/review.json'
    sources={str(HOME/n):core.sha(HOME/n) for n in ('external_spatial.py','external_arithmetic.py','scalar_outcomes.py','prepare_scalar_baselines.py','scalar_baselines.json','check_external_spatial.py')};sources[str(Path(__file__))]=core.sha(__file__)
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append({'name':name,'passed':True})
    def reject(name,call):
        try:call()
        except (ValueError,KeyError,AssertionError,StopIteration):check(name,True)
        else:raise AssertionError('Unexpected acceptance: '+name)
    try:
        frames=[{'id':f'synthetic_{i:03d}','acquisition_date':'synthetic_date'} for i in range(26)]
        inp={'records':frames,'source_manifest':{'sha256':'d'*64}}
        entry={'run_id':'synthetic_hard','variant':'hard_seg','seed':17,'best_epoch':1,'checkpoint_sha256':'a'*64,'config_sha256':'b'*64}
        meta={'run_id':entry['run_id'],'variant':entry['variant'],'seed':entry['seed'],'original_selected_epoch':entry['best_epoch'],'checkpoint_sha256':entry['checkpoint_sha256'],'config_sha256':entry['config_sha256'],'export_sha256':'c'*64,'source_manifest_sha256':'d'*64}
        ranks=[];choices=[]
        for R in (1,2,4,6):
            for frame in frames:
                for g in (4,8,16):
                    ids=[f"{frame['id']}:g{g:02d}:r{r:02d}:c{c:02d}" for r in range(g) for c in range(g)]
                    units=[{'id':i,'A':1/g**2,'P':.25/g**2} for i in ids]
                    key=f"synthetic_hard|R{R}|{frame['id']}|g{g:02d}"
                    common={**meta,'R':R,'frame_id':frame['id'],'grid_g':g,'acquisition_date':frame['acquisition_date'],'ranking_key':key}
                    ranks.append({**common,'units':units,'ordered_ids':ids});seen={}
                    for budget in a.budget_rows(g):
                        k=budget['k'];ck=f"{key}|{budget['budget_basis']}|{budget['parent_nominal_budget_code']}"
                        choices.append({**common,**budget,'selected_ids':ids[:k],'selection_key':f'{key}|k{k:03d}','choice_key':ck,'duplicate_selection_alias':seen.get(k),'selected_area_FAE':k/g**2,'selected_predicted_weed_FAE':.25*k/g**2});seen.setdefault(k,ck)
        check('full26_fourR_allgrids_case_validator',ext.validate_inventory(entry,inp,ranks,choices,'c'*64) and len(ranks)==312 and len(choices)==3120)
        for name,mut in [('duplicate_rank',lambda rr,cc:rr.__setitem__(-1,rr[0])),('missing_R6',lambda rr,cc:rr[-1].update(R=3)),('checkpoint_mismatch',lambda rr,cc:rr[0].update(checkpoint_sha256='0'*64)),('manifest_mismatch',lambda rr,cc:rr[0].update(source_manifest_sha256='0'*64)),('wrong_date',lambda rr,cc:rr[0].update(acquisition_date='wrong')),('wrong_ascii_tie',lambda rr,cc:rr[0]['ordered_ids'].reverse()),('duplicate_case',lambda rr,cc:cc.__setitem__(-1,cc[0])),('bad_budget_integer',lambda rr,cc:cc[0].update(k=2)),('wrong_prefix',lambda rr,cc:cc[0].update(selected_ids=[])),('choice_model_mismatch',lambda rr,cc:cc[0].update(seed=29))]:
            rr,cc=copy.deepcopy(ranks),copy.deepcopy(choices);mut(rr,cc)
            reject('reject_'+name,lambda rr=rr,cc=cc:ext.validate_inventory(entry,inp,rr,cc,'c'*64))
        # Execute the exact new scalar CSV identity block with synthetic rows; no full evaluator,
        # fake reserved caches, model loading or downstream files are necessary for this guard.
        scalar_source=(HOME/'scalar_outcomes.py').read_text(encoding='utf-8-sig');tree=ast.parse(scalar_source)
        evaluation=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        loops=[n for n in ast.walk(evaluation) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='row' and isinstance(n.iter,ast.Name) and n.iter.id=='rows']
        check('unique_exact_scalar_identity_guard',len(loops)==1)
        module=ast.fix_missing_locations(ast.Module(body=[loops[0]],type_ignores=[]));code=compile(module,'<exact_root_scalar_identity_guard>','exec')
        model={'run_id':'synthetic','dataset':'local','encoder':'resnet18','variant':'hard_seg','seed':17,'best_epoch':3}
        row={k:str(v) for k,v in model.items()}
        exec(code,{'require':core.require,'rows':[row],'model':model});check('exact_scalar_identity_guard_accepts_match',True)
        for key in model:
            bad={**row,key:('99' if key in ('seed','best_epoch') else 'wrong')}
            reject('scalar_identity_rejects_'+key,lambda bad=bad:exec(code,{'require':core.require,'rows':[bad],'model':model}))
        for filename,text in [('external_spatial.py','Truth dataset/source differs'),('scalar_outcomes.py','Truth source membership differs')]:
            source=(HOME/filename).read_text(encoding='utf-8-sig')
            guards=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='require' and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value==text]
            check(filename+'_explicit_truth_source_guard',len(guards)==1)
            code2=compile(ast.fix_missing_locations(ast.Module(body=guards,type_ignores=[])),'<exact_truth_guard>','exec')
            env={'require':core.require,'tm':{'dataset':'weeds','source_manifest_sha256':'d'*64},'inp':{'source_manifest':{'sha256':'d'*64}}}
            exec(code2,env)
            env['tm']['source_manifest_sha256']='e'*64;reject(filename+'_rejects_wrong_truth_manifest',lambda:exec(code2,env))
        unchanged=all(core.sha(p)==h for p,h in sources.items());check('final_sources_unchanged_during_checks',unchanged)
        check('no_real_freeze_or_model_created',not(HOME/'FREEZE.json').exists() and 'torch' not in sys.modules)
        (out/'source').mkdir()
        for p in sources:shutil.copyfile(p,out/'source'/Path(p).name)
        report={'passed':True,'check_count':len(checks),'checks':checks,'source_sha256':sources,'prior_independent_arithmetic_review':core.binding(original_review),'prior_arithmetic_checks':64,'all_previous_guard_suggestions_addressed':True,'test_manifest_or_arrays_read':False,'actual_inference_or_R6':False,'GPU_or_solver':False,'formal_confirmation_authorized':False,'scope':'Final static/synthetic consumer integrity review; accepted preparatory implementation, actual future outputs require separate audit.'}
        core.write(out/'review.json',report);core.write(out/'artifact_manifest.json',core.stage_manifest(out));print(json.dumps({'passed':True,'checks':len(checks),'path':str(out/'review.json'),'sha256':core.sha(out/'review.json')}))
    except BaseException as error:
        core.write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'checks':checks,'sources':sources});raise

if __name__=='__main__':main()
