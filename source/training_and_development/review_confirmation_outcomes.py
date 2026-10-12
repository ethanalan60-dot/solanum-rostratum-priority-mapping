"""Independent pure arithmetic/identity preparation review; no reserved files or models."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,math,os,shutil,statistics,sys,traceback
ROOT=Path(__file__).resolve().parent;HOME=ROOT/'confirmation_v1'
sys.path.insert(0,str(HOME))
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import core

def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=ROOT/'provenance'/('confirmation_outcomes_independent_'+stamp);out.mkdir()
    checks=[];files=['external_spatial.py','external_arithmetic.py','check_external_spatial.py','scalar_outcomes.py','prepare_scalar_baselines.py','scalar_baselines.json']
    sources={str(HOME/n):core.sha(HOME/n) for n in files};sources[str(Path(__file__))]=core.sha(__file__)
    (out/'source').mkdir()
    for p in sources:shutil.copyfile(p,out/'source'/Path(p).name)
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append({'name':name,'passed':True})
    def reject(name,call):
        try:call()
        except (ValueError,AssertionError,FileNotFoundError,KeyError):check(name,True)
        else:raise AssertionError('Unexpected acceptance: '+name)
    try:
        import external_spatial as ext,external_arithmetic as a,scalar_outcomes as s
        import numpy as np
        check('no_real_freeze_or_model',not(HOME/'FREEZE.json').exists() and 'torch' not in sys.modules)
        for name,fn in [('external_select',lambda:ext.select_run('synthetic',out/'absent')),('external_evaluate',lambda:ext.evaluate_all(out/'absent',out/'absent')),('scalar_evaluate',lambda:s.evaluate(out/'absent',out/'absent'))]:reject(name+'_requires_freeze_before_input',fn)
        extract=core.read(HOME/'external_arithmetic_extraction.json');core.bound_file(extract['original_path'],extract['original_sha256'])
        original=Path(extract['original_path']).read_text(encoding='utf-8-sig');copy_source=(HOME/'external_arithmetic.py').read_text(encoding='utf-8-sig')
        def functions(text):return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        original_functions=functions(original);new_functions=functions(copy_source)
        for name,digest in extract['function_bodies'].items():check('reviewed_arithmetic_exact_AST_'+name,new_functions[name]==original_functions[name] and hashlib.sha256(new_functions[name].encode()).hexdigest()==digest)
        constants=core.read(HOME/'scalar_baselines.json');baselines={}
        for dataset,folder,count,pinned in [('local','local_data_v2',985,'815098fa5c5378c6163d13a11a8996b8a89ece3d8f209b55bdcd3e949f93cd96'),('weeds','weedsgalore_data_v1',104,'02fefe6e1097a141d38f996904476eb13ad2dbd2b1b402341655463f3e175ed7')]:
            path=ROOT/folder/'train.json';core.bound_file(path,pinned);rows=core.read(path)
            check(dataset+'_train_population',len(rows)==count and all(x['split']=='train' for x in rows))
            values=[float(np.float32(x['coverage'])) if dataset=='weeds' else float(x['coverage']) for x in rows]
            check(dataset+'_train_range',all(math.isfinite(x) and 0<=x<=1 for x in values))
            independent=math.fsum(values)/count;declared=constants['datasets'][dataset]
            check(dataset+'_constant_original_precision',abs(independent-declared['train_mean'])<1e-16 and declared['zero']==0 and declared['n_train']==count and declared['training_manifest_sha256']==pinned)
            baselines[dataset]={'independent_fsum_mean':independent,'declared_mean':declared['train_mean'],'absolute_difference':abs(independent-declared['train_mean']),'n_train':count,'training_manifest_sha256':pinned}
        mixed=s.summarize([.1,.3,.2],[0,.2,.4]);expected_errors=[.1,.1,-.2]
        check('scalar_mae_and_bias',abs(mixed['mae']-sum(abs(x) for x in expected_errors)/3)<1e-15 and abs(mixed['bias'])<1e-15)
        check('scalar_stratification',mixed['n_zero']==1 and mixed['n_positive']==2 and abs(mixed['balanced_mae']-.125)<1e-15)
        positive=s.summarize([.1,.3],[.2,.2]);zero=s.summarize([.1,.3],[0,0])
        check('missing_zero_is_NA',positive['balanced_mae'] is None and positive['mae_zero'] is None and positive['balanced_mae_na_reason']=='missing_zero_stratum')
        check('missing_positive_is_NA',zero['balanced_mae'] is None and zero['mae_positive'] is None and zero['balanced_mae_na_reason']=='missing_positive_stratum')
        for label,p,t in [('nan',[float('nan')],[0]),('range',[1.1],[0]),('mismatch',[.1],[0,.2]),('empty',[],[])]:reject('scalar_reject_'+label,lambda p=p,t=t:s.summarize(p,t))
        contract=core.read(HOME/'contract.json');rows=[]
        for model in contract['roster']:
            for R in (1,2,4,6):
                q=.01+.0001*model['seed']+.001*R+(.002 if model['variant']=='soft_seg' else 0)
                rows.append({**model,'R':R,**s.summarize([q,.2+q],[0,.2])})
        pair=s.paired(rows);groups=s.seed_summaries(rows)
        check('all148_scales_60_pairs_60_seed_groups',len(rows)==148 and len(pair)==60 and len(groups)==60)
        check('all_hard_soft_pp_signs',all(abs(x['soft_minus_hard_mae_pp']-.2)<1e-12 for x in pair))
        check('seed_group_sizes_local_regressors_not_replicated',sum(x['n_models']==1 for x in groups)==16 and sum(x['n_models']==3 for x in groups)==44)
        expected_sd=statistics.stdev([.0001*s0 for s0 in (17,29,43)])
        check('descriptive_seed_sample_SD_ddof1',all(x['mae_sample_sd'] is None if x['n_models']==1 else abs(x['mae_sample_sd']-expected_sd)<1e-14 for x in groups))
        seg=next(x for x in rows if x['variant']=='hard_seg');reject('duplicate_pair_rejected',lambda:s.paired(rows+[seg]))
        missing=next(x for x in rows if x['variant']=='soft_seg');reject('missing_pair_rejected',lambda:s.paired([x for x in rows if x is not missing]))
        rng=np.random.default_rng(6181);w=rng.random((256,256))*.4;c=rng.random((256,256))*.3;b=1-w-c;p=rng.random((256,256));truth={};frame={'id':'synthetic','acquisition_date':'one_date'}
        meta={'run_id':'synthetic_hard','variant':'hard_seg','seed':17,'original_selected_epoch':1,'checkpoint_sha256':'a'*64,'config_sha256':'b'*64,'export_sha256':'c'*64,'source_manifest_sha256':'d'*64}
        max_cell=0.;unique=0
        nominal={4:[1,3,4,8,16],8:[6,12,19,32,64],16:[25,51,76,128,256]};matched={4:[1,3,4,8,16],8:[4,12,16,32,64],16:[16,48,64,128,256]}
        for g in (4,8,16):
            budgets=a.budget_rows(g);check('exact_integer_budget_g'+str(g),[x['k'] for x in budgets[:5]]==nominal[g] and [x['k'] for x in budgets[5:]]==matched[g])
            arrays=[a.aggregate_plane(x,frame['id'],g) for x in (b,c,w)]
            truth[(frame['id'],g)]={i:{'A':1/g**2,'B':arrays[0][i],'C':arrays[1][i],'W':arrays[2][i]} for i in arrays[0]}
            side=256//g;independent=w.reshape(g,side,g,side).sum((1,3),dtype=np.float64)/65536
            err=max(abs(v-independent[r,c0]) for (r,c0),v in zip(np.ndindex(g,g),arrays[2].values()));max_cell=max(err,max_cell)
            check('independent_reshape_aggregation_g'+str(g),err<1e-14)
            rank,choices=a.ranking_record(meta,frame,6,g,p)
            curve=a.make_frame_curve(rank,truth);curve_fixed=a.fixed_rows(curve)
            check('saved_prefix_choice_equivalence_g'+str(g),all(x['choice_key']==y['choice_key'] and x['selected_area_FAE']==y['selected_area_FAE'] and x['selected_predicted_weed_FAE']==y['selected_predicted_weed_FAE'] for x,y in zip(choices,curve_fixed)))
            unique+=len({x['selection_key'] for x in choices})
            vals=[x['weed_capture_ratio'] for x in curve];check('full_step_auc_g'+str(g),abs(a.step_auc(curve,1)-math.fsum(vals[:-1])/g**2)<1e-14)
            check('half_step_auc_g'+str(g),abs(a.step_auc(curve,.5)-math.fsum(vals[:g**2//2])/(g**2//2))<1e-14)
            oracle=ext.reference_rankings([frame],truth,g,'label_oracle','d'*64)[0];oc=a.make_frame_curve(oracle,truth,'label_oracle')
            check('oracle_wholeunit_optimal_g'+str(g),all(o['selected_weed_FAE']+1e-14>=x['selected_weed_FAE'] for o,x in zip(oc,curve)))
            rr=ext.reference_rankings([frame],truth,g,'random_expectation','d'*64)[0];rc=a.make_frame_curve(rr,truth,'random_expectation',True)
            check('random_k_over_N_g'+str(g),all(abs(x['weed_capture_ratio']-x['k']/g**2)<1e-14 for x in rc))
            check('reference_R_is_None_g'+str(g),rr['R'] is None and oracle['R'] is None)
        nested=a.check_nested_truth_cells(truth,['synthetic']);check('all240_nested_BCW_checks',nested['scalar_comparisons']==240 and nested['max_absolute_residual_FAE']<1e-12)
        check('full_confirmation_case_arithmetic',26*4*3==312 and 26*4*sum(g*g for g in (4,8,16))==34944 and 26*4*unique==2080 and 6*26*4*sum(g*g+1 for g in (4,8,16))==211536)
        check('no_model_module_or_real_freeze_after_checks','torch' not in sys.modules and not(HOME/'FREEZE.json').exists())
        source_changes=[p for p,h in sources.items() if core.sha(p)!=h]
        report={'passed':True,'scope':'Independent arithmetic/metadata preparation review; no actual confirmation inference or outcomes','checks':checks,'check_count':len(checks),'source_sha256_at_start':sources,'source_changes_during_check':source_changes,'training_baselines':baselines,'max_independent_cell_error_FAE':max_cell,'reserved_manifest_or_arrays_read':False,'real_forward_or_R6_forward':False,'solver':False,'GPU':False,'consumer_guard_suggestions_reported_to_root':['Scalar CSV row model/config identity should be explicit','External ranking/choice unique case keys and model bindings should be explicit','Truth manifest dataset/source/input bindings should be explicit'],'note':'These are consumer integrity guard suggestions; reviewed producer loops and numerical formulas are correct. Final acceptance must bind any subsequent guard revision separately.'}
        core.write(out/'review.json',report);core.write(out/'artifact_manifest.json',core.stage_manifest(out));print(json.dumps({'passed':True,'checks':len(checks),'output':str(out),'sha256':core.sha(out/'review.json')}))
    except BaseException as error:
        core.write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'checks':checks,'source_sha256_at_start':sources});raise

if __name__=='__main__':main()
