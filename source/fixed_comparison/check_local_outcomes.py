"""Artificial native labels + mocked MILP end-to-end; no real reserved data."""
from pathlib import Path
from unittest.mock import patch
import argparse,datetime,itertools,json,math,sys,traceback
import core
import local_management as m
import local_outcomes as q
OUT=q.ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_outcomes_checks_%Y%m%d_%H%M%S_%f');OUT.mkdir();checks=[]
parser=argparse.ArgumentParser();parser.add_argument('--reuse-synthetic-population');args=parser.parse_args()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);core.write(p,v)
def check(name,v=True):
 if not v:raise AssertionError(name)
 checks.append({'name':name,'passed':True})
def rejects(name,f):
 try:f()
 except (AssertionError,ValueError,FileNotFoundError,FileExistsError,RuntimeError,OSError):check(name);return
 raise AssertionError('Expected refusal: '+name)
try:
 check('stdlib_import_before_gate','numpy' not in sys.modules and 'torch' not in sys.modules)
 rejects('missing_freeze_blocks_truth_and_solver',lambda:q.execute(OUT/'absent_review',OUT/'absent_choices'))
 check('no_scientific_dependency_before_denied_gate','numpy' not in sys.modules)
 d=q.dependencies();np=d.np;from scipy.optimize import OptimizeResult
 check('resource_and_bound_only_label_AST_adapter',d.label_ledger['explicit_AST_adapters']==['label_solver_threads_2_to_1','large_raw_upper_inversion_uses_totalT_pending_uniform_downgrade'])
 check('bound_module_never_imported','audit_common_bound_consistency' not in sys.modules)
 home=OUT/'h';home.mkdir();geo=home/'stages/local_geometry';(geo/'tiles').mkdir(parents=True)
 native_raw=np.zeros(262144);native_raw[:2]=1.;native_weight=d.sparse.csr_matrix(([1.,1.],([0,1],[0,1])),shape=(2,262144));gpath=geo/'tiles/toy.npz'
 np.savez_compressed(gpath,cells=np.array([[0,0],[1,0]]),area=np.array([1.,1.]),native_weights=native_weight.data,native_indices=native_weight.indices,native_indptr=native_weight.indptr,native_shape=np.array(native_weight.shape),raw_valid_by_native=native_raw)
 from PIL import Image
 artificial=np.zeros((512,512),np.uint8);artificial[0,0]=1;artificial[3,3]=255
 mask=home/'artificial.png';Image.fromarray(artificial).save(mask)
 source={'source_id':'ARTIFICIAL','mask':str(mask),'mask_sha256':core.sha(mask)}
 window={'id':'toy','source_id':'ARTIFICIAL','x':0,'y':0,'file':'tiles/toy.npz','sha256':core.sha(gpath)}
 geometry={'windows':[window],'source_headers':[{'source_id':'ARTIFICIAL','width':512,'height':512}],'area_m2':2.}
 truth={'dataset':'local','records':[{'id':'toy','native_reference':{'source_id':'ARTIFICIAL','mask_source':str(mask),'native_window_xywh':[0,0,512,512]}}]}
 area=[{'M':1,'id':'M1_E0_N0','ix':0,'iy':0,'area_m2':1.},{'M':1,'id':'M1_E1_N0','ix':1,'iy':0,'area_m2':1.}]+[{'M':i,'id':f'M{i}_E0_N0','ix':0,'iy':0,'area_m2':2.} for i in (2,4)]
 core.table(geo/'area_units.csv',area);summary={'sources':[source]}
 unit,audit=q.join_native_truth(truth,geometry,geo,summary,d,lambda s,h:artificial)
 check('native_grayscale_positive_one_and_alpha_intersection',audit['native_target_pixels']==1 and audit['native_valid_pixels']==2 and audit['target_m2']==1.)
 base=[x['target_m2'] for x in unit if x['M']==1];check('native_CSR_preserves_within2x2_boundary',base==[1.,0.] and base!=[.5,.5])
 check('nested_M_target_totals_equal',all(math.fsum(x['target_m2'] for x in unit if x['M']==i)==1 for i in (1,2,4)))
 check('domain_is_artificial_not_val_total',audit['area_m2']==2 and audit['validation_T_reused'] is False)
 valid=np.zeros((512,512),bool);valid[0,0]=True
 check('native_edge_padding_false',q.native_target(np.ones((1,1),np.uint8),0,0,valid,np).sum()==1)
 rejects('RGB_mask_refused',lambda:q.native_target(np.zeros((512,512,3),np.uint8),0,0,valid,np))
 altered={'dataset':'local','records':[{'id':'toy','native_reference':dict(truth['records'][0]['native_reference'],mask_source=str(home/'wrong.png'))}]}
 rejects('mask_path_not_original_summary_rejected',lambda:q.join_native_truth(altered,geometry,geo,summary,d,lambda s,h:artificial))
 rejects('mask_source_dimensions_rejected',lambda:q.join_native_truth(truth,geometry,geo,summary,d,lambda s,h:np.zeros((4,4),np.uint8)))
 calls=[]
 def backend(c,**kw):
  cost=np.asarray(kw['constraints'].A)[0];b=float(kw['constraints'].ub[0]);n=len(c)
  candidates=[np.asarray(z,float) for z in itertools.product((0,1),repeat=n) if math.fsum(cost*np.asarray(z))<=b]
  x=min(candidates,key=lambda z:float(c@z));fun=float(c@x);calls.append(dict(kw['options']))
  return OptimizeResult(x=x,fun=fun,mip_dual_bound=fun,mip_gap=0.,mip_node_count=0,status=0,message='MOCK BACKEND ONLY')
 # Full24 synthetic selection stages; actual solver is patched before any selection.
 population={'models':[]};allowed=[{'M':x['M'],'id':x['id'],'area':x['area_m2'],**{f'predicted_R{r}':(.1 if x['M']==1 and x['ix']==0 else .9 if x['M']==1 else 1.) for r in m.RS}} for x in area]
 if args.reuse_synthetic_population:
  reused=Path(args.reuse_synthetic_population).resolve()
  assert reused.is_relative_to(q.ROOT/'provenance') and reused.parent.name=='local_selections_seal'
  population=core.read(reused);assert [x['run_id'] for x in population['models']]==[f'SYNTHETIC_{i:02d}' for i in range(1,25)]
  for item in population['models']:
   folder=Path(item['selection_seal']['path']).parent;core.completed_stage(folder);m.verify_model_selection(folder,d)
  check('prior1440_mock_selection_cases_reused_unchanged',len(population['models'])==24 and len(calls)==0)
 else:
  with patch.object(d.policy,'milp',side_effect=backend):
   for index in range(1,25):
    folder=home/'stages'/f'local_select_{index:02d}';folder.mkdir();rid=f'SYNTHETIC_{index:02d}';context={k:'0'*64 for k in ('freeze_sha256','all_predictions_sha256','geometry_metadata_sha256','checkpoint_sha256')};context['run_id']=rid
    m.select_arrays(folder,allowed,2.,context,d);write(folder/'status.json',{'state':'completed'});write(folder/'artifact_manifest.json',core.stage_manifest(folder))
    population['models'].append({'run_id':rid,'case_count':120,'selection_seal':core.binding(folder/'selection_seal.json'),'artifact_manifest':core.binding(folder/'artifact_manifest.json')})
  check('full24_synthetic_selection_population_1440_mock_cases',len(calls)==1440)
 summary_path=home/'source_summary.json';write(summary_path,summary)
 contract={'datasets':{'local':{'source_summary_path':str(summary_path),'source_summary_sha256':core.sha(summary_path)}}}
 for file in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):write(home/file,{'synthetic_only':True})
 write(home/'stages/truth_local/truth_manifest.json',truth);write(home/'stages/local_selections_seal/all_local_selections.json',population);write(geo/'artifact_manifest.json',{'synthetic_only':True})
 for stage in ('truth_local','local_selections_seal','predictions_seal'):write(home/'stages'/stage/'artifact_manifest.json',{'synthetic_only':True})
 write(geo/'metadata.json',geometry);write(home/'stages/predictions_seal/all_predictions.json',{'synthetic_only':True})
 root_review=home/'root.json';choice_review=home/'choice.json';write(root_review,{'synthetic_only':True});write(choice_review,{'synthetic_only':True})
 original_attempt=core.attempt
 def attempt(name):return original_attempt(name,home)
 before=len(calls)
 with patch.object(q,'HERE',home),patch.object(q,'gate',return_value=(contract,population,truth)),patch.object(core,'attempt',side_effect=attempt),patch.object(q,'dependencies',return_value=d),patch.object(m,'validate_geometry',return_value=geometry),patch.object(core,'validate_source_lock',return_value={}),patch.object(m,'truth_gate',return_value=population),patch.object(core,'completed_stage',return_value={} ),patch.dict(d.label_namespace,{'milp':backend}):
  result=q.execute(root_review,choice_review)
  check('full_executor_end_to_end_completed',core.read(result/'status.json')['state']=='completed')
  rejects('outcome_permanent_claim_refuses_repeat',lambda:q.execute(root_review,choice_review))
 check('shared_reference_exact15_mock_calls_for24_models',len(calls)-before==15)
 check('label_resource_threads1_time30_gap1e8',all(x['threads']==1 and x['time_limit']==30 and x['mip_rel_gap']==1e-8 for x in calls[-15:]))
 accepted=core.read(result/'summary.json');check('all24_full_matrix_and_AUC',accepted['models']['fixed_rows']==2880 and accepted['models']['auc_rows']==288 and accepted['models']['models']==24)
 core.completed_stage(result);check('final_complete_artifact_manifest_verified')
 fixed=m.read_rows(result/'models/all_fixed_budgets.csv');check('adverse_model_capture_retained',any(x['policy']=='model_prefix' and float(x['budget'])==.5 and int(x['M'])==1 and float(x['capture_fraction'])==0 for x in fixed))
 for ms in (1,2,4):
  path=result/'shared_reference'/f'random_full_curves_M{ms}.npz'
  with np.load(path,allow_pickle=False) as saved:
   n=sum(x['M']==ms for x in unit);check(f'M{ms}_all1000_full_steps',saved['area_fraction'].shape==(1000,n+1) and saved['order_indices'].shape==(1000,n))
   rng=np.random.Generator(np.random.PCG64(q.RANDOM_SEED));check(f'M{ms}_fixed_PCG64_first_order',np.array_equal(saved['order_indices'][0],rng.permutation(n)))
 # Explicit raw-bound inversion: safe totalT bound, IDs unchanged, no retry.
 target=np.array([.7,.3]);areas=np.ones(2);ids=['a','b'];prefix=q.strict_order_budget(areas,target,np.arange(2),.5,d)
 abnormal=OUT/'abnormal';abnormal.mkdir();extra=[]
 def inverted(c,**kw):
  extra.append(1);return OptimizeResult(x=np.array([1.,0.]),fun=-.7,mip_dual_bound=-.1,mip_gap=0.,status=0,message='MOCK INVERTED BOUND')
 with patch.dict(d.label_namespace,{'milp':inverted}):r=q.label_case(abnormal,1,ids,areas,target,.5,prefix,d)
 check('raw_bound_inversion_safe_totalT_not_proven',r['bound_anomaly'] and r['updated_upper_m2']==1 and not r['updated_proven_flag'])
 check('bound_recovery_does_not_reselect_or_retry',r['selected_ids']==['a'] and len(extra)==1 and (abnormal/'M1_B050_solver_raw.json').exists())
 zero=OUT/'zero';zero_unit=[{**x,'target_m2':0.} for x in unit]
 with patch.dict(d.label_namespace,{'milp':lambda *a,**k:(_ for _ in ()).throw(AssertionError('ZERO T MUST NOT SOLVE'))}):z=q.shared_references(zero_unit,zero,d)
 check('zeroT15cases_no_solver',z['unique_label_cases']==15 and z['actual_label_solver_calls']==0)
 check('zeroT_label_ratio_gap_missing',all(x['capture_fraction'] is None and x['updated_relative_gap'] is None for x in core.read(zero/'label_MILP_budgets.json')))
 check('zeroT_random_ratio_AUC_missing',all(x['capture_fraction_mean'] is None for x in core.read(zero/'random_summary.json')) and all(x['auc_0_1'] is None for x in core.read(zero/'random_auc.json')))
 # The full production gate is not patched here: incomplete freeze fails before PIL.
 rejects('unpatched_production_gate_still_refuses',lambda:q.gate(root_review,choice_review))
 check('real_confirmation_claim_not_created',not (q.HERE/'claims/local_outcomes.json').exists())
 check('no_model_module_imported','torch' not in sys.modules)
 report={'passed':True,'implementation_sha256':core.sha(q.__file__),'management_sha256':core.sha(m.__file__),'checker_sha256':core.sha(__file__),'checks':checks,'check_count':len(checks),'mock_selection_backend_returns_this_attempt':0 if args.reuse_synthetic_population else 1440,'reused_synthetic_selection_population':core.binding(args.reuse_synthetic_population) if args.reuse_synthetic_population else None,'mock_label_backend_returns':16,'real_MILP_calls':0,'real_mask_or_test_array_files_opened':0,'artificial_PNG_used':str(mask),'real_R6_forward_calls':0,'model_forward_calls':0,'source_lock_changed':False,'FREEZE_created':False,'label_adapter':d.label_ledger}
 write(OUT/'focused_checks.json',report);print(json.dumps({'passed':True,'directory':str(OUT),'checks':len(checks),'sha256':core.sha(OUT/'focused_checks.json')}))
except BaseException as e:
 write(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'checks':checks});raise
