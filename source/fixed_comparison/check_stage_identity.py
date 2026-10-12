"""Standard-library-only supplemental preparation checks; no real data access."""
from pathlib import Path
import ast, datetime, json, sys
import core
import local_management as m
import local_outcomes as q

root = Path(__file__).resolve().parent.parent
out = root/'provenance'/datetime.datetime.now().strftime('confirmation_identity_checks_%Y%m%d_%H%M%S_%f')
out.mkdir(); checks=[]
def check(name, condition=True):
    assert condition, name
    checks.append({'name':name,'passed':True})
def rejects(name, fn):
    try: fn()
    except (ValueError, FileNotFoundError): check(name); return
    raise AssertionError(name)
def funcs(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}

old = root/'provenance/confirmation_identity_preexecution_20260916_103008_984028'
for name, allowed in [('local_management.py',{'execute_run','bind_stage_identity'}),('local_outcomes.py',{'execute'})]:
    a,b=funcs(old/name),funcs(root/'confirmation_v1'/name)
    changed={key for key in set(a)|set(b) if a.get(key)!=b.get(key)}
    check(name+'_only_provenance_functions_changed',changed==allowed)
check('imports_no_numpy_torch_scipy',not any(x in sys.modules for x in ('numpy','torch','scipy')))
toy=out/'toy';toy.mkdir()
for name in ('metadata.json','artifact_manifest.json','area_units.csv'):(toy/name).write_text('ORIGINAL',encoding='utf-8')
watch={};m.bind_stage_identity(watch,toy,('metadata.json','artifact_manifest.json','area_units.csv'))
check('complete_initial_identity_inventory',set(watch)=={str((toy/n).resolve()) for n in ('metadata.json','artifact_manifest.json','area_units.csv')})
m.verify_execution_hashes(watch);check('unchanged_stage_passes')
for name in ('metadata.json','artifact_manifest.json','area_units.csv'):
    (toy/name).write_text('NEW_SELF_CONSISTENT_STAGE',encoding='utf-8')
    rejects(name+'_replacement_detected',lambda:m.verify_execution_hashes(watch))
    (toy/name).write_text('ORIGINAL',encoding='utf-8')
rejects('missing_stage_file_refused',lambda:m.bind_stage_identity({},toy,('missing.json',)))
rejects('path_escape_refused',lambda:m.bind_stage_identity({},toy,('../outside.json',)))
rejects('conflicting_initial_identity_refused',lambda:m.bind_stage_identity({str((toy/'metadata.json').resolve()):'0'*64},toy,('metadata.json',)))
text=(root/'confirmation_v1/local_management.py').read_text(encoding='utf-8')
check('selection_watches_geometry_and_prediction',"bind_stage_identity(watched, HERE/'stages'/'local_geometry'" in text and "bind_stage_identity(watched, Path(predicted['stage'])" in text)
text=(root/'confirmation_v1/local_outcomes.py').read_text(encoding='utf-8')
check('outcomes_all_four_stage_identities_watched',all("('"+name+"', (" in text for name in ('truth_local','local_selections_seal','local_geometry','predictions_seal')) and "management.bind_stage_identity(watched, HERE/'stages'/stage, names)" in text)
report={'passed':True,'check_count':len(checks),'checks':checks,'implementation_sha256':{'local_management.py':core.sha(m.__file__),'local_outcomes.py':core.sha(q.__file__)},'checker_sha256':core.sha(__file__),
        'previous_full_scientific_readiness':{'management43':core.binding(root/'provenance/confirmation_management_checks_20260916_101012_515708/focused_checks.json'),'outcomes35':core.binding(root/'provenance/confirmation_outcomes_checks_20260916_102007_531527/focused_checks.json')},
        'scientific_functions_AST_unchanged':True,'real_dataset_files_opened':0,'model_forward_calls':0,'solver_calls':0,'scope':'identity-only guard supplement; previous full synthetic receipts retained; no production execution'}
core.write(out/'focused_checks.json',report)
print(json.dumps({'directory':str(out),'passed':True,'checks':len(checks),'sha256':core.sha(out/'focused_checks.json')}))
