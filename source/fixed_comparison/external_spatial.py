"""Confirmation equal-area allocation. Frozen stages keep reference data after all choices."""
from pathlib import Path
import argparse,csv,json,math
from core import HERE,R_VALUES,attempt,binding,bound_file,completed_stage,configure_cpu,read,require,sha,table,validate_freeze,validate_source_lock,write
GRIDS=(4,8,16)

def lines(path,rows):
    with Path(path).open('x',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def read_lines(path):
    with Path(path).open(encoding='utf-8') as stream:return [json.loads(x) for x in stream]
def code_review(path,scope):
    review=read(path)
    require(review.get('passed') is True and review['scope']==scope,'Wrong spatial execution grant')
    require(review['implementation_sha256']==sha(__file__) and review['freeze_sha256']==sha(HERE/'FREEZE.json'),'Spatial grant changed')
    require(review['arithmetic_sha256']==sha(HERE/'external_arithmetic.py'),'Arithmetic changed')
    return review

def selection_paths(run_id):
    out=HERE/'stages'/('select_'+run_id)
    return out,out/'selection_seal.json'

def validate_inventory(entry,inp,rankings,choices,export_sha256):
    import external_arithmetic as arithmetic
    frames={x['id']:x for x in inp['records']}
    expected={(r,f,g) for r in R_VALUES for f in frames for g in GRIDS}
    keys=[(x['R'],x['frame_id'],x['grid_g']) for x in rankings]
    require(len(keys)==len(expected) and set(keys)==expected,'Duplicate or missing complete ranking')
    metadata={'run_id':entry['run_id'],'variant':entry['variant'],'seed':entry['seed'],'original_selected_epoch':entry['best_epoch'],
              'checkpoint_sha256':entry['checkpoint_sha256'],'config_sha256':entry['config_sha256'],
              'export_sha256':export_sha256,'source_manifest_sha256':inp['source_manifest']['sha256']}
    by_key={}
    for row in rankings:
        for name,value in metadata.items():require(row[name]==value,'Ranking identity differs: '+name)
        g=row['grid_g'];frame=row['frame_id'];units=row['units']
        require(row['acquisition_date']==frames[frame]['acquisition_date'],'Ranking date differs')
        ids={f'{frame}:g{g:02d}:r{r:02d}:c{c:02d}' for r in range(g) for c in range(g)}
        require(len(units)==g*g and {x['id'] for x in units}==ids,'Incomplete grid cells')
        require(row['ordered_ids']==[x['id'] for x in arithmetic.rank_units(units)],'Ranking order differs')
        key=f"{entry['run_id']}|R{row['R']}|{frame}|g{g:02d}"
        require(row['ranking_key']==key,'Ranking key differs');by_key[key]=row
    case_keys=[]
    for row in choices:
        for name,value in metadata.items():require(row[name]==value,'Choice identity differs: '+name)
        key=(row['R'],row['frame_id'],row['grid_g'],row['budget_basis'],row['parent_nominal_budget_code'])
        case_keys.append(key)
        rank=by_key[row['ranking_key']]
        require((rank['R'],rank['frame_id'],rank['grid_g'])==key[:3],'Choice ranking identity differs')
        require(row['acquisition_date']==rank['acquisition_date'],'Choice date differs')
        budget=next(x for x in arithmetic.budget_rows(row['grid_g']) if x['budget_basis']==key[3] and x['parent_nominal_budget_code']==key[4])
        for name,value in budget.items():require(row[name]==value,'Choice budget differs: '+name)
        require(row['selected_ids']==rank['ordered_ids'][:budget['k']],'Choice prefix differs')
        require(row['selection_key']==rank['ranking_key']+f"|k{budget['k']:03d}",'Canonical selection key differs')
    expected_cases={(r,f,g,b['budget_basis'],b['parent_nominal_budget_code']) for r,f,g in expected for b in arithmetic.budget_rows(g)}
    require(len(case_keys)==len(expected_cases) and set(case_keys)==expected_cases,'Duplicate or missing budget case')
    return True

def validate_selection(entry):
    from data_io import validate_input_manifest
    out,seal_path=selection_paths(entry['run_id']);completed_stage(out);seal=read(seal_path)
    require(seal['scope']=='confirmation_external_label_free_selection' and seal['run_id']==entry['run_id'],'Wrong selection identity')
    require(seal['freeze_sha256']==sha(HERE/'FREEZE.json') and seal['all_predictions_sha256']==sha(HERE/'stages/predictions_seal/all_predictions.json'),'Selection freeze/predictions differ')
    require(seal['R']==list(R_VALUES) and seal['grids']==list(GRIDS) and seal['case_count']==3120 and seal['rankings']==312,'Selection inventory differs')
    require(seal['truth_used_for_choices'] is False,'Reference data used in selection')
    for relative,digest in seal['files_sha256'].items():bound_file(out/relative,digest)
    inp=validate_input_manifest('weeds')
    prediction=read(HERE/'stages/predictions_seal/all_predictions.json')
    export=next(x for x in prediction['models'] if x['run_id']==entry['run_id'])
    validate_inventory(entry,inp,read_lines(out/'rankings.jsonl'),read_lines(out/'choices.jsonl'),export['artifact_manifest']['sha256'])
    return out,seal

def select_run(run_id,review_path):
    from data_io import validate_input_manifest
    from pipeline import validate_global_predictions
    contract,models,freeze=validate_freeze();code_review(review_path,'confirmation_external_selection_execution')
    review_digest=sha(review_path)
    global_seal=validate_global_predictions()
    entry=next((x for x in models if x['run_id']==run_id),None)
    require(entry is not None and entry['dataset']=='weeds' and entry['variant']!='regression','Wrong external model')
    inp=validate_input_manifest('weeds');frames=inp['records']
    require(len(frames)==26,'Wrong frame count')
    prediction=next(x for x in global_seal['models'] if x['run_id']==run_id)
    bound_file(prediction['map_index']['path'],prediction['map_index']['sha256'])
    maps=read(prediction['map_index']['path'])
    require([x['id'] for x in maps]==[x['id'] for x in frames],'Map membership/order changed')
    with attempt('select_'+run_id) as out:
        configure_cpu(out)
        import numpy as np
        import external_arithmetic as arithmetic
        meta={'run_id':run_id,'variant':entry['variant'],'seed':entry['seed'],'original_selected_epoch':entry['best_epoch'],
              'checkpoint_sha256':entry['checkpoint_sha256'],'config_sha256':entry['config_sha256'],
              'export_sha256':prediction['artifact_manifest']['sha256'],'source_manifest_sha256':inp['source_manifest']['sha256']}
        write(out/'input_binding.json',{'freeze':binding(HERE/'FREEZE.json'),'all_predictions':binding(HERE/'stages/predictions_seal/all_predictions.json'),
              'input_manifest':binding(HERE/'stages/inputs_weeds/input_manifest.json'),'map_index':prediction['map_index'],'root_review':binding(review_path)})
        rankings=[];choices=[];units=[]
        for frame,item in zip(frames,maps):
            bound_file(item['path'],item['sha256'])
            with np.load(item['path'],allow_pickle=False) as z:
                require(set(z.files)=={'probabilities','valid','R','id'},'Wrong prediction keys')
                p=z['probabilities'];v=z['valid']
                require(np.array_equal(z['R'],np.asarray(R_VALUES,dtype=np.int64)) and str(z['id'])==frame['id'],'Map R/ID mismatch')
            require(p.dtype==np.float32 and p.shape==(4,256,256) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid map')
            require(v.dtype==np.float32 and v.shape==(256,256) and np.all(v==1),'External support must be all valid')
            for ri,factor in enumerate(R_VALUES):
                for g in GRIDS:
                    ranking,selected=arithmetic.ranking_record(meta,frame,factor,g,p[ri])
                    rankings.append(ranking);choices.extend(selected)
                    units.extend({'run_id':run_id,'R':factor,'frame_id':frame['id'],'grid_g':g,**u} for u in ranking['units'])
        require(len(rankings)==312 and len(choices)==3120 and len(units)==34944,'Incomplete external selection')
        validate_inventory(entry,inp,rankings,choices,prediction['artifact_manifest']['sha256'])
        require(len({x['selection_key'] for x in choices})==2080,'Unexpected alias accounting')
        table(out/'prediction_units.csv',units);lines(out/'rankings.jsonl',rankings);lines(out/'choices.jsonl',choices)
        validate_source_lock()
        bound_file(review_path,review_digest);code_review(review_path,'confirmation_external_selection_execution')
        for item in maps:bound_file(item['path'],item['sha256'])
        files=['input_binding.json','prediction_units.csv','rankings.jsonl','choices.jsonl']
        write(out/'selection_seal.json',{'scope':'confirmation_external_label_free_selection','run_id':run_id,'dataset':'weeds',
              'freeze_sha256':sha(HERE/'FREEZE.json'),'all_predictions_sha256':sha(HERE/'stages/predictions_seal/all_predictions.json'),
              'R':list(R_VALUES),'grids':list(GRIDS),'case_count':3120,'rankings':312,'unique_selections':2080,'prediction_unit_rows':34944,
              'truth_used_for_choices':False,'files_sha256':{name:sha(out/name) for name in files},
              'unit':'frame-area equivalent; separate whole-unit budget in each frame'})
    return out

def truth_cells(frames,truth_manifest):
    import numpy as np
    import external_arithmetic as arithmetic
    require([x['id'] for x in frames]==[x['id'] for x in truth_manifest['records']],'Truth frame join differs')
    result={};checks=[]
    for frame,entry in zip(frames,truth_manifest['records']):
        bound_file(entry['truth_cache'],entry['truth_cache_sha256'])
        with np.load(entry['truth_cache'],allow_pickle=False) as z:
            require(set(z.files)=={'soft','soft3','native_class_counts','valid','coverage'},'Truth keys differ')
            soft=z['soft'];a=z['soft3'];counts=z['native_class_counts'];v=z['valid']
        require(a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'Truth classes invalid')
        require(np.array_equal(a[2],soft) and v.shape==(256,256) and np.all(v==1),'Pooled weed/valid differ')
        require(counts.dtype==np.uint32 and counts.shape==(3,) and sum(int(x) for x in counts)==360000,'Native class totals invalid')
        error=[abs(math.fsum(a[i].ravel())/65536-int(counts[i])/360000) for i in range(3)]
        require(max(error)<=1e-7 and np.max(abs(a.sum(0,dtype=np.float64)-1))<=1e-6,'Truth conservation invalid')
        checks.append({'id':frame['id'],'native_to_cache_class_error':error})
        reference=None
        for g in GRIDS:
            vals=[arithmetic.aggregate_plane(a[i],frame['id'],g) for i in range(3)]
            cells={identifier:{'A':1/(g*g),'B':vals[0][identifier],'C':vals[1][identifier],'W':vals[2][identifier]} for identifier in vals[0]}
            for cell in cells.values():require(abs(math.fsum(cell[k] for k in ('B','C','W'))-cell['A'])<=1e-6*cell['A'],'Unit class sum mismatch')
            totals={k:math.fsum(cells[i][k] for i in sorted(cells)) for k in ('B','C','W')}
            if reference is not None:require(max(abs(totals[k]-reference[k]) for k in totals)<=1e-12,'Grid totals differ')
            reference=totals;result[(frame['id'],g)]=cells
    nested=arithmetic.check_nested_truth_cells(result,[x['id'] for x in frames])
    return result,{'frames':checks,'nested':nested}

def auc_row(curve):
    import external_arithmetic as arithmetic
    fields=('run_id','variant','seed','original_selected_epoch','checkpoint_sha256','config_sha256','export_sha256',
            'source_manifest_sha256','R','grid_g','policy','scope','scope_id','frame_id','acquisition_date')
    result={key:curve[0][key] for key in fields}
    result.update(prefix_points=len(curve),frame_n=curve[0]['frame_n'],weed_eligible_frame_n=curve[0]['weed_eligible_frame_n'],
          total_weed_FAE=curve[0]['total_weed_FAE'],auc_0_1=arithmetic.step_auc(curve,1),
          auc_0_0p5_normalized=arithmetic.step_auc(curve,.5),auc_na_reason=curve[0]['weed_capture_na_reason'])
    if result['policy']=='random_expectation' and result['auc_0_1'] is not None:
        n=len(curve)-1
        require(abs(result['auc_0_1']-(n-1)/(2*n))<=1e-12 and abs(result['auc_0_0p5_normalized']-(.25-1/(2*n)))<=1e-12,'Random analytic AUC mismatch')
    return result

def reference_rankings(frames,truth,g,policy,manifest_sha):
    require(policy in ('label_oracle','random_expectation'),'Unknown reference')
    result=[]
    for frame in frames:
        cells=truth[(frame['id'],g)]
        ids=sorted(cells,key=lambda i:(-cells[i]['W'],i)) if policy=='label_oracle' else sorted(cells)
        result.append({'run_id':policy,'variant':'reference','seed':None,'original_selected_epoch':None,'checkpoint_sha256':'',
            'config_sha256':'','export_sha256':'','source_manifest_sha256':manifest_sha,'R':None,'grid_g':g,'frame_id':frame['id'],
            'acquisition_date':frame['acquisition_date'],'ranking_key':policy+'|'+frame['id']+'|g'+str(g),'ordered_ids':ids,
            'units':[{'id':i,'A':cells[i]['A'],'P':0.0} for i in sorted(cells)]})
    return result

def save_family(out,stem,frame_curves):
    import external_arithmetic as arithmetic
    groups={}
    for curve in frame_curves:
        row=curve[0];groups.setdefault((row['run_id'],row['R'],row['grid_g']),[]).append(curve)
    pooled=[];dates=[]
    for group in groups.values():
        pooled.append(arithmetic.group_curves(group,'pooled','all'+str(len(group))))
        for date in sorted({c[0]['acquisition_date'] for c in group}):
            dates.append(arithmetic.group_curves([c for c in group if c[0]['acquisition_date']==date],'date',date))
    result={}
    for scope,curves in [('frame',frame_curves),('date',dates),('pooled',pooled)]:
        points=[r for c in curves for r in c];fixed=[r for c in curves for r in arithmetic.fixed_rows(c)];aucs=[auc_row(c) for c in curves]
        table(out/(stem+'_'+scope+'_complete_prefix_curves.csv'),points)
        table(out/(stem+'_'+scope+'_budget_metrics.csv'),fixed);table(out/(stem+'_'+scope+'_auc.csv'),aucs)
        result[scope]={'budget_rows':fixed,'auc_rows':aucs,'curve_points':len(points),'curve_count':len(curves)}
    return result

def evaluate_all(choice_review,execution_review):
    from data_io import validate_input_manifest
    from pipeline import validate_global_predictions,validate_choices
    contract,models,freeze=validate_freeze();code_review(execution_review,'confirmation_external_evaluation_execution')
    review_digest=sha(execution_review)
    validate_global_predictions();choices_acceptance=validate_choices(choice_review)
    entries=[x for x in models if x['dataset']=='weeds' and x['variant']!='regression']
    inp=validate_input_manifest('weeds');frames=inp['records'];selections={}
    for entry in entries:
        out,seal=validate_selection(entry);accepted=next(x for x in choices_acceptance['models'] if x['run_id']==entry['run_id'])
        require(accepted['selection_seal']==binding(out/'selection_seal.json'),'Accepted external choices changed')
        selections[entry['run_id']]=out
    truth_home=HERE/'stages/truth_weeds';completed_stage(truth_home);tm=read(truth_home/'truth_manifest.json')
    require(tm['dataset']=='weeds' and tm['source_manifest_sha256']==inp['source_manifest']['sha256'],'Truth dataset/source differs')
    require(tm['choice_review']==binding(choice_review) and tm['all_predictions_seal']==binding(HERE/'stages/predictions_seal/all_predictions.json'),'Truth gate bindings differ')
    with attempt('external_outcomes') as out:
        configure_cpu(out)
        import external_arithmetic as arithmetic
        truth,validation=truth_cells(frames,tm);write(out/'truth_checks.json',validation)
        table(out/'truth_units.csv',[{'frame_id':frame,'grid_g':g,'id':i,**cell} for (frame,g),cells in sorted(truth.items()) for i,cell in sorted(cells.items())])
        curves=[];all_choices=[]
        for entry in entries:
            selected=selections[entry['run_id']];rankings=read_lines(selected/'rankings.jsonl');choices=read_lines(selected/'choices.jsonl')
            require(len(rankings)==312 and len(choices)==3120,'Selection cases changed')
            local_curves=[arithmetic.make_frame_curve(r,truth) for r in rankings]
            ranks={r['ranking_key']:r for r in rankings}
            lookup={(c[0]['R'],c[0]['frame_id'],c[0]['grid_g']):c for c in local_curves}
            for choice in choices:
                require(choice['selected_ids']==ranks[choice['ranking_key']]['ordered_ids'][:choice['k']],'Selected prefix differs')
                row=lookup[(choice['R'],choice['frame_id'],choice['grid_g'])][choice['k']]
                require(row['selected_area_FAE']==choice['selected_area_FAE'] and row['selected_predicted_weed_FAE']==choice['selected_predicted_weed_FAE'],'Choice measurement changed')
            curves.extend(local_curves);all_choices.extend(choices)
        learned=save_family(out,'learned',curves)
        fixed_lookup={x['choice_key']:x for x in learned['frame']['budget_rows']}
        for choice in all_choices:
            row=fixed_lookup[choice['choice_key']]
            for key in ('selection_key','duplicate_selection_alias','selected_area_FAE','selected_predicted_weed_FAE'):
                require(row[key]==choice[key],'Sealed choice differs from outcome '+key)
        require(len(curves)==1872 and len(all_choices)==18720 and learned['frame']['curve_points']==211536,'Global external outcome inventory')
        for scope,result in learned.items():
            table(out/('hard_soft_'+scope+'_budget_pairs.csv'),arithmetic.paired_rows(result['budget_rows']))
            table(out/('hard_soft_'+scope+'_auc_pairs.csv'),arithmetic.paired_auc_rows(result['auc_rows']))
        reference_summary={}
        for policy in ('label_oracle','random_expectation'):
            ranks=[r for g in GRIDS for r in reference_rankings(frames,truth,g,policy,inp['source_manifest']['sha256'])]
            reference_curves=[arithmetic.make_frame_curve(r,truth,policy,policy=='random_expectation') for r in ranks]
            refs=save_family(out,policy,reference_curves)
            lines(out/(policy+'_reference_rankings.jsonl'),ranks)
            reference_summary[policy]={scope:{k:v for k,v in result.items() if k not in ('budget_rows','auc_rows')} for scope,result in refs.items()}
        validate_source_lock()
        bound_file(execution_review,review_digest);code_review(execution_review,'confirmation_external_evaluation_execution')
        for entry in entries:validate_selection(entry)
        completed_stage(truth_home)
        write(out/'summary.json',{'models':[x['run_id'] for x in entries],'R':list(R_VALUES),'grids':list(GRIDS),'frames':26,
             'truth_units':8736,'learned_rankings':1872,'learned_budget_rows':18720,'learned_frame_curve_points':211536,
             'learned':{scope:{k:v for k,v in result.items() if k not in ('budget_rows','auc_rows')} for scope,result in learned.items()},
             'reference':reference_summary,'choice_review':binding(choice_review),'truth_manifest':binding(truth_home/'truth_manifest.json'),
             'execution_review':binding(execution_review),'freeze':binding(HERE/'FREEZE.json'),
             'interpretation':'Conditional frame-area accounting; crop inclusion is not crop injury or field efficacy.'})
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('select');p.add_argument('--run-id',required=True);p.add_argument('--root-review',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--choice-review',required=True);p.add_argument('--root-review',required=True)
    a=parser.parse_args()
    result=select_run(a.run_id,a.root_review) if a.stage=='select' else evaluate_all(a.choice_review,a.root_review)
    print(result)
if __name__=='__main__':main()
