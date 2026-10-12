"""Bounded synthetic checks and metadata-only readiness; never use real maps/labels."""
import argparse
import ast
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/"evaluate_external_weeds_spatial.py"
spec = importlib.util.spec_from_file_location("spatial_checked", SOURCE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
np = s.np


def run(acceptance=None, acceptance_sha=None):
    out = s.HOME/dt.datetime.now().strftime("readiness_%Y%m%d_%H%M%S_%f")
    out.mkdir(exist_ok=False)
    shutil.copy2(SOURCE, out/"evaluator.py")
    shutil.copy2(__file__, out/"checker.py")
    passed = []

    def check(name, function):
        function()
        passed.append(name)

    def fails(function, exception=AssertionError):
        try:
            function()
        except exception:
            return
        raise AssertionError("Expected rejection")

    def budget_test():
        actual = [[x["k"] for x in s.budget_rows(g)] for g in s.GRIDS]
        assert actual == [[1,3,4,8,16,1,3,4,8,16], [6,12,19,32,64,4,12,16,32,64], [25,51,76,128,256,16,48,64,128,256]]
        assert len({(g, row["k"]) for g in s.GRIDS for row in s.budget_rows(g)}) == 20
    check("all30 rational budgets,20 unique grid/k,matched exact areas", budget_test)

    def partition_test():
        plane = (np.arange(256*256, dtype=np.float64).reshape(256,256)/(256*256))
        totals = []
        for g in s.GRIDS:
            values = s.aggregate_plane(plane, "toy", g)
            assert len(values) == g*g
            totals.append(math.fsum(values.values()))
        assert max(totals)-min(totals) <= 1e-12
        for g, fine in ((4,8),(8,16)):
            coarse = s.aggregate_plane(plane,"toy",g)
            finer = s.aggregate_plane(plane,"toy",fine)
            for r in range(g):
                for c in range(g):
                    children = [f"toy:g{fine:02d}:r{2*r+dr:02d}:c{2*c+dc:02d}" for dr in (0,1) for dc in (0,1)]
                    assert abs(coarse[f"toy:g{g:02d}:r{r:02d}:c{c:02d}"]-math.fsum(finer[i] for i in children)) <= 1e-12
    check("complete half-open grids and exact nested support", partition_test)

    def child_mapping_test():
        plane=np.arange(256*256,dtype=np.float64).reshape(256,256)/(256*256)
        truth={}
        for g in s.GRIDS:
            values=s.aggregate_plane(plane,"mapping",g)
            truth[("mapping",g)]={i:dict(A=1/(g*g),W=v,C=0.0,B=1/(g*g)-v) for i,v in values.items()}
        result=s.check_nested_truth_cells(truth,["mapping"])
        assert result["scalar_comparisons"]==240
        cells=truth[("mapping",8)]
        identifiers=sorted(cells)
        a,b=identifiers[0],identifiers[-1]
        before={key:math.fsum(row[key] for row in cells.values()) for key in ("B","C","W")}
        cells[a],cells[b]=cells[b],cells[a]
        assert all(math.fsum(row[key] for row in cells.values())==before[key] for key in before)
        fails(lambda:s.check_nested_truth_cells(truth,["mapping"]))
    check("per-cell nested B/C/W rejects wrong child-ID mapping despite unchanged totals",child_mapping_test)

    units = [dict(id=f"cell{i:02d}", A=.25, P=.125) for i in (3,1,2,0)]
    check("ASCII exact ties", lambda: _assert([x["id"] for x in s.rank_units(units)] == [f"cell{i:02d}" for i in range(4)]))
    check("selector rejects target/crop fields", lambda: fails(lambda: s.rank_units([dict(x,W=0) for x in units])))
    check("selector rejects duplicate IDs", lambda: fails(lambda: s.rank_units([units[0]]*4)))
    check("selector rejects nonfinite prediction", lambda: fails(lambda: s.rank_units([dict(x,P=math.nan) for x in units])))
    check("zero denominator is NA and positive-denominator zero remains0", lambda: _assert(s.ratio(0,0,"zero")== (None,"zero") and s.ratio(0,1,"zero")== (0.0,"")))
    check("crop proxy denominators differ", lambda: _assert(s.measurements(.5,0,.1,.2,.2,.4,.25,1,1,1)["crop_capture_ratio"]==.8 and s.measurements(.5,0,.1,.2,.2,.4,.25,1,1,1)["crop_fraction_of_selected_area"]==.4))
    check("empty selected-area crop composition NA", lambda: _assert(s.measurements(0,0,0,0,0,.4,.25,1,1,1)["crop_fraction_of_selected_area"] is None))

    def auc_test():
        for n in (2,4,16,64,256):
            random = [dict(k=k,weed_capture_ratio=k/n) for k in range(n+1)]
            assert abs(s.step_auc(random,1)-(n-1)/(2*n)) < 1e-12
            assert abs(s.step_auc(random,.5)-(.25-1/(2*n))) < 1e-12
        curve = [dict(k=i,weed_capture_ratio=v) for i,v in enumerate([0,.75,1])]
        assert s.step_auc(curve,1)==.375 and s.step_auc(curve,.5)==0
        assert s.step_auc([dict(k=k,weed_capture_ratio=None) for k in range(3)],1) is None
    check("complete-step AUC analytic examples/random formulas/NA", auc_test)

    def zero_curve_test():
        cells={f"zero:g04:r{r:02d}:c{c:02d}":dict(A=1/16,B=1/16,C=0.0,W=0.0) for r in range(4) for c in range(4)}
        ranking=dict(run_id="zero",variant="hard_seg",seed=17,original_selected_epoch=1,
                     checkpoint_sha256="toy",config_sha256="toy",export_sha256="toy",val_manifest_sha256="toy",
                     R=1,grid_g=4,frame_id="zero",acquisition_date="toy",ranking_key="toy",
                     ordered_ids=sorted(cells),units=[dict(id=i,A=1/16,P=0.0) for i in sorted(cells)])
        curve=s.make_frame_curve(ranking,{("zero",4):cells})
        assert all(x["weed_capture_ratio"] is None and x["crop_capture_ratio"] is None for x in curve)
        assert curve[0]["crop_fraction_of_selected_area"] is None
        assert curve[1]["crop_fraction_of_selected_area"]==0
        assert s.step_auc(curve,1) is None and s.step_auc(curve,.5) is None
    check("zero-weed/crop complete curves and AUC retain NA",zero_curve_test)

    fixture = out/"synthetic_fixture"
    fixture.mkdir()
    output = out/"synthetic_output"
    output.mkdir()
    (output/"source").mkdir()
    shutil.copy2(SOURCE, output/"source"/"evaluator.py")
    s.dump(output/"input_binding.json", {"synthetic_only":True})
    frames = []
    for j in range(2):
        identifier = f"toy_frame{j}"
        soft3 = np.zeros((3,256,256), np.float32)
        if j == 0:
            soft3[2,:128] = .25
            soft3[1,128:] = .5
        else:
            soft3[2,:128,:128] = .75
        soft3[0] = 1-soft3[1]-soft3[2]
        path = fixture/(identifier+".npz")
        native_counts = np.array([round(float(soft3[i].mean(dtype=np.float64))*600*600) for i in range(3)],np.uint32)
        np.savez_compressed(path,soft3=soft3,soft=soft3[2],valid=np.ones((256,256),np.float32),native_class_counts=native_counts)
        frames.append(dict(id=identifier,acquisition_date=f"toy_date{j}",cache=str(path),cache_sha256=s.sha(path)))
    models = []
    metadata = {}
    maps = []
    for variant in ("hard_seg","soft_seg"):
        run_id = "synthetic_"+variant
        models.append(dict(run_id=run_id))
        metadata[run_id] = dict(run_id=run_id,variant=variant,seed=17,original_selected_epoch=1,
                               checkpoint_sha256="synthetic",config_sha256="synthetic",export_sha256="synthetic",val_manifest_sha256="synthetic")
        for frame in frames:
            plane = np.tile(np.linspace(0,1,256,dtype=np.float32)[:,None],(1,256))
            if variant == "soft_seg": plane=plane[::-1].copy()
            folder = fixture/run_id
            folder.mkdir(exist_ok=True)
            path = folder/(frame["id"]+".npz")
            np.savez_compressed(path,probabilities=plane[None],R=np.array([1],np.int64),id=np.array(frame["id"]))
            maps.append(dict(run_id=run_id,id=frame["id"],path=path.relative_to(fixture).as_posix(),sha256=s.sha(path)))
    counts = dict(learned_complete_rankings=12,learned_frame_budget_rows=120,
                  unique_learned_frame_model_R_grid_k_choices=80,prediction_unit_rows=1344,
                  unique_truth_unit_rows_after_selection_seal=672)
    for scope, framefactor in (("frame",2),("date",2),("pooled",1)):
        counts[f"learned_{scope}_budget_rows"] = 2*3*10*framefactor
        counts[f"learned_{scope}_auc_rows_two_auc_columns"] = 2*3*framefactor
        counts[f"learned_{scope}_complete_prefix_points_including_zero"] = 2*339*framefactor
        counts[f"hard_soft_paired_{scope}_budget_rows"] = 3*10*framefactor
    toyprotocol = dict(eligible_models=models, planned_counts_not_observed_results=counts)
    real_rs = s.RS
    s.RS = (1,)
    watched = {}
    original_np_load = s.np.load
    calls = []
    predicted_paths = {(fixture/x["path"]).resolve() for x in maps}

    def guarded_prediction_load(path,*args,**kwargs):
        assert Path(path).resolve() in predicted_paths, "PhaseA tried a non-prediction array"
        calls.append(str(path))
        return original_np_load(path,*args,**kwargs)

    try:
        s.np.load = guarded_prediction_load
        seal_sha = s.phase_a(output,toyprotocol,frames,metadata,maps,fixture,watched)
        assert len(calls)==4
        passed.append("synthetic orchestration: all2model/2frame/3grid rankings sealed before any labels")
        choices = s.read_lines(output/"learned_fixed_budget_choices.jsonl")
        by_key = {}
        for row in choices:
            old = by_key.setdefault(row["selection_key"],row)
            assert old["selected_ids"]==row["selected_ids"] and old["selected_area_FAE"]==row["selected_area_FAE"] and old["selected_predicted_weed_FAE"]==row["selected_predicted_weed_FAE"]
        assert len(by_key)==80
        passed.append("bothbudget aliases use identical IDs/A/P and strict area")
        original_choice_bytes = (output/"learned_fixed_budget_choices.jsonl").read_bytes()
        (output/"learned_fixed_budget_choices.jsonl").write_bytes(original_choice_bytes+b" ")
        fails(lambda:s.load_truth_after_seal(output,seal_sha,frames,watched))
        (output/"learned_fixed_budget_choices.jsonl").write_bytes(original_choice_bytes)
        passed.append("seal mutation rejected before any label load")
        s.np.load = original_np_load
        summary = s.phase_b(output,seal_sha,toyprotocol,frames,watched)
        assert summary["truth_unit_rows"]==672 and summary["nested_oracle_checks"]==15
        passed.append("synthetic full label-join/frame/date/pooled/paired/oracle/random/AUC/count closure")
        pooled = list(__import__("csv").DictReader((output/"learned_pooled_budget_metrics.csv").open(encoding="utf-8")))
        frame_metrics = list(__import__("csv").DictReader((output/"learned_frame_budget_metrics.csv").open(encoding="utf-8")))
        differs=False
        for row in pooled:
            pair=[x for x in frame_metrics if all(x[k]==row[k] for k in ("run_id","R","grid_g","budget_basis","parent_nominal_budget_code"))]
            mean=math.fsum(float(x["weed_capture_ratio"]) for x in pair)/len(pair)
            if abs(mean-float(row["weed_capture_ratio"]))>1e-8: differs=True
        assert differs
        passed.append("ratio-of-sums demonstrably differs from mean-of-frame ratios")
    finally:
        s.np.load=original_np_load
        s.RS=real_rs

    check("source hash mismatch hard fails",lambda:fails(lambda:s.checked(SOURCE,"0"*64,{})))
    check("path escape rejected",lambda:fails(lambda:s.confined(fixture/".."/"outside",fixture)))
    check("exclusive product write refuses overwrite",lambda:fails(lambda:s.dump(output/"input_binding.json",{}),FileExistsError))
    tree=ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(n.name.split('.')[0] for n in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append((node.module or '').split('.')[0])
    check("no torch/scipy/solver/model imports",lambda:_assert(not(set(imports)&{"torch","scipy","train_external_weeds","models_common"})))

    def init_failure_test(stage):
        # Exercise execute() initialization with synthetic metadata, before any array stage.
        home=out/("synthetic_init_failure_"+stage)
        home.mkdir()
        original_home=s.HOME
        original_inputs=s.read_allowed_inputs
        original_mkdir=Path.mkdir
        old_env={key:s.os.environ.get(key) for key in ("TEMP","TMP")}
        def failing_mkdir(path,*args,**kwargs):
            if (stage=="output" and path.parent==home and path.name.startswith("results_")) or (stage=="temp" and path.name=="temp" and path.parent.parent==home):
                raise OSError("Synthetic initialization failure at "+stage)
            return original_mkdir(path,*args,**kwargs)
        s.HOME=home
        s.read_allowed_inputs=lambda *args,**kwargs: ({},[],{},{},home,{})
        Path.mkdir=failing_mkdir
        args=SimpleNamespace(acceptance="toy",acceptance_sha256="toy",authorization="toy",authorization_sha256="toy")
        try:
            fails(lambda:s.execute(args),OSError)
            assert (home/"SPATIAL_EXECUTION_CLAIM.json").exists()
            if stage=="output":
                receipts=list(home.glob("execution_failure_*.json"))
                assert len(receipts)==1 and not s.read(receipts[0])["output_owned"]
            else:
                folders=list(home.glob("results_*"))
                assert len(folders)==1 and s.read(folders[0]/"failure.json")["output_owned"]
                assert (folders[0]/"status.json").exists() and (folders[0]/"artifact_manifest.json").exists()
        finally:
            s.HOME=original_home
            s.read_allowed_inputs=original_inputs
            Path.mkdir=original_mkdir
            for key,value in old_env.items():
                if value is None:s.os.environ.pop(key,None)
                else:s.os.environ[key]=value
    check("postclaim output mkdir failure preserves sibling receipt and claim",lambda:init_failure_test("output"))
    check("postclaim temp mkdir failure preserves output failure/status/manifest",lambda:init_failure_test("temp"))

    metadata_check=None
    if acceptance:
        # This gate actively forbids np.load: actual maps and all cache arrays stay unopened.
        def prohibit(*args,**kwargs):
            raise AssertionError("Actual array access is forbidden in metadata readiness")
        s.np.load=prohibit
        try:
            protocol,ff,mm,idx,export,ww=s.read_allowed_inputs(acceptance,acceptance_sha)
            assert len(ff)==26 and len(mm)==6 and len(idx)==156
            s.recheck(ww)
            metadata_check=dict(passed=True,frames=26,models=6,map_records=156,
                                watched_source_files=len(ww),source_sha256=ww,
                                actual_arrays_opened=False,production_authorization_present=False)
            passed.append("actual sanitized metadata+156map hashes accepted while np.load prohibited")
        finally:
            s.np.load=original_np_load
    s.dump(out/"readiness.json",dict(passed=True,created_utc=s.now(),source_sha256=s.sha(SOURCE),
           checker_sha256=s.sha(__file__),protocol_sha256=s.PROTOCOL_SHA,
           acceptance_sha256=acceptance_sha,checks=passed,check_count=len(passed),
           actual_model_predictions_opened=False,actual_label_arrays_opened=False,
           actual_spatial_selection=False,production_execution_authorized=False,
           synthetic_only_outputs=True,metadata_check=metadata_check))
    s.dump(out/"artifact_manifest.json",dict(file_sha256={p.relative_to(out).as_posix():s.sha(p) for p in out.rglob("*") if p.is_file()}))
    print(json.dumps(dict(output=str(out),passed=True,check_count=len(passed),source_sha256=s.sha(SOURCE),readiness_sha256=s.sha(out/"readiness.json")),ensure_ascii=False))


def _assert(value):
    assert value


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--acceptance")
    parser.add_argument("--acceptance-sha256")
    args=parser.parse_args()
    assert bool(args.acceptance)==bool(args.acceptance_sha256)
    run(args.acceptance,args.acceptance_sha256)
