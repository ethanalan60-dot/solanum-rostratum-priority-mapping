"""Read completed spatial CSVs only; prepare a descriptive report and full g8 figure."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT/"external_weeds_v1"/"spatial_v1"
RESULT = BASE/"results_20260916_065643_813785"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(path, rows):
    assert rows
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def num(row, key):
    return None if row[key] == "" else float(row[key])


def direction(rows, key, tolerance=1e-10):
    vals=[num(row,key) for row in rows]
    present=[x for x in vals if x is not None]
    return dict(rows=len(rows), eligible=len(present), NA=len(vals)-len(present),
                soft_higher=sum(x>tolerance for x in present),
                soft_lower=sum(x < -tolerance for x in present),
                tied=sum(abs(x)<=tolerance for x in present),
                minimum=min(present), maximum=max(present), tolerance=tolerance)


def main():
    state=read(RESULT/"status.json")
    assert state["state"]=="completed" and not state["test_accessed"] and not state["R6_accessed"]
    manifest=read(RESULT/"artifact_manifest.json")
    source_files={name: sha(RESULT/name) for name in ("artifact_manifest.json","status.json","selection_seal.json")}
    for name in ("status.json","selection_seal.json"):
        assert source_files[name]==manifest["file_sha256"][name]
    names=[
        "learned_pooled_complete_prefix_curves.csv", "learned_pooled_budget_metrics.csv", "learned_pooled_auc.csv",
        "label_oracle_pooled_complete_prefix_curves.csv", "label_oracle_pooled_budget_metrics.csv", "label_oracle_pooled_auc.csv",
        "random_expectation_pooled_complete_prefix_curves.csv", "random_grid_budget_expectations.csv",
        "hard_soft_paired_frame_budget_metrics.csv", "hard_soft_paired_date_budget_metrics.csv", "hard_soft_paired_pooled_budget_metrics.csv",
        "hard_soft_paired_frame_auc.csv", "hard_soft_paired_date_auc.csv", "hard_soft_paired_pooled_auc.csv",
        "learned_date_budget_metrics.csv", "learned_frame_budget_metrics.csv", "nested_oracle_matched_actual_checks.csv",
    ]
    data={}
    for name in names:
        assert sha(RESULT/name)==manifest["file_sha256"][name], name
        source_files[name]=sha(RESULT/name)
        data[name]=load_csv(RESULT/name)
    out=BASE/"reports"/dt.datetime.now().strftime("full6_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    (out/"source_tables").mkdir()
    (out/"qa").mkdir()
    (out/"temp").mkdir()
    for key in ("TEMP","TMP","MPLCONFIGDIR"):
        os.environ[key]=str(out/"temp")
    for name in source_files:
        shutil.copy2(RESULT/name,out/"source_tables"/name)
    shutil.copy2(__file__,out/"build_external_weeds_spatial_report.py")
    source_files["../protocol.json"]=sha(BASE/"protocol.json")
    shutil.copy2(BASE/"protocol.json",out/"protocol.json")

    pooled=data["learned_pooled_budget_metrics.csv"]
    pairs=data["hard_soft_paired_pooled_budget_metrics.csv"]
    aucpairs=data["hard_soft_paired_pooled_auc.csv"]
    curves=data["learned_pooled_complete_prefix_curves.csv"]
    assert len(pooled)==540 and len(pairs)==270 and len(aucpairs)==27 and len(curves)==6102
    assert len({(x["run_id"],x["R"],x["grid_g"],x["k"]) for x in pooled})==360
    primary=[x for x in pooled if x["grid_g"]=="8" and x["budget_basis"]=="nominal_primary"]
    primarypairs=[x for x in pairs if x["grid_g"]=="8" and x["budget_basis"]=="nominal_primary"]
    assert len(primary)==90 and len(primarypairs)==45
    assert {float(x["actual_fraction"]) for x in primary}=={6/64,12/64,19/64,32/64,1}
    table(out/"primary_g8_nominal_all90.csv",primary)
    table(out/"primary_g8_nominal_paired45.csv",primarypairs)
    table(out/"all_grids_both_budget_bases_pooled540.csv",pooled)
    table(out/"all_grids_both_budget_bases_paired270.csv",pairs)
    table(out/"all_grids_full_prefix_auc_paired27.csv",aucpairs)
    summaries=[]
    for g in (4,8,16):
        for basis in ("nominal_primary","matched_actual_secondary"):
            group=[x for x in pairs if x["grid_g"]==str(g) and x["budget_basis"]==basis]
            assert len(group)==45
            for metric in ("weed_capture_ratio","crop_capture_ratio","crop_fraction_of_selected_area"):
                summaries.append(dict(grid_g=g,budget_basis=basis,metric=metric,
                    unit="percentage_points",**direction(group,"soft_minus_hard_"+metric+"_pp")))
    table(out/"paired_direction_descriptions.csv",summaries)
    auc_summaries=[]
    for g in (4,8,16):
        group=[x for x in aucpairs if x["grid_g"]==str(g)]
        assert len(group)==9
        for metric in ("auc_0_1","auc_0_0p5_normalized"):
            auc_summaries.append(dict(grid_g=g,metric=metric,unit="dimensionless",**direction(group,"soft_minus_hard_"+metric,1e-12)))
    table(out/"paired_auc_direction_descriptions.csv",auc_summaries)
    epochs={}
    for row in pooled:
        entry={key:row[key] for key in ("run_id","variant","seed","original_selected_epoch","checkpoint_sha256","config_sha256","export_sha256","val_manifest_sha256")}
        assert epochs.setdefault(row["run_id"],entry)==entry
    assert len(epochs)==6
    table(out/"original_selected_checkpoint_ledger.csv",list(epochs.values()))

    # Primary 2x3 panel: all complete-prefix points, no seed averaging or intervals.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import fitz
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,
        "axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"path",
        "axes.spines.top":False,"axes.spines.right":False})
    colors={17:"#0072B2",29:"#D55E00",43:"#009E73"}
    styles={"hard_seg":(0,(5,2.5)),"soft_seg":"solid"}
    metrics=("weed_capture_ratio","crop_fraction_of_selected_area")
    references={"label_oracle":data["label_oracle_pooled_complete_prefix_curves.csv"],
                "random_expectation":data["random_expectation_pooled_complete_prefix_curves.csv"]}
    series=[]
    for r in (1,2,4):
        for seed in (17,29,43):
            for variant in ("hard_seg","soft_seg"):
                rows=sorted([x for x in curves if x["grid_g"]=="8" and x["R"]==str(r) and x["seed"]==str(seed) and x["variant"]==variant],key=lambda x:int(x["k"]))
                assert [int(x["k"]) for x in rows]==list(range(65))
                for metric in metrics:
                    assert (rows[0][metric]=="")==(metric=="crop_fraction_of_selected_area")
                    series.append(dict(R=r,metric=metric,policy="learned",run_id=rows[0]["run_id"],seed=seed,variant=variant,
                                       x=[float(x["actual_fraction"]) for x in rows],y=[num(x,metric) for x in rows]))
        for policy, ref in references.items():
            rows=sorted([x for x in ref if x["grid_g"]=="8"],key=lambda x:int(x["k"]))
            assert [int(x["k"]) for x in rows]==list(range(65))
            for metric in metrics:
                series.append(dict(R=r,metric=metric,policy=policy,run_id=policy,seed=None,variant="reference",
                                   x=[float(x["actual_fraction"]) for x in rows],y=[num(x,metric) for x in rows]))
    assert len(series)==48
    points=[]
    for item in series:
        for k,(x,y) in enumerate(zip(item["x"],item["y"])):
            points.append(dict(R=item["R"],metric=item["metric"],policy=item["policy"],run_id=item["run_id"],seed=item["seed"],variant=item["variant"],k=k,nominal_budget_percent=100*x,value_fraction=y,value_percent=None if y is None else 100*y))
    assert len(points)==3120
    table(out/"figure_bound_all3120_points.csv",points)
    cropmax=max(y*100 for item in series if item["metric"]==metrics[1] for y in item["y"] if y is not None)
    crop_upper=max(5,math.ceil(cropmax/5)*5)
    fig,axes=plt.subplots(2,3,figsize=(13.8,8.1),sharex=True,sharey="row")
    fig.subplots_adjust(left=.085,right=.985,bottom=.215,top=.865,hspace=.28,wspace=.16)
    drawn=[]
    for col,r in enumerate((1,2,4)):
        for row,metric in enumerate(metrics):
            ax=axes[row,col]
            relevant=[item for item in series if item["R"]==r and item["metric"]==metric]
            assert len(relevant)==8
            for item in relevant:
                if item["policy"]=="learned":
                    color=colors[item["seed"]];style=styles[item["variant"]];width=1.5
                elif item["policy"]=="label_oracle":
                    color="#222222";style=(0,(6,2,1,2));width=1.6
                else:
                    color="#777777";style=(0,(1.5,2));width=1.8
                x=np.array(item["x"],float)*100
                y=np.array([np.nan if v is None else v*100 for v in item["y"]],float)
                line,=ax.step(x,y,where="post",color=color,linestyle=style,linewidth=width)
                assert np.array_equal(line.get_xdata(),x) and np.array_equal(line.get_ydata(),y,equal_nan=True)
                drawn.append(dict(R=r,metric=metric,run_id=item["run_id"],point_count=65,nan_points=int(np.isnan(y).sum()),drawstyle=line.get_drawstyle()))
            ax.set_xlim(0,100)
            ax.set_ylim(0,100 if row==0 else crop_upper)
            ax.set_xticks([0,20,40,60,80,100])
            ax.grid(True,color="#dddddd",linewidth=.55,alpha=.8)
            ax.set_axisbelow(True)
            ax.text(.025,.95,chr(ord("a")+3*row+col),transform=ax.transAxes,va="top",ha="left",fontweight="bold",fontsize=11)
            if row==0:ax.set_title(f"R = {r}",pad=9)
            else:ax.set_xlabel("Per-frame area budget (%)")
    axes[0,0].set_ylabel("Pooled weed capture (%)")
    axes[1,0].set_ylabel("Crop fraction within selected area (%)")
    title=fig.suptitle("WeedsGalore: complete-prefix allocation on the primary 8 × 8 grid",x=.52,y=.974,fontsize=15,fontweight="bold")
    subtitle=fig.text(.52,.927,"26 validation frames · one field · original MAE-selected checkpoints · no seed averaging",ha="center",fontsize=10.5)
    handles=[]
    for seed in (17,29,43):
        for variant,label in (("hard_seg","Hard"),("soft_seg","Soft")):
            handles.append(Line2D([],[],color=colors[seed],linestyle=styles[variant],linewidth=1.8,label=f"{label}, seed {seed}"))
    handles.extend([Line2D([],[],color="#222222",linestyle=(0,(6,2,1,2)),linewidth=1.8,label="Weed-label top-k reference"),Line2D([],[],color="#777777",linestyle=(0,(1.5,2)),linewidth=1.8,label="Random expectation")])
    legend=fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.535,.077),ncol=4,frameon=False,columnspacing=1.8,handlelength=3.5,fontsize=9.5)
    footer=fig.text(.52,.026,"Crop composition is undefined at zero selected area. Lower crop composition is not automatically better.\nDevelopment description; numerical acceptance pending independent audit.",ha="center",va="center",fontsize=9,color="#444444")
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    layout=[]
    for label,artist in [("title",title),("subtitle",subtitle),("legend",legend),("footer",footer)]:
        box=artist.get_window_extent(renderer)
        bounds=fig.bbox
        inside=bool(box.x0>=bounds.x0 and box.y0>=bounds.y0 and box.x1<=bounds.x1 and box.y1<=bounds.y1)
        assert inside,label
        layout.append(dict(label=label,inside_figure=inside,bbox=[float(v) for v in box.bounds]))
    for extension in ("png","svg","pdf"):
        fig.savefig(out/("primary_g8_complete_prefix_2x3."+extension),dpi=300,facecolor="white")
    plt.close(fig)
    for extension in ("pdf","svg"):
        document=fitz.open(out/("primary_g8_complete_prefix_2x3."+extension))
        if extension=="svg":
            converted=document.convert_to_pdf();document.close();document=fitz.open(stream=converted,filetype="pdf")
        assert document.page_count==1
        pix=document[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
        pix.save(out/"qa"/("primary_g8_"+extension+"_render.png"))
        document.close()
    dump(out/"figure_binding.json",dict(source_results=str(RESULT),source_sha256=source_files,
        series=series,display_series=drawn,total_bound_points=len(points),x_range_percent=[0,100],
        weed_y_range_percent=[0,100],crop_y_range_percent=[0,crop_upper],crop_observed_max_percent=cropmax,
        axes_are_shared_within_rows=True,where="post",k0_crop_is_NA=True,
        seed_color=colors,hard_soft_distinct_linestyle=True,confidence_intervals=False,seed_aggregation=False,
        references_repeated_across_R_because_labels_and_geometry_fixed=True,layout_checks=layout,
        matplotlib=matplotlib.__version__,python=sys.version,numeric_acceptance="pending_independent_audit",
        figure_sha256={ext:sha(out/("primary_g8_complete_prefix_2x3."+ext)) for ext in ("png","svg","pdf")}))

    weed_primary=direction(primarypairs,"soft_minus_hard_weed_capture_ratio_pp")
    crop_primary=direction(primarypairs,"soft_minus_hard_crop_fraction_of_selected_area_pp")
    wt={float(x["total_weed_FAE"]) for x in pooled};ct={float(x["total_crop_FAE"]) for x in pooled}
    assert max(wt)-min(wt)<1e-12 and max(ct)-min(ct)<1e-12
    totals=dict(frame_area_equivalents=26,total_weed_FAE=next(iter(wt)),total_crop_FAE=next(iter(ct)))
    checkpoint_rows="\n".join(f'| {x["variant"]} | {x["seed"]} | {x["original_selected_epoch"]} |' for x in epochs.values())
    direction_rows="\n".join(f'| {x["grid_g"]}×{x["grid_g"]} | {x["budget_basis"]} | {x["soft_higher"]} | {x["soft_lower"]} | {x["tied"]} | {x["minimum"]:+.4f} to {x["maximum"]:+.4f} |' for x in summaries if x["metric"]=="weed_capture_ratio")
    auc_rows="\n".join(f'| {x["grid_g"]}×{x["grid_g"]} | {x["metric"]} | {x["soft_higher"]} | {x["soft_lower"]} | {x["tied"]} | {x["minimum"]:+.6f} to {x["maximum"]:+.6f} |' for x in auc_summaries)
    primary_tables=[]
    for metric,label in (("weed_capture_ratio","Weed capture (%)"),("crop_fraction_of_selected_area","Crop fraction within selected area (%)")):
        rows=[]
        for seed in (17,29,43):
            for r in (1,2,4):
                for variant in ("hard_seg","soft_seg"):
                    values=[]
                    for code in ("b10","b20","b30","b50","b100"):
                        subset=[x for x in primary if x["seed"]==str(seed) and x["R"]==str(r) and x["variant"]==variant and x["parent_nominal_budget_code"]==code]
                        assert len(subset)==1
                        val=num(subset[0],metric)
                        values.append("NA" if val is None else f"{100*val:.3f}")
                    rows.append(f"| {seed} | {r} | {variant.replace('_seg','')} | "+" | ".join(values)+" |")
        primary_tables.append(f"### {label}\n\n| Seed | R | Supervision | Nominal 10% | 20% | 30% | 50% | 100% |\n|---:|---:|---|---:|---:|---:|---:|---:|\n"+"\n".join(rows))
    report=f"""# External WeedsGalore spatial report: all six segmentation models

**Status: descriptive report prepared from completed tables; independent numerical acceptance is pending.** Figure rendering checks are recorded separately. This report performs no new prediction, ranking, cell selection, label aggregation, inference or solver work.

## Scope and interpretation

All six original MAE-selected hard/soft models (seeds17/29/43) are shown at R1/2/4, with all26 official validation frames. The primary partition is8×8;4×4 and16×16 are relative-grid sensitivity analyses. Each frame receives its own budget. This is within-dataset retraining on one field over four dates, not local Solanum checkpoint transfer. Frame-area equivalents may include repeated or overlapping ground support. They are not hectares or unique ground area, and absolute capture levels must not be compared or statistically pooled with the local geographic study.

The source tables contain {totals['total_weed_FAE']:.12f} weed FAE and {totals['total_crop_FAE']:.12f} crop FAE across26 frame-area equivalents. Capture uses the ratio of pooled annotation-area sums after independent per-frame selections. The bottom figure row is selected crop annotation divided by selected image area; it is different from crop capture (selected crop divided by all crop). Both proxies and their denominators remain in the complete tables. Lower crop composition alone is not evidence of a better allocation, less injury, higher yield or successful weed control.

## Primary full-prefix figure

![Primary complete-prefix figure](primary_g8_complete_prefix_2x3.png)

Columns show R1/2/4. Every panel includes all six models; seed is color and hard/soft supervision is line style. Curves are right-continuous complete-prefix steps on the full0–100% per-frame budget range. Each curve includes all65 prefix points. Crop composition is NA at the empty selection, so no artificial zero is drawn there. Shared references are annotation-informed equal-cost weed top-k and analytic uniform-random expectation, with no Monte Carlo intervals. The weed-first reference does not minimize crop. Axes share their scale within each row; all displayed values are included. No seed averaging, confidence intervals or statistical significance is asserted.

Figure sources and all3120 displayed coordinate records are in [figure_bound_all3120_points.csv](figure_bound_all3120_points.csv) and [figure_binding.json](figure_binding.json). Download [PDF](primary_g8_complete_prefix_2x3.pdf) or [SVG](primary_g8_complete_prefix_2x3.svg).

## All primary fixed budgets

The five nominal budgets10/20/30/50/100% select6/12/19/32/64 of64 cells per frame: actual fractions **9.375/18.75/29.6875/50/100%**. These are2.4375/4.875/7.71875/13/26 FAE across26 frames. Every seed/R/model has the same area at a matched budget. All90 source rows are retained in [primary_g8_nominal_all90.csv](primary_g8_nominal_all90.csv); the following tables show every model/R/budget combination.

{chr(10).join(primary_tables)}

## Complete paired descriptions, including sensitivity

For the45 primary nominal seed/R/budget pairs, soft weed capture is higher in **{weed_primary['soft_higher']}**, lower in **{weed_primary['soft_lower']}**, and tied in **{weed_primary['tied']}** cases at an arithmetic display tolerance of1e-10 percentage points. Differences span **{weed_primary['minimum']:+.4f} to {weed_primary['maximum']:+.4f} pp**. This count includes the structurally identical100% allocation. Crop composition is higher/lower/tied in {crop_primary['soft_higher']}/{crop_primary['soft_lower']}/{crop_primary['tied']} cases; no direction is assigned an agricultural benefit. All45 exact pairs are in [primary_g8_nominal_paired45.csv](primary_g8_nominal_paired45.csv).

The following counts retain all45 pairs for each grid and budget basis, including unfavorable results. Matched-actual secondary anchors are6.25/18.75/25/50/100%, derived from the coarsest grid before spatial outcomes. They control the different actual area produced by nominal-budget floors. They do not replace the primary nominal analysis.

| Grid | Budget basis | Soft higher | Soft lower | Tied | Weed capture difference, pp |
|---|---|---:|---:|---:|---|
{direction_rows}

Both budget bases contain270 labelled paired rows but only180 distinct seed/R/grid/k comparisons. Repeated50/100% cases, all4×4 cases and the8×8 20% anchor are aliases of the same selection; they are not additional independent evidence. [Complete270 pairs](all_grids_both_budget_bases_paired270.csv), [all540 model rows](all_grids_both_budget_bases_pooled540.csv), and [all crop/weed direction descriptions](paired_direction_descriptions.csv) retain both bases and every grid. Per-frame and per-date paired tables are linked below.

## Both complete-prefix AUC intervals

These are the already computed exact staircase integrals on[0,1] and normalized[0,0.5], never trapezoids through five budget points. Each model/R/grid has one full curve; fixed-budget bases do not duplicate AUC evidence. The27 hard/soft pairs below are descriptive across three seeds and three R values per grid, with an arithmetic tie tolerance1e-12 AUC.

| Grid | AUC metric | Soft higher | Soft lower | Tied | Soft-minus-hard range |
|---|---|---:|---:|---:|---|
{auc_rows}

[All27 AUC pairs](all_grids_full_prefix_auc_paired27.csv) retain every value. Seeds describe training variability, not independent fields. No strongest seed, grid, epoch or model is chosen from these spatial outcomes.

## Original checkpoint ledger

| Supervision | Seed | Original selected epoch |
|---|---:|---:|
{checkpoint_rows}

Exact checkpoint/config/export/val-manifest hashes are in [original_selected_checkpoint_ledger.csv](original_selected_checkpoint_ledger.csv). This report has not reopened checkpoints or raw probability/label arrays.

## Complete companion evidence

- [All14,040 learned frame-budget rows](source_tables/learned_frame_budget_metrics.csv), [all2,160 learned date-budget rows](source_tables/learned_date_budget_metrics.csv).
- [All7,020 paired frame rows](source_tables/hard_soft_paired_frame_budget_metrics.csv), [all1,080 paired date rows](source_tables/hard_soft_paired_date_budget_metrics.csv).
- [Frame AUC pairs](source_tables/hard_soft_paired_frame_auc.csv), [date AUC pairs](source_tables/hard_soft_paired_date_auc.csv).
- [All54 learned pooled AUC rows](source_tables/learned_pooled_auc.csv), [all6,102 pooled complete-prefix points](source_tables/learned_pooled_complete_prefix_curves.csv).
- [Annotation-informed fixed-budget reference](source_tables/label_oracle_pooled_budget_metrics.csv), [reference AUC](source_tables/label_oracle_pooled_auc.csv), [random expectations](source_tables/random_grid_budget_expectations.csv).
- [Nested oracle checks](source_tables/nested_oracle_matched_actual_checks.csv), [source hashes and report evidence](report_evidence.json), [frozen protocol copy](protocol.json).

The source result directory remains unchanged. Its full frame/date curves, selection seal and truth tables remain available through the absolute paths in the evidence ledger. Independent numerical audit and root acceptance remain a separate gate before manuscript evidence use.
"""
    (out/"report.md").write_text(report,encoding="utf-8",newline="\n")
    evidence=dict(status="prepared_pending_independent_numeric_acceptance",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_result_directory=str(RESULT),source_sha256=source_files,
        all_source_artifact_sha256=manifest["file_sha256"],builder_sha256=sha(__file__),
        counts=dict(primary_model_rows=90,primary_pairs=45,pooled_both_basis_rows=540,pooled_both_basis_pairs=270,
                    unique_pooled_model_grid_k=360,unique_paired_seed_R_grid_k=180,pooled_auc_pairs=27,plotted_series=48,plotted_points=3120),
        totals_FAE=totals,primary_weed_pair_direction=weed_primary,primary_crop_composition_pair_direction=crop_primary,
        direction_descriptions=summaries,auc_direction_descriptions=auc_summaries,
        no_new_model_prediction_selection_or_label_aggregation=True,source_table_only=True,
        independent_numeric_audit_accepted=False,source_result_status_sha256=source_files["status.json"])
    dump(out/"report_evidence.json",evidence)
    for name,digest in source_files.items():
        path=RESULT/name
        assert sha(path)==digest,name
    dump(out/"build_receipt.json",dict(status="built_pending_visual_inspection_and_independent_numeric_acceptance",source_unchanged=True,
        product_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob("*") if p.is_file() and "temp" not in p.relative_to(out).parts},
        source_result=str(RESULT),builder_sha256=sha(__file__)))
    print(json.dumps({"output":str(out),"status":"built_pending_visual_and_numeric_acceptance","primary_weed_pair_direction":weed_primary,"crop_axis_upper_percent":crop_upper},ensure_ascii=False))


if __name__=="__main__":
    main()
