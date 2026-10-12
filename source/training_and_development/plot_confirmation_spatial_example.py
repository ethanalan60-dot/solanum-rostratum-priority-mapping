"""Display one prospectively fixed saved selection; no prediction or optimization.

Production consumes accepted CSV/JSON only. Import performs no data access.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import traceback

ROOT = Path(__file__).resolve().parent
PLAN = ROOT/'provenance/confirmation_spatial_example_plan_20260916_124026_730827/display_plan.json'
PLAN_SHA = '2c35c4dd45c07ff6289fed0e937495021720ffa71b778861e1f82521b3902563'
MODEL = 'dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17'
RS = (1, 2, 4, 6)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def number(x):
    value = float(x)
    require(math.isfinite(value), 'Nonfinite display value')
    return value


def integer(x):
    value = number(x)
    require(value == int(value), 'Noninteger grid identity')
    return int(value)


def close(a, b):
    require(math.isclose(number(a), number(b), rel_tol=1e-9, abs_tol=1e-8), 'Saved display arithmetic mismatch')


class Inputs:
    def __init__(self):
        self.watched = {}

    def bytes(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT.resolve()), 'Input escapes campaign')
        require(path.suffix.lower() in ('.json', '.csv', '.py'), 'Only explicit CSV/JSON/source inputs allowed')
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        require(expected is None or digest == expected, 'Input hash mismatch: '+str(path))
        require(str(path) not in self.watched or self.watched[str(path)] == digest, 'Input changed while reading')
        self.watched[str(path)] = digest
        return data

    def json(self, path, expected=None):
        return json.loads(self.bytes(path, expected).decode('utf-8-sig'))

    def table(self, path, expected):
        return list(csv.DictReader(self.bytes(path, expected).decode('utf-8-sig').splitlines()))

    def stage(self, folder, expected=None):
        manifest = self.json(folder/'artifact_manifest.json', expected)
        status = self.json(folder/'status.json', manifest['files']['status.json'])
        require(status.get('state') == 'completed', 'Stage is not completed')
        return manifest['files']

    def unchanged(self):
        for name, digest in self.watched.items():
            require(sha(name) == digest, 'Display source changed: '+name)


def load_accepted(args):
    src = Inputs(); src.bytes(__file__, args.source_sha256)
    plan = src.json(PLAN, PLAN_SHA)
    require(plan['model'] == MODEL and plan['M'] == 2 and plan['budget'] == .1 and plan['policy'] == 'prefix' and plan['R'] == list(RS), 'Fixed display plan changed')
    home = ROOT/'confirmation_v1'
    freeze = src.json(home/'FREEZE.json'); fh = src.watched[str((home/'FREEZE.json').resolve())]
    require(freeze.get('passed') is True and freeze.get('frozen') is True and freeze.get('scope') == 'confirmation_v1', 'Actual formal freeze required')
    require(MODEL in freeze['run_ids'] and len(freeze['run_ids']) == 37 and freeze['R'] == list(RS), 'Wrong fixed confirmation population')
    pointer = src.json(ROOT/'confirmation_runtime/report_output_acceptance.json')
    require(Path(pointer['path']).resolve() == Path(args.report_review).resolve() and pointer['sha256'] == args.report_review_sha256, 'Actual root report acceptance differs')
    report = src.json(args.report_review, args.report_review_sha256)
    require(report.get('passed') is True and report.get('scope') == 'confirmation_report_tables_independent' and report.get('stage') == 'report_tables' and report.get('freeze_sha256') == fh, 'Actual accepted report review required')
    require(report.get('numeric_comparisons', 0) > 0, 'Report review has no numerical acceptance')
    report_dir = home/'stages/report_tables'
    require(Path(report['output_directory']).resolve() == report_dir.resolve(), 'Wrong report output')
    rfiles = src.stage(report_dir, report['audited_artifact_manifest_sha256'])
    src.json(report_dir/'summary.json', report['summary_sha256'])
    require(rfiles['summary.json'] == report['summary_sha256'], 'Report summary mismatch')
    local_review_path = home/'stages/audit_local_outcomes/review.json'
    require(Path(args.local_review).resolve() == local_review_path.resolve(), 'Only canonical local-outcomes audit allowed')
    afiles = src.stage(local_review_path.parent)
    require(afiles['review.json'] == args.local_review_sha256, 'Local review not in its completed audit manifest')
    local_review = src.json(local_review_path, args.local_review_sha256)
    require(local_review.get('passed') is True and local_review.get('stage') == 'local_outcomes' and local_review.get('scope') == 'confirmation_local_outcomes_independent' and local_review.get('freeze_sha256') == fh, 'Actual local-outcomes acceptance required')
    require(any(Path(x['path']).resolve() == local_review_path.resolve() and x['sha256'] == args.local_review_sha256 for x in report['source_acceptances']), 'Report did not use this exact local acceptance')
    local_dir = home/'stages/local_outcomes'
    lfiles = src.stage(local_dir, local_review['audited_artifact_manifest_sha256'])
    end = src.json(local_dir/'end_binding.json', lfiles['end_binding.json'])
    geometry_dir = home/'stages/local_geometry'; gfiles = src.stage(geometry_dir)
    geometry_path = (geometry_dir/'metadata.json').resolve()
    require(end['watched_sha256'][str(geometry_path)] == gfiles['metadata.json'], 'Accepted outcomes used different geometry metadata')
    geometry = src.json(geometry_path, gfiles['metadata.json'])
    require(geometry['state'] == 'completed' and geometry['expected_windows'] == 338 and geometry['complete_declared_domain'] is True and geometry['crs'] == 'EPSG:32649', 'Wrong complete confirmation domain')
    # All acceptance/identity gates above precede the first saved numerical CSV.
    budgets = src.table(report_dir/'local/spatial_budget_all.csv', rfiles['local/spatial_budget_all.csv'])
    require(len(budgets) == 2880, 'Incomplete local report matrix')
    units = src.table(local_dir/'unit_truth.csv', lfiles['unit_truth.csv'])
    display = prepare_display(units, budgets, geometry)
    src.unchanged()
    return src, plan, display, units, {'report_review': pointer, 'local_review': {'path': str(local_review_path), 'sha256': args.local_review_sha256}, 'freeze_sha256': fh}


def prepare_display(unit_rows, budget_rows, geometry):
    origin = [number(x) for x in geometry['origin_utm_m']]
    require(len(origin) == 2, 'Wrong UTM origin')
    units = []
    for raw in unit_rows:
        if integer(raw['M']) != 2:
            continue
        ix, iy = integer(raw['ix']), integer(raw['iy']); a, t = number(raw['area_m2']), number(raw['target_m2'])
        require(raw['id'] == f'M2_E{ix}_N{iy}' and 0 < a <= 4+1e-8 and 0 <= t <= a, 'Invalid saved unit')
        units.append(dict(raw=raw, id=raw['id'], ix=ix, iy=iy, area_m2=a, target_m2=t,
                          x_m=2*ix, y_m=2*iy, east_utm_m=origin[0]+2*ix, north_utm_m=origin[1]+2*iy,
                          target_pct=100*t/a, partial=a < 4-1e-8))
    lookup = {u['id']:u for u in units}
    require(units and len(lookup) == len(units), 'Missing or duplicate units')
    domain_a = math.fsum(u['area_m2'] for u in units); domain_t = math.fsum(u['target_m2'] for u in units)
    close(domain_a, geometry['area_m2'])
    cases = [r for r in budget_rows if r['run_id'] == MODEL and integer(r['M']) == 2 and number(r['budget']) == .1 and r['policy'] == 'prefix']
    require(len(cases) == 4 and {integer(r['R']) for r in cases} == set(RS), 'Exactly four fixed-R cases required')
    choices = {}
    for raw in cases:
        ids = json.loads(raw['selected_ids_json'])
        require(isinstance(ids, list) and all(isinstance(x,str) for x in ids) and len(set(ids)) == len(ids) and set(ids) <= set(lookup), 'Invalid saved selection IDs')
        a = math.fsum(lookup[i]['area_m2'] for i in ids); t = math.fsum(lookup[i]['target_m2'] for i in ids)
        require(a <= .1*domain_a, 'Saved selection exceeds original observed-area budget')
        close(a, raw['actual_area_m2']); close(t, raw['captured_target_m2']); close(domain_a, raw['domain_area_m2']); close(domain_t, raw['domain_target_m2'])
        if domain_t:
            close(100*t/domain_t, raw['capture_pct'])
        else:
            require(raw['capture_pct'] in ('', None), 'Zero target must retain NA')
        choices[integer(raw['R'])] = dict(raw=raw, ids=ids, area_m2=a, target_m2=t)
    extent = [min(u['x_m'] for u in units), max(u['x_m'] for u in units)+2,
              min(u['y_m'] for u in units), max(u['y_m'] for u in units)+2]
    return dict(units=units, choices=choices, origin_utm_m=origin, extent_m=extent,
                domain_area_m2=domain_a, domain_target_m2=domain_t, crs=geometry['crs'])


def table(path, rows):
    require(rows, 'Empty display export')
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render(display, out, synthetic=False):
    os.environ['MPLCONFIGDIR'] = str(out/'mpl_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2, 2, figsize=(174/25.4,180/25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12,right=.98,top=.95,bottom=.22,hspace=.30,wspace=.24)
    norm = Normalize(0, 100); units = display['units']; extent = display['extent_m']; mappable=None; coordinates=[]
    for ax, r in zip(axes.flat, RS):
        rects = [Rectangle((u['x_m'],u['y_m']),2,2) for u in units]
        fill = PatchCollection(rects,cmap='cividis',norm=norm,edgecolor='#a8a8a8',linewidth=.15)
        fill.set_array([u['target_pct'] for u in units]);ax.add_collection(fill);mappable=fill
        chosen = set(display['choices'][r]['ids'])
        for u in units:
            if u['partial']:
                ax.add_patch(Rectangle((u['x_m'],u['y_m']),2,2,fill=False,hatch='///',edgecolor='#858585',linewidth=.2))
            if u['id'] in chosen:
                ax.add_patch(Rectangle((u['x_m'],u['y_m']),2,2,fill=False,edgecolor='white',linewidth=1.5))
                ax.add_patch(Rectangle((u['x_m'],u['y_m']),2,2,fill=False,edgecolor='black',linewidth=.7))
            coordinates.append({k:v for k,v in u.items() if k!='raw'} | {'R':r,'selected':u['id'] in chosen})
        ax.set_xlim(extent[0]-.7,extent[1]+.7);ax.set_ylim(extent[2]-.7,extent[3]+.7);ax.set_aspect('equal',adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel('Easting from grid origin (m)');ax.set_ylabel('Northing from grid origin (m)')
        ax.set_title(f'{chr(97+RS.index(r))}  R = {r} ({256//r} px)'+(' — synthetic' if synthetic else ''),loc='left')
    bar = fig.colorbar(mappable,cax=fig.add_axes([.18,.115,.64,.025]),orientation='horizontal',ticks=[0,25,50,75,100])
    bar.set_label('Annotated target / observed unit area (%)')
    fig.legend([Patch(facecolor='none',edgecolor='black',linewidth=.7),Patch(facecolor='none',edgecolor='#858585',hatch='///')],
               ['Saved selected unit','Partially observed nominal unit'],loc='lower center',bbox_to_anchor=(.5,.012),ncol=2,frameon=False)
    fig.canvas.draw()
    from matplotlib.text import Text
    fonts=[t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    require(min(fonts)>=8,'Figure text below8pt')
    for suffix in ('png','pdf','svg'):
        fig.savefig(out/('spatial_example.'+suffix),dpi=600,facecolor='white')
    plt.close(fig)
    table(out/'plotted_unit_selection_coordinates.csv',coordinates)
    table(out/'source_selected_cases_R1246.csv',[display['choices'][r]['raw'] for r in RS])
    table(out/'source_units_M2.csv',[u['raw'] for u in units])
    caption = ('Fixed local confirmation example: ResNet18 hard segmentation, learning rate 1e-4, seed 17; '
        'prefix policy at M=2 m and nominal budget 10%, shown for all four fixed information degradations. '
        'R1/2/4/6 correspond to 256/128/64/42 effective input pixels before resizing to 256. '
        'The same saved reference annotation fraction T/A is used as background in every panel; black outlines show saved selected IDs. '
        'Axes show offsets from UTM49N grid origin '+str(tuple(display['origin_utm_m']))+' m. '
        'Squares depict nominal 2×2 m units, not the detailed observed footprint. Hatched units have observed area below 4 m² '
        '(tolerance 1e-8 m²); budget cost uses observed A only. Missing units are unobserved. '
        'The coarse region annotations are not individual plant maps; selection does not demonstrate spraying efficacy, crop injury, yield or savings.')
    (out/'caption.md').write_text(caption+'\n',encoding='utf-8')
    return {'passed_automatic_display_checks':True,'synthetic_only':synthetic,'model':MODEL,'R':list(RS),'M':2,'budget':.1,'policy':'prefix',
      'unit_count_per_panel':len(units),'coordinate_rows':len(coordinates),'same_extent_m':extent,'shared_color_range_pct':[0,100],
      'figure_width_mm':174,'figure_height_mm':180,'minimum_visible_font_pt':min(fonts),'new_model_solver_selection_calls':0,
      'manual_visual_review':'pending_root_actual_view','publication_ready':False,
      'manual_QA_required':['View complete PNG and rendered PDF at final174mm width: no clipped labels/legend or ambiguous selected outlines.',
        'Confirm all four panels, identical extent/color range, R6=42px and nominal-square/partial-footprint caption.',
        'Bind final view receipt to PNG/PDF/SVG and plotted CSV hashes; automated checks alone are not visual acceptance.']}


def execute(args):
    src, plan, display, unit_rows, evidence = load_accepted(args)
    stamp = dt.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT/'precision_agriculture'/('confirmation_spatial_example_'+stamp)
    claim = ROOT/'provenance/confirmation_spatial_example_v1.claim.json'
    with claim.open('x',encoding='utf-8') as f:
        try:
            json.dump({'output':str(out),'source_sha256':args.source_sha256,'plan_sha256':PLAN_SHA},f);f.flush()
            out.mkdir();shutil.copyfile(__file__,out/'plot_source.py');shutil.copyfile(PLAN,out/'display_plan.json')
            write(out/'input_binding.json',{'source_sha256':src.watched,'accepted_evidence':evidence})
            table(out/'source_unit_truth_all_M.csv',unit_rows)
            qa = render(display,out);src.unchanged();write(out/'QA_gates.json',qa)
            write(out/'status.json',{'state':'completed_pending_root_visual_review','no_new_selection':True,'no_new_inference':True})
            products=[p for p in out.iterdir() if p.is_file()]
            write(out/'artifact_manifest.json',{'files_sha256':{p.name:sha(p) for p in products}})
        except BaseException as error:
            failed={'state':'failed','error':repr(error),'traceback':traceback.format_exc(),'no_retry':True}
            write(out/'FAILED.json' if out.exists() else claim.with_name(claim.name+'.failure.json'),failed)
            raise
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--report-review',required=True);parser.add_argument('--report-review-sha256',required=True)
    parser.add_argument('--local-review',required=True);parser.add_argument('--local-review-sha256',required=True)
    args=parser.parse_args();require(args.execute,'Actual plot requires explicit --execute and all accepted inputs')
    print(execute(args))


if __name__=='__main__':
    main()
