"""Source-bound diagnostic figure; plots existing results without new selection."""
from pathlib import Path
import json,csv,hashlib,shutil,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'precision_agriculture'/'RM_policy_diagnostic_20260916_021512_266975'
status=json.loads((OUT/'status.json').read_text(encoding='utf-8-sig'))
assert status['state']=='completed'
hashes=json.loads((OUT/'product_hashes.json').read_text(encoding='utf-8-sig'))
source=OUT/'all_45_deltas.csv'
assert hashlib.sha256(source.read_bytes()).hexdigest()==hashes['all_45_deltas.csv']
with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==45
by={(int(r['M']),int(r['R'])):r for r in rows if float(r['budget'])==.1}
assert len(by)==9
values=np.array([[float(by[(m,r)]['delta_capture_percentage_points']) for r in (1,2,4)] for m in (1,2,4)])
fig,ax=plt.subplots(figsize=(8,6.7))
fig.subplots_adjust(left=.17,right=.85,top=.78,bottom=.22)
im=ax.imshow(values,cmap='PuOr',vmin=-10,vmax=10,aspect='auto')
ax.set_xticks(range(3),['R1','R2','R4']); ax.set_yticks(range(3),['1 m','2 m','4 m'])
ax.set_xlabel('Image degradation factor',labelpad=10)
ax.set_ylabel('Management grid side length',labelpad=12)
for i,m in enumerate((1,2,4)):
    for j,r in enumerate((1,2,4)):
        row=by[(m,r)];d=float(row['delta_capture_percentage_points'])
        if abs(d)<1e-8:d=0.
        color='white' if abs(d)>7 else '#222222'
        ax.text(j,i-.10,f'{d:+.2f} pp',ha='center',va='center',fontsize=16,color=color,fontweight='bold')
        ax.text(j,i+.18,f"{100*float(row['prefix_capture_fraction']):.2f}% → {100*float(row['pmilp_capture_fraction']):.2f}%",ha='center',va='center',fontsize=9,color=color)
ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
for spine in ax.spines.values():spine.set_visible(False)
cax=fig.add_axes([.88,.22,.025,.56]);cb=fig.colorbar(im,cax=cax);cb.set_label('Capture change (percentage points)')
fig.suptitle('Optimizing predicted area can reduce target capture',fontsize=15,y=.95)
fig.text(.5,.865,'10% nominal valid-area budget · fixed checkpoint · development validation',ha='center',fontsize=10)
fig.text(.17,.075,'Cells show prefix → predicted-area optimization; annotation is used only for evaluation.\nAll 45 budget combinations: 3 improved, 2 worsened, 40 unchanged.\nAll solutions feasible; 38/45 meet the recorded optimality tolerance.\nOne scene; target-region capture is not field control efficacy.',fontsize=8.5,va='center',linespacing=1.6)
dest=OUT/'figures';dest.mkdir(exist_ok=True)
for ext in ('png','svg','pdf'):fig.savefig(dest/f'policy_capture_change.{ext}',dpi=180,facecolor='white')
plt.close(fig)
shutil.copyfile(__file__,dest/Path(__file__).name)
provenance={'created_local':datetime.datetime.now().astimezone().isoformat(),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matrix':values.tolist(),'selected_budget':.1,
  'display_rounding_pp':2,'caveat':'Post-hoc diagnostic, original validation checkpoint, no new model or threshold selection',
  'test_accessed':False,'visual_qa':'pending'}
(dest/'figure_source.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
print(dest/'policy_capture_change.png')
