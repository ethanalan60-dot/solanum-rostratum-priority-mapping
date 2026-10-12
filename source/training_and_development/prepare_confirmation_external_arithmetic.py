"""Extract exact pure external grid arithmetic; never import old production entrypoints."""
from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
path=root/'evaluate_external_weeds_spatial.py'
text=path.read_text(encoding='utf-8')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(path.read_bytes())=='af13294cdff8e5da059cb4f8f3bec4cb4b0c2f7c24c05226278724a8a79e1810'
names=['budget_rows','aggregate_plane','rank_units','ranking_record','ratio','measurements','step_auc','check_nested_truth_cells','make_frame_curve','group_curves','fixed_rows','paired_rows','paired_auc_rows']
nodes={node.name:node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)}
parts=[ast.get_source_segment(text,nodes[name]) for name in names]
header='"""Byte-exact pure grid and outcome function bodies from the accepted development evaluator. No file readers or entrypoints."""\nimport math\nimport numpy as np\nGRIDS=(4,8,16)\nBUDGETS=(("b10",1,10),("b20",1,5),("b30",3,10),("b50",1,2),("b100",1,1))\n\n'
out=root/'confirmation_v1/external_arithmetic.py'
with out.open('x',encoding='utf-8',newline='\n') as f:f.write(header+'\n\n'.join(parts)+'\n')
audit={'original_path':str(path),'original_sha256':sha(path.read_bytes()),'output_sha256':sha(out.read_bytes()),'function_bodies':{name:sha(part.encode()) for name,part in zip(names,parts)},'changes':'None within function bodies. Header only imports math/numpy and original GRIDS/BUDGETS. No source reader, training, model, solver, protocol, or production entrypoint extracted.'}
with (out.parent/'external_arithmetic_extraction.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2)
print(out)

