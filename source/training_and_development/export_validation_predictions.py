"""Export source-linked development validation predictions, never sealed test data."""
from pathlib import Path
import argparse
import csv
import datetime
import json
import time
import torch
from torch.utils.data import DataLoader
import train_local as training

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def main(args):
    export_script_hash = training.sha(__file__)
    run_dir = ROOT / "runs" / args.run_id
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed" or config["stage"] != "development":
        raise ValueError("Only completed development runs are supported")
    if config.get("max_train", 0) or config.get("max_val", 0):
        raise ValueError("Subset runs require their exact selection subset; this exporter accepts full development manifests only")
    if (run_dir / "USE_RESTRICTION.json").exists():
        raise ValueError("Restricted diagnostic run")
    assert config["script_sha256"] == training.sha(training.__file__), "Use the matching archived model definition"
    data = Path(config["data_dir"])
    if not data.is_absolute():
        data = ROOT / data
    manifest = data / "val.json"
    assert config["val_manifest_sha256"] == training.sha(manifest)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(records) == config["n_val"], "Validation population differs from checkpoint selection"
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.16)
    checkpoint_path = run_dir / "best.pt"
    # This is a checkpoint produced locally by this campaign, not an untrusted external pickle.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["config"]["script_sha256"] == config["script_sha256"]
    regression = config["variant"] == "regression"
    model = training.Model(regression=regression, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    loader = DataLoader(training.Tiles(records), batch_size=args.batch_size, num_workers=0)
    rows, summaries = [], {}
    start = time.time()
    for factor in (1, 2, 4):
        predictions, truths = [], []
        index = 0
        for image, soft, valid, hard, target in loader:
            image, soft, valid = image.to(device), soft.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prob = model(training.degrade(image, factor, valid), valid).float().sigmoid()
                predicted = prob if regression else training.coverage(prob, valid)
            batch_estimates = predicted.cpu().tolist()
            for offset, estimate in enumerate(batch_estimates):
                source = records[index + offset]
                truth = float(target[offset])
                row = {"id": source["id"], "source_id": source["source_id"],
                       "scene_group": source["scene_group"], "split": "val",
                       "scale_factor": factor, "effective_input_pixels": 256 // factor,
                       "coverage_true": truth, "coverage_pred": estimate,
                       "absolute_error": abs(estimate-truth), "signed_error": estimate-truth,
                       "cache_valid_area": float(valid[offset].sum())}
                for name in ("global_x", "global_y", "x", "y", "latitude", "longitude", "pca_m", "east_m", "north_m"):
                    if name in source:
                        row[name] = source[name]
                if not regression:
                    binary = (prob[offset] > .5).float()
                    row.update(tp_area=float((binary*soft[offset]*valid[offset]).sum()),
                               fp_area=float((binary*(1-soft[offset])*valid[offset]).sum()),
                               fn_area=float(((1-binary)*soft[offset]*valid[offset]).sum()))
                rows.append(row)
                predictions.append(estimate)
                truths.append(truth)
            index += len(batch_estimates)
        assert index == len(records)
        summaries[str(factor)] = training.summarize(predictions, truths)
        print(json.dumps({"factor":factor,"n":index,"seconds":round(time.time()-start,1)}),flush=True)
    out = run_dir / ("validation_export_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(exist_ok=False)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (out / "predictions.json").write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
    summary = {"run_id":args.run_id,"checkpoint_epoch":checkpoint["epoch"],"device":str(device),
               "precision":"CUDA AMP" if device.type=="cuda" else "CPU float32",
               "created_at":datetime.datetime.now().astimezone().isoformat(),
               "checkpoint_sha256":training.sha(checkpoint_path),"manifest_sha256":training.sha(manifest),
               "export_script_sha256":export_script_hash,"test_accessed":False,
               "area_units":"valid/TP/FP/FN columns are effective pixels on the 256x256 cache grid, not square metres",
               "scales":summaries,"selection_balanced_mae":sum(x["balanced_mae"] for x in summaries.values())/3,
               "notes":["Validation diagnostics only; these scores can inform further development.",
                        "CPU float32 and GPU AMP may differ slightly; original selection remains recorded best_validation.json.",
                        "Source-linked records support spatial analyses; windows and seeds do not create independent fields."]}
    (out / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),"summary":summary},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--batch-size",type=int,default=4)
    main(parser.parse_args())
