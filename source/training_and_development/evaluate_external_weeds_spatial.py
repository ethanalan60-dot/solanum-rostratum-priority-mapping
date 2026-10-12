"""Frozen-protocol relative grids. No inference, training, solver, or test access.

Phase A only receives allowlisted metadata and probability maps. Phase B opens
label arrays only through a gate that verifies the complete selection seal.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"
import numpy as np

ROOT = Path(__file__).resolve().parent
HOME = ROOT / "external_weeds_v1" / "spatial_v1"
PROTOCOL = HOME / "protocol.json"
PROTOCOL_SHA = "a96c28da40f5b2b336651af4826ff8b1476ac56233d08ef2c19e00596d9410e8"
FREEZE_SHA = "455e943e2c67a3090cbe356231e20169813bef384ac1b1c1544d42de469e3688"
RS = (1, 2, 4)
GRIDS = (4, 8, 16)
BUDGETS = (("b10", 1, 10), ("b20", 1, 5), ("b30", 3, 10), ("b50", 1, 2), ("b100", 1, 1))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def lines(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def read_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def table(path, rows):
    assert rows, path
    fields = list(rows[0])
    assert all(set(row) == set(fields) for row in rows), str(path)
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def confined(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    assert path.is_relative_to(directory), (path, directory)
    return path


def relpath(path):
    return confined(path, ROOT).relative_to(ROOT).as_posix()


def checked(path, digest, watched):
    path = confined(path, ROOT)
    assert sha(path) == digest, str(path)
    key = relpath(path)
    assert key not in watched or watched[key] == digest
    watched[key] = digest
    return path


def recheck(watched):
    for path, digest in watched.items():
        assert sha(ROOT / path) == digest, path


def budget_rows(g):
    n = g * g
    rows = []
    for basis in ("nominal_primary", "matched_actual_secondary"):
        for code, num, den in BUDGETS:
            nominal = num / den
            if basis == "nominal_primary":
                k = n * num // den
                assert k * den <= n * num
                assert k == n or (k + 1) * den > n * num
                effective = nominal
            else:
                coarse = 16 * num // den
                assert n * coarse % 16 == 0
                k = n * coarse // 16
                effective = coarse / 16
            rows.append(dict(budget_basis=basis, parent_nominal_budget_code=code,
                             parent_nominal_budget_fraction=nominal,
                             effective_budget_fraction=effective, k=k, N=n,
                             actual_fraction=k / n, unused_effective_fraction=effective-k/n))
    return rows


def aggregate_plane(array, frame_id, g):
    array = np.asarray(array)
    assert array.ndim == 2 and array.shape[0] == array.shape[1]
    side = array.shape[0]
    assert side % g == 0 and frame_id.isascii()
    assert np.isfinite(array).all() and np.all((array >= 0) & (array <= 1))
    s = side // g
    values = {}
    for r in range(g):
        for c in range(g):
            identifier = f"{frame_id}:g{g:02d}:r{r:02d}:c{c:02d}"
            cell = array[r*s:(r+1)*s, c*s:(c+1)*s]
            values[identifier] = math.fsum(float(v) for v in cell.ravel(order="C")) / (side*side)
    total = math.fsum(float(v) for v in array.ravel(order="C")) / (side*side)
    assert abs(math.fsum(values.values())-total) <= 1e-12
    return values


def rank_units(units):
    """Deliberately narrow selection API: extra/label keys are rejected."""
    assert units and all(set(x) == {"id", "A", "P"} for x in units)
    assert len({x["id"] for x in units}) == len(units)
    n = len(units)
    for item in units:
        assert item["id"].isascii() and item["A"] == 1/n
        assert math.isfinite(item["P"]) and 0 <= item["P"] <= item["A"]
    return sorted(units, key=lambda row: (-row["P"], row["id"]))


def ranking_record(meta, frame, factor, g, prediction):
    values = aggregate_plane(prediction, frame["id"], g)
    units = [{"id": identifier, "A": 1/(g*g), "P": value} for identifier, value in sorted(values.items())]
    order = rank_units(units)
    key = f'{meta["run_id"]}|R{factor}|{frame["id"]}|g{g:02d}'
    ranking = dict(meta, R=factor, grid_g=g, frame_id=frame["id"],
                   acquisition_date=frame["acquisition_date"], ranking_key=key,
                   ordered_ids=[x["id"] for x in order], units=units)
    choices, seen = [], {}
    for budget in budget_rows(g):
        k = budget["k"]
        ids = [x["id"] for x in order[:k]]
        canonical = f"{key}|k{k:03d}"
        choice_key = f'{key}|{budget["budget_basis"]}|{budget["parent_nominal_budget_code"]}'
        p = math.fsum(values[identifier] for identifier in sorted(ids))
        item = dict(meta, R=factor, grid_g=g, frame_id=frame["id"],
                    acquisition_date=frame["acquisition_date"], ranking_key=key,
                    choice_key=choice_key, selection_key=canonical,
                    duplicate_selection_alias=seen.get(k), **budget,
                    selected_ids=ids, selected_area_FAE=k/(g*g), selected_predicted_weed_FAE=p)
        assert item["selected_area_FAE"] <= item["effective_budget_fraction"]
        choices.append(item)
        seen.setdefault(k, choice_key)
    return ranking, choices


def ratio(num, den, reason):
    assert math.isfinite(num) and math.isfinite(den) and den >= 0
    return (None, reason) if den == 0 else (num/den, "")


def measurements(area, prediction, weed, crop, background, total_weed, total_crop, frame_n, weed_eligible, crop_eligible):
    wc, wr = ratio(weed, total_weed, "zero_total_weed")
    cc, cr = ratio(crop, total_crop, "zero_total_crop")
    comp, cpr = ratio(crop, area, "zero_selected_area")
    return dict(selected_area_FAE=area, selected_predicted_weed_FAE=prediction,
                selected_weed_FAE=weed, total_weed_FAE=total_weed,
                unselected_weed_FAE=total_weed-weed, selected_crop_FAE=crop,
                total_crop_FAE=total_crop, selected_background_FAE=background,
                weed_capture_ratio=wc, weed_capture_na_reason=wr,
                crop_capture_ratio=cc, crop_capture_na_reason=cr,
                crop_fraction_of_selected_area=comp, crop_composition_na_reason=cpr,
                frame_n=frame_n, weed_eligible_frame_n=weed_eligible,
                crop_eligible_frame_n=crop_eligible,
                zero_weed_frame_n=frame_n-weed_eligible, zero_crop_frame_n=frame_n-crop_eligible)


def step_auc(curve, upper):
    assert upper in (1, .5) and len(curve) >= 2
    n = len(curve)-1
    assert [x["k"] for x in curve] == list(range(n+1))
    values = [x["weed_capture_ratio"] for x in curve]
    if values[0] is None:
        assert all(v is None for v in values)
        return None
    assert all(v is not None and math.isfinite(v) for v in values)
    return math.fsum(values[k]*max(0.0, min((k+1)/n, upper)-k/n) for k in range(n)) / upper


def verify_seal(out, expected_sha=None):
    path = out / "selection_seal.json"
    if expected_sha is not None:
        assert sha(path) == expected_sha
    seal = read(path)
    assert seal["phase"] == "all_predictions_sealed_before_labels"
    for relative, digest in seal["file_sha256"].items():
        assert sha(confined(out/relative, out)) == digest
    return seal


def check_nested_truth_cells(truth, frame_ids):
    count, maximum = 0, 0.0
    for frame in frame_ids:
        for g, finer_g in ((4, 8), (8, 16)):
            coarse, finer = truth[(frame, g)], truth[(frame, finer_g)]
            assert len(coarse) == g*g and len(finer) == finer_g*finer_g
            for r in range(g):
                for c in range(g):
                    parent = f"{frame}:g{g:02d}:r{r:02d}:c{c:02d}"
                    children = [f"{frame}:g{finer_g:02d}:r{2*r+dr:02d}:c{2*c+dc:02d}" for dr in (0,1) for dc in (0,1)]
                    assert parent in coarse and all(child in finer for child in children)
                    for channel in ("B", "C", "W"):
                        residual = abs(coarse[parent][channel]-math.fsum(finer[child][channel] for child in sorted(children)))
                        assert residual <= 1e-12, (frame, parent, channel, residual)
                        count += 1
                        maximum = max(maximum, residual)
    assert count == len(frame_ids)*(16+64)*3
    return dict(scalar_comparisons=count, max_absolute_residual_FAE=maximum,
                absolute_tolerance_FAE=1e-12, comparisons="each g4 parent vs four g8 children, each g8 parent vs four g16 children, separately B/C/W")


def load_truth_after_seal(out, seal_sha, frames, watched):
    verify_seal(out, seal_sha)
    recheck(watched)
    truth = {}
    validation = []
    for frame in frames:
        path = checked(frame["cache"], frame["cache_sha256"], watched)
        with np.load(path, allow_pickle=False) as z:
            soft3, soft, valid = z["soft3"], z["soft"], z["valid"]
            native_counts = z["native_class_counts"]
        assert soft3.dtype == np.float32 and soft3.shape == (3, 256, 256)
        assert soft.dtype == np.float32 and soft.shape == (256, 256)
        assert valid.dtype == np.float32 and valid.shape == (256, 256) and np.all(valid == 1)
        assert np.isfinite(soft3).all() and np.all((soft3 >= 0) & (soft3 <= 1))
        assert np.max(np.abs(soft3.astype(np.float64).sum(axis=0)-1)) <= 1e-6
        assert np.array_equal(soft, soft3[2])
        assert native_counts.dtype == np.uint32 and native_counts.shape == (3,)
        assert sum(int(v) for v in native_counts) == 600*600
        class_error = [abs(math.fsum(float(v) for v in soft3[i].ravel(order="C"))/(256*256)-int(native_counts[i])/(600*600)) for i in range(3)]
        assert max(class_error) <= 1e-7
        validation.append(dict(frame_id=frame["id"], cache_sha256=frame["cache_sha256"],
                               native_to_cache_class_fraction_abs_error=class_error,
                               sum_soft3_max_abs_error=float(np.max(np.abs(soft3.astype(np.float64).sum(axis=0)-1))),
                               spatial_denominator="soft3 sum; native counts used for conservation check only"))
        reference = None
        for g in GRIDS:
            vals = [aggregate_plane(soft3[i], frame["id"], g) for i in range(3)]
            cells = {identifier: dict(A=1/(g*g), B=vals[0][identifier],
                                      C=vals[1][identifier], W=vals[2][identifier]) for identifier in vals[0]}
            for row in cells.values():
                assert abs(math.fsum((row["B"], row["C"], row["W"]))-row["A"]) <= 1e-6*row["A"]
            totals = {k: math.fsum(cells[i][k] for i in sorted(cells)) for k in ("B", "C", "W")}
            if reference is not None:
                assert max(abs(totals[k]-reference[k]) for k in totals) <= 1e-12
            reference = totals
            truth[(frame["id"], g)] = cells
    nested = check_nested_truth_cells(truth, [frame["id"] for frame in frames])
    dump(out/"truth_validation_after_seal.json", dict(selection_seal_sha256=seal_sha, frames=validation,
                                                       nested_cell_conservation=nested))
    return truth


def make_frame_curve(ranking, truth, policy="learned", random_expected=False):
    cells = truth[(ranking["frame_id"], ranking["grid_g"])]
    ids = ranking["ordered_ids"]
    assert len(ids) == len(cells) and set(ids) == set(cells)
    n = len(cells)
    p = {x["id"]: x["P"] for x in ranking["units"]}
    total_w = math.fsum(cells[i]["W"] for i in sorted(cells))
    total_c = math.fsum(cells[i]["C"] for i in sorted(cells))
    total_b = math.fsum(cells[i]["B"] for i in sorted(cells))
    meta = {k: v for k, v in ranking.items() if k not in ("ordered_ids", "units", "ranking_key")}
    meta.update(policy=policy, scope="frame", scope_id=ranking["frame_id"])
    rows = []
    for k in range(n+1):
        selected = sorted(ids[:k])
        if random_expected:
            w, c, b, pred = (k/n)*total_w, (k/n)*total_c, (k/n)*total_b, None
        else:
            w, c, b = (math.fsum(cells[i][key] for i in selected) for key in ("W", "C", "B"))
            pred = math.fsum(p[i] for i in selected) if policy == "learned" else None
        rows.append(dict(meta, k=k, N=n, actual_fraction=k/n,
                         **measurements(k/n, pred, w, c, b, total_w, total_c, 1, int(total_w>0), int(total_c>0))))
    return rows


def group_curves(frame_curves, scope, group_id):
    assert scope in ("date", "pooled") and frame_curves
    selected = sorted(frame_curves, key=lambda cur: cur[0]["frame_id"])
    n = len(selected[0])-1
    assert all(len(cur) == n+1 for cur in selected)
    meta = dict(selected[0][0])
    meta.update(scope=scope, scope_id=group_id, frame_id="", acquisition_date=group_id if scope == "date" else "")
    out = []
    for k in range(n+1):
        rr = [cur[k] for cur in selected]
        values = {key: math.fsum(x[key] for x in rr) for key in (
            "selected_area_FAE", "selected_weed_FAE", "selected_crop_FAE", "selected_background_FAE", "total_weed_FAE", "total_crop_FAE")}
        pred = math.fsum(x["selected_predicted_weed_FAE"] for x in rr) if rr[0]["selected_predicted_weed_FAE"] is not None else None
        row = dict(meta)
        row.update(k=k, N=n, actual_fraction=k/n)
        row.update(measurements(values["selected_area_FAE"], pred, values["selected_weed_FAE"],
                                values["selected_crop_FAE"], values["selected_background_FAE"],
                                values["total_weed_FAE"], values["total_crop_FAE"], len(rr),
                                sum(x["weed_eligible_frame_n"] for x in rr), sum(x["crop_eligible_frame_n"] for x in rr)))
        out.append(row)
    return out


def fixed_rows(curve):
    out, seen = [], {}
    for budget in budget_rows(curve[0]["grid_g"]):
        k = budget["k"]
        row = dict(curve[k])
        row.update(budget)
        scope_key = row["frame_id"] if row["scope"] == "frame" else row["scope"]+"|"+row["scope_id"]
        base = f'{row["run_id"]}|R{row["R"]}|{scope_key}|g{row["grid_g"]:02d}'
        key = f"{base}|k{k:03d}"
        alias_key = base + "|" + budget["budget_basis"] + "|" + budget["parent_nominal_budget_code"]
        row.update(selection_key=key, choice_key=alias_key, duplicate_selection_alias=seen.get(k),
                   unused_effective_budget_area_FAE=budget["unused_effective_fraction"]*row["frame_n"])
        seen.setdefault(k, alias_key)
        out.append(row)
    return out


def auc_row(curve):
    keep = ("run_id", "variant", "seed", "original_selected_epoch", "checkpoint_sha256", "config_sha256",
            "export_sha256", "val_manifest_sha256", "R", "grid_g", "policy", "scope", "scope_id", "frame_id", "acquisition_date")
    row = {key: curve[0][key] for key in keep}
    row.update(prefix_points=len(curve), frame_n=curve[0]["frame_n"],
               weed_eligible_frame_n=curve[0]["weed_eligible_frame_n"],
               total_weed_FAE=curve[0]["total_weed_FAE"],
               auc_0_1=step_auc(curve, 1), auc_0_0p5_normalized=step_auc(curve, .5),
               auc_na_reason=curve[0]["weed_capture_na_reason"])
    if row["policy"] == "random_expectation" and row["auc_0_1"] is not None:
        n = len(curve)-1
        assert abs(row["auc_0_1"]-(n-1)/(2*n)) <= 1e-12
        assert abs(row["auc_0_0p5_normalized"]-(.25-1/(2*n))) <= 1e-12
    return row


def paired_rows(rows):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in ("seed", "R", "grid_g", "scope", "scope_id", "budget_basis", "parent_nominal_budget_code"))
        assert row["variant"] in ("hard_seg", "soft_seg")
        group = index.setdefault(key, {})
        assert row["variant"] not in group
        group[row["variant"]] = row
    result = []
    for key, both in sorted(index.items()):
        assert set(both) == {"hard_seg", "soft_seg"}
        h, s = both["hard_seg"], both["soft_seg"]
        assert h["selected_area_FAE"] == s["selected_area_FAE"]
        keep = ("seed", "R", "grid_g", "scope", "scope_id", "budget_basis", "parent_nominal_budget_code", "effective_budget_fraction", "k", "N", "frame_n", "selected_area_FAE")
        row = {k: h[k] for k in keep}
        row.update(hard_run_id=h["run_id"], soft_run_id=s["run_id"],
                   hard_selection_key=h["selection_key"], soft_selection_key=s["selection_key"],
                   duplicate_budget_selection=h["duplicate_selection_alias"] is not None)
        for metric in ("weed_capture_ratio", "crop_capture_ratio", "crop_fraction_of_selected_area"):
            row["hard_"+metric] = h[metric]
            row["soft_"+metric] = s[metric]
            row["soft_minus_hard_"+metric+"_pp"] = None if h[metric] is None or s[metric] is None else 100*(s[metric]-h[metric])
        result.append(row)
    return result


def paired_auc_rows(rows):
    index = {}
    keys = ("seed", "R", "grid_g", "scope", "scope_id")
    for row in rows:
        group = index.setdefault(tuple(row[k] for k in keys), {})
        assert row["variant"] not in group
        group[row["variant"]] = row
    result = []
    for key, both in sorted(index.items()):
        assert set(both) == {"hard_seg", "soft_seg"}
        h, soft = both["hard_seg"], both["soft_seg"]
        row = {k: h[k] for k in keys}
        row.update(hard_run_id=h["run_id"], soft_run_id=soft["run_id"],
                   budget_basis="complete_prefix_once", frame_n=h["frame_n"])
        for metric in ("auc_0_1", "auc_0_0p5_normalized"):
            row["hard_"+metric] = h[metric]
            row["soft_"+metric] = soft[metric]
            row["soft_minus_hard_"+metric] = None if h[metric] is None or soft[metric] is None else soft[metric]-h[metric]
        result.append(row)
    return result


def read_allowed_inputs(acceptance_path, acceptance_sha, authorization_path=None, authorization_sha=None):
    """No exporter inputs/summary/scalar or per-model metadata JSON is parsed."""
    watched = {}
    checked(PROTOCOL, PROTOCOL_SHA, watched)
    checked(HOME/"freeze_receipt.json", FREEZE_SHA, watched)
    protocol = read(PROTOCOL)
    checked(acceptance_path, acceptance_sha, watched)
    accepted = read(acceptance_path)
    assert accepted["passed"] is True and accepted["valid_all_ones_attested"] is True
    if authorization_path is not None:
        checked(authorization_path, authorization_sha, watched)
        auth = read(authorization_path)
        assert auth["passed"] is True and auth["production_execution_authorized"] is True
        assert auth["protocol_sha256"] == PROTOCOL_SHA and auth["source_sha256"] == sha(__file__)
        assert auth["acceptance_sha256"] == acceptance_sha
    else:
        assert authorization_sha is None  # metadata-only readiness; execute() requires a separate authorization
    checked(__file__, sha(__file__), watched)
    for path, digest in protocol["basis_source_sha256"].items():
        checked(ROOT/path, digest, watched)  # hash bytes only; do not parse annotation metadata
    frames = protocol["dataset"]["frame_metadata"]
    ids = protocol["dataset"]["frame_ids_in_official_order"]
    assert [x["id"] for x in frames] == ids == accepted["val_ids"] and len(ids) == 26
    assert accepted["val_manifest_sha256"] == protocol["basis_source_sha256"]["weedsgalore_data_v1/val.json"]
    export = confined(accepted["export_root"], ROOT/"external_exports")
    checked(export/"artifact_manifest.json", accepted["artifact_manifest_sha256"], watched)
    manifest = read(export/"artifact_manifest.json")
    for relative, digest in manifest["file_sha256"].items():
        checked(confined(export/relative, export), digest, watched)
    status = read(export/"status.json")
    assert status["state"] == "completed" and status["map_count"] == 156 and status["scalar_rows"] == 702 and status["run_count"] == 9
    assert not status["test_accessed"] and not status["R6_accessed"] and not status["GPU_started"]
    checked(export/"map_index.json", accepted["map_index_sha256"], watched)
    checked(ROOT/"export_external_weeds_validation.py", accepted["exporter_sha256"], watched)
    checked(accepted["training_audit"]["path"], accepted["training_audit"]["sha256"], watched)
    maps = read(export/"map_index.json")
    assert len(maps) == 156
    declared = {x["run_id"]: x for x in protocol["eligible_models"]}
    assert len(accepted["segmentation_models"]) == 6 and {x["run_id"] for x in accepted["segmentation_models"]} == set(declared)
    metadata = {}
    for entry in accepted["segmentation_models"]:
        run = entry["run_id"]
        assert entry["completed24"] is True
        assert (entry["encoder"], entry["variant"], entry["seed"]) == ("resnet18", declared[run]["variant"], declared[run]["seed"])
        assert isinstance(entry["best_epoch"], int) and 1 <= entry["best_epoch"] <= 24
        run_dir = ROOT/"external_weeds_v1"/"runs"/run
        assert confined(entry["config_path"], run_dir) == (run_dir/"config.json").resolve()
        assert confined(entry["checkpoint_path"], run_dir) == (run_dir/"best.pt").resolve()
        checked(entry["config_path"], entry["config_sha256"], watched)
        checked(entry["checkpoint_path"], entry["checkpoint_sha256"], watched)
        metadata[run] = dict(run_id=run, variant=entry["variant"], seed=entry["seed"],
                             original_selected_epoch=entry["best_epoch"], checkpoint_sha256=entry["checkpoint_sha256"],
                             config_sha256=entry["config_sha256"], export_sha256=accepted["artifact_manifest_sha256"],
                             val_manifest_sha256=accepted["val_manifest_sha256"])
    expected = {(run, identifier) for run in declared for identifier in ids}
    assert {(x["run_id"], x["id"]) for x in maps} == expected
    by_frame = {x["id"]: x for x in frames}
    for item in maps:
        assert item["R"] == list(RS) and item["shape"] == [3, 256, 256] and item["dtype"] == "float32"
        assert item["source_cache_sha256"] == by_frame[item["id"]]["cache_sha256"]
        path = confined(export/item["path"], export/item["run_id"]/"maps")
        assert path.name == item["id"]+".npz"
        checked(path, item["sha256"], watched)
    return protocol, frames, metadata, maps, export, watched


def phase_a(out, protocol, frames, metadata, maps, export, watched):
    rankings, choices, unit_rows = [], [], []
    lookup = {(x["run_id"], x["id"]): x for x in maps}
    for model in protocol["eligible_models"]:
        run = model["run_id"]
        for frame in frames:
            item = lookup[(run, frame["id"])]
            path = checked(export/item["path"], item["sha256"], watched)
            with np.load(path, allow_pickle=False) as z:
                assert set(z.files) == {"probabilities", "R", "id"}
                probability = z["probabilities"]
                assert np.array_equal(z["R"], np.array(RS, dtype=np.int64)) and str(z["id"]) == frame["id"]
            assert probability.dtype == np.float32 and probability.shape == (len(RS), 256, 256)
            assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
            for r_index, factor in enumerate(RS):
                for g in GRIDS:
                    ranking, selected = ranking_record(metadata[run], frame, factor, g, probability[r_index])
                    rankings.append(ranking)
                    choices.extend(selected)
                    for unit in ranking["units"]:
                        unit_rows.append(dict(run_id=run, R=factor, frame_id=frame["id"], grid_g=g, **unit))
    counts = protocol["planned_counts_not_observed_results"]
    assert len(rankings) == counts["learned_complete_rankings"]
    assert len(choices) == counts["learned_frame_budget_rows"]
    unique_choices = len({x["selection_key"] for x in choices})
    assert unique_choices == counts["unique_learned_frame_model_R_grid_k_choices"]
    assert len(unit_rows) == counts["prediction_unit_rows"]
    recheck(watched)
    table(out/"prediction_units.csv", unit_rows)
    lines(out/"learned_rankings.jsonl", rankings)
    lines(out/"learned_fixed_budget_choices.jsonl", choices)
    files = ["prediction_units.csv", "learned_rankings.jsonl", "learned_fixed_budget_choices.jsonl", "input_binding.json"]
    files += [p.relative_to(out).as_posix() for p in (out/"source").iterdir() if p.is_file()]
    dump(out/"selection_seal.json", dict(phase="all_predictions_sealed_before_labels", created_utc=now(),
         rankings=len(rankings), budget_rows=len(choices), unique_choices=unique_choices,
         file_sha256={name: sha(out/name) for name in files}, source_input_sha256=watched,
         label_arrays_opened=False))
    return sha(out/"selection_seal.json")


def reference_rankings(frames, truth, g, policy):
    rankings = []
    for frame in frames:
        cells = truth[(frame["id"], g)]
        order = sorted(cells, key=lambda i: (-cells[i]["W"], i)) if policy == "label_oracle" else sorted(cells)
        meta = dict(run_id=policy, variant="reference", seed=None, original_selected_epoch=None,
                    checkpoint_sha256="", config_sha256="", export_sha256="",
                    val_manifest_sha256=read(PROTOCOL)["basis_source_sha256"]["weedsgalore_data_v1/val.json"],
                    R=None, grid_g=g, frame_id=frame["id"], acquisition_date=frame["acquisition_date"],
                    ranking_key=f'{policy}|{frame["id"]}|g{g:02d}', ordered_ids=order,
                    units=[dict(id=i, A=cells[i]["A"], P=0.0) for i in sorted(cells)])
        rankings.append(meta)
    return rankings


def save_curve_family(out, stem, frame_curves):
    by_group = {}
    for curve in frame_curves:
        first = curve[0]
        key = (first["run_id"], first["R"], first["grid_g"])
        by_group.setdefault(key, []).append(curve)
    pooled, dates = [], []
    for _, group in by_group.items():
        pooled.append(group_curves(group, "pooled", f"all{len(group)}"))
        for date in sorted({x[0]["acquisition_date"] for x in group}):
            dates.append(group_curves([x for x in group if x[0]["acquisition_date"] == date], "date", date))
    result = {}
    for scope, curves in [("frame", frame_curves), ("date", dates), ("pooled", pooled)]:
        curve_rows = [x for curve in curves for x in curve]
        fixed = [x for curve in curves for x in fixed_rows(curve)]
        auc = [auc_row(curve) for curve in curves]
        table(out/f"{stem}_{scope}_complete_prefix_curves.csv", curve_rows)
        table(out/f"{stem}_{scope}_budget_metrics.csv", fixed)
        table(out/f"{stem}_{scope}_auc.csv", auc)
        result[scope] = dict(curves=curves, budget_rows=fixed, auc_rows=auc, curve_points=len(curve_rows))
    return result


def phase_b(out, seal_sha, protocol, frames, watched):
    truth = load_truth_after_seal(out, seal_sha, frames, watched)
    truth_rows = []
    for (frame, g), cells in sorted(truth.items()):
        for identifier, values in sorted(cells.items()):
            truth_rows.append(dict(frame_id=frame, grid_g=g, id=identifier, **values))
    assert len(truth_rows) == protocol["planned_counts_not_observed_results"]["unique_truth_unit_rows_after_selection_seal"]
    table(out/"truth_units.csv", truth_rows)
    # Deliberately reload the exact sealed bytes instead of reusing mutable phase-A objects.
    rankings = read_lines(out/"learned_rankings.jsonl")
    choices = read_lines(out/"learned_fixed_budget_choices.jsonl")
    curves = [make_frame_curve(r, truth) for r in rankings]
    curve_index = {(c[0]["run_id"], c[0]["R"], c[0]["frame_id"], c[0]["grid_g"]): c for c in curves}
    ranking_index = {r["ranking_key"]: r for r in rankings}
    for choice in choices:
        rank = ranking_index[choice["ranking_key"]]
        assert choice["selected_ids"] == rank["ordered_ids"][:choice["k"]]
        row = curve_index[(choice["run_id"], choice["R"], choice["frame_id"], choice["grid_g"])][choice["k"]]
        assert row["selected_area_FAE"] == choice["selected_area_FAE"]
        assert row["selected_predicted_weed_FAE"] == choice["selected_predicted_weed_FAE"]
    learned = save_curve_family(out, "learned", curves)
    fixed_index = {row["choice_key"]: row for row in learned["frame"]["budget_rows"]}
    assert len(fixed_index) == len(choices)
    for choice in choices:
        row = fixed_index[choice["choice_key"]]
        for key in ("selection_key", "duplicate_selection_alias", "selected_area_FAE", "selected_predicted_weed_FAE"):
            assert row[key] == choice[key], key
    counts = protocol["planned_counts_not_observed_results"]
    for scope in ("frame", "date", "pooled"):
        result = learned[scope]
        assert len(result["budget_rows"]) == counts[f"learned_{scope}_budget_rows"]
        assert len(result["auc_rows"]) == counts[f"learned_{scope}_auc_rows_two_auc_columns"]
        assert result["curve_points"] == counts[f"learned_{scope}_complete_prefix_points_including_zero"]
        paired = paired_rows(result["budget_rows"])
        assert len(paired) == counts[f"hard_soft_paired_{scope}_budget_rows"]
        table(out/f"hard_soft_paired_{scope}_budget_metrics.csv", paired)
        paired_auc = paired_auc_rows(result["auc_rows"])
        assert len(paired_auc)*2 == len(result["auc_rows"])
        table(out/f"hard_soft_paired_{scope}_auc.csv", paired_auc)
    refs = {}
    for policy in ("label_oracle", "random_expectation"):
        ranks = [r for g in GRIDS for r in reference_rankings(frames, truth, g, policy)]
        if policy == "label_oracle":
            lines(out/"label_oracle_rankings_after_seal.jsonl", ranks)
        ref_curves = [make_frame_curve(r, truth, policy, policy == "random_expectation") for r in ranks]
        refs[policy] = save_curve_family(out, policy, ref_curves)
    oracle = {(c[0]["frame_id"], c[0]["grid_g"]): c for c in refs["label_oracle"]["frame"]["curves"]}
    nested_checks = []
    for frame in frames:
        for code, num, den in BUDGETS:
            cnum = 16*num//den
            values = {g: oracle[(frame["id"], g)][g*g*cnum//16]["selected_weed_FAE"] for g in GRIDS}
            assert values[8] >= values[4]-1e-12 and values[16] >= values[8]-1e-12
            nested_checks.append(dict(scope="frame", frame_id=frame["id"], parent_nominal_budget_code=code,
                                      matched_fraction=cnum/16, g4_W=values[4], g8_W=values[8], g16_W=values[16]))
    pooled_oracle = {c[0]["grid_g"]: c for c in refs["label_oracle"]["pooled"]["curves"]}
    for code, num, den in BUDGETS:
        cnum = 16*num//den
        values = {g: pooled_oracle[g][g*g*cnum//16]["selected_weed_FAE"] for g in GRIDS}
        assert values[8] >= values[4]-1e-12 and values[16] >= values[8]-1e-12
        nested_checks.append(dict(scope="pooled", frame_id="", parent_nominal_budget_code=code,
                                  matched_fraction=cnum/16,g4_W=values[4],g8_W=values[8],g16_W=values[16]))
    table(out/"nested_oracle_matched_actual_checks.csv", nested_checks)
    for curve in curves:
        reference = oracle[(curve[0]["frame_id"], curve[0]["grid_g"])]
        assert all(reference[k]["selected_weed_FAE"] >= row["selected_weed_FAE"]-1e-12 for k, row in enumerate(curve))
    random_canonical = []
    for g in GRIDS:
        for budget in budget_rows(g):
            random_canonical.append(dict(grid_g=g, **budget,
                expected_capture_if_denominator_positive=budget["k"]/(g*g),
                auc_0_1_if_denominator_positive=(g*g-1)/(2*g*g),
                auc_0_0p5_normalized_if_denominator_positive=.25-1/(2*g*g), Monte_Carlo_draws=0))
    table(out/"random_grid_budget_expectations.csv", random_canonical)
    verify_seal(out, seal_sha)
    recheck(watched)
    return {"phase": "completed_after_seal", "truth_unit_rows": len(truth_rows),
            "learned": {scope: {"fixed_budget_rows": len(v["budget_rows"]), "auc_rows": len(v["auc_rows"]), "complete_prefix_points": v["curve_points"]} for scope, v in learned.items()},
            "oracle_frame_rankings": len(frames)*len(GRIDS), "nested_oracle_checks": len(nested_checks),
            "two_budget_bases_share_rankings": True, "original_checkpoints_unchanged": True,
            "test_accessed": False, "R6_accessed": False, "solver_or_model_inference": False}


def execute(args):
    assert args.authorization and args.authorization_sha256
    protocol, frames, metadata, maps, export, watched = read_allowed_inputs(
        args.acceptance, args.acceptance_sha256, args.authorization, args.authorization_sha256)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = HOME / ("results_"+stamp)
    claim = HOME/"SPATIAL_EXECUTION_CLAIM.json"
    dump(claim, dict(output=relpath(out), pid=os.getpid(), started_utc=now(), source_sha256=sha(__file__),
                     protocol_sha256=PROTOCOL_SHA, acceptance_sha256=args.acceptance_sha256,
                     failure_policy="permanent claim; no automatic retry or overwrite"))
    output_owned = False
    try:
        out.mkdir(exist_ok=False)
        output_owned = True
        for name in ("TEMP", "TMP"):
            os.environ[name] = str(out/"temp")
        (out/"temp").mkdir()
        source = out/"source"
        source.mkdir()
        for src, name in [(Path(__file__), "evaluator.py"), (PROTOCOL, "protocol.json"),
                          (HOME/"freeze_receipt.json", "freeze_receipt.json"),
                          (Path(args.acceptance), "acceptance.json"), (Path(args.authorization), "authorization.json"),
                          (export/"map_index.json", "map_index.json"), (export/"artifact_manifest.json", "export_artifact_manifest.json")]:
            shutil.copy2(src, source/name)
        dump(out/"input_binding.json", dict(source_input_sha256=watched, models=metadata,
             python=sys.version, numpy=np.__version__, argv=sys.argv, pid=os.getpid(),
             protocol_sha256=PROTOCOL_SHA, acceptance_sha256=args.acceptance_sha256,
             CPU_only=True, label_arrays_opened=False))
        dump(out/"phase_a_started.json", dict(created_utc=now(), state="prediction_only"))
        seal_sha = phase_a(out, protocol, frames, metadata, maps, export, watched)
        dump(out/"phase_b_started.json", dict(created_utc=now(), selection_seal_sha256=seal_sha))
        summary = phase_b(out, seal_sha, protocol, frames, watched)
        dump(out/"summary.json", summary)
        dump(out/"status.json", dict(state="completed", completed_utc=now(), source_sha256=sha(__file__),
             protocol_sha256=PROTOCOL_SHA, selection_seal_sha256=seal_sha,
             acceptance_sha256=args.acceptance_sha256, **summary))
    except BaseException as error:
        failure = dict(state="failed", failed_utc=now(), error=repr(error), traceback=traceback.format_exc(),
                       no_retry=True, source_sha256=sha(__file__), protocol_sha256=PROTOCOL_SHA,
                       claim_path=relpath(claim), attempted_output=relpath(out), output_owned=output_owned)
        try:
            if not output_owned:
                raise OSError("Output directory was not created by this attempt")
            dump(out/"failure.json", failure)
            if not (out/"status.json").exists():
                dump(out/"status.json", failure)
        except BaseException as receipt_error:
            # The permanent claim remains. Do not write into a pre-existing output.
            failure["output_receipt_error"] = repr(receipt_error)
            dump(HOME/("execution_failure_"+stamp+".json"), failure)
        raise
    finally:
        if output_owned:
            dump(out/"artifact_manifest.json", dict(created_utc=now(),
                 file_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"},
                 source_sha256=sha(__file__), protocol_sha256=PROTOCOL_SHA))
    print(json.dumps({"state": "completed", "output": str(out)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    for key in ("acceptance", "acceptance-sha256", "authorization", "authorization-sha256"):
        parser.add_argument("--"+key)
    args = parser.parse_args()
    if not args.execute or not all(getattr(args, key) for key in ("acceptance", "acceptance_sha256", "authorization", "authorization_sha256")):
        parser.error("Execution requires explicit --execute and hash-bound accepted export + source authorization; use the separate toy checker for readiness.")
    execute(args)


if __name__ == "__main__":
    main()
