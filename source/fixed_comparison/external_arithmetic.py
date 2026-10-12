"""Byte-exact pure grid and outcome function bodies from the accepted development evaluator. No file readers or entrypoints."""
import math
import numpy as np
GRIDS=(4,8,16)
BUDGETS=(("b10",1,10),("b20",1,5),("b30",3,10),("b50",1,2),("b100",1,1))

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
