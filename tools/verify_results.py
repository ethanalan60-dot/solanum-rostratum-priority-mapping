"""Independent standard-library checks of public grid-origin result tables."""
from pathlib import Path
import csv, math, statistics
root = Path(__file__).resolve().parents[1] / "analysis"
def read(name):
    with (root / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
rows = read("results_all.csv")
assert len(rows) == 4608
keys = [(r["run_id"], r["R"], r["M"], r["budget"], r["shift_x_fraction"], r["shift_y_fraction"]) for r in rows]
assert len(set(keys)) == 4608
assert len({r["run_id"] for r in rows}) == 24
summary = read("origin_range_summary.csv")
for grid in (1, 2, 4):
    for budget in (.1, .2, .3, .5):
        groups = {}
        for r in rows:
            assert 0 <= float(r["capture_pct"]) <= 100 + 1e-9
            if int(r["M"]) == grid and float(r["budget"]) == budget:
                groups.setdefault((r["run_id"], r["R"]), []).append(float(r["capture_pct"]))
        assert len(groups) == 96 and all(len(v) == 4 for v in groups.values())
        ranges = [max(v) - min(v) for v in groups.values()]
        saved = next(r for r in summary if int(r["M"]) == grid and float(r["budget"]) == budget)
        assert math.isclose(statistics.median(ranges), float(saved["range_pp_median"]), abs_tol=1e-10)
lookup = {(r["run_id"], r["R"], r["M"], float(r["budget"]), r["shift_x_fraction"], r["shift_y_fraction"]): float(r["capture_pct"]) for r in rows}
for saved in read("grid_size_comparison_summary.csv"):
    b = float(saved["budget"])
    diffs = []
    for r in rows:
        if r["M"] == "4" and float(r["budget"]) == b:
            other = lookup[r["run_id"], r["R"], "2", b, r["shift_x_fraction"], r["shift_y_fraction"]]
            diffs.append(float(r["capture_pct"]) - other)
    assert len(diffs) == 384
    assert sum(v < -1e-9 for v in diffs) == int(saved["M4_lower_than_M2"])
    assert sum(v > 1e-9 for v in diffs) == int(saved["M4_higher_than_M2"])
print("PASS: all 4608 rows, 12 median summaries and 4 paired-comparison groups")
