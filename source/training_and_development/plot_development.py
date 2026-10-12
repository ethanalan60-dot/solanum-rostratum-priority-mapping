"""Plot a recorded development report without opening datasets or checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [r for r in report["runs"] if r.get("curve")]
    if not rows:
        raise ValueError("No recorded development metrics")
    if len({r["val_manifest_sha256"] for r in rows}) != 1:
        raise ValueError("Do not plot different validation splits together")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#75507B", "#333333"]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.9))
    for i, r in enumerate(rows):
        label = r["run_id"].replace("dev_v2_", "").replace("_s17", "")
        if r["status"] != "completed":
            label += " (running)"
        color = colors[i % len(colors)]
        axes[0].plot([x["epoch"] for x in r["curve"]],
                     [100*x["selection_balanced_mae"] for x in r["curve"]],
                     color=color, linewidth=1.6, label=label,
                     linestyle="-" if r["status"] == "completed" else "--")
        axes[0].scatter([r["best_epoch"]], [r["best_balanced_mae_pp"]],
                        s=35, facecolors="white", edgecolors=color, zorder=4)
        axes[1].plot([1, 2, 4], [100*r["best_scales"][str(f)]["balanced_mae"] for f in (1, 2, 4)],
                     color=color, marker="o", linewidth=1.6,
                     linestyle="-" if r["status"] == "completed" else "--")
    zero = rows[0]["constants"]["zero"]
    for ax in axes:
        ax.axhline(100*zero["balanced_mae"], color="#666666", linestyle=":", linewidth=1.5)
        ax.set_ylabel("Balanced coverage MAE (percentage points)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="A  Validation curves; circles mark selected epochs", xlabel="Epoch")
    axes[1].set(title="B  Scale errors at the selected epoch", xlabel="Synthetic degradation factor")
    axes[1].set_xticks([1, 2, 4], ["1 (256 px)", "2 (128 px)", "4 (64 px)"])
    axes[0].legend(fontsize=8, loc="upper right")
    seed_label = ", ".join(str(s) for s in sorted({r["seed"] for r in rows}))
    fig.suptitle(f"Development only | one spatial split | seed(s) {seed_label}", fontsize=14, y=.98)
    fig.text(.5, .035,
             f"Same footprint and target at every scale. Dotted line: zero predictor. Validation: {zero['n']} windows / {zero['n_positive']} positive.\n"
             "Validation-selected checkpoints; no final-test scores or confirmation-seed results are shown.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.20, wspace=.26)
    out = report_path.parent
    for suffix in ("png", "svg", "pdf"):
        path = out / f"validation_curves.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180 if suffix == "png" else 300)
    plt.close(fig)
    (out / "figure_source.json").write_text(json.dumps({"input": str(report_path),
          "input_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
          "matplotlib": matplotlib.__version__, "test_accessed": False}, indent=2), encoding="utf-8")
    print(str(out / "validation_curves.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or Path(json.loads((ROOT / "latest_development_report.json").read_text(encoding="utf-8"))["path"]) / "summary.json"
    main(report_path)
