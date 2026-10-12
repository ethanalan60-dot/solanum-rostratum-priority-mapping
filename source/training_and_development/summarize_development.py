"""Audit and summarize recorded validation runs; never opens test data or checkpoints."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(plan_path):
    plan = read_json(plan_path)
    records, files = [], {str(plan_path): digest(plan_path)}
    for item in plan["runs"]:
        run_dir = ROOT / "runs" / item["run_id"]
        row = {"run_id": item["run_id"], "variant": item["variant"],
               "seed": item["seed"], "status": "pending"}
        if (run_dir / "USE_RESTRICTION.json").exists():
            row["status"] = "excluded_by_use_restriction"
            records.append(row)
            continue
        if not (run_dir / "config.json").exists():
            records.append(row)
            continue
        cfg = read_json(run_dir / "config.json")
        assert cfg["stage"] == "development" and cfg["test_accessed"] is False
        for key in ("variant", "seed", "epochs", "lr", "coverage_weight",
                    "consistency_weight", "scale_augmentation"):
            assert cfg[key] == item[key], (row["run_id"], key)
        if (run_dir / "failure.json").exists():
            row.update(status="failed", failure=read_json(run_dir / "failure.json")["error"])
        else:
            row["status"] = "running"
        for name in ("config.json", "status.json", "failure.json", "metrics.jsonl",
                     "best_validation.json", "constant_baselines.json"):
            if (run_dir / name).exists():
                files[str(run_dir / name)] = digest(run_dir / name)
        metrics_path = run_dir / "metrics.jsonl"
        history = []
        if metrics_path.exists():
            # A running writer may leave one partial trailing line; preserve all complete records.
            lines = metrics_path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    if i != len(lines) - 1 or row["status"] != "running":
                        raise
        if history:
            assert [x["epoch"] for x in history] == list(range(1, len(history) + 1))
            assert all(math.isfinite(x["selection_balanced_mae"]) for x in history)
            best = min(history, key=lambda x: x["selection_balanced_mae"])
            last = history[-1]
            row.update(epochs_observed=len(history), best_epoch=best["epoch"],
                       best_balanced_mae_pp=100 * best["selection_balanced_mae"],
                       last_balanced_mae_pp=100 * last["selection_balanced_mae"],
                       best_scales=best["scales"], curve=history,
                       train_manifest_sha256=cfg["train_manifest_sha256"],
                       val_manifest_sha256=cfg["val_manifest_sha256"],
                       script_sha256=cfg["script_sha256"],
                       constants=read_json(run_dir / "constant_baselines.json"))
            for metric in ("mae", "mae_zero", "mae_positive", "bias"):
                values = [x[metric] for x in best["scales"].values() if x[metric] is not None]
                row[f"{metric}_pp"] = 100 * sum(values) / len(values) if values else None
            for metric in ("target_iou", "dice"):
                values = [x[metric] for x in best["scales"].values() if metric in x]
                row[metric] = sum(values) / len(values) if values else None
        if (run_dir / "status.json").exists():
            status = read_json(run_dir / "status.json")
            if status["status"] == "completed":
                assert len(history) == cfg["epochs"]
                assert status["test_accessed"] is False
                saved_best = read_json(run_dir / "best_validation.json")
                assert saved_best["epoch"] == row["best_epoch"] == status["best_epoch"]
                assert abs(saved_best["selection_balanced_mae"] * 100 - row["best_balanced_mae_pp"]) < 1e-10
                assert abs(status["best_validation_balanced_mae"] * 100 - row["best_balanced_mae_pp"]) < 1e-10
                row.update(status="completed", elapsed_seconds=status["elapsed_seconds"])
        records.append(row)
    return {"created_at": dt.datetime.now().astimezone().isoformat(), "plan": str(plan_path),
            "protocol": plan["protocol"], "test_accessed": False,
            "selection": plan["selection"], "source_sha256": files, "runs": records}


def output(report):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "development_reports" / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["run_id", "variant", "seed", "status", "epochs_observed", "best_epoch",
               "best_balanced_mae_pp", "last_balanced_mae_pp", "mae_pp", "mae_zero_pp",
               "mae_positive_pp", "bias_pp", "target_iou", "dice", "elapsed_seconds"]
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["runs"])
    lines = ["# 开发验证实验汇总", "", f"生成时间：{report['created_at']}", "",
             "仅汇总当前计划的开发验证成绩；不读取测试数据或模型权重。所有计划运行均列出，失败和未完成运行保留。",
             "误差单位为百分点；主指标为零/非零窗口分层 MAE 的均值，再平均 1、2、4 倍退化尺度。最佳 epoch 只由该验证指标选择。",
             "这不是确认实验或最终测试结论，尚不能据此声称达到期刊要求。", "",
             "| 运行 | 状态 | 最佳轮/已完成轮 | 平衡MAE | 整体MAE | 零区MAE | 非零区MAE | 最末轮平衡MAE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: f"{v:.3f}" if v is not None else "—"
    for r in report["runs"]:
        lines.append("| " + " | ".join([r["run_id"], r["status"],
                     f"{r.get('best_epoch', '—')}/{r.get('epochs_observed', 0)}",
                     *[fmt(r.get(k)) for k in ("best_balanced_mae_pp", "mae_pp", "mae_zero_pp",
                                               "mae_positive_pp", "last_balanced_mae_pp")]]) + " |")
    complete = [r for r in report["runs"] if r["status"] == "completed"]
    lines.extend(["", "## 解释边界", ""])
    if complete:
        groups = {(r["train_manifest_sha256"], r["val_manifest_sha256"]) for r in complete}
        if len(groups) == 1:
            constants = complete[0]["constants"]
            for name, val in constants.items():
                lines.append(f"- 常数基线 `{name}`：平衡MAE {100*val['balanced_mae']:.3f}、整体MAE {100*val['mae']:.3f} 个百分点；验证窗口 {val['n']} 个，其中非零 {val['n_positive']} 个。")
            winner = min(complete, key=lambda r: r["best_balanced_mae_pp"])
            lines.append(f"- 已完成运行中验证主指标最低的是 `{winner['run_id']}`，为 {winner['best_balanced_mae_pp']:.3f} 个百分点。这里只描述当前开发排序；未完成对照可能改变排序。")
        else:
            lines.append("- 本表包含不同数据划分，禁止跨划分直接排名。")
    lines.extend(["- 最佳轮与最后一轮差异用于识别过拟合；不把最好开发值当作无偏泛化估计。",
                  "- 分割 IoU/Dice 使用固定 0.5 概率阈值及原目标面积口径；没有按测试集选阈值。",
                  "- 当前单种子、单地块验证不能证明跨地区刺萼龙葵泛化；多次训练的种子不是独立地块。",
                  "- 提交论文前还需冻结方法、完整多种子确认、外部数据复现、强基线及最终封存测试。", ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "latest_development_report.json").write_text(json.dumps({"path": str(out), "created_at": report["created_at"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "statuses": {r["run_id"]: r["status"] for r in report["runs"]}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "development_plan.json")
    args = parser.parse_args()
    output(summarize(args.plan.resolve()))
