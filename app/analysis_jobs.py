"""Persisted asynchronous UMAP and evidence-analysis jobs."""

from __future__ import annotations

import json
import threading
import time
import uuid
import re

import pandas as pd
from app import core
from science import visualization as viz
from app.exploration_analysis import summarize_clusters, summarize_pair

LOCK = threading.Lock()
ACTIVE = {}


def directory(ident):
    if not isinstance(ident, str) or not re.fullmatch(r"[a-f0-9]{12}", ident):
        raise ValueError("空间分析任务编号无效。")
    path = core.ROOT / "outputs/analysis_jobs" / ident
    if not path.is_dir():
        raise ValueError("空间分析任务不存在。")
    return path


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def get_job(ident):
    value = core.read_json(directory(ident) / "job.json")
    if value["status"] == "running" and ident not in ACTIVE:
        value.update(status="interrupted", stage="服务重启；已保存的结果仍保留")
    if value["status"] == "completed":
        value["result"] = core.read_json(directory(ident) / "result.json")
    return value


def list_jobs():
    paths = sorted(
        (core.ROOT / "outputs/analysis_jobs").glob("*/job.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    rows = []
    for p in paths[:30]:
        try:
            value = core.read_json(p)
            if value["status"] == "running" and p.parent.name not in ACTIVE:
                value.update(status="interrupted", stage="服务重启；已保存结果保留")
            rows.append(value)
        except (OSError, ValueError):
            continue
    return rows


def urls(out):
    return {
        p.name: "/download/" + p.relative_to(core.ROOT / "outputs").as_posix()
        for p in out.iterdir()
        if p.is_file() and p.name not in ["job.json", "request.json"]
    }


def export_result(result, out):
    result["software_version"] = "1.0.0"
    write(out / "result.json", result)
    kind = result["kind"]
    if kind in ["atlas", "sample_projection", "pair_projection"]:
        rows = []
        for channel in result["channels"]:
            for r in channel["points"]:
                rows.append({"channel": channel["channel"], "role": "reference", **r})
            for state in channel.get("states", []):
                rows.append(
                    {
                        "channel": channel["channel"],
                        "role": "prediction",
                        **{k: v for k, v in state.items() if k != "references"},
                    }
                )
                for r in state["references"]:
                    rows.append(
                        {
                            "channel": channel["channel"],
                            "role": "top5",
                            "state_id": state["state_id"],
                            **r,
                        }
                    )
        pd.DataFrame(rows).to_csv(
            out / "channel_coordinates.csv", index=False, encoding="utf-8-sig"
        )
        if result.get("global", {}).get("points"):
            pd.DataFrame(result["global"]["points"]).to_csv(
                out / "global_coordinates.csv", index=False, encoding="utf-8-sig"
            )
        from science.visual_export import export_plots

        export_plots(result, out)
    if kind in ["cluster_summary", "pair_summary"]:
        write(out / "prompt_record.json", result["prompt_records"])
        lines = [
            "# "
            + ("六通道簇解释" if kind == "cluster_summary" else "候选对机制差异解释"),
            "模型：" + result["model"],
            "结果为近邻支持的解释，支持关系需人工核对。",
        ]
        if result["mode"] == "demo":
            lines.append("合成演示，非研究结果。")
        if kind == "cluster_summary":
            for c in result["clusters"]:
                lines += [
                    f"## CH {c['channel']} Cluster {c['cluster_id']}",
                    str(c.get("short_label", "")),
                    str(c.get("mechanism_summary", "")),
                    "引用：" + ", ".join(c.get("supporting_reference_ids", [])),
                ]
                for r in c["representatives"]:
                    lines += [
                        f"### {r['reference_id']} | 来源 {r['Ref_source_num']} | 样本 {r['sample_id']}",
                        r["text"],
                    ]
        else:
            lines += ["## 整体机制差异", result["overall"]["summary"]]
            for c in result["channels"]:
                lines += [
                    f"## CH {c['channel']} {c['name']}",
                    c["summary"],
                    "共性：" + "；".join(c["common_features"]),
                    "差异：" + "；".join(c["differences"]),
                    "引用：" + ", ".join(c["supporting_reference_ids"]),
                    "分歧：" + "；".join(c["uncertainties"]),
                ]
            for c in result["evidence"]["channels"]:
                for state in c["states"]:
                    for r in state["references"]:
                        lines += [
                            f"### {r['reference_id']} | 来源 {r['Ref_source_num']} | 样本 {r['sample_id']}",
                            r["text"],
                        ]
        (out / "analysis_report.md").write_text("\n\n".join(lines), encoding="utf-8")


def start_job(request):
    allowed = {
        "atlas",
        "sample_projection",
        "pair_projection",
        "cluster_summary",
        "pair_summary",
    }
    kind = request.get("kind")
    if kind not in allowed:
        raise ValueError("未知空间分析任务。")
    cfg = viz.get_config()  # Snapshot settings before the worker starts.
    mode = request.get("mode", "demo")
    scope = request.get("scope", "train")
    if mode not in ["demo", "real"] or scope not in ["train", "train_val"]:
        raise ValueError("空间分析模式或参考范围无效。")
    if kind in ["cluster_summary", "pair_summary"]:
        source = get_job(request.get("source_job"))
        wanted = "atlas" if kind == "cluster_summary" else "pair_projection"
        if source["status"] != "completed" or source["result"]["kind"] != wanted:
            raise ValueError("请先完成对应空间或配对投影任务。")
    with LOCK:
        if ACTIVE:
            raise ValueError("已有空间或机制比较任务运行，请等待完成。")
        ident = uuid.uuid4().hex[:12]
        out = core.ROOT / "outputs/analysis_jobs" / ident
        out.mkdir(parents=True)
        job = {
            "id": ident,
            "kind": kind,
            "status": "running",
            "stage": "已提交",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write(out / "job.json", job)
        write(out / "request.json", {"request": request, "visualization_config": cfg})
        ACTIVE[ident] = job

    def progress(message):
        with LOCK:
            job["stage"] = message
            write(out / "job.json", job)

    def worker():
        try:
            if kind == "atlas":
                result = viz.atlas(mode, scope, cfg, progress)
            elif kind == "sample_projection":
                result = viz.project_run(
                    request["run_id"],
                    cfg,
                    sample_index=request.get("sample_index"),
                    progress=progress,
                )
            elif kind == "pair_projection":
                result = viz.project_run(
                    request["run_id"],
                    cfg,
                    pair_index=request.get("pair_index"),
                    progress=progress,
                )
            elif kind == "cluster_summary":
                result = summarize_clusters(source["result"], progress)
            else:
                result = summarize_pair(source["result"], progress)
            progress("保存坐标、图表或分析报告")
            export_result(result, out)
            with LOCK:
                job.update(status="completed", stage="完成", downloads=urls(out))
                write(out / "job.json", job)
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(
                    exc, (ValueError, ImportError, FileNotFoundError, OSError)
                )
                else f"空间分析失败：{type(exc).__name__}。检查数据、依赖与日志。"
            )
            with LOCK:
                job.update(status="failed", stage="失败", error=message)
                write(out / "job.json", job)
        finally:
            with LOCK:
                ACTIVE.pop(ident, None)

    threading.Thread(target=worker, daemon=True).start()
    return dict(job)
