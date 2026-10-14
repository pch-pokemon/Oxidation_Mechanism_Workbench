"""Persisted task records and explicit subprocess execution without shell commands."""

import json, os, subprocess, sys, threading, time, uuid
from pathlib import Path
from app.core import ROOT, read_json
from app.training_workflow import get_training_config, write_json

JOBS = {}
LOCK = threading.Lock()


def list_jobs():
    rows = []
    for path in sorted((ROOT / "outputs/jobs").glob("*/job.json"), reverse=True):
        try:
            j = read_json(path)
            progress = path.parent / "progress.json"
            if progress.is_file():
                j["stage"] = read_json(progress).get("stage", "")
            if j["status"] == "running" and j["id"] not in JOBS:
                j["status"] = "unknown_after_restart"
            rows.append(j)
        except (OSError, ValueError):
            continue
    return rows[:30]


def start_job(kind):
    if kind not in ["prepare", "pretrain", "transfer", "pipeline", "optimize"]:
        raise ValueError("未知任务。")
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise ValueError("已有数据或训练任务正在运行。")
        if any(j["status"] == "unknown_after_restart" for j in list_jobs()):
            raise ValueError(
                "检测到上次服务关闭时未结束的任务。请先确认旧进程已结束，并检查该任务 job.json 状态，避免并行写入。"
            )
        cfg = get_training_config()
        ident = uuid.uuid4().hex[:12]
        out = ROOT / "outputs/jobs" / ident
        out.mkdir(parents=True)
        snapshot = out / "task_config.json"
        write_json(snapshot, cfg)
        if kind == "optimize":
            from science.optimization import get_config

            write_json(out / "optimization_config.json", get_config())
        with (out / "task.log").open("wb") as stream:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "tools/workflow.py"),
                    kind,
                    str(snapshot),
                    str(out),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        j = {
            "id": ident,
            "kind": kind,
            "status": "running",
            "pid": proc.pid,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log": (out / "task.log").relative_to(ROOT).as_posix(),
        }
        JOBS[ident] = j
        write_json(out / "job.json", j)

    def monitor():
        code = proc.wait()
        with LOCK:
            j.update(status="completed" if code == 0 else "failed", returncode=code)
            write_json(out / "job.json", j)

    threading.Thread(target=monitor, daemon=True).start()
    return dict(j)


def job_log(ident):
    if (
        not isinstance(ident, str)
        or len(ident) != 12
        or any(c not in "0123456789abcdef" for c in ident)
    ):
        raise ValueError("任务编号无效。")
    path = ROOT / "outputs/jobs" / ident / "task.log"
    if not path.is_file():
        raise ValueError("任务日志不存在。")
    return {"text": path.read_text(encoding="utf-8", errors="replace")[-50000:]}
