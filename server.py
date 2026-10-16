"""Local browser UI. Bind to loopback. Only explicitly requested LLM analysis sends selected evidence."""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import argparse, json, mimetypes, io, uuid, platform, importlib.util, time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from app.core import (
    ROOT,
    FEATURES,
    KEYS,
    CHANNELS,
    config,
    path_for,
    parse_table,
    validate,
    predict,
    retrieve,
    screen,
    generate,
)
from app.task_manager import start_job, list_jobs, job_log
from app.training_workflow import workflow_status, save_training_config
from app.mechanism_analysis import (
    llm_status,
    save_llm_config,
    list_analysis_runs,
    build_evidence_bundle,
    start_analysis,
    get_job,
)
from app.candidate_generation import defaults as candidate_defaults, generate_candidates
from app import analysis_jobs
from science import visualization, optimization


def records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def export_run(df, y, vectors, meta, result):
    out = (
        ROOT
        / "outputs"
        / ("run_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    )
    out.mkdir(parents=True)
    pred = df.copy()
    pred["lgkp_pred"] = y
    pred["run_mode"] = meta["mode"]
    pred["result_label"] = meta["label"]
    for ch, v in vectors.items():
        pred = pd.concat(
            [
                pred,
                pd.DataFrame(
                    v,
                    columns=[f"ch{ch}_pred_{j}" for j in range(v.shape[1])],
                    index=pred.index,
                ),
            ],
            axis=1,
        )
    pred.to_csv(out / "prediction.csv", index=False, encoding="utf-8-sig")
    result["run_id"] = out.name
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    workspace = config()
    reference_snapshot = None
    if meta["mode"] == "real":
        try:
            reference_snapshot = visualization.snapshot(workspace)
        except (FileNotFoundError, KeyError):
            pass
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "features": FEATURES,
                "config": workspace,
                "reference_snapshot": reference_snapshot,
                "metadata": meta,
                "candidate_generation": result.get("candidate_generation"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if "pairs" in result:
        pd.DataFrame(result["pairs"]).to_csv(
            out / "pairs.csv", index=False, encoding="utf-8-sig"
        )
    if "candidate_generation" in result:
        (out / "candidate_generation.json").write_text(
            json.dumps(result["candidate_generation"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        n: "/download/" + (out / n).relative_to(ROOT / "outputs").as_posix()
        for n in ["prediction.csv", "result.json", "manifest.json"]
        + (["pairs.csv"] if "pairs" in result else [])
        + (["candidate_generation.json"] if "candidate_generation" in result else [])
    }


def status():
    c = config()
    files = {
        k: {
            "path": v,
            "exists": (
                path_for(v).is_file()
                if k in ["scaler_json", "pretrain_split"]
                else path_for(v).is_dir()
            ),
        }
        for k, v in c.items()
        if k != "references"
    }
    files["transfer_config"] = {
        "path": c["kinetics_dir"] + "/transfer_config.json",
        "exists": (path_for(c["kinetics_dir"]) / "transfer_config.json").is_file(),
    }
    files["kinetics_weights"] = {
        "path": c["kinetics_dir"] + "/best_model_state.npz 或 .pt",
        "exists": any(
            (path_for(c["kinetics_dir"]) / n).is_file()
            for n in ["best_model_state.npz", "best_model_state.pt"]
        ),
    }
    for ch, value in c["references"].items():
        files["ch" + ch] = {"path": value, "exists": path_for(value).is_file()}
    return {
        "version": "1.0.0",
        "python": platform.python_version(),
        "torch_available": bool(importlib.util.find_spec("torch")),
        "files": files,
        "config": c,
        "channels": CHANNELS,
        "features": FEATURES,
        "jobs": list_jobs(),
        "demo_records": 96,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def allowed(self):
        host = self.headers.get("Host", "").split(":")[0]
        if host not in ["127.0.0.1", "localhost"]:
            return False
        origin = self.headers.get("Origin")
        return not origin or origin in [
            f"http://127.0.0.1:{self.server.server_port}",
            f"http://localhost:{self.server.server_port}",
        ]

    def send(self, value, code=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def file(self, path, base):
        path = path.resolve()
        base = base.resolve()
        if not path.is_relative_to(base) or not path.is_file():
            return self.send({"error": "文件不存在"}, 404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream"
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.allowed():
            return self.send({"error": "仅接受本机访问"}, 403)
        p = urlparse(self.path).path
        try:
            if p == "/api/candidates/config":
                return self.send({"defaults": candidate_defaults()})
            if p == "/api/status":
                return self.send(status())
            if p == "/api/visualization/config":
                return self.send({"config": visualization.get_config()})
            if p == "/api/visualization/jobs":
                return self.send(analysis_jobs.list_jobs())
            if p.startswith("/api/visualization/job/"):
                return self.send(analysis_jobs.get_job(p.rsplit("/", 1)[-1]))
            if p == "/api/optimization/config":
                return self.send(optimization.status())
            if p == "/api/training/config":
                return self.send(workflow_status())
            if p == "/api/demo":
                return self.send(
                    {
                        "csv": generate(96, 7).to_csv(index=False),
                        "rows": records(generate(96, 7)),
                    }
                )
            if p == "/api/jobs":
                return self.send(list_jobs())
            if p == "/api/analysis/config":
                return self.send(llm_status())
            if p == "/api/analysis/runs":
                return self.send(list_analysis_runs())
            if p.startswith("/api/analysis/job/"):
                return self.send(get_job(p.rsplit("/", 1)[-1]))
            if p.startswith("/api/log/"):
                return self.send(job_log(p.rsplit("/", 1)[-1]))
            if p.startswith("/download/"):
                return self.file(
                    ROOT / "outputs" / p[len("/download/") :], ROOT / "outputs"
                )
            if p == "/template.csv":
                return self.file(
                    ROOT / "data" / "demo" / "input_template.csv",
                    ROOT / "data" / "demo",
                )
            return self.file(
                ROOT / "web" / ("index.html" if p == "/" else p.lstrip("/")),
                ROOT / "web",
            )
        except Exception as e:
            self.send({"error": str(e)}, 400)

    def do_POST(self):
        if not self.allowed():
            return self.send({"error": "仅接受同源本机请求"}, 403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 34 * 1024 * 1024:
                raise ValueError("请求体大小应为 1 字节至 34 MiB。")
            body = json.loads(self.rfile.read(length))
            p = urlparse(self.path).path
            if p == "/api/config":
                required = {
                    "pretrained_dir",
                    "kinetics_dir",
                    "scaler_json",
                    "pretrain_split",
                    "references",
                }
                if set(body) != required or set(body["references"]) != {
                    str(c) for c in range(1, 7)
                }:
                    raise ValueError("配置字段不完整。")
                if any(
                    not isinstance(v, str) or not v.strip()
                    for k, v in body.items()
                    if k != "references"
                ) or any(
                    not isinstance(v, str) or not v.strip()
                    for v in body["references"].values()
                ):
                    raise ValueError("路径必须是非空字符串。")
                (ROOT / "config" / "workspace.json").write_text(
                    json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return self.send({"saved": True})
            if p == "/api/training/config":
                return self.send(save_training_config(body))
            if p == "/api/visualization/config":
                return self.send(visualization.save_config(body))
            if p == "/api/visualization/start":
                return self.send(analysis_jobs.start_job(body))
            if p == "/api/optimization/config":
                return self.send(optimization.save_config(body))
            if p == "/api/training/upload":
                import base64, shutil

                name = body.get("name")
                allowed = {
                    "joint_six_channel.json": "data/private",
                    "oxidation_dataset.xlsx": "data/private",
                    "train.xlsx": "data/manual",
                    "val.xlsx": "data/manual",
                    "test.xlsx": "data/manual",
                }
                if name not in allowed:
                    raise ValueError(
                        "请选择联合 JSON、总数据 Excel 或 train/val/test.xlsx。"
                    )
                data = base64.b64decode(body["base64"], validate=True)
                if len(data) > 24 * 1024 * 1024:
                    raise ValueError("单个文件最大 24 MiB。")
                if name.endswith(".json"):
                    json.loads(data.decode("utf-8-sig"))
                elif not data.startswith(b"PK"):
                    raise ValueError("该文件不是有效的 XLSX 工作簿。")
                target = ROOT / allowed[name] / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    backup = ROOT / "outputs/uploads" / uuid.uuid4().hex / target.name
                    backup.parent.mkdir(parents=True)
                    shutil.copy2(target, backup)
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(target)
                return self.send(
                    {"saved": True, "path": target.relative_to(ROOT).as_posix()}
                )
            if p == "/api/analysis/config":
                return self.send(save_llm_config(body))
            if p == "/api/analysis/preview":
                return self.send(
                    build_evidence_bundle(body["run_id"], body["sample_index"])
                )
            if p == "/api/analysis/start":
                return self.send(start_analysis(body["run_id"], body["sample_index"]))
            if p == "/api/job":
                return self.send(start_job(body["kind"]))
            if p == "/api/validate":
                df = validate(parse_table(body["csv"]))
                return self.send(
                    {
                        "valid": True,
                        "count": len(df),
                        "features": len(FEATURES),
                        "preview": records(df.head(8)),
                    }
                )
            if p == "/api/predict":
                raw = (
                    pd.DataFrame([body["sample"]])
                    if "sample" in body
                    else parse_table(body["csv"])
                )
                df, y, v, meta = predict(raw, body.get("mode", "demo"))
                result = {
                    "metadata": meta,
                    "count": len(df),
                    "predictions": [
                        {
                            "sample_id": str(df.iloc[i].sample_id),
                            "lgkp_pred": float(value),
                        }
                        for i, value in enumerate(y)
                    ],
                }
                result["reference_scope"] = body.get("reference_scope", "train")
                result["retrieval_policy"] = {
                    "exclude_source": body.get("exclude_source", True),
                    "exclude_self": True,
                    "top_k": 5,
                }
                if body.get("retrieve", True):
                    result["retrieval"] = retrieve(
                        df,
                        v,
                        meta["mode"],
                        body.get("exclude_source", True),
                        reference_scope=result["reference_scope"],
                    )
                result["downloads"] = export_run(df, y, v, meta, result)
                return self.send(result)
            if p == "/api/screen":
                source = body.get(
                    "candidate_source", "csv" if "csv" in body else "generated"
                )
                if source == "csv":
                    csv = body.get("csv")
                    if not isinstance(csv, str) or not csv.strip():
                        raise ValueError("请先在数据管理页导入 CSV，再运行候选筛选。")
                    raw = parse_table(csv)
                    generation = {
                        "source": "csv",
                        "input_count": len(raw),
                        "notes": "直接使用导入状态，不应用生成范围、步长或固定条件。",
                    }
                elif source == "generated":
                    raw, generation = generate_candidates(
                        body.get("n", 180),
                        body.get("seed", 42),
                        body.get("generation_config"),
                    )
                else:
                    raise ValueError("未知候选来源。")
                df, y, v, meta = predict(raw, body.get("mode", "demo"))
                result = screen(
                    df,
                    y,
                    v,
                    meta["mode"],
                    float(body.get("kinetic", 0.05)),
                    float(body.get("divergence", 0.60)),
                    float(body.get("prescreen", 0.10)),
                    bool(body.get("cross_source", True)),
                )
                result["candidate_generation"] = generation
                result["metadata"] = meta
                result["downloads"] = export_run(df, y, v, meta, result)
                return self.send(result)
            self.send({"error": "未知接口"}, 404)
        except Exception as e:
            self.send({"error": str(e)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"氧化机制表征学习与动力学分析软件 V1.0\n打开 http://127.0.0.1:{args.port}\n按 Ctrl+C 退出。",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
