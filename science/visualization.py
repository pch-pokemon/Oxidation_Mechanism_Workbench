"""Six-channel spaces, train-fitted UMAP projections and compatible research imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from app import core

DEFAULTS = {
    "n_neighbors": 20,
    "min_dist": 0.15,
    "metric": "cosine",
    "seed": 42,
    "cluster_method": "kmeans",
    "n_clusters": 6,
    "min_cluster_size": 15,
    "min_samples": 5,
    "representatives": 5,
    "raw_embeddings_dir": "",
    "research_output_root": "",
    "channel_directories": {str(c): "" for c in range(1, 7)},
    "global_coordinates_path": "",
    "global_raw_variant": "combined",
    "reuse_existing": True,
}


def get_config():
    path = core.ROOT / "config/visualization.json"
    return validate_config(core.read_json(path) if path.is_file() else DEFAULTS)


def validate_config(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError("可视化配置字段不完整，请保留默认配置的全部字段。")
    for name, lo, hi in [
        ("n_neighbors", 2, 200),
        ("seed", 0, 100000),
        ("n_clusters", 2, 100),
        ("min_cluster_size", 2, 10000),
        ("min_samples", 1, 1000),
        ("representatives", 1, 20),
    ]:
        v = value[name]
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError("可视化整数参数无效：" + name)
    if (
        not isinstance(value["min_dist"], (float, int))
        or not 0 <= value["min_dist"] <= 1
    ):
        raise ValueError("UMAP min_dist 应在 0—1 之间。")
    if value["metric"] != "cosine":
        raise ValueError("当前通道投影统一使用 cosine 距离。")
    if value["cluster_method"] not in ["kmeans", "hdbscan"]:
        raise ValueError("聚类方法应为 kmeans 或 hdbscan。")
    if type(value["reuse_existing"]) is not bool:
        raise ValueError("reuse_existing 应为布尔值。")
    for field in [
        "raw_embeddings_dir",
        "research_output_root",
        "global_coordinates_path",
    ]:
        if not isinstance(value[field], str):
            raise ValueError("目录或文件路径应为字符串。")
    if value["global_raw_variant"] not in ["combined", "raw"]:
        raise ValueError("原始向量模式应为 combined 或 raw。")
    dirs = value["channel_directories"]
    if (
        not isinstance(dirs, dict)
        or set(dirs) != {str(c) for c in range(1, 7)}
        or any(not isinstance(p, str) for p in dirs.values())
    ):
        raise ValueError("channel_directories 应包含六通道字符串路径。")
    return dict(value)


def save_config(value):
    cfg = validate_config(value)
    p = core.ROOT / "config/visualization.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True, "config": cfg}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(cfg):
    paths = {
        "split": cfg["pretrain_split"],
        **{f"ch{c}": cfg["references"][str(c)] for c in range(1, 7)},
    }
    return {
        k: {"path": str(core.path_for(p)), "sha256": digest_file(core.path_for(p))}
        for k, p in paths.items()
    }


def run_inputs(run_id):
    from app.mechanism_analysis import run_directory

    directory = run_directory(run_id)
    result = core.read_json(directory / "result.json")
    manifest = core.read_json(directory / "manifest.json")
    table = core.normalize_keys(
        pd.read_csv(directory / "prediction.csv", dtype={k: str for k in core.KEYS})
    )
    cfg = manifest["config"]
    if result["metadata"]["mode"] == "real":
        saved = manifest.get("reference_snapshot")
        if not saved:
            raise ValueError(
                "旧运行缺少参考文件指纹，请在 V1.0 中重新预测或筛选后再做投影与比较。"
            )
        for record in saved.values():
            path = Path(record["path"])
            if not path.is_file() or digest_file(path) != record["sha256"]:
                raise ValueError(
                    "该运行对应的参考库或划分已改变，请重新预测；禁止混用旧向量与新参考。"
                )
        cfg = {
            **cfg,
            "pretrain_split": saved["split"]["path"],
            "references": {str(c): saved[f"ch{c}"]["path"] for c in range(1, 7)},
        }
    return directory, result, table, cfg


def library(mode, cfg=None, scope="train"):
    if mode == "demo":
        return core.references("demo", 16, scope)
    cfg = cfg or core.config()
    header = pd.read_csv(core.path_for(cfg["references"]["1"]), nrows=0)
    cols = [c for c in header if c.startswith("emb_") and c[4:].isdigit()]
    if not cols or set(cols) != {f"emb_{i}" for i in range(len(cols))}:
        raise ValueError("参考库 emb_0… 列不连续。")
    return core.references(mode, len(cols), scope, cfg)


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def reference_text(row):
    return str(
        next(
            (
                row[k]
                for k in ["embedding_text", "text", "mechanism_text"]
                if k in row and pd.notna(row[k])
            ),
            "",
        )
    )


def reference_records(frame, xy, clusters=None):
    items = []
    for index, row in frame.iterrows():
        item = {k: str(row[k]) for k in core.KEYS}
        item.update(
            x=float(xy[index, 0]),
            y=float(xy[index, 1]),
            split=str(row["split"]),
            text=reference_text(row),
        )
        if clusters is not None:
            item["cluster_id"] = int(clusters[index])
        for src, dest in [
            ("canonical_tags", "tags"),
            ("tags", "tags"),
            ("evidence_status", "evidence_status"),
            ("evidence_basis", "evidence_basis"),
        ]:
            if src in row and pd.notna(row[src]):
                item[dest] = str(row[src])
        items.append(item)
    return items


def coordinate_hash(frame, matrix, cfg):
    h = hashlib.sha256()
    h.update(
        json.dumps(
            frame[core.KEYS].astype(str).values.tolist(), ensure_ascii=False
        ).encode()
    )
    h.update(np.ascontiguousarray(matrix, dtype="float32").tobytes())
    h.update(
        json.dumps(
            {k: cfg[k] for k in ["n_neighbors", "min_dist", "metric", "seed"]},
            sort_keys=True,
        ).encode()
    )
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP 需要安装 requirements-analysis.txt；不会用 PCA 冒充 UMAP。"
        ) from exc
    h.update(str(umap.__version__).encode())
    return h.hexdigest()


def fit_training_umap(frame, matrix, cfg, progress=lambda _: None):
    fit = frame["split"].eq("train").to_numpy()
    if int(fit.sum()) < 3:
        raise ValueError("UMAP 至少需要三条有效训练参考。")
    train_frame = frame.loc[fit].reset_index(drop=True)
    train_x = matrix[fit]
    key = coordinate_hash(train_frame, train_x, cfg)
    cache = core.ROOT / "outputs/visualization_cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    model_file = cache / "umap.joblib"
    if model_file.is_file():
        # Only load caches produced by this app, never a user-supplied pickle.
        progress("复用本软件生成的训练 UMAP 缓存")
        reducer = joblib.load(model_file)
    else:
        import umap

        progress("拟合训练参考 UMAP（首次运行包含 JIT 编译）")
        reducer = umap.UMAP(
            n_neighbors=min(cfg["n_neighbors"], len(train_x) - 1),
            min_dist=cfg["min_dist"],
            metric="cosine",
            random_state=cfg["seed"],
            transform_seed=cfg["seed"],
            n_jobs=1,
            init="random",
        )
        with threadpool_limits(limits=2):
            reducer.fit(train_x)
        joblib.dump(reducer, model_file)
        (cache / "provenance.json").write_text(
            json.dumps(
                {
                    "fit_scope": "train_only",
                    "key": key,
                    "train_records": len(train_x),
                    "parameters": {
                        k: cfg[k] for k in ["n_neighbors", "min_dist", "metric", "seed"]
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    xy = np.empty((len(frame), 2), dtype=float)
    xy[fit] = reducer.embedding_
    if (~fit).any():
        xy[~fit] = reducer.transform(matrix[~fit])
    if not np.isfinite(xy).all():
        raise ValueError("UMAP 坐标包含非有限值，请检查参考向量。")
    return reducer, xy, key


def imported_channel_files(ch, cfg):
    root = (
        core.path_for(cfg["research_output_root"])
        if cfg["research_output_root"]
        else None
    )
    directories = (
        [core.path_for(cfg["channel_directories"][str(ch)])]
        if cfg["channel_directories"][str(ch)]
        else []
    )
    if root:
        directories.extend(
            [
                root / f"ch{ch}",
                root / f"channel_{ch}",
                root / f"mechanism_embedding{ch}",
            ]
        )
    return [d for d in directories if d.is_dir()]


def coordinate_columns(frame):
    for pair in [
        ("umap_x", "umap_y"),
        ("umap_1", "umap_2"),
        ("UMAP1", "UMAP2"),
        ("x", "y"),
    ]:
        if set(pair).issubset(frame):
            return list(pair)
    raise ValueError("坐标文件应包含 umap_x/y 或 umap_1/2。")


def import_coordinates(ch, ref, cfg):
    if not cfg["reuse_existing"]:
        return None
    for directory in imported_channel_files(ch, cfg):
        # Names exported by the supplied single-channel and Fig.6 scripts.
        candidates = [
            directory / name
            for name in [
                "umap_coordinates.csv",
                "umap_coordinates_and_clusters.csv",
                "semantic_umap_coordinates.csv",
                "umap_training_reference_coordinates.csv",
            ]
        ]
        candidates += sorted(directory.glob("*coordinates*.csv"))
        for path in dict.fromkeys(candidates):
            if not path.is_file():
                continue
            data = core.normalize_keys(
                pd.read_csv(path, dtype={k: str for k in core.KEYS})
            )
            channel_col = next(
                (k for k in ["channel_id", "channel"] if k in data), None
            )
            if channel_col:
                data = data[pd.to_numeric(data[channel_col], errors="coerce").eq(ch)]
            if not set(core.KEYS).issubset(data):
                continue
            cols = coordinate_columns(data)
            data = data[
                core.KEYS + cols + (["cluster_id"] if "cluster_id" in data else [])
            ]
            if data.duplicated(core.KEYS).any():
                raise ValueError(f"导入 CH {ch} 坐标存在重复四元标识。")
            merged = ref[core.KEYS].merge(
                data, on=core.KEYS, how="left", validate="one_to_one"
            )
            xy = merged[cols].to_numpy(float)
            if not np.isfinite(xy).all():
                raise ValueError(
                    f"CH {ch} 导入坐标没有覆盖当前参考范围，请更换文件或关闭复用。"
                )
            return xy, merged, path
    return None


def cluster_matrix(matrix, cfg):
    if cfg["cluster_method"] == "kmeans":
        if len(matrix) <= cfg["n_clusters"]:
            raise ValueError(
                "共同簇数必须小于每个通道的有效参考数，请降低 n_clusters。"
            )
        with threadpool_limits(limits=2):
            return KMeans(
                n_clusters=cfg["n_clusters"], random_state=cfg["seed"], n_init=10
            ).fit_predict(matrix)
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN 需要安装 requirements-clustering.txt，或选择 kmeans。"
        ) from exc
    return hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
        metric="euclidean",
        core_dist_n_jobs=2,
    ).fit_predict(matrix)


def representatives(ch, frame, matrix, labels, cfg):
    result = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        ids = np.flatnonzero(labels == label)
        center = matrix[ids].mean(0)
        norm = np.linalg.norm(center)
        scores = matrix[ids] @ (center / norm) if norm > 1e-12 else np.zeros(len(ids))
        order = ids[np.argsort(-scores, kind="stable")[: cfg["representatives"]]]
        refs = []
        for rank, index in enumerate(order, 1):
            row = frame.iloc[index]
            refs.append(
                {
                    "reference_id": f"CH{ch}-C{int(label)}-R{rank}",
                    **{k: str(row[k]) for k in core.KEYS},
                    "text": reference_text(row),
                    "split": str(row["split"]),
                }
            )
        result.append(
            {
                "cluster_id": int(label),
                "size": len(ids),
                "representatives": refs,
                "label": f"Cluster {int(label)}",
            }
        )
    return result


def reuse_cluster_summaries(ch, labels, imported, clusters, cfg):
    """Reuse a summary only when imported memberships exactly match current ones."""
    if imported is None or "cluster_id" not in imported[1]:
        return
    old = pd.to_numeric(imported[1]["cluster_id"], errors="coerce").to_numpy()
    if not np.array_equal(old, labels):
        return
    paths = [
        imported[2].parent / "cluster_llm_summary.json",
        imported[2].parent / "clusters/cluster_llm_summary.json",
    ]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        return
    summaries = core.read_json(path)
    for cluster in clusters:
        previous = (
            summaries.get(str(cluster["cluster_id"]))
            if isinstance(summaries, dict)
            else None
        )
        if isinstance(previous, dict):
            cluster["imported_summary"] = previous
            cluster["label"] = str(previous.get("short_label", cluster["label"]))
            cluster["summary_source"] = str(path)


def raw_channel(ch, ref, cfg, workspace):
    candidates = []
    if cfg["raw_embeddings_dir"]:
        candidates.append(core.path_for(cfg["raw_embeddings_dir"]) / f"ch{ch}_raw.csv")
    if workspace:
        candidates.append(
            core.path_for(workspace["references"][str(ch)]).parent / f"ch{ch}_raw.csv"
        )
    for directory in imported_channel_files(ch, cfg):
        candidates.extend(
            [
                directory / f"ch{ch}_raw.csv",
                directory
                / f"mechanism_embedding{ch}_scibert_{cfg['global_raw_variant']}.csv",
            ]
        )
        # Original research encoders can use scibert_0... rather than raw_0...
        candidates.extend(
            sorted(directory.glob(f"*scibert*{cfg['global_raw_variant']}*.csv"))
        )
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        raw = core.normalize_keys(pd.read_csv(path, dtype={k: str for k in core.KEYS}))
        cols = next(
            (
                sorted(
                    [
                        c
                        for c in raw
                        if c.startswith(prefix) and c[len(prefix) :].isdigit()
                    ],
                    key=lambda c: int(c[len(prefix) :]),
                )
                for prefix in ["raw_", "scibert_"]
                if any(c.startswith(prefix) and c[len(prefix) :].isdigit() for c in raw)
            ),
            [],
        )
        if not cols or not set(core.KEYS).issubset(raw):
            continue
        matched = ref[core.KEYS].merge(
            raw[core.KEYS + cols], on=core.KEYS, how="left", validate="one_to_one"
        )
        x = matched[cols].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f"CH {ch} 原始文本向量与当前参考不完整匹配。")
        return core.unit_rows(x), str(path)
    return None


def global_space(lib, mode, cfg, workspace, progress):
    if cfg["reuse_existing"] and cfg["global_coordinates_path"]:
        path = core.path_for(cfg["global_coordinates_path"])
        data = core.normalize_keys(pd.read_csv(path, dtype={k: str for k in core.KEYS}))
        chcol = next((k for k in ["channel_id", "channel"] if k in data), None)
        if chcol is None or not set(core.KEYS).issubset(data):
            raise ValueError("整体坐标 CSV 须包含 channel_id 和完整四元标识。")
        xycols = coordinate_columns(data)
        result = []
        for ch, (ref, _) in lib.items():
            sub = data[pd.to_numeric(data[chcol], errors="coerce").eq(ch)]
            matched = ref[core.KEYS].merge(
                sub[core.KEYS + xycols], how="left", on=core.KEYS, validate="one_to_one"
            )
            xy = matched[xycols].to_numpy(float)
            if not np.isfinite(xy).all():
                raise ValueError("导入整体坐标未完整覆盖六通道当前参考。")
            result.extend([{**r, "channel": ch} for r in reference_records(ref, xy)])
        return {
            "points": result,
            "source": str(path),
            "fit_scope": "imported_research_coordinates",
            "note": "整体图复用已有研究坐标；原始拟合范围以该研究输出配置为准。",
        }
    arrays = []
    frames = []
    sources = []
    for ch, (ref, _) in lib.items():
        raw = raw_channel(ch, ref, cfg, workspace) if mode == "real" else None
        if mode == "demo":
            base = core.generate(96, 7)[core.FEATURES].to_numpy(float)
            scale = np.maximum(np.std(base, axis=0), 1)
            shared = (base / scale) @ np.random.default_rng(11).normal(0, 0.1, (29, 24))
            shared[:, ch - 1] += 2.0
            raw = (core.unit_rows(shared), "synthetic_shared_space_not_scibert")
        if raw is None:
            return {
                "points": [],
                "note": "整体跨通道图需要同一固定文本编码器的原始向量，或已有整体坐标 CSV。六套独立 PCA 向量不会混合绘图。",
            }
        x, source = raw
        arrays.append(x)
        frames.append(ref.assign(channel=ch))
        sources.append(source)
    if len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("六通道原始向量维度不同，不能构建共同语义空间。")
    matrix = np.vstack(arrays)
    frame = pd.concat(frames, ignore_index=True)
    # Extend keys for the cache, since a sample appears once per channel.
    cache_frame = frame.copy()
    cache_frame["condition_id"] = (
        cache_frame["condition_id"].astype(str)
        + "-CH"
        + cache_frame["channel"].astype(str)
    )
    _, xy, key = fit_training_umap(cache_frame, matrix, cfg, progress)
    points = [
        {**r, "channel": int(frame.iloc[i]["channel"])}
        for i, r in enumerate(reference_records(frame, xy))
    ]
    return {
        "points": points,
        "source": sources,
        "cache_key": key,
        "fit_scope": (
            "train_only_raw_text" if mode == "real" else "synthetic_train_only"
        ),
        "note": (
            "六通道共同原始文本空间；需由用户确认六份向量来自同一固定编码器及同一文本模式。"
            if mode == "real"
            else "合成共享空间展示，非 SciBERT 或论文结果。"
        ),
    }


def atlas(mode, scope, cfg, progress=lambda _: None):
    workspace = core.config() if mode == "real" else None
    lib = library(mode, workspace, scope)
    channels = []
    for ch, (frame, matrix) in lib.items():
        progress(f"CH {ch}：空间坐标与高维聚类")
        imported = import_coordinates(ch, frame, cfg) if mode == "real" else None
        if imported is None:
            _, xy, key = fit_training_umap(frame, matrix, cfg, progress)
            source = "train_fitted_umap"
        else:
            xy, _, path = imported
            key = None
            source = str(path)
        labels = cluster_matrix(matrix, cfg)
        clusters = representatives(ch, frame, matrix, labels, cfg)
        reuse_cluster_summaries(ch, labels, imported, clusters, cfg)
        channels.append(
            {
                "channel": ch,
                "name": core.CHANNELS[ch - 1],
                "points": reference_records(frame, xy, labels),
                "clusters": clusters,
                "noise_count": int((labels < 0).sum()),
                "coordinate_source": source,
                "cache_key": key,
                "fit_scope": "imported_unknown" if imported else "train_only",
                "cluster_scope": scope,
                "cluster_method": cfg["cluster_method"],
            }
        )
    progress("构建整体六通道原始文本空间")
    global_plot = global_space(lib, mode, cfg, workspace, progress)
    return {
        "kind": "atlas",
        "mode": mode,
        "scope": scope,
        "config": cfg,
        "channels": channels,
        "global": global_plot,
        "note": "六通道共用一套聚类参数，在各通道原始高维表示内独立聚类；不同通道的同号簇不是同一物理类别。",
    }


def pair_indices(result, table, pair_index):
    pairs = result.get("pairs", [])
    if type(pair_index) is not int or not 0 <= pair_index < len(pairs):
        raise ValueError("请选择有效的筛选配对。")
    pair = pairs[pair_index]
    ids = []
    for position in [1, 2]:
        key = f"sample_index_{position}"
        if key in pair:
            index = int(pair[key])
        else:
            candidates = np.flatnonzero(
                table.sample_id.astype(str).eq(str(pair[f"sample_{position}"]))
            )
            if len(candidates) != 1:
                raise ValueError("旧配对缺少行号且样本 ID 不唯一，请重新筛选。")
            index = int(candidates[0])
        if not 0 <= index < len(table) or str(table.iloc[index].sample_id) != str(
            pair[f"sample_{position}"]
        ):
            raise ValueError("配对行号与预测记录身份不一致。")
        ids.append(index)
    return ids, pair


def selected_evidence(run_id, indices, result, table, workspace, scope):
    dim = result["metadata"]["embedding_dim"]
    vectors = {
        ch: table.iloc[indices][[f"ch{ch}_pred_{j}" for j in range(dim)]].to_numpy(
            float
        )
        for ch in range(1, 7)
    }
    selected = table.iloc[indices].reset_index(drop=True)
    if "pairs" in result:
        retrieved = core.retrieve(
            selected, vectors, result["metadata"]["mode"], True, 5, scope, workspace
        )
    else:
        stored = result.get("retrieval", [])
        if not stored:
            raise ValueError("该预测没有 Top-5，请启用检索后重新预测。")
        retrieved = [stored[i] for i in indices]
    states = []
    for pos, index in enumerate(indices):
        row = table.iloc[index]
        states.append(
            {
                "state_id": f"S{pos+1}",
                "sample_index": index,
                "sample_id": str(row.sample_id),
                "context": {
                    **{k: str(row[k]) for k in core.KEYS if k in row},
                    **{k: float(row[k]) for k in core.FEATURES},
                    "lgkp_pred": float(row.lgkp_pred),
                },
                "retrieval": retrieved[pos],
            }
        )
    return vectors, states


def project_run(
    run_id, cfg, sample_index=None, pair_index=None, progress=lambda _: None
):
    _, result, table, workspace = run_inputs(run_id)
    pair = None
    if pair_index is not None:
        indices, pair = pair_indices(result, table, pair_index)
        scope = "train"
    else:
        if type(sample_index) is not int or not 0 <= sample_index < len(table):
            raise ValueError("预测样本序号无效。")
        indices = [sample_index]
        scope = result.get("reference_scope", "train")
    vectors, states = selected_evidence(
        run_id, indices, result, table, workspace, scope
    )
    lib = library(result["metadata"]["mode"], workspace, scope)
    channels = []
    for ch, (frame, matrix) in lib.items():
        progress(f"CH {ch}：预测向量投影与 Top-5 定位")
        reducer, xy, key = fit_training_umap(frame, matrix, cfg, progress)
        query = core.unit_rows(vectors[ch])
        qxy = reducer.transform(query)
        points = reference_records(frame, xy)
        projected = []
        for pos, state in enumerate(states):
            entry = next(
                c for c in state["retrieval"]["channels"] if c["channel"] == ch
            )
            refs = []
            for neighbor in entry["neighbors"]:
                match = np.ones(len(frame), dtype=bool)
                for k in core.KEYS:
                    match &= frame[k].astype(str).eq(str(neighbor[k])).to_numpy()
                found = np.flatnonzero(match)
                if len(found) != 1:
                    raise ValueError("保存的 Top-5 无法唯一匹配该运行的参考库。")
                idx = int(found[0])
                refs.append(
                    {
                        **neighbor,
                        "reference_id": f"{state['state_id']}-CH{ch}-R{neighbor['rank']}",
                        "x": float(xy[idx, 0]),
                        "y": float(xy[idx, 1]),
                    }
                )
            projected.append(
                {
                    "state_id": state["state_id"],
                    "sample_id": state["sample_id"],
                    "x": float(qxy[pos, 0]),
                    "y": float(qxy[pos, 1]),
                    "references": refs,
                }
            )
        if not np.isfinite(qxy).all():
            raise ValueError("预测 UMAP 投影出现非有限坐标。")
        channels.append(
            {
                "channel": ch,
                "name": core.CHANNELS[ch - 1],
                "points": points,
                "states": projected,
                "cache_key": key,
            }
        )
    return {
        "kind": "pair_projection" if pair else "sample_projection",
        "run_id": run_id,
        "mode": result["metadata"]["mode"],
        "scope": scope,
        "config": cfg,
        "states": [{k: v for k, v in s.items() if k != "retrieval"} for s in states],
        "channels": channels,
        "pair": pair,
        "note": "UMAP 仅拟合有效训练参考；验证与预测点通过 transform 投影。Top-5 和 D_mech 仍按高维向量计算。",
    }
