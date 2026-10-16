"""Validation, deterministic demonstration and calibrated six-channel retrieval."""

from pathlib import Path
import json, io, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "solu_temp",
    "solu_time",
    "middle_temp",
    "middle_time",
    "aging_temp",
    "aging_time",
    "Co",
    "Al",
    "W",
    "Ni",
    "Cr",
    "Mo",
    "Fe",
    "Nb",
    "C",
    "Hf",
    "Si",
    "Ta",
    "Ti",
    "Y",
    "V",
    "B",
    "Zr",
    "Mn",
    "Sc",
    "La",
    "Re",
    "Temperature",
    "E_Time",
]
ELEMENTS = FEATURES[6:27]
KEYS = ["Ref_source_num", "sample_id", "alloy_id", "condition_id"]
CHANNELS = [
    "保护性氧化层形成",
    "内氧化与选择性氧化",
    "非保护性与过渡氧化物",
    "传输与缺陷动力学",
    "氧化层失效与剥落",
    "合金元素相互作用",
]
RANGES = {
    "Al": (7, 15),
    "Cr": (3, 15),
    "Ni": (20, 35),
    "W": (0, 8),
    "Ti": (0, 8),
    "Mo": (0, 4),
    "Ta": (0, 6),
    "Si": (0, 3),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def config():
    return read_json(ROOT / "config" / "workspace.json")


def path_for(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def normalize_keys(df):
    df = df.copy()
    for key in KEYS:
        if key in df:
            df[key] = df[key].map(
                lambda v: (
                    "" if pd.isna(v) else re.sub(r"^(\d+)\.0$", r"\1", str(v).strip())
                )
            )
    return df


def parse_table(text):
    return pd.read_csv(io.StringIO(text), dtype={k: str for k in KEYS})


def validate(df):
    if not 1 <= len(df) <= 2000:
        raise ValueError("输入记录数须在 1–2000 之间。")
    df = normalize_keys(df)
    for element in ELEMENTS:
        alias = "x_" + element
        if element not in df and alias in df:
            df[element] = df[alias]
    missing = [c for c in FEATURES if c not in df]
    if missing:
        raise ValueError("缺少必需特征：" + ", ".join(missing))
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("特征包含空值、非数值或无穷值。请明确补全，软件不自动填零。")
    if (df[FEATURES] < 0).any().any():
        raise ValueError("成分、温度和时间不能为负。")
    if ((df[ELEMENTS].sum(axis=1) - 100).abs() > 0.2).any():
        raise ValueError("元素含量须采用 at.% 且总和为 100±0.2。wt.% 数据请先转换。")
    if (df["Temperature"] <= 0).any() or (df["E_Time"] <= 0).any():
        raise ValueError("氧化温度和暴露时间须大于零。")
    if all(k in df for k in KEYS) and df.duplicated(KEYS).any():
        raise ValueError("样本四元标识重复。")
    if "sample_id" not in df:
        df["sample_id"] = [f"INPUT{i+1:05d}" for i in range(len(df))]
    return df


def generate(n=180, seed=42, generation_config=None):
    if generation_config is not None:
        from app.candidate_generation import generate_candidates

        return generate_candidates(n, seed, generation_config)[0]
    if not 2 <= n <= 800:
        raise ValueError("候选数量须在 2–800 之间。")
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        row = dict.fromkeys(FEATURES, 0.0)
        for k, (lo, hi) in RANGES.items():
            row[k] = float(rng.integers(round(lo * 2), round(hi * 2) + 1) / 2)
        row["Co"] = 100 - sum(row[k] for k in RANGES)
        if not 40 <= row["Co"] <= 80:
            continue
        row.update(
            solu_temp=1200,
            solu_time=12,
            aging_temp=900,
            aging_time=100,
            Temperature=900,
            E_Time=100,
            sample_id=f"DEMO{len(rows)+1:04d}",
            alloy_id=f"SYN{len(rows)+1:04d}",
            Ref_source_num=f"DEMO-SRC{len(rows)%12+1:02d}",
            condition_id="900C_100h",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def demo_predict(df):
    """Synthetic analytic outputs, never a trained research predictor."""
    x = df[FEATURES].to_numpy(float)
    y = (
        -10.2
        - 0.045 * df.Al.to_numpy()
        - 0.035 * df.Cr.to_numpy()
        + 0.015 * df.W.to_numpy()
        + 0.012 * df.Mo.to_numpy()
        + 0.002 * (df.Temperature.to_numpy() - 900)
    )
    vectors = {}
    for ch in range(1, 7):
        rng = np.random.default_rng(100 + ch)
        scaled = x / np.array(
            [1200, 12, 1000, 100, 1000, 100] + [30] * 21 + [1000, 100]
        )
        vectors[ch] = np.tanh(
            scaled @ rng.normal(0, 1.2, (len(FEATURES), 16)) + rng.normal(0, 0.1, 16)
        )
    return (
        y,
        vectors,
        {"mode": "demo", "label": "合成演示 / 非研究模型", "embedding_dim": 16},
    )


def predict(df, mode):
    df = validate(df)
    if mode == "demo":
        y, v, meta = demo_predict(df)
    elif mode == "real":
        from app.prediction_engine import predict_real

        y, v, meta = predict_real(df, config())
    else:
        raise ValueError("未知运行模式。")
    return df, y, v, meta


def unit_rows(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) < 1e-12).any():
        raise ValueError("机制向量含无效值或零向量，无法计算余弦距离。")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def reference_split_labels(series):
    aliases = {
        "training": "train",
        "validation": "val",
        "valid": "val",
        "dev": "val",
        "testing": "test",
    }
    return series.astype(str).str.strip().str.lower().replace(aliases)


def references(mode, dim, reference_scope="train", workspace_config=None):
    if reference_scope not in ["train", "train_val"]:
        raise ValueError("参考范围必须为 train 或 train_val。")
    allowed_splits = ["train"] if reference_scope == "train" else ["train", "val"]
    if mode == "demo":
        base = generate(96, 7)
        _, vs, _ = demo_predict(base)
        result = {}
        text = [
            "合成示例：保护性外氧化层的形成。",
            "合成示例：次表层的内氧化与选择性氧化。",
            "合成示例：过渡性或非保护性氧化物的生成。",
            "合成示例：氧与阳离子的耦合传输。",
            "合成示例：氧化层开裂及剥落特征。",
            "合成示例：合金元素间的作用关系。",
        ]
        for ch in range(1, 7):
            d = base[KEYS].copy()
            d["embedding_text"] = text[ch - 1]
            d["split"] = "train"
            d["status"] = "synthetic_demo"
            result[ch] = (d, unit_rows(vs[ch]))
        return result
    cfg = workspace_config or config()
    result = {}
    for ch in range(1, 7):
        value = cfg["references"].get(str(ch))
        if not value:
            raise ValueError(f"尚未配置通道 {ch} 的训练参考 CSV。")
        d = normalize_keys(pd.read_csv(path_for(value), dtype={k: str for k in KEYS}))
        for key in KEYS:
            if key not in d or d[key].eq("").any():
                raise ValueError(f"通道 {ch} 缺少完整标识 {key}。")
        if d.duplicated(KEYS).any():
            raise ValueError(f"通道 {ch} 存在重复四元标识。")
        # Always attach the authoritative split; never trust an imported split column.
        split = normalize_keys(
            pd.read_csv(path_for(cfg["pretrain_split"]), dtype={k: str for k in KEYS})
        )
        join = [k for k in KEYS if k in split]
        if (
            not {"Ref_source_num", "sample_id", "alloy_id"}.issubset(join)
            or "split" not in split
        ):
            raise ValueError("预训练划分缺少标识或 split 列。")
        if split.duplicated(join).any():
            raise ValueError("预训练划分键重复。")
        split["split"] = reference_split_labels(split["split"])
        labels = set(split["split"])
        if reference_scope == "train_val" and "val" in labels and "test" not in labels:
            raise ValueError(
                "当前划分只有 train/val，无法区分验证集与测试集。请先使用训练参考；训练＋验证需要明确包含 train/val/test 的真实划分。"
            )
        d = d.drop(columns=["split"], errors="ignore").merge(
            split[join + ["split"]], on=join, how="left", validate="many_to_one"
        )
        if d["split"].isna().any():
            raise ValueError(f"通道 {ch} 有参考记录未匹配预训练划分。")
        if "is_empty" in d:
            valid = d["is_empty"].astype(str).str.lower().isin(["0", "0.0", "false"])
        else:
            status = next((k for k in ["evidence_status", "status"] if k in d), None)
            if status is None:
                raise ValueError(f"通道 {ch} 缺少证据状态列。")
            valid = (
                ~d[status]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin(["", "not_reported", "none", "nan"])
            )
        d["split"] = reference_split_labels(d["split"])
        d = d[valid & d["split"].isin(allowed_splits)].copy().reset_index(drop=True)
        cols = [f"emb_{i}" for i in range(dim)]
        if not set(cols).issubset(d.columns):
            raise ValueError(f"通道 {ch} 向量维度不匹配，期望 {dim} 维。")
        if len(d) < 3:
            raise ValueError(f"通道 {ch} 在所选参考范围内至少需要三条有效记录。")
        result[ch] = (d, unit_rows(d[cols].to_numpy(float)))
    return result


def retrieve(
    df,
    vectors,
    mode,
    exclude_source=True,
    k=5,
    reference_scope="train",
    workspace_config=None,
):
    if len(df) > 200:
        raise ValueError("含检索的批量预测最多 200 条。")
    if exclude_source and (
        "Ref_source_num" not in df or df["Ref_source_num"].eq("").any()
    ):
        raise ValueError("排除同源检索须提供 Ref_source_num。")
    lib = references(
        mode, next(iter(vectors.values())).shape[1], reference_scope, workspace_config
    )
    out = []
    for idx, (_, row) in enumerate(df.iterrows()):
        sample = {
            "sample_id": str(row.sample_id),
            "reference_scope": reference_scope,
            "channels": [],
            "warnings": [],
        }
        for ch in range(1, 7):
            ref, x = lib[ch]
            query = unit_rows(vectors[ch][idx : idx + 1])[0]
            sim = np.clip(x @ query, -1, 1)
            keep = np.ones(len(ref), dtype=bool)
            if exclude_source:
                keep &= ref.Ref_source_num.ne(
                    str(row.get("Ref_source_num", ""))
                ).to_numpy()
            if all(key in row and str(row[key]) for key in KEYS):
                same = np.ones(len(ref), dtype=bool)
                for key in KEYS:
                    same &= ref[key].eq(str(row[key])).to_numpy()
                keep &= ~same
            order = np.where(keep)[0]
            order = order[np.argsort(-sim[order], kind="stable")[:k]]
            neighbors = []
            for j in order:
                r = ref.iloc[j]
                text = next(
                    (
                        str(r[key])
                        for key in ["embedding_text", "text", "mechanism_text"]
                        if key in r and pd.notna(r[key])
                    ),
                    "未提供参考描述",
                )
                neighbor = {
                    "rank": len(neighbors) + 1,
                    "cosine": float(sim[j]),
                    "text": text,
                    "reference_split": str(r["split"]),
                    **{key: str(r[key]) for key in KEYS},
                }
                for dest, candidates in {
                    "tags": ["canonical_tags", "tags"],
                    "evidence_status": ["evidence_status", "status"],
                    "evidence_basis": ["evidence_basis"],
                }.items():
                    for field in candidates:
                        if field in r and pd.notna(r[field]):
                            neighbor[dest] = str(r[field])
                            break
                neighbors.append(neighbor)
            if reference_scope == "train_val" and not ref["split"].eq("val").any():
                sample["warnings"].append(
                    f"通道 {ch} 未发现有效验证参考，当前仅使用训练记录。"
                )
            sample["channels"].append(
                {"channel": ch, "name": CHANNELS[ch - 1], "neighbors": neighbors}
            )
        out.append(sample)
    return out


def screen(
    df,
    y,
    vectors,
    mode,
    kinetic=0.05,
    divergence=0.60,
    prescreen=0.10,
    cross_source=False,
):
    if len(df) > 800:
        raise ValueError("交互配对筛选最多 800 个候选；更大规模使用研究脚本。")
    if not 0 <= kinetic <= prescreen <= 2 or not 0 <= divergence <= 1:
        raise ValueError("筛选阈值无效。")
    refs = references(mode, next(iter(vectors.values())).shape[1])
    calibration = {}
    for ch, (r, x) in refs.items():
        i, j = np.triu_indices(len(x), 1)
        if cross_source:
            keep = r.Ref_source_num.to_numpy()[i] != r.Ref_source_num.to_numpy()[j]
            i, j = i[keep], j[keep]
        if not len(i):
            raise ValueError("参考库缺少可用于距离校准的样本对。")
        calibration[ch] = np.sort(1 - np.clip(np.sum(x[i] * x[j], axis=1), -1, 1))
    i, j = np.triu_indices(len(df), 1)
    delta = np.abs(y[i] - y[j])
    stage2 = int((delta <= prescreen).sum())
    keep = delta <= kinetic
    i, j, delta = i[keep], j[keep], delta[keep]
    pct = np.zeros((len(i), 6))
    dist = np.zeros_like(pct)
    for col, ch in enumerate(range(1, 7)):
        x = unit_rows(vectors[ch])
        dist[:, col] = 1 - np.clip(np.sum(x[i] * x[j], axis=1), -1, 1)
        pct[:, col] = np.searchsorted(
            calibration[ch], dist[:, col], side="right"
        ) / len(calibration[ch])
    sums = pct.sum(1)
    squares = (pct**2).sum(1)
    eff = np.divide(sums**2, squares, out=np.zeros_like(sums), where=squares > 1e-12)
    score = pct.mean(1) * np.sqrt(eff / 6)
    rows = []
    for t in np.where(score >= divergence)[0]:
        rows.append(
            {
                "sample_index_1": int(i[t]),
                "sample_index_2": int(j[t]),
                "sample_1": str(df.iloc[i[t]].sample_id),
                "sample_2": str(df.iloc[j[t]].sample_id),
                "lgkp_1": float(y[i[t]]),
                "lgkp_2": float(y[j[t]]),
                "abs_delta_lgkp": float(delta[t]),
                "D_mech": float(score[t]),
                "effective_channels": float(eff[t]),
                "n_high_channels": int((pct[t] >= 0.75).sum()),
                **{f"ch{ch}_percentile": float(pct[t, ch - 1]) for ch in range(1, 7)},
            }
        )
    rows.sort(key=lambda r: (-r["D_mech"], r["abs_delta_lgkp"]))
    return {
        "counts": {
            "candidates": len(df),
            "all_pairs": len(df) * (len(df) - 1) // 2,
            "kinetic_prescreen_pairs": stage2,
            "target_pairs": len(rows),
            "unique_states": len(
                {r[k] for r in rows for k in ["sample_1", "sample_2"]}
            ),
        },
        "thresholds": {
            "prescreen": prescreen,
            "kinetic": kinetic,
            "divergence": divergence,
            "cross_source_calibration": cross_source,
        },
        "pairs": rows,
        "mode": mode,
        "formula": "D_mech = mean(p_ch) × sqrt(N_eff / 6); N_eff = (Σp_ch)² / Σp_ch²",
    }
