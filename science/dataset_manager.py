"""One authoritative partition for all stages; automatic group or manual files."""

from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from app.core import FEATURES, KEYS, ELEMENTS, normalize_keys, validate

CHANNEL_KEYS = [
    "protective_scale_formation",
    "internal_oxidation_and_selective_oxidation",
    "nonprotective_or_transient_oxide_formation",
    "transport_and_defect_kinetics",
    "scale_failure_and_spallation",
    "alloying_element_interaction",
]


def load_joint(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("联合 JSON 须为对象或对象列表。")
    rows = []
    for block in data:
        if not isinstance(block, dict):
            raise ValueError("联合 JSON 条目须为对象。")
        samples = block.get("target_samples", [block] if "sample_id" in block else [])
        for sample in samples:
            row = {k: sample.get(k, block.get(k, "")) for k in KEYS}
            channels = sample.get("channels", {})
            for ch, name in enumerate(CHANNEL_KEYS, 1):
                item = channels.get(name, channels.get(str(ch), {})) or {}
                tags = item.get("tags", item.get("canonical_tags", []))
                tags = tags if isinstance(tags, list) else [str(tags)]
                text = str(
                    item.get("text", item.get("embedding_text", "")) or ""
                ).strip()
                status = str(
                    item.get("status", item.get("evidence_status", "not_reported"))
                    or "not_reported"
                ).lower()
                valid = bool(text) and status not in [
                    "not_reported",
                    "not_selected",
                    "none",
                    "null",
                    "",
                ]
                row.update(
                    {
                        f"ch{ch}_text": text,
                        f"ch{ch}_tags": json.dumps(tags, ensure_ascii=False),
                        f"ch{ch}_status": status,
                        f"ch{ch}_valid": valid,
                        f"ch{ch}_basis": str(item.get("evidence_basis", "")),
                    }
                )
            rows.append(row)
    df = normalize_keys(pd.DataFrame(rows))
    if df.empty:
        raise ValueError("联合 JSON 没有样本。")
    if any(df[k].eq("").any() for k in KEYS):
        raise ValueError("联合 JSON 必须包含完整四元标识。")
    if df.duplicated(KEYS).any():
        raise ValueError("联合 JSON 有重复四元标识。")
    return df


def read_workbook(path, joint):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    if path.suffix.lower() == ".csv":
        sheets = {"data": pd.read_csv(path, dtype={k: str for k in KEYS})}
    else:
        book = pd.ExcelFile(path)
        names = [n for n in ["Sheet3", "Sheet1"] if n in book.sheet_names]
        if not names:
            names = [book.sheet_names[0]]
        sheets = {
            n: pd.read_excel(book, sheet_name=n, dtype={k: str for k in KEYS})
            for n in names
        }
    frames = []
    for name, df in sheets.items():
        df = normalize_keys(df)
        for e in ELEMENTS:
            if e not in df and "x_" + e in df:
                df[e] = df["x_" + e]
        if not set(KEYS[:3]).issubset(df):
            raise ValueError(f"{path.name}/{name} 缺少来源、样本或合金标识。")
        if "condition_id" not in df or df.condition_id.eq("").any():
            simple = joint[KEYS].drop_duplicates()
            if simple.duplicated(KEYS[:3]).any():
                raise ValueError(
                    "Excel 缺少 condition_id，且 JSON 同一三元标识有多个条件，无法唯一对齐。"
                )
            df = df.drop(columns=["condition_id"], errors="ignore").merge(
                simple, on=KEYS[:3], how="left", validate="many_to_one"
            )
        df = normalize_keys(df)
        if any(df[k].eq("").any() for k in KEYS):
            raise ValueError(f"{path.name}/{name} 标识为空。")
        if df.duplicated(KEYS).any():
            raise ValueError(f"{path.name}/{name} 有重复四元标识。")
        missing = [k for k in FEATURES if k not in df]
        if missing:
            raise ValueError(f"{path.name}/{name} 缺少特征：" + ", ".join(missing))
        if "lgkp" not in df:
            df["lgkp"] = np.nan
        df["lgkp"] = pd.to_numeric(df["lgkp"], errors="coerce")
        # Prediction validation limits only apply to UI input; validate workbook in chunks.
        for start in range(0, len(df), 2000):
            validate(df.iloc[start : start + 2000].copy())
        for k in FEATURES:
            df[k] = pd.to_numeric(df[k], errors="raise")
        frames.append(df[KEYS + FEATURES + ["lgkp"]])
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for key, group in combined.groupby(KEYS, sort=False, dropna=False):
        if len(group) > 1:
            for col in FEATURES:
                if np.ptp(group[col].to_numpy(float)) > 1e-6:
                    raise ValueError(f"跨表同一标识的 {col} 不一致：{key}")
            targets = group.lgkp.dropna().to_numpy(float)
            if len(targets) > 1 and np.ptp(targets) > 1e-6:
                raise ValueError(f"同一标识的 lgkp 不一致：{key}")
        row = group.iloc[0].copy()
        t = group.lgkp.dropna()
        row["lgkp"] = float(t.iloc[0]) if len(t) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def auto_partition(df, seed=42, group_by_source=True):
    rng = np.random.default_rng(seed)
    units = df.Ref_source_num.unique() if group_by_source else np.arange(len(df))
    if len(units) < 3:
        raise ValueError("60/20/20 划分至少需要三个文献来源（或三条记录）。")
    nv = max(1, round(len(units) * 0.2))
    nt = max(1, round(len(units) * 0.2))
    nv = min(nv, len(units) - nt - 1)
    best = None
    for _ in range(128):
        order = rng.permutation(units)
        mapping = {
            str(u): ("test" if i < nt else "val" if i < nt + nv else "train")
            for i, u in enumerate(order)
        }
        labels = (
            df.Ref_source_num.astype(str)
            if group_by_source
            else pd.Series(np.arange(len(df)), index=df.index).astype(str)
        ).map(mapping)
        fractions = labels.value_counts(normalize=True)
        score = sum(
            (fractions.get(s, 0) - r) ** 2
            for s, r in [("train", 0.6), ("val", 0.2), ("test", 0.2)]
        )
        if best is None or score < best[0]:
            best = (score, labels)
    return best[1].to_numpy()


def prepare_dataset(cfg, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joint = load_joint(cfg["joint_json_path"])
    warnings = []
    if cfg["dataset_mode"] == "auto":
        master = read_workbook(cfg["excel_path"], joint)
        master["split"] = auto_partition(
            master, int(cfg["seed"]), cfg["split_strategy"] == "source"
        )
    elif cfg["dataset_mode"] == "manual":
        parts = []
        for split, field in [
            ("train", "train_path"),
            ("val", "val_path"),
            ("test", "test_path"),
        ]:
            p = cfg.get(field, "")
            if split == "val" and (not p or not Path(p).is_file()):
                continue
            part = read_workbook(p, joint)
            part["split"] = split
            parts.append(part)
        master = pd.concat(parts, ignore_index=True)
        if master.duplicated(KEYS).any():
            raise ValueError("手动文件的四元标识跨集合重复。")
        if master.groupby("sample_id")["split"].nunique().gt(1).any():
            raise ValueError("同一 sample_id 出现在多个集合中。")
    else:
        raise ValueError("dataset_mode 应为 auto 或 manual。")
    if master.groupby("Ref_source_num")["split"].nunique().gt(1).any():
        warnings.append(
            "文献来源跨集合重叠：当前为手动/按记录划分，评估不具备文献来源隔离。"
        )
    known = set(map(tuple, master[KEYS].to_numpy()))
    json_keys = set(map(tuple, joint[KEYS].to_numpy()))
    if not (json_keys & known):
        raise ValueError(
            "JSON 与 Excel 的四元标识没有匹配，请检查来源、样本、合金和条件编号。"
        )
    extra = len(json_keys - known)
    if extra:
        warnings.append(
            f"JSON 中 {extra} 个样本不在当前材料数据范围内，未用于本轮训练。"
        )
    master = master.merge(joint, on=KEYS, how="left", validate="one_to_one")
    for ch in range(1, 7):
        master[f"ch{ch}_valid"] = master[f"ch{ch}_valid"].fillna(False).astype(bool)
        for suffix in ["text", "tags", "status", "basis"]:
            master[f"ch{ch}_{suffix}"] = master[f"ch{ch}_{suffix}"].fillna("")
    summary = {
        "mode": cfg["dataset_mode"],
        "strategy": cfg["split_strategy"],
        "seed": cfg["seed"],
        "requested_ratios": (
            {"train": 0.6, "val": 0.2, "test": 0.2}
            if cfg["dataset_mode"] == "auto"
            else None
        ),
        "warnings": warnings,
        "counts": {},
        "channel_train_valid": {},
    }
    for split in ["train", "val", "test"]:
        part = master[master.split.eq(split)]
        summary["counts"][split] = {
            "records": len(part),
            "sources": int(part.Ref_source_num.nunique()),
            "kinetics": int(np.isfinite(part.lgkp).sum()),
            "mechanisms": int(
                part[[f"ch{ch}_valid" for ch in range(1, 7)]].any(axis=1).sum()
            ),
        }
        if len(part):
            with pd.ExcelWriter(out / f"{split}.xlsx") as writer:
                part[np.isfinite(part.lgkp)][KEYS + FEATURES + ["lgkp"]].to_excel(
                    writer, sheet_name="Sheet1", index=False
                )
                part[KEYS + FEATURES].to_excel(writer, sheet_name="Sheet3", index=False)
    if (
        not summary["counts"]["train"]["records"]
        or not summary["counts"]["test"]["records"]
    ):
        raise ValueError("训练和测试集合都必须非空。")
    for ch in range(1, 7):
        summary["channel_train_valid"][str(ch)] = int(
            (master.split.eq("train") & master[f"ch{ch}_valid"]).sum()
        )
    summary["has_validation"] = bool(summary["counts"]["val"]["records"])
    master.to_csv(out / "master.csv", index=False, encoding="utf-8-sig")
    master[KEYS + ["split"]].to_csv(
        out / "split.csv", index=False, encoding="utf-8-sig"
    )
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
