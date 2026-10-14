"""Frozen local SciBERT text encoding; PCA is fitted exclusively on training references."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from science.dataset_manager import CHANNEL_KEYS
from app.core import KEYS, normalize_keys

_MODEL_CACHE = {}


def encode_scibert(texts, cfg):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError("SciBERT 编码需要安装 requirements-model.txt。") from e
    device = (
        "cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    )
    path = cfg["scibert_path"]
    key = (str(Path(path).resolve()), device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = (
            AutoTokenizer.from_pretrained(path, local_files_only=True),
            AutoModel.from_pretrained(path, local_files_only=True).to(device).eval(),
        )
    tokenizer, model = _MODEL_CACHE[key]
    parts = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            token = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            token = {k: v.to(device) for k, v in token.items()}
            hidden = model(**token).last_hidden_state
            mask = token["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            parts.append(pooled.cpu().numpy())
    x = np.concatenate(parts).astype("float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_references(master_path, out_dir, cfg, encoder=None):
    master = normalize_keys(pd.read_csv(master_path, dtype={k: str for k in KEYS}))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = [
        int(
            (
                master.split.eq("train")
                & master[f"ch{ch}_valid"].astype(str).str.lower().isin(["true", "1"])
            ).sum()
        )
        for ch in range(1, 7)
    ]
    if min(counts) < 3:
        raise ValueError(f"每个通道至少需三条有效训练机制；目前为 {counts}。")
    requested = int(cfg["embedding_dim"])
    raw_by_channel = {}
    for ch in range(1, 7):
        mask = (
            master[f"ch{ch}_valid"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            .to_numpy()
        )
        indices = np.flatnonzero(mask)
        if cfg.get("raw_embeddings_dir"):
            raw = normalize_keys(
                pd.read_csv(
                    Path(cfg["raw_embeddings_dir"]) / f"ch{ch}_raw.csv",
                    dtype={k: str for k in KEYS},
                )
            )
            cols = sorted(
                [c for c in raw if c.startswith("raw_") and c[4:].isdigit()],
                key=lambda s: int(s[4:]),
            )
            if not cols or cols != [f"raw_{i}" for i in range(len(cols))]:
                raise ValueError("原始向量列应连续命名为 raw_0、raw_1…")
            matched = master.loc[mask, KEYS].merge(
                raw[KEYS + cols], on=KEYS, how="left", validate="one_to_one"
            )
            x = matched[cols].to_numpy("float32")
        else:
            texts = []
            for _, row in master.loc[mask].iterrows():
                tags = json.loads(row[f"ch{ch}_tags"] or "[]")
                texts.append(
                    " ".join(str(t) for t in tags) + " " + str(row[f"ch{ch}_text"])
                )
            x = (encoder or encode_scibert)(texts, cfg)
        if not np.isfinite(x).all():
            raise ValueError(f"通道 {ch} 原始向量无效或未对齐。")
        raw_by_channel[ch] = (indices, x)
    dim = min(
        requested, min(counts) - 1, min(x.shape[1] for _, x in raw_by_channel.values())
    )
    if dim < 2:
        raise ValueError("有效训练参考不足以构建通道表示。")
    summary = {
        "requested_dim": requested,
        "embedding_dim": dim,
        "pca_fit_scope": "train_only",
        "train_valid_counts": counts,
        "encoder": "precomputed_raw" if cfg.get("raw_embeddings_dir") else "scibert",
        "channels": {},
    }
    for ch, (indices, x) in raw_by_channel.items():
        fit = master.iloc[indices].split.eq("train").to_numpy()
        pca = PCA(n_components=dim, random_state=int(cfg["seed"]))
        pca.fit(x[fit])
        z = pca.transform(x)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        if (norms <= 1e-12).any():
            raise ValueError(f"通道 {ch} 有零范数 PCA 向量；请检查文本或降低维度。")
        z = z / norms
        all_z = np.zeros((len(master), dim), dtype="float32")
        all_z[indices] = z
        ref = master[KEYS + ["split"]].copy()
        ref["embedding_text"] = master[f"ch{ch}_text"]
        ref["canonical_tags"] = master[f"ch{ch}_tags"]
        ref["evidence_status"] = master[f"ch{ch}_status"]
        ref["evidence_basis"] = master[f"ch{ch}_basis"]
        ref["is_empty"] = 1
        ref.loc[indices, "is_empty"] = 0
        ref = pd.concat(
            [ref, pd.DataFrame(all_z, columns=[f"emb_{i}" for i in range(dim)])], axis=1
        )
        ref.to_csv(out / f"ch{ch}.csv", index=False, encoding="utf-8-sig")
        raw_df = master.iloc[indices][KEYS].reset_index(drop=True)
        raw_df = pd.concat(
            [raw_df, pd.DataFrame(x, columns=[f"raw_{i}" for i in range(x.shape[1])])],
            axis=1,
        )
        raw_df.to_csv(out / f"ch{ch}_raw.csv", index=False, encoding="utf-8-sig")
        joblib.dump(pca, out / f"ch{ch}_pca.joblib")
        summary["channels"][str(ch)] = {
            "valid": len(indices),
            "training": int(fit.sum()),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }
    (out / "reference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("参考表示完成：", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
