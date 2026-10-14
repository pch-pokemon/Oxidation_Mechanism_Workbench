"""Synthetic numerical fixtures; these are not literature data."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.core import FEATURES, KEYS, generate
from science.dataset_manager import CHANNEL_KEYS


def write_inputs(root, n=45, seed=1):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    d = generate(n, seed)
    d["Ref_source_num"] = [f"SRC{i//3:02d}" for i in range(n)]
    d["lgkp"] = -10 - 0.06 * d.Al - 0.04 * d.Cr + 0.01 * d.Ti
    with pd.ExcelWriter(root / "oxidation_dataset.xlsx") as w:
        d[KEYS + FEATURES].to_excel(w, sheet_name="Sheet3", index=False)
        d[KEYS + FEATURES + ["lgkp"]].to_excel(w, sheet_name="Sheet1", index=False)
    payload = []
    for _, row in d.iterrows():
        sample = {k: str(row[k]) for k in KEYS}
        sample["channels"] = {
            name: {
                "tags": [f"demo-tag-{i+1}"],
                "text": f"Synthetic channel {i+1} sample {row.sample_id} mechanism.",
                "status": "reported",
                "evidence_basis": "synthetic fixture",
            }
            for i, name in enumerate(CHANNEL_KEYS)
        }
        payload.append(sample)
    (root / "joint_six_channel.json").write_text(json.dumps(payload), encoding="utf-8")
    raw = root / "raw"
    raw.mkdir()
    rng = np.random.default_rng(seed)
    for ch in range(1, 7):
        vals = rng.normal(size=(n, 12)).astype("float32")
        pd.concat(
            [d[KEYS], pd.DataFrame(vals, columns=[f"raw_{i}" for i in range(12)])],
            axis=1,
        ).to_csv(raw / f"ch{ch}_raw.csv", index=False)
    return d
