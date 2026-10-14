"""Convert an existing, trusted local sklearn scaler to non-executable JSON."""

import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import FEATURES

p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output")
p.add_argument("--trust-local-file", action="store_true")
a = p.parse_args()
if not a.trust_local_file:
    p.error(
        "joblib 会反序列化 Python 对象。仅对自己训练生成的可信文件使用 --trust-local-file。"
    )
import joblib

s = joblib.load(a.input)
if int(s.n_features_in_) != len(FEATURES):
    raise ValueError("特征数量不一致")
if hasattr(s, "feature_names_in_") and list(s.feature_names_in_) != FEATURES:
    raise ValueError("特征顺序不一致")
Path(a.output).write_text(
    json.dumps(
        {
            "feature_cols": FEATURES,
            "mean": s.mean_.tolist(),
            "scale": s.scale_.tolist(),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("已导出标准化参数：", a.output)
