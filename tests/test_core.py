import unittest, tempfile, json, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.core as core


class WorkflowTests(unittest.TestCase):
    def test_input_composition_and_nonfinite(self):
        d = core.generate(8)
        core.validate(d)
        for value in [float("inf"), float("nan"), -1]:
            bad = d.copy()
            bad.loc[0, "Al"] = value
            with self.assertRaises(ValueError):
                core.validate(bad)
        bad = d.copy()
        bad.loc[0, "Co"] += 1
        with self.assertRaises(ValueError):
            core.validate(bad)
        aliased = d.rename(columns={c: "x_" + c for c in core.ELEMENTS})
        self.assertEqual(
            list(core.validate(aliased)[core.FEATURES].columns), core.FEATURES
        )

    def test_source_exclusion_and_train_only(self):
        d = core.generate(3)
        _, v, _ = core.demo_predict(d)
        result = core.retrieve(d.head(1), {c: x[:1] for c, x in v.items()}, "demo")
        for ch in result[0]["channels"]:
            self.assertEqual(len(ch["neighbors"]), 5)
            self.assertTrue(
                all(
                    n["Ref_source_num"] != d.iloc[0].Ref_source_num
                    for n in ch["neighbors"]
                )
            )
            self.assertTrue(
                all(
                    ch["neighbors"][i]["cosine"] >= ch["neighbors"][i + 1]["cosine"]
                    for i in range(4)
                )
            )

    def test_reference_mask_authoritative_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            rows = []
            for i in range(5):
                rows.append(
                    dict(
                        Ref_source_num=str(i),
                        sample_id=f"s{i}",
                        alloy_id=f"a{i}",
                        condition_id="900C",
                        status="not_reported" if i == 4 else "reported",
                        split="train",
                        emb_0=1,
                        emb_1=i + 0.1,
                    )
                )
            d = pd.DataFrame(rows)
            d.to_csv(root / "ref.csv", index=False)
            split = d[core.KEYS].copy()
            split["split"] = ["train", "train", "train", "val", "train"]
            split.to_csv(root / "split.csv", index=False)
            (root / "config/workspace.json").write_text(
                json.dumps(
                    {
                        "pretrain_split": "split.csv",
                        "references": {str(c): "ref.csv" for c in range(1, 7)},
                    }
                )
            )
            with patch.object(core, "ROOT", root):
                refs = core.references("real", 2)
                for ch, (r, x) in refs.items():
                    self.assertEqual(r.sample_id.tolist(), ["s0", "s1", "s2"])

    def test_screen_counts_against_brute_force(self):
        d, y, v, meta = core.predict(core.generate(20), "demo")
        result = core.screen(d, y, v, "demo", 0.05, 0, 0.10, True)
        brute = sum(
            abs(y[i] - y[j]) <= 0.10
            for i in range(len(y))
            for j in range(i + 1, len(y))
        )
        target = sum(
            abs(y[i] - y[j]) <= 0.05
            for i in range(len(y))
            for j in range(i + 1, len(y))
        )
        self.assertEqual(result["counts"]["all_pairs"], 190)
        self.assertEqual(result["counts"]["kinetic_prescreen_pairs"], brute)
        self.assertEqual(result["counts"]["target_pairs"], target)
        for r in result["pairs"]:
            p = np.array([r[f"ch{c}_percentile"] for c in range(1, 7)])
            eff = p.sum() ** 2 / (p * p).sum() if (p * p).sum() else 0
            self.assertAlmostEqual(r["D_mech"], p.mean() * np.sqrt(eff / 6))

    def test_real_missing_dependency_or_files_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(core, "ROOT", Path(temporary)):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    core.predict(core.generate(3), "real")


if __name__ == "__main__":
    unittest.main()
