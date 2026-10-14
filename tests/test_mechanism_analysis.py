import copy, json, tempfile, time, unittest, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import core
from app import mechanism_analysis as analysis


class FakeGLM:
    def __init__(self, wrong_reference=False):
        self.calls = []
        self.wrong_reference = wrong_reference

    def invoke(self, messages):
        self.calls.append(messages)
        payload = json.loads(messages[1][1])
        if "evidence" in payload:
            cs = []
            for c in payload["evidence"]["channels"]:
                ids = [r["reference_id"] for r in c["references"][:2]]
                if self.wrong_reference and c["channel"] == 1:
                    ids = ["CH2-R1"]
                cs.append(
                    {
                        "channel": c["channel"],
                        "summary": "合成示例：邻域支持的通道解释。",
                        "agreement": "consistent" if ids else "insufficient",
                        "supporting_reference_ids": ids,
                        "uncertainties": ["细节需参考证据进一步核对。"],
                    }
                )
            return SimpleNamespace(
                content=json.dumps({"channels": cs}, ensure_ascii=False)
            )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "overall": {
                        "summary": "合成示例：六通道综合机制解释。",
                        "supporting_channel_ids": [1, 2, 3, 4, 5, 6],
                        "uncertainties": ["不是实测机制验证。"],
                    }
                },
                ensure_ascii=False,
            )
        )


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        (self.root / "outputs").mkdir()
        self.patch = patch.object(analysis, "ROOT", self.root)
        self.patch.start()
        analysis.JOBS.clear()
        self.run_id = "run_20261012_120000_a1b2c3"
        self.directory = self.root / "outputs" / self.run_id
        self.directory.mkdir()
        d, y, v, meta = core.predict(core.generate(3), "demo")
        case = core.retrieve(d.head(1), {c: x[:1] for c, x in v.items()}, "demo")[0]
        table = d.head(1).copy()
        table["lgkp_pred"] = y[:1]
        table.to_csv(self.directory / "prediction.csv", index=False)
        (self.directory / "result.json").write_text(
            json.dumps(
                {
                    "metadata": meta,
                    "reference_scope": "train",
                    "retrieval": [case],
                    "retrieval_policy": {"exclude_source": True},
                }
            )
        )
        self.bundle = analysis.build_evidence_bundle(self.run_id, 0)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_two_stages_and_citation_validation(self):
        fake = FakeGLM()
        with patch.object(analysis, "build_metadata_llm", return_value=fake):
            result = analysis.generate_analysis(self.bundle, analysis.DEFAULT_CONFIG)
        self.assertEqual(len(result["channels"]), 6)
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("channels", json.loads(fake.calls[1][1][1]))
        self.assertNotIn("evidence", json.loads(fake.calls[1][1][1]))
        self.assertEqual(
            self.bundle["channels"][0]["references"][0]["reference_id"], "CH1-R1"
        )

    def test_reject_cross_channel_and_invalid_overall(self):
        fake = FakeGLM(True)
        with patch.object(analysis, "build_metadata_llm", return_value=fake):
            with self.assertRaisesRegex(ValueError, "跨通道"):
                analysis.generate_analysis(self.bundle, analysis.DEFAULT_CONFIG)
        self.assertEqual(len(fake.calls), 1)
        fake = FakeGLM()
        payload = analysis.parse_response(
            fake.invoke([("", ""), ("", json.dumps({"evidence": self.bundle}))]).content
        )
        cs = analysis.validate_channel_output(payload, self.bundle)
        with self.assertRaises(ValueError):
            analysis.validate_overall_output(
                {
                    "overall": {
                        "summary": "bad",
                        "supporting_channel_ids": [7],
                        "uncertainties": [],
                    }
                },
                cs,
            )

    def test_missing_channel_unsupported_and_path_guard(self):
        payload = {"channels": []}
        with self.assertRaises(ValueError):
            analysis.validate_channel_output(payload, self.bundle)
        for run in ["../config", "run_20261012_120000_a1b2c3/../.."]:
            with self.assertRaises(ValueError):
                analysis.build_evidence_bundle(run, 0)
        with self.assertRaises(ValueError):
            analysis.build_evidence_bundle(self.run_id, 9)

    def test_worker_exports_and_no_key(self):
        fake = FakeGLM()
        with patch.object(
            analysis, "resolve_api_key", return_value=("private-test-key", "API_KEY")
        ), patch.object(
            analysis.importlib.util, "find_spec", return_value=object()
        ), patch.object(
            analysis, "build_metadata_llm", return_value=fake
        ):
            job = analysis.start_analysis(self.run_id, 0)
            deadline = time.monotonic() + 5
            while job["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
                job = analysis.get_job(job["id"])
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(len(job["downloads"]), 3)
        for p in self.directory.glob("analysis_*/*"):
            self.assertNotIn("private-test-key", p.read_text())
        self.assertIn(
            "CH1-R1", job["result"]["channels"][0]["supporting_reference_ids"]
        )

    def test_provider_error_does_not_echo_secrets(self):
        fake = SimpleNamespace(
            invoke=lambda messages: (_ for _ in ()).throw(
                ValueError("secret-key-in-provider-error")
            )
        )
        with self.assertRaises(ValueError) as e:
            analysis.invoke_json(fake, "sys", {})
        self.assertNotIn("secret-key", str(e.exception))

    def test_key_candidates_and_config(self):
        with patch.object(analysis, "_load_env_file"), patch.dict(
            "os.environ", {"API_KEY": "own-key"}, clear=True
        ):
            self.assertEqual(analysis.resolve_api_key(), ("own-key", "API_KEY"))
        with self.assertRaises(ValueError):
            analysis.validate_config({**analysis.DEFAULT_CONFIG, "api_key": "secret"})
        with self.assertRaises(ValueError):
            analysis.validate_config(
                {**analysis.DEFAULT_CONFIG, "api_key_env_candidates": ["BAD-NAME"]}
            )

    def test_source_context_matches_saved_prediction(self):
        row = pd.read_csv(self.directory / "prediction.csv")
        row.loc[0, "sample_id"] = "tampered"
        row.to_csv(self.directory / "prediction.csv", index=False)
        with self.assertRaisesRegex(ValueError, "不一致"):
            analysis.build_evidence_bundle(self.run_id, 0)


class ReferenceScopeTests(unittest.TestCase):
    def test_true_train_val_test_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            d = pd.DataFrame(
                [
                    dict(
                        Ref_source_num=str(i),
                        sample_id=f"s{i}",
                        alloy_id=f"a{i}",
                        condition_id="900C",
                        status="reported",
                        text=f"mechanism {i}",
                        emb_0=1,
                        emb_1=i + 0.1,
                    )
                    for i in range(6)
                ]
            )
            d.to_csv(root / "ref.csv", index=False)
            split = d[core.KEYS].copy()
            split["split"] = ["train", "train", "train", "validation", "test", "test"]
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
                refs = core.references("real", 2, "train_val")
                for r, x in refs.values():
                    self.assertEqual(r.sample_id.tolist(), ["s0", "s1", "s2", "s3"])
                for r, x in core.references("real", 2).values():
                    self.assertEqual(len(r), 3)
                split["split"] = ["train", "train", "train", "val", "val", "val"]
                split.to_csv(root / "split.csv", index=False)
                with self.assertRaisesRegex(ValueError, "无法区分"):
                    core.references("real", 2, "train_val")

    def test_train_test_only_produces_warning_and_never_uses_test(self):
        d = core.generate(3)
        _, v, _ = core.demo_predict(d)
        result = core.retrieve(
            d.head(1),
            {c: x[:1] for c, x in v.items()},
            "demo",
            reference_scope="train_val",
        )
        self.assertEqual(len(result[0]["warnings"]), 6)
        self.assertTrue(
            all(
                n["reference_split"] == "train"
                for c in result[0]["channels"]
                for n in c["neighbors"]
            )
        )


if __name__ == "__main__":
    unittest.main()
