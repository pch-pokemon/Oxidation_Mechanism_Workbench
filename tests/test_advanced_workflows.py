"""Real numeric UMAP/Optuna tests; GLM responses are explicit fixtures, never research evidence."""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import write_inputs
from app import core, training_workflow, exploration_analysis, analysis_jobs
from app import mechanism_analysis as llm
from science import visualization as viz, optimization
from science.dataset_manager import prepare_dataset
from science.reference_encoder import build_references
from science.training import train_mechanism_encoder, train_kinetics_model
from app.prediction_engine import predict_real
import server


class FakeAnalysisGLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        system = messages[0][1]
        payload = json.loads(messages[1][1])
        if system == exploration_analysis.CLUSTER_SYSTEM:
            refs = payload["representatives"]
            value = {
                "short_label": "Synthetic cluster",
                "mechanism_summary": "Synthetic fixture mechanism.",
                "heterogeneity": "consistent",
                "supporting_reference_ids": [refs[0]["reference_id"]],
                "uncertainties": [],
            }
        elif system == exploration_analysis.PAIR_SYSTEM:
            value = {
                "channels": [
                    {
                        "channel": c["channel"],
                        "summary": "Synthetic fixture comparison.",
                        "common_features": ["Synthetic common feature"],
                        "differences": ["Synthetic difference"],
                        "comparison_status": "different",
                        "supporting_reference_ids": [
                            s["references"][0]["reference_id"] for s in c["states"]
                        ],
                        "uncertainties": [],
                    }
                    for c in payload["evidence"]["channels"]
                ]
            }
        else:
            value = {
                "overall": {
                    "summary": "Synthetic overall comparison.",
                    "supporting_channel_ids": list(range(1, 7)),
                    "uncertainties": [],
                }
            }
        return SimpleNamespace(content=json.dumps(value))


class AdvancedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "config").mkdir()
        (cls.root / "outputs").mkdir()
        cls.stack = contextlib.ExitStack()
        for module in [core, training_workflow, server, llm]:
            cls.stack.enter_context(patch.object(module, "ROOT", cls.root))
        cls.data = write_inputs(cls.root / "data")
        cls.cfg = {
            **training_workflow.DEFAULTS,
            "joint_json_path": str(cls.root / "data/joint_six_channel.json"),
            "excel_path": str(cls.root / "data/oxidation_dataset.xlsx"),
            "raw_embeddings_dir": str(cls.root / "data/raw"),
            "embedding_dim": 4,
            "hidden_dims": [8, 6],
            "kinetics_hidden_dims": [4],
            "pretrain_epochs": 3,
            "kinetics_epochs": 4,
            "batch_size": 16,
        }
        cls.prepared = cls.root / "outputs/training_fixture"
        prepare_dataset(cls.cfg, cls.prepared / "dataset")
        build_references(
            cls.prepared / "dataset/master.csv", cls.prepared / "references", cls.cfg
        )
        train_mechanism_encoder(cls.prepared, cls.cfg)
        train_kinetics_model(cls.prepared, cls.cfg)
        training_workflow.write_json(cls.prepared / "training_config.json", cls.cfg)
        training_workflow.activate(cls.prepared)
        cls.vcfg = {
            **viz.DEFAULTS,
            "n_neighbors": 5,
            "n_clusters": 2,
            "representatives": 2,
        }

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        cls.tmp.cleanup()

    def save_prediction(self, pair=False):
        table = self.data.iloc[:3].copy()
        table.sample_id = ["NEW1", "NEW2", "NEW3"]
        table.Ref_source_num = "NEW-SOURCE"
        y, vectors, meta = predict_real(table, core.config())
        if pair:
            result = core.screen(table, y, vectors, "real", 2.0, 0.0, 2.0, True)
        else:
            result = {
                "retrieval": core.retrieve(
                    table, vectors, "real", True, 5, "train_val"
                ),
                "reference_scope": "train_val",
            }
        result["metadata"] = meta
        server.export_run(table, y, vectors, meta, result)
        return result["run_id"]

    def test_train_cache_excludes_validation_and_projects_new_points(self):
        ref, matrix = viz.library("real", scope="train_val")[1]
        model, xy, key = viz.fit_training_umap(ref, matrix, self.vcfg)
        changed = matrix.copy()
        changed[ref.split.eq("val")] = core.unit_rows(
            np.random.default_rng(17).normal(size=changed[ref.split.eq("val")].shape)
        )
        other, z, key2 = viz.fit_training_umap(ref, changed, self.vcfg)
        self.assertEqual(key, key2)
        np.testing.assert_array_equal(
            xy[ref.split.eq("train")], z[ref.split.eq("train")]
        )
        result = viz.project_run(self.save_prediction(), self.vcfg, sample_index=1)
        self.assertEqual(len(result["channels"]), 6)
        self.assertEqual(result["states"][0]["sample_id"], "NEW2")
        self.assertTrue(
            all(len(c["states"][0]["references"]) == 5 for c in result["channels"])
        )
        self.assertTrue(
            all(
                np.isfinite([c["states"][0]["x"], c["states"][0]["y"]]).all()
                for c in result["channels"]
            )
        )

    def test_pair_projection_two_sided_citations_and_export(self):
        run_id = self.save_prediction(pair=True)
        result = viz.project_run(run_id, self.vcfg, pair_index=0)
        self.assertEqual(len(result["states"]), 2)
        self.assertTrue(all(len(c["states"]) == 2 for c in result["channels"]))
        fake = FakeAnalysisGLM()
        with patch.object(llm, "build_metadata_llm", return_value=fake):
            analysis = exploration_analysis.summarize_pair(result)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(analysis["channels"]), 6)
        bad = {
            "channels": [
                {k: v for k, v in c.items() if k != "name"}
                for c in analysis["channels"]
            ]
        }
        bad["channels"][0]["supporting_reference_ids"] = ["S1-CH1-R1"]
        with self.assertRaisesRegex(ValueError, "两侧"):
            exploration_analysis.validate_pair_channels(bad, analysis["evidence"])
        out = self.root / "outputs/pair_export"
        out.mkdir()
        analysis_jobs.export_result(result, out)
        self.assertTrue((out / "six_channel_umap.png").is_file())
        self.assertTrue((out / "six_channel_umap.svg").is_file())
        self.assertTrue((out / "channel_coordinates.csv").is_file())

    def test_reference_mutation_blocks_old_run(self):
        run_id = self.save_prediction()
        path = core.path_for(core.config()["references"]["1"])
        previous = path.read_bytes()
        try:
            path.write_bytes(previous + b"\n")
            with self.assertRaisesRegex(ValueError, "已改变"):
                viz.project_run(run_id, self.vcfg, sample_index=0)
        finally:
            path.write_bytes(previous)

    def test_atlas_global_and_imported_cluster_summaries_reused(self):
        atlas = viz.atlas("real", "train", self.vcfg)
        self.assertEqual(len(atlas["global"]["points"]), 6 * 27)
        self.assertEqual(len(atlas["channels"]), 6)
        fake = FakeAnalysisGLM()
        with patch.object(llm, "build_metadata_llm", return_value=fake):
            summary = exploration_analysis.summarize_clusters(atlas)
        self.assertEqual(len(summary["clusters"]), 12)
        paths = {}
        for c in atlas["channels"]:
            directory = self.root / f"import_ch{c['channel']}"
            directory.mkdir()
            frame = pd.DataFrame(c["points"]).rename(
                columns={"x": "umap_1", "y": "umap_2"}
            )
            frame.to_csv(directory / "umap_coordinates_and_clusters.csv", index=False)
            items = {
                str(s["cluster_id"]): s
                for s in summary["clusters"]
                if s["channel"] == c["channel"]
            }
            (directory / "cluster_llm_summary.json").write_text(json.dumps(items))
            paths[str(c["channel"])] = str(directory)
        cfg = {**self.vcfg, "channel_directories": paths}
        reused = viz.atlas("real", "train", cfg)
        self.assertTrue(
            all(c["coordinate_source"].endswith(".csv") for c in reused["channels"])
        )
        with patch.object(
            llm, "build_metadata_llm", side_effect=AssertionError("unexpected API")
        ):
            imported = exploration_analysis.summarize_clusters(reused)
        self.assertTrue(
            all(
                c["source"] == "imported_existing_summary" for c in imported["clusters"]
            )
        )

    def test_tpe_search_no_trial_test_metrics_and_activation(self):
        search = {
            **optimization.DEFAULTS,
            "pretrain_trials": 2,
            "kinetics_trials": 2,
            "pretrain_epochs": 3,
            "kinetics_epochs": 4,
            "pruning": False,
            "pretrain_hidden_options": [[8, 6]],
            "kinetics_hidden_options": [[4]],
            "batch_sizes": [16],
            "seed": 7,
        }
        report = optimization.run_search(self.prepared, self.cfg, search)
        run = Path(report["run_dir"])
        self.assertEqual(set(report["studies"]), {"pretrain", "kinetics"})
        for path in run.glob("optimization/pretrain/trial_*/pretrained/metrics.json"):
            self.assertIsNone(json.loads(path.read_text())["test_cosine"])
        for path in run.glob("optimization/kinetics/trial_*/kinetics/metrics.json"):
            metrics = json.loads(path.read_text())
            self.assertFalse(metrics["test_evaluated"])
            self.assertIsNone(metrics["test"]["RMSE"])
        final = json.loads((run / "kinetics/metrics.json").read_text())
        self.assertTrue(final["test_evaluated"])
        self.assertIn("optimized_", core.config()["kinetics_dir"])
        pred, _, _ = predict_real(self.data.head(2), core.config())
        self.assertTrue(np.isfinite(pred).all())
        expected = {
            k: v.copy()
            for k, v in np.load(
                run / "kinetics/best_model_state.npz", allow_pickle=False
            ).items()
        }
        master_path = self.prepared / "dataset/master.csv"
        previous = master_path.read_bytes()
        try:
            changed = pd.read_csv(master_path)
            changed.loc[changed.split.eq("test"), "lgkp"] += 500
            changed.to_csv(master_path, index=False)
            repeated = optimization.run_search(self.prepared, self.cfg, search)
            for stage in report["studies"]:
                self.assertEqual(
                    report["studies"][stage]["best_value"],
                    repeated["studies"][stage]["best_value"],
                )
            state = np.load(
                Path(repeated["run_dir"]) / "kinetics/best_model_state.npz",
                allow_pickle=False,
            )
            for name, value in expected.items():
                np.testing.assert_array_equal(value, state[name])
        finally:
            master_path.write_bytes(previous)
        training_workflow.activate(
            self.prepared
        )  # Other tests retain their reference version.

    def test_optimizer_rejects_two_set_mode_before_search(self):
        root = self.root / "two_set"
        (root / "dataset").mkdir(parents=True)
        master = pd.read_csv(self.prepared / "dataset/master.csv")
        master.loc[master.split.eq("val"), "split"] = "train"
        master.to_csv(root / "dataset/master.csv", index=False)
        with self.assertRaisesRegex(ValueError, "验证集"):
            optimization.run_search(root, self.cfg, optimization.DEFAULTS)


if __name__ == "__main__":
    unittest.main()
