import contextlib, json, shutil, subprocess, sys, tempfile, unittest, threading, urllib.request
from pathlib import Path
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import write_inputs
from app.training_workflow import DEFAULTS
from app import core, training_workflow, task_manager, mechanism_analysis
import server
from http.server import ThreadingHTTPServer


class PipelineActivationTests(unittest.TestCase):
    def test_cli_pipeline_activates_real_predict_retrieve_and_analysis_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = Path(__file__).resolve().parents[1]
            for name in ["app", "science", "tools"]:
                shutil.copytree(
                    src / name,
                    root / name,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            (root / "config").mkdir()
            (root / "outputs").mkdir()
            d = write_inputs(root / "data")
            cfg = {
                **DEFAULTS,
                "joint_json_path": str(root / "data/joint_six_channel.json"),
                "excel_path": str(root / "data/oxidation_dataset.xlsx"),
                "raw_embeddings_dir": str(root / "data/raw"),
                "embedding_dim": 4,
                "hidden_dims": [12, 8],
                "kinetics_hidden_dims": [8],
                "pretrain_epochs": 3,
                "kinetics_epochs": 4,
                "batch_size": 16,
            }
            (root / "task.json").write_text(json.dumps(cfg))
            job = root / "outputs/job"
            job.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "tools/workflow.py"),
                    "pipeline",
                    str(root / "task.json"),
                    str(job),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            active = json.loads((root / "config/workspace.json").read_text())
            self.assertTrue(
                (root / active["kinetics_dir"] / "best_model_state.npz").is_file()
            )
            with contextlib.ExitStack() as stack:
                for mod in [
                    server,
                    core,
                    training_workflow,
                    task_manager,
                    mechanism_analysis,
                ]:
                    stack.enter_context(patch.object(mod, "ROOT", root))
                http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
                threading.Thread(target=http.serve_forever, daemon=True).start()
                base = "http://127.0.0.1:" + str(http.server_port)
                try:
                    sample = d.iloc[0].to_dict()
                    sample["sample_id"] = "NEW-INPUT"
                    sample["Ref_source_num"] = "UNSEEN-SOURCE"

                    def call(path, body=None):
                        req = urllib.request.Request(
                            base + path,
                            data=None if body is None else json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(req) as r:
                            return json.load(r)

                    prediction = call(
                        "/api/predict",
                        {
                            "sample": sample,
                            "mode": "real",
                            "retrieve": True,
                            "reference_scope": "train_val",
                        },
                    )
                    self.assertEqual(prediction["metadata"]["backend"], "numpy")
                    run_id = prediction["run_id"]
                    self.assertEqual(len(prediction["retrieval"][0]["channels"]), 6)
                    with urllib.request.urlopen(
                        base + prediction["downloads"]["result.json"]
                    ) as r:
                        saved = json.load(r)
                    self.assertEqual(saved["run_id"], run_id)
                    self.assertTrue(
                        all(
                            "\\" not in path
                            for path in prediction["downloads"].values()
                        )
                    )
                    recent = call("/api/analysis/runs")
                    self.assertEqual(
                        recent["runs"][0]["samples"][0]["sample_id"], "NEW-INPUT"
                    )
                    evidence = call(
                        "/api/analysis/preview", {"run_id": run_id, "sample_index": 0}
                    )
                    self.assertFalse(evidence["synthetic_demo"])
                    self.assertEqual(
                        evidence["sample_context"]["sample_id"], "NEW-INPUT"
                    )
                    for c in evidence["channels"]:
                        self.assertEqual(len(c["references"]), 5)
                        self.assertTrue(
                            all(
                                r["reference_split"] in ["train", "val"]
                                for r in c["references"]
                            )
                        )
                finally:
                    http.shutdown()
                    http.server_close()


if __name__ == "__main__":
    unittest.main()
