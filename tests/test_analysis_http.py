import json, sys, time, threading, unittest, urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from http.server import ThreadingHTTPServer
from server import Handler
from app import core, mechanism_analysis as analysis
from test_mechanism_analysis import FakeGLM


class AnalysisHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, path, body=None):
        request = urllib.request.Request(
            self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as r:
            return json.load(r)

    def test_http_prediction_preview_async_analysis_export(self):
        sample = core.generate(2).iloc[0].to_dict()
        p = self.call(
            "/api/predict", {"mode": "demo", "sample": sample, "retrieve": True}
        )
        body = {"run_id": p["run_id"], "sample_index": 0}
        evidence = self.call("/api/analysis/preview", body)
        self.assertEqual(len(evidence["channels"]), 6)
        self.assertTrue(
            any(
                r["run_id"] == p["run_id"]
                for r in self.call("/api/analysis/runs")["runs"]
            )
        )
        analysis.JOBS.clear()
        fake = FakeGLM()
        with patch.object(
            analysis,
            "resolve_api_key",
            return_value=("http-private-test-key", "API_KEY"),
        ), patch.object(
            analysis.importlib.util, "find_spec", return_value=object()
        ), patch.object(
            analysis, "build_metadata_llm", return_value=fake
        ):
            job = self.call("/api/analysis/start", body)
            deadline = time.monotonic() + 5
            while job["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.03)
                job = self.call("/api/analysis/job/" + job["id"])
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(len(fake.calls), 2)
        for url in job["downloads"].values():
            with urllib.request.urlopen(self.base + url) as r:
                data = r.read().decode()
            self.assertNotIn("http-private-test-key", data)

    def test_local_config_only_and_analysis_js_served(self):
        result = self.call("/api/analysis/config")
        self.assertEqual(result["config"]["model"], "glm-4-plus")
        self.assertNotIn("api_key", result)
        with urllib.request.urlopen(self.base + "/analysis.js") as r:
            self.assertIn("renderMechanismAnalysis", r.read().decode())


if __name__ == "__main__":
    unittest.main()
