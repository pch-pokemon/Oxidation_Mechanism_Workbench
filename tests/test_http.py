import unittest, json, threading, urllib.request, urllib.error, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from http.server import ThreadingHTTPServer
from server import Handler


class HttpWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, data=None, headers=None):
        req = urllib.request.Request(
            self.base + path,
            data=None if data is None else json.dumps(data).encode(),
            headers=headers or {"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=20)

    def test_demo_predict_screen_export(self):
        with self.request("/api/demo") as r:
            d = json.load(r)
        with self.request("/api/validate", {"csv": d["csv"]}) as r:
            self.assertEqual(json.load(r)["count"], 96)
        with self.request(
            "/api/predict",
            {"csv": d["csv"], "mode": "demo", "retrieve": True, "exclude_source": True},
        ) as r:
            p = json.load(r)
        self.assertEqual(p["count"], 96)
        self.assertEqual(len(p["retrieval"]), 96)
        with self.request(
            "/api/screen", {"mode": "demo", "n": 180, "cross_source": True}
        ) as r:
            s = json.load(r)
        self.assertEqual(s["counts"]["all_pairs"], 16110)
        self.assertGreater(s["counts"]["target_pairs"], 0)
        for path in s["downloads"].values():
            with self.request(path) as r:
                self.assertTrue(r.read())

    def test_assets_and_local_origin(self):
        for path in ["/", "/app.js", "/style.css", "/template.csv"]:
            with self.request(path) as r:
                self.assertEqual(r.status, 200)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.request(
                "/api/config",
                {},
                headers={
                    "Origin": "https://example.com",
                    "Content-Type": "application/json",
                },
            )
        self.assertEqual(e.exception.code, 403)

    def test_invalid_features_return_error(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.request("/api/predict", {"sample": {"Al": 10}, "mode": "demo"})
        self.assertEqual(e.exception.code, 400)
        self.assertIn("缺少", json.load(e.exception)["error"])


if __name__ == "__main__":
    unittest.main()
