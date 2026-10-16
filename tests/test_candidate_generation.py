"""Behavior checks for grid precision, feasibility, provenance, and CSV handling."""

import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import candidate_generation as cg, core
import server
from http.server import ThreadingHTTPServer


def small_config():
    c = cg.defaults()
    for s in c["elements"].values():
        s.update(min=0, max=0, step=0.1)
    c["elements"]["Al"].update(min=7, max=8, step=0.3)
    c["elements"]["Ni"].update(min=30, max=30)
    c["elements"]["Nb"].update(min=0.05, max=0.05)
    c["co"] = {"min": 60, "max": 65}
    c["conditions"].update(Temperature=1000, E_Time=24)
    return c


class CandidateGridTests(unittest.TestCase):
    def test_grid_precision_fixed_elements_conditions_and_reproducibility(self):
        a, info = cg.generate_candidates(4, 123, small_config())
        b, other = cg.generate_candidates(4, 123, small_config())
        self.assertTrue(a.equals(b))
        self.assertEqual(info, other)
        self.assertEqual(set(a.Al), {7.0, 7.3, 7.6, 7.9})
        self.assertTrue((a.Nb == 0.05).all())
        self.assertTrue((a.Temperature == 1000).all())
        self.assertTrue((a.E_Time == 24).all())
        np.testing.assert_allclose(a[core.ELEMENTS].sum(axis=1), 100)
        self.assertEqual(info["eligible_grid_combinations"], 4)
        self.assertEqual(info["effective_grid_max"]["Al"], 7.9)
        self.assertEqual(len(core.validate(a)), 4)
        self.assertEqual(len(a.drop_duplicates(core.ELEMENTS)), 4)

    def test_default_candidates_valid_unique_and_bounded(self):
        a, info = cg.generate_candidates(180, 42)
        self.assertEqual(len(a), 180)
        self.assertEqual(len(a.drop_duplicates(core.ELEMENTS)), 180)
        self.assertTrue(a.Co.between(40, 80).all())
        self.assertLessEqual(info["grid_draws"], cg.MAX_DRAWS)
        self.assertEqual(len(core.validate(a)), 180)
        for e, s in info["settings"]["elements"].items():
            self.assertTrue(a[e].between(s["min"], s["max"]).all())
            np.testing.assert_allclose(
                (a[e] - s["min"]) / s["step"],
                np.rint((a[e] - s["min"]) / s["step"]),
                atol=1e-10,
            )

    def test_impossible_co_and_exhausted_grids_fail_explicitly(self):
        c = small_config()
        c["co"] = {"min": 0, "max": 10}
        with self.assertRaisesRegex(ValueError, "没有交集"):
            cg.generate_candidates(2, 1, c)
        c = small_config()
        with self.assertRaisesRegex(ValueError, "只有 4"):
            cg.generate_candidates(5, 1, c)
        c["co"] = {"min": 62.4, "max": 62.4}  # falls between grid points
        with self.assertRaisesRegex(ValueError, "只有 0"):
            cg.generate_candidates(2, 1, c)
        c = small_config()
        c["co"] = {"min": 62.05, "max": 62.05}
        with self.assertRaisesRegex(ValueError, "只有 1"):
            cg.generate_candidates(2, 1, c)

    def test_invalid_values_are_not_coerced_or_silently_rounded(self):
        for bad in [0, -1, float("nan"), float("inf"), True, "", 1e-7]:
            c = small_config()
            c["elements"]["Al"]["step"] = bad
            with self.subTest(step=bad), self.assertRaises(ValueError):
                cg.generate_candidates(2, 1, c)
        for bad in [2.5, True, None, 0, 801]:
            with self.subTest(n=bad), self.assertRaises(ValueError):
                cg.generate_candidates(bad)
        for bad in [-1, 2.3, 2**32, True]:
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                cg.generate_candidates(2, bad)
        for c in [
            {"elements": {"Unknown": {"min": 0}}},
            {"co": {"step": 0.5}},
            {"conditions": {"E_Time": 0}},
            {"elements": {"Al": {"min": 9, "max": 8}}},
        ]:
            with self.subTest(config=c), self.assertRaises(ValueError):
                cg.generate_candidates(2, 1, c)

    def test_large_grid_rejection_has_explicit_work_limit(self):
        c = small_config()
        c["elements"]["Al"].update(min=0, max=9, step=0.00001)
        c["elements"]["Ni"].update(min=0, max=1, step=1)
        c["co"] = {"min": 95.123456, "max": 95.123456}
        with patch.object(cg, "MAX_DRAWS", 200):
            with self.assertRaisesRegex(ValueError, "抽样上限"):
                cg.generate_candidates(2, 1, c)


class CandidateHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root_patch = patch.object(server, "ROOT", Path(cls.tmp.name))
        cls.root_patch.start()
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.http.server_port}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.root_patch.stop()
        cls.tmp.cleanup()

    def request(self, path, data=None):
        req = urllib.request.Request(
            self.base + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        return self.opener.open(req, timeout=20)

    def test_custom_parameters_reach_prediction_and_exports(self):
        with self.request("/api/candidates/config") as r:
            self.assertEqual(json.load(r)["defaults"], cg.defaults())
        with self.request(
            "/api/screen",
            {
                "mode": "demo",
                "candidate_source": "generated",
                "n": 4,
                "seed": 123,
                "generation_config": small_config(),
            },
        ) as r:
            result = json.load(r)
        self.assertEqual(result["counts"]["candidates"], 4)
        self.assertEqual(result["counts"]["all_pairs"], 6)
        info = result["candidate_generation"]
        with self.request(result["downloads"]["candidate_generation.json"]) as r:
            self.assertEqual(json.load(r), info)
        with self.request(result["downloads"]["manifest.json"]) as r:
            self.assertEqual(json.load(r)["candidate_generation"], info)
        with self.request(result["downloads"]["result.json"]) as r:
            self.assertEqual(json.load(r)["candidate_generation"], info)
        with self.request(result["downloads"]["prediction.csv"]) as r:
            frame = core.parse_table(r.read().decode("utf-8-sig"))
        self.assertEqual(set(frame.Al), {7.0, 7.3, 7.6, 7.9})
        self.assertTrue((frame.Temperature == 1000).all())

    def test_csv_uses_own_states_ignoring_generation_config(self):
        frame, _ = cg.generate_candidates(4, 123, small_config())
        with self.request(
            "/api/screen",
            {
                "mode": "demo",
                "candidate_source": "csv",
                "csv": frame.to_csv(index=False),
                "generation_config": {"bad": "ignored"},
                "n": 800,
            },
        ) as r:
            result = json.load(r)
        self.assertEqual(result["counts"]["candidates"], 4)
        self.assertEqual(result["candidate_generation"]["source"], "csv")
        self.assertNotIn("settings", result["candidate_generation"])

    def test_empty_csv_and_fractional_count_rejected(self):
        for data in [
            {"candidate_source": "csv", "csv": ""},
            {"csv": "   "},
            {"candidate_source": "generated", "n": 2.5},
        ]:
            with self.subTest(data=data), self.assertRaises(
                urllib.error.HTTPError
            ) as err:
                self.request("/api/screen", data)
            self.assertEqual(err.exception.code, 400)
            self.assertTrue(json.load(err.exception)["error"])


if __name__ == "__main__":
    unittest.main()
