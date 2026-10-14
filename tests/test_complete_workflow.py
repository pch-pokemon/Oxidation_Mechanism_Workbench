import json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import write_inputs
from science.dataset_manager import prepare_dataset
from science.reference_encoder import build_references
from science.training import train_mechanism_encoder, train_kinetics_model
from app.prediction_engine import predict_real
from app.training_workflow import DEFAULTS
from app import core
from science.neural_network import SharedNetwork, cosine_loss


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.d = write_inputs(self.root)
        self.cfg = {
            **DEFAULTS,
            "excel_path": str(self.root / "oxidation_dataset.xlsx"),
            "joint_json_path": str(self.root / "joint_six_channel.json"),
            "embedding_dim": 4,
            "raw_embeddings_dir": str(self.root / "raw"),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_auto_partition_shared_and_source_isolated(self):
        s = prepare_dataset(self.cfg, self.root / "run/dataset")
        master = pd.read_csv(self.root / "run/dataset/master.csv")
        self.assertTrue(s["has_validation"])
        self.assertEqual(master.groupby("Ref_source_num").split.nunique().max(), 1)
        self.assertEqual(sum(c["records"] for c in s["counts"].values()), 45)

    def test_manual_two_and_three_modes_and_overlap(self):
        for split, part in [
            ("train", self.d.iloc[:27]),
            ("val", self.d.iloc[27:36]),
            ("test", self.d.iloc[36:]),
        ]:
            part.to_excel(self.root / f"{split}.xlsx", index=False)
        cfg = {
            **self.cfg,
            "dataset_mode": "manual",
            "train_path": str(self.root / "train.xlsx"),
            "val_path": "",
            "test_path": str(self.root / "test.xlsx"),
        }
        self.assertFalse(prepare_dataset(cfg, self.root / "two")["has_validation"])
        cfg["val_path"] = str(self.root / "val.xlsx")
        self.assertTrue(prepare_dataset(cfg, self.root / "three")["has_validation"])
        cfg["test_path"] = cfg["train_path"]
        with self.assertRaisesRegex(ValueError, "重复"):
            prepare_dataset(cfg, self.root / "bad")

    def test_pca_fit_training_only(self):
        prepare_dataset(self.cfg, self.root / "run/dataset")
        build_references(
            self.root / "run/dataset/master.csv", self.root / "ref1", self.cfg
        )
        master = pd.read_csv(self.root / "run/dataset/master.csv")
        test = set(master.loc[master.split.ne("train"), "sample_id"])
        for ch in range(1, 7):
            path = self.root / f"raw/ch{ch}_raw.csv"
            d = pd.read_csv(path)
            cols = [c for c in d if c.startswith("raw_")]
            d.loc[d.sample_id.isin(test), cols] += 100
            d.to_csv(path, index=False)
        build_references(
            self.root / "run/dataset/master.csv", self.root / "ref2", self.cfg
        )
        for ch in range(1, 7):
            a = pd.read_csv(self.root / f"ref1/ch{ch}.csv")
            b = pd.read_csv(self.root / f"ref2/ch{ch}.csv")
            cols = [f"emb_{i}" for i in range(4)]
            np.testing.assert_allclose(
                a.loc[a.split.eq("train"), cols],
                b.loc[b.split.eq("train"), cols],
                atol=1e-6,
            )

    def test_real_training_prediction_and_test_label_isolation(self):
        run = self.root / "run"
        cfg = {
            **self.cfg,
            "hidden_dims": [12, 8],
            "kinetics_hidden_dims": [8, 4],
            "pretrain_epochs": 8,
            "kinetics_epochs": 40,
            "batch_size": 12,
            "patience": 10,
        }
        prepare_dataset(cfg, run / "dataset")
        build_references(run / "dataset/master.csv", run / "references", cfg)
        pm = train_mechanism_encoder(run, cfg)
        km = train_kinetics_model(run, cfg)
        self.assertEqual(km["selection"], "validation_RMSE")
        self.assertGreater(km["test"]["n"], 0)
        config = {
            "pretrained_dir": str(run / "pretrained"),
            "kinetics_dir": str(run / "kinetics"),
            "scaler_json": str(run / "pretrained/scaler.json"),
        }
        pred, v, meta = predict_real(self.d.head(3), config)
        self.assertEqual(meta["backend"], "numpy")
        self.assertEqual(v[1].shape, (3, 4))
        self.assertTrue(np.isfinite(pred).all())
        state1 = {
            k: x.copy()
            for k, x in np.load(run / "kinetics/best_model_state.npz").items()
        }
        master = pd.read_csv(run / "dataset/master.csv")
        master.loc[master.split.eq("test"), "lgkp"] += 500
        master.to_csv(run / "dataset/master.csv", index=False)
        train_kinetics_model(run, cfg)
        for k, v in np.load(run / "kinetics/best_model_state.npz").items():
            np.testing.assert_array_equal(v, state1[k])

    def test_no_validation_fixed_epochs(self):
        self.d.iloc[:30].to_excel(self.root / "train.xlsx", index=False)
        self.d.iloc[30:].to_excel(self.root / "test.xlsx", index=False)
        cfg = {
            **self.cfg,
            "dataset_mode": "manual",
            "train_path": str(self.root / "train.xlsx"),
            "val_path": "",
            "test_path": str(self.root / "test.xlsx"),
            "hidden_dims": [8],
            "kinetics_hidden_dims": [8],
            "pretrain_epochs": 3,
            "kinetics_epochs": 4,
        }
        run = self.root / "run"
        prepare_dataset(cfg, run / "dataset")
        build_references(run / "dataset/master.csv", run / "references", cfg)
        p = train_mechanism_encoder(run, cfg)
        k = train_kinetics_model(run, cfg)
        self.assertEqual(p["best_epoch"], 3)
        self.assertEqual(k["best_epoch"], 4)
        self.assertEqual(k["selection"], "fixed_epochs_no_validation")
        self.assertIsNone(k["val"]["RMSE"])


class GradientTests(unittest.TestCase):
    def test_shared_cosine_backprop_finite_difference(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(5, 4)).astype("float32")
        targets = {ch: rng.normal(size=(5, 3)).astype("float32") for ch in range(1, 7)}
        masks = {ch: np.array([True, True, True, True, False]) for ch in targets}
        net = SharedNetwork(4, [5], 3, dropout=0, seed=8)
        p = net.forward(x, True)
        loss, g = cosine_loss(p, targets, masks)
        grad = net.backward(g)
        for key, idx in [
            ("encoder.net.0.weight", (0, 0)),
            ("encoder.net.1.weight", (0,)),
            ("embedding_heads.1.weight", (0, 0)),
        ]:
            orig = net.state[key][idx]
            eps = 0.001
            net.state[key][idx] = orig + eps
            a = cosine_loss(net.forward(x, True), targets, masks)[0]
            net.state[key][idx] = orig - eps
            b = cosine_loss(net.forward(x, True), targets, masks)[0]
            net.state[key][idx] = orig
            self.assertAlmostEqual(
                float(grad[key][idx]), (a - b) / (2 * eps), delta=0.002
            )


if __name__ == "__main__":
    unittest.main()
