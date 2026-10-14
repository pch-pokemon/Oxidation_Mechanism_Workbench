"""Fixed-parameter mechanism pretraining and frozen kinetics transfer on one saved split."""

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app.core import FEATURES, KEYS, normalize_keys
from science.neural_network import (
    SharedNetwork,
    KineticsHead,
    Adam,
    cosine_loss,
    clone_state,
    finite_state,
)


def dump_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def metrics(y, p):
    if not len(y):
        return {"n": 0, "R2": None, "MAE": None, "RMSE": None}
    residual = y - p
    den = float(((y - y.mean()) ** 2).sum())
    return {
        "n": len(y),
        "R2": (
            float(1 - (residual**2).sum() / den) if len(y) > 1 and den > 1e-12 else None
        ),
        "MAE": float(np.abs(residual).mean()),
        "RMSE": float(np.sqrt((residual**2).mean())),
    }


def data_arrays(run_dir):
    run = Path(run_dir)
    master = normalize_keys(
        pd.read_csv(run / "dataset/master.csv", dtype={k: str for k in KEYS})
    )
    targets = {}
    masks = {}
    info = json.loads((run / "references/reference_summary.json").read_text())
    dim = info["embedding_dim"]
    for ch in range(1, 7):
        ref = normalize_keys(
            pd.read_csv(run / f"references/ch{ch}.csv", dtype={k: str for k in KEYS})
        )
        ref = master[KEYS].merge(ref, on=KEYS, how="left", validate="one_to_one")
        cols = [f"emb_{i}" for i in range(dim)]
        targets[ch] = ref[cols].to_numpy("float32")
        masks[ch] = ref.is_empty.eq(0).to_numpy()
    return master, targets, masks, dim


def mechanism_score(model, x, indices, targets, masks):
    if not len(indices):
        return None
    p = model.forward(x[indices], False)
    scores = []
    for ch in range(1, 7):
        mask = masks[ch][indices]
        if mask.any():
            q = p[ch][mask]
            t = targets[ch][indices][mask]
            scores.append(
                float(
                    (
                        (q * t).sum(1)
                        / np.maximum(
                            np.linalg.norm(q, axis=1) * np.linalg.norm(t, axis=1), 1e-8
                        )
                    ).mean()
                )
            )
    return float(np.mean(scores)) if scores else None


def train_mechanism_encoder(run_dir, cfg, evaluate_test=True, epoch_callback=None):
    run = Path(run_dir)
    out = run / "pretrained"
    out.mkdir(exist_ok=True)
    master, targets, masks, dim = data_arrays(run)
    train_all = np.flatnonzero(master.split.eq("train"))
    has = np.column_stack(list(masks.values())).any(1)
    train = np.flatnonzero(master.split.eq("train").to_numpy() & has)
    val = np.flatnonzero(master.split.eq("val").to_numpy() & has)
    test = np.flatnonzero(master.split.eq("test").to_numpy() & has)
    if len(train) < 3:
        raise ValueError("有效机制训练样本至少需要三条。")
    scaler = StandardScaler().fit(master.iloc[train_all][FEATURES].to_numpy(float))
    x = scaler.transform(master[FEATURES].to_numpy(float)).astype("float32")
    dump_json(
        out / "scaler.json",
        {
            "feature_cols": FEATURES,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "fit_split": "train",
        },
    )
    dims = cfg["hidden_dims"]
    model = SharedNetwork(
        len(FEATURES), dims, dim, float(cfg["dropout"]), int(cfg["seed"])
    )
    opt = Adam(model.state, float(cfg["pretrain_lr"]), float(cfg["weight_decay"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    best = None
    best_score = -np.inf
    wait = 0
    history = []
    best_epoch = 0
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["pretrain_epochs"]) + 1):
        order = rng.permutation(train)
        losses = []
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            pred = model.forward(x[ids], True)
            loss, g = cosine_loss(
                pred,
                {ch: t[ids] for ch, t in targets.items()},
                {ch: m[ids] for ch, m in masks.items()},
            )
            opt.step(model.backward(g))
            losses.append(loss)
        score = mechanism_score(model, x, val, targets, masks)
        finite_state(model.state)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_cosine": score,
            }
        )
        if epoch_callback is not None:
            epoch_callback(epoch, score)
        if score is None:
            best = clone_state(model.state)
            best_epoch = epoch
        elif score > best_score + 1e-8:
            best_score = score
            best = clone_state(model.state)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f'预训练 epoch={epoch} loss={history[-1]["train_loss"]:.5f} val_cos={score}',
                flush=True,
            )
        if score is not None and wait >= int(cfg["patience"]):
            break
    model.state = best
    np.savez_compressed(out / "best_pretrain_model.npz", **best)
    architecture = {
        "hidden_dims": dims,
        "dropout": cfg["dropout"],
        "emb_dim": dim,
        "selected_channels": list(range(1, 7)),
        "feature_cols": FEATURES,
        "backend": "numpy",
        "best_epoch": best_epoch,
        "selection": "validation_cosine" if len(val) else "fixed_epochs_no_validation",
        "training_parameters": cfg,
    }
    dump_json(out / "pretrain_config.json", architecture)
    master[KEYS + ["split"]].to_csv(
        out / "shared_data_split.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    channel_scores = {}
    for ch in range(1, 7):
        active = masks[ch][val]
        if active.any():
            predicted = model.forward(x[val], False)[ch][active]
            target = targets[ch][val][active]
            channel_scores[str(ch)] = float(
                (
                    (predicted * target).sum(1)
                    / np.maximum(
                        np.linalg.norm(predicted, axis=1)
                        * np.linalg.norm(target, axis=1),
                        1e-8,
                    )
                ).mean()
            )
        else:
            channel_scores[str(ch)] = None
    result = {
        "train_cosine": mechanism_score(model, x, train, targets, masks),
        "val_cosine": mechanism_score(model, x, val, targets, masks),
        "test_cosine": (
            mechanism_score(model, x, test, targets, masks) if evaluate_test else None
        ),
        "validation_channels": channel_scores,
        "test_evaluated": evaluate_test,
        "best_epoch": best_epoch,
        "selection": architecture["selection"],
    }
    dump_json(out / "metrics.json", result)
    print("预训练完成：", result, flush=True)
    return result


def train_kinetics_model(run_dir, cfg, evaluate_test=True, epoch_callback=None):
    run = Path(run_dir)
    out = run / "kinetics"
    out.mkdir(exist_ok=True)
    master, targets, masks, dim = data_arrays(run)
    pre = json.loads((run / "pretrained/pretrain_config.json").read_text())
    scaler = json.loads((run / "pretrained/scaler.json").read_text())
    backbone = SharedNetwork(
        len(FEATURES), pre["hidden_dims"], dim, pre["dropout"], int(cfg["seed"])
    )
    backbone.state = {
        k: v.copy()
        for k, v in np.load(
            run / "pretrained/best_pretrain_model.npz", allow_pickle=False
        ).items()
    }
    x = (
        (master[FEATURES].to_numpy(float) - np.asarray(scaler["mean"]))
        / np.asarray(scaler["scale"])
    ).astype("float32")
    z = backbone.encode(x, False).copy()
    y = pd.to_numeric(master.lgkp, errors="coerce").to_numpy(float).astype("float32")
    usable = np.isfinite(y)
    splits = {
        s: np.flatnonzero(master.split.eq(s).to_numpy() & usable)
        for s in ["train", "val", "test"]
    }
    if len(splits["train"]) < 2 or not len(splits["test"]):
        raise ValueError("迁移训练需要至少两条有效训练 lgkp 和至少一条测试 lgkp。")
    if master.split.eq("val").any() and not len(splits["val"]):
        raise ValueError(
            "提供了验证集，但没有有效验证 lgkp；请补全标签或明确使用两集合模式。"
        )
    model = KineticsHead(
        z.shape[1],
        cfg["kinetics_hidden_dims"],
        float(cfg["dropout"]),
        int(cfg["seed"]) + 1,
    )
    # Initialize output bias at training mean, without reading validation/test labels.
    last = f'kinetics_head.{len(cfg["kinetics_hidden_dims"])*3}.bias'
    model.state[last][:] = float(y[splits["train"]].mean())
    opt = Adam(model.state, float(cfg["kinetics_lr"]), float(cfg["weight_decay"]))
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    history = []
    best = None
    best_rmse = np.inf
    best_epoch = 0
    wait = 0
    for epoch in range(1, int(cfg["kinetics_epochs"]) + 1):
        order = rng.permutation(splits["train"])
        losses = []
        for start in range(0, len(order), int(cfg["batch_size"])):
            ids = order[start : start + int(cfg["batch_size"])]
            pred = model.forward(z[ids], True)
            res = pred - y[ids]
            losses.append(float((res * res).mean()))
            opt.step(model.backward(2 * res / len(ids)))
        finite_state(model.state)
        vp = model.forward(z[splits["val"]], False)
        vm = metrics(y[splits["val"]], vp)
        rmse = vm["RMSE"]
        history.append(
            {"epoch": epoch, "train_mse": float(np.mean(losses)), "val_RMSE": rmse}
        )
        if epoch_callback is not None:
            epoch_callback(epoch, rmse)
        if rmse is None:
            best = clone_state(model.state)
            best_epoch = epoch
        elif rmse < best_rmse - 1e-8:
            best_rmse = rmse
            best = clone_state(model.state)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 20 == 0:
            print(
                f'迁移训练 epoch={epoch} loss={history[-1]["train_mse"]:.5f} val_RMSE={rmse}',
                flush=True,
            )
        if rmse is not None and wait >= int(cfg["patience"]):
            break
    model.state = best
    state = {"backbone." + k: v for k, v in backbone.state.items()}
    state.update(best)
    np.savez_compressed(out / "best_model_state.npz", **state)
    selection = (
        "validation_RMSE" if len(splits["val"]) else "fixed_epochs_no_validation"
    )
    dump_json(
        out / "transfer_config.json",
        {
            "kinetics_hidden_dims": cfg["kinetics_hidden_dims"],
            "kinetics_dropout": cfg["dropout"],
            "transfer_mode": "frozen",
            "backend": "numpy",
            "best_epoch": best_epoch,
            "selection": selection,
        },
    )
    result = {
        s: (
            metrics(y[ids], model.forward(z[ids], False))
            if (evaluate_test or s != "test")
            else {"n": len(ids), "R2": None, "MAE": None, "RMSE": None}
        )
        for s, ids in splits.items()
    }
    result.update(
        best_epoch=best_epoch, selection=selection, test_evaluated=evaluate_test
    )
    dump_json(out / "metrics.json", result)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    if evaluate_test:
        pred = master[KEYS + ["split"]].copy()
        pred["lgkp_true"] = y
        pred["lgkp_pred"] = model.forward(z, False)
        pred.to_csv(out / "prediction.csv", index=False, encoding="utf-8-sig")
    print(
        "迁移训练完成，测试集仅最终评估：",
        json.dumps(result, ensure_ascii=False),
        flush=True,
    )
    return result
