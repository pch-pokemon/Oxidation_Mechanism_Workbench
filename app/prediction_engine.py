"""CPU inference matching the supplied encoder and transfer state_dict names."""

import hashlib, json
from pathlib import Path
import numpy as np
from app.core import FEATURES, path_for, read_json


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def predict_torch(df, cfg):
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise ValueError("真实模型需要 PyTorch。请安装 requirements-model.txt。") from e
    torch.set_num_threads(2)
    root = path_for(cfg["pretrained_dir"])
    ft = path_for(cfg["kinetics_dir"])
    arch_path = next(
        (
            root / n
            for n in [
                "trial_summary.json",
                "pretrain_config.json",
                "base_search_config.json",
            ]
            if (root / n).is_file()
        ),
        None,
    )
    if arch_path is None:
        raise ValueError("缺少预训练架构配置。")
    saved = read_json(arch_path)
    arch = dict(saved.get("config", saved))
    if "hidden_dims" in saved:
        arch["hidden_dims"] = saved["hidden_dims"]
    for key in ["hidden_dims", "dropout", "emb_dim", "selected_channels"]:
        if key not in arch:
            raise ValueError("架构缺少字段：" + key)
    if arch.get("feature_cols", FEATURES) != FEATURES:
        raise ValueError("模型特征顺序与软件的 29 个特征不一致。")
    if sorted(arch["selected_channels"]) != list(range(1, 7)):
        raise ValueError("本软件的联合预测要求完整六通道模型。")
    tc = read_json(ft / "transfer_config.json")
    scaler_path = path_for(cfg["scaler_json"])
    scaler = read_json(scaler_path)
    if scaler["feature_cols"] != FEATURES:
        raise ValueError("标准化器特征顺序不一致。")
    mean = np.array(scaler["mean"], float)
    scale = np.array(scaler["scale"], float)
    if (
        mean.shape != (29,)
        or scale.shape != (29,)
        or not np.isfinite([mean, scale]).all()
        or (scale <= 0).any()
    ):
        raise ValueError("标准化器参数无效。")

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            width = 29
            for h in arch["hidden_dims"]:
                layers += [
                    nn.Linear(width, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(arch["dropout"]),
                ]
                width = h
            self.net = nn.Sequential(*layers)
            self.out_dim = width

        def forward(self, x):
            return self.net(x)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.embedding_heads = nn.ModuleDict(
                {
                    str(c): nn.Linear(self.encoder.out_dim, arch["emb_dim"])
                    for c in range(1, 7)
                }
            )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            layers = []
            width = self.backbone.encoder.out_dim
            for h in tc["kinetics_hidden_dims"]:
                layers += [
                    nn.Linear(width, h),
                    nn.ReLU(),
                    nn.Dropout(tc["kinetics_dropout"]),
                ]
                width = h
            self.kinetics_head = nn.Sequential(*layers, nn.Linear(width, 1))

        def forward(self, x):
            z = self.backbone.encoder(x)
            return self.kinetics_head(z).squeeze(-1), {
                int(c): head(z) for c, head in self.backbone.embedding_heads.items()
            }

    state_path = ft / "best_model_state.pt"
    if not state_path.is_file():
        raise ValueError("迁移模型 best_model_state.pt 不存在。")
    # No unsafe pickle fallback. This app never accepts a model binary via web upload.
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state = {
        k: v
        for k, v in state.items()
        if k.startswith(
            ("backbone.encoder.", "backbone.embedding_heads.", "kinetics_head.")
        )
    }
    model = Model()
    model.load_state_dict(state, strict=True)
    model.eval()
    x = ((df[FEATURES].to_numpy(float) - mean) / scale).astype("float32")
    ys = []
    vs = {c: [] for c in range(1, 7)}
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            y, v = model(torch.from_numpy(x[start : start + 128]))
            ys.append(y.numpy())
            for c in vs:
                vs[c].append(v[c].numpy())
    y = np.concatenate(ys)
    vectors = {c: np.concatenate(vs[c]) for c in vs}
    if not np.isfinite(y).all() or any(
        not np.isfinite(v).all() for v in vectors.values()
    ):
        raise ValueError("模型输出包含非有限数值。")
    return (
        y,
        vectors,
        {
            "mode": "real",
            "label": "本地研究模型 / CPU",
            "embedding_dim": int(arch["emb_dim"]),
            "weights_sha256": file_hash(state_path),
            "architecture_sha256": file_hash(arch_path),
            "transfer_config_sha256": file_hash(ft / "transfer_config.json"),
            "scaler_sha256": file_hash(scaler_path),
            "transfer_mode": tc.get("transfer_mode", "unknown"),
        },
    )


def predict_real(df, cfg):
    from science.neural_network import SharedNetwork, KineticsHead

    root = path_for(cfg["pretrained_dir"])
    ft = path_for(cfg["kinetics_dir"])
    state_path = ft / "best_model_state.npz"
    tc = read_json(ft / "transfer_config.json")
    if tc.get("backend") != "numpy":
        return predict_torch(df, cfg)
    if not state_path.is_file():
        raise ValueError("NumPy 模型权重不存在。")
    arch_path = root / "pretrain_config.json"
    arch = read_json(arch_path)
    scaler_path = path_for(cfg["scaler_json"])
    scaler = read_json(scaler_path)
    if scaler["feature_cols"] != FEATURES or arch.get("feature_cols") != FEATURES:
        raise ValueError("模型特征顺序不一致。")
    mean = np.asarray(scaler["mean"])
    scale = np.asarray(scaler["scale"])
    if (
        mean.shape != (29,)
        or scale.shape != (29,)
        or not np.isfinite([mean, scale]).all()
        or (scale <= 0).any()
    ):
        raise ValueError("标准化参数无效。")
    with np.load(state_path, allow_pickle=False) as data:
        state = {k: data[k].copy() for k in data.files}
    net = SharedNetwork(29, arch["hidden_dims"], arch["emb_dim"], arch["dropout"])
    head = KineticsHead(
        arch["hidden_dims"][-1], tc["kinetics_hidden_dims"], tc["kinetics_dropout"]
    )
    for key, value in net.state.items():
        stored = state.get("backbone." + key)
        if stored is None or stored.shape != value.shape:
            raise ValueError("共享主干权重与架构不匹配：" + key)
        net.state[key] = stored
    for key, value in head.state.items():
        stored = state.get(key)
        if stored is None or stored.shape != value.shape:
            raise ValueError("动力学权重与架构不匹配：" + key)
        head.state[key] = stored
    x = ((df[FEATURES].to_numpy(float) - mean) / scale).astype("float32")
    v = net.forward(x, False)
    y = head.forward(net.z, False)
    if not np.isfinite(y).all() or any(not np.isfinite(a).all() for a in v.values()):
        raise ValueError("模型输出非有限。")
    return (
        y,
        v,
        {
            "mode": "real",
            "label": "本地研究模型 / NumPy MLP",
            "embedding_dim": arch["emb_dim"],
            "backend": "numpy",
            "weights_sha256": file_hash(state_path),
            "architecture_sha256": file_hash(arch_path),
            "transfer_config_sha256": file_hash(ft / "transfer_config.json"),
            "scaler_sha256": file_hash(scaler_path),
            "transfer_mode": "frozen",
        },
    )
