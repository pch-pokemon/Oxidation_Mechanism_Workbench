"""Inspectable NumPy MLP, BatchNorm, dropout and Adam; safe NPZ checkpoints."""

from __future__ import annotations
import numpy as np


class SharedNetwork:
    def __init__(self, in_dim, hidden_dims, embedding_dim, dropout=0.1, seed=42):
        self.hidden_dims = list(hidden_dims)
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.state = {}
        self.params = []
        width = in_dim
        for i, h in enumerate(hidden_dims):
            linear = f"encoder.net.{i*4}"
            bn = f"encoder.net.{i*4+1}"
            self._linear(linear, width, h)
            self.state[bn + ".weight"] = np.ones(h, dtype="float32")
            self.state[bn + ".bias"] = np.zeros(h, dtype="float32")
            self.params.extend([bn + ".weight", bn + ".bias"])
            self.state[bn + ".running_mean"] = np.zeros(h, dtype="float32")
            self.state[bn + ".running_var"] = np.ones(h, dtype="float32")
            width = h
        for ch in range(1, 7):
            self._linear(f"embedding_heads.{ch}", width, embedding_dim)

    def _linear(self, name, ni, no):
        self.state[name + ".weight"] = self.rng.normal(
            0, np.sqrt(2 / ni), (no, ni)
        ).astype("float32")
        self.state[name + ".bias"] = np.zeros(no, dtype="float32")
        self.params.extend([name + ".weight", name + ".bias"])

    def encode(self, x, training=False):
        a = x
        self.cache = []
        for i, h in enumerate(self.hidden_dims):
            ln = f"encoder.net.{i*4}"
            bn = f"encoder.net.{i*4+1}"
            w = self.state[ln + ".weight"]
            u = a @ w.T + self.state[ln + ".bias"]
            if training:
                mean = u.mean(0)
                var = u.var(0)
                self.state[bn + ".running_mean"] = (
                    0.9 * self.state[bn + ".running_mean"] + 0.1 * mean
                )
                unbiased = var * len(u) / max(len(u) - 1, 1)
                self.state[bn + ".running_var"] = (
                    0.9 * self.state[bn + ".running_var"] + 0.1 * unbiased
                )
            else:
                mean = self.state[bn + ".running_mean"]
                var = self.state[bn + ".running_var"]
            inv = 1 / np.sqrt(var + 1e-5)
            uhat = (u - mean) * inv
            v = uhat * self.state[bn + ".weight"] + self.state[bn + ".bias"]
            relu = v > 0
            b = np.maximum(v, 0)
            drop = (
                (self.rng.random(b.shape) >= self.dropout).astype("float32")
                / (1 - self.dropout)
                if training and self.dropout
                else np.ones_like(b)
            )
            self.cache.append((ln, bn, a, uhat, inv, relu, drop))
            a = b * drop
        return a

    def forward(self, x, training=False):
        z = self.encode(x, training)
        self.z = z
        return {
            ch: z @ self.state[f"embedding_heads.{ch}.weight"].T
            + self.state[f"embedding_heads.{ch}.bias"]
            for ch in range(1, 7)
        }

    def backward(self, channel_grads):
        grads = {}
        dz = np.zeros_like(self.z)
        for ch, g in channel_grads.items():
            name = f"embedding_heads.{ch}"
            grads[name + ".weight"] = g.T @ self.z
            grads[name + ".bias"] = g.sum(0)
            dz += g @ self.state[name + ".weight"]
        for ln, bn, a, uhat, inv, relu, drop in reversed(self.cache):
            dv = dz * drop * relu
            grads[bn + ".weight"] = (dv * uhat).sum(0)
            grads[bn + ".bias"] = dv.sum(0)
            duh = dv * self.state[bn + ".weight"]
            du = inv * (duh - duh.mean(0) - uhat * (duh * uhat).mean(0))
            grads[ln + ".weight"] = du.T @ a
            grads[ln + ".bias"] = du.sum(0)
            dz = du @ self.state[ln + ".weight"]
        return grads


class KineticsHead:
    def __init__(self, in_dim, hidden_dims=(64, 32), dropout=0.1, seed=43):
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.state = {}
        self.params = []
        width = in_dim
        for i, h in enumerate([*hidden_dims, 1]):
            name = f"kinetics_head.{i*3}"
            self.state[name + ".weight"] = self.rng.normal(
                0, np.sqrt(2 / width), (h, width)
            ).astype("float32")
            self.state[name + ".bias"] = np.zeros(h, dtype="float32")
            self.params.extend([name + ".weight", name + ".bias"])
            width = h

    def forward(self, x, training=False):
        a = x
        self.cache = []
        for i in range(len(self.hidden_dims) + 1):
            name = f"kinetics_head.{i*3}"
            u = a @ self.state[name + ".weight"].T + self.state[name + ".bias"]
            last = i == len(self.hidden_dims)
            relu = np.ones_like(u, dtype=bool) if last else u > 0
            drop = (
                (self.rng.random(u.shape) >= self.dropout).astype("float32")
                / (1 - self.dropout)
                if training and self.dropout and not last
                else np.ones_like(u)
            )
            self.cache.append((name, a, relu, drop))
            a = u if last else np.maximum(u, 0) * drop
        return a[:, 0]

    def backward(self, g):
        g = g[:, None]
        out = {}
        for name, a, relu, drop in reversed(self.cache):
            du = g * relu * drop
            out[name + ".weight"] = du.T @ a
            out[name + ".bias"] = du.sum(0)
            g = du @ self.state[name + ".weight"]
        return out


class Adam:
    def __init__(self, state, lr=0.001, weight_decay=1e-5):
        self.state = state
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, grads):
        self.t += 1
        norm = np.sqrt(
            sum(float((g.astype("float64") ** 2).sum()) for g in grads.values())
        )
        factor = min(1.0, 5.0 / max(norm, 1e-12))
        for key, g in grads.items():
            g = g * factor + self.weight_decay * self.state[key]
            self.m[key] = 0.9 * self.m.get(key, np.zeros_like(g)) + 0.1 * g
            self.v[key] = 0.999 * self.v.get(key, np.zeros_like(g)) + 0.001 * g * g
            m = self.m[key] / (1 - 0.9**self.t)
            v = self.v[key] / (1 - 0.999**self.t)
            self.state[key] -= self.lr * m / (np.sqrt(v) + 1e-8)


def cosine_loss(predictions, targets, masks):
    valid = [ch for ch in predictions if masks[ch].any()]
    if not valid:
        raise ValueError("该批次没有有效机制监督。")
    grads = {}
    loss = 0.0
    for ch in valid:
        mask = masks[ch]
        p = predictions[ch][mask]
        t = targets[ch][mask]
        pn = np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        tn = np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
        tu = t / tn
        cos = (p * tu).sum(1, keepdims=True) / pn
        loss += float((1 - cos).mean()) / len(valid)
        g = (cos * p / (pn * pn) - tu / pn) / (len(p) * len(valid))
        all_g = np.zeros_like(predictions[ch])
        all_g[mask] = g
        grads[ch] = all_g
    return loss, grads


def clone_state(state):
    return {k: v.copy() for k, v in state.items()}


def finite_state(state):
    if any(not np.isfinite(v).all() for v in state.values()):
        raise ValueError("训练出现非有限参数，请检查数据与学习率。")
