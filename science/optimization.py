"""Optuna TPE searches with validation-only objectives and final held-out evaluation."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from app import core
from app.training_workflow import write_json

DEFAULTS = {
    "target": "both",
    "pretrain_trials": 10,
    "kinetics_trials": 10,
    "pretrain_epochs": 150,
    "kinetics_epochs": 400,
    "seed": 42,
    "pretrain_hidden_options": [[128, 128, 64], [64, 64, 32], [128, 64]],
    "kinetics_hidden_options": [[64, 32], [32, 16], [64]],
    "batch_sizes": [16, 32, 64],
    "dropout_range": [0.0, 0.3],
    "pretrain_lr_range": [0.0001, 0.005],
    "kinetics_lr_range": [0.0001, 0.005],
    "weight_decay_range": [0.000001, 0.001],
    "pruning": True,
}


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS):
        raise ValueError("优化配置字段不完整。")
    if cfg["target"] not in ["pretrain", "kinetics", "both"]:
        raise ValueError("优化目标应为 pretrain、kinetics 或 both。")
    for field in [
        "pretrain_trials",
        "kinetics_trials",
        "pretrain_epochs",
        "kinetics_epochs",
    ]:
        if type(cfg[field]) is not int or not 1 <= cfg[field] <= 5000:
            raise ValueError("优化次数或轮数无效：" + field)
    if (
        type(cfg["seed"]) is not int
        or not 0 <= cfg["seed"] <= 100000
        or type(cfg["pruning"]) is not bool
    ):
        raise ValueError("优化种子或 pruning 无效。")
    for field in ["pretrain_hidden_options", "kinetics_hidden_options"]:
        options = cfg[field]
        if (
            not isinstance(options, list)
            or not options
            or any(
                not isinstance(row, list)
                or not row
                or any(type(v) is not int or not 2 <= v <= 1024 for v in row)
                for row in options
            )
        ):
            raise ValueError("隐藏层选项无效。")
    if (
        not isinstance(cfg["batch_sizes"], list)
        or not cfg["batch_sizes"]
        or any(type(v) is not int or not 2 <= v <= 1024 for v in cfg["batch_sizes"])
    ):
        raise ValueError("batch_sizes 无效。")
    for field in [
        "dropout_range",
        "pretrain_lr_range",
        "kinetics_lr_range",
        "weight_decay_range",
    ]:
        pair = cfg[field]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in pair)
            or pair[0] > pair[1]
        ):
            raise ValueError("优化参数范围无效：" + field)
    if not 0 <= cfg["dropout_range"][0] <= cfg["dropout_range"][1] < 0.8:
        raise ValueError("Dropout 搜索范围应在 0—0.8 内。")
    for field in ["pretrain_lr_range", "kinetics_lr_range"]:
        if not 0 < cfg[field][0] <= cfg[field][1] <= 0.1:
            raise ValueError("学习率搜索范围无效。")
    if not 0 < cfg["weight_decay_range"][0] <= cfg["weight_decay_range"][1] < 1:
        raise ValueError("Weight decay 搜索范围须大于零且小于 1。")
    return dict(cfg)


def get_config():
    path = core.ROOT / "config/optimization.json"
    return validate_config(core.read_json(path) if path.is_file() else DEFAULTS)


def save_config(cfg):
    cfg = validate_config(cfg)
    write_json(core.ROOT / "config/optimization.json", cfg)
    return {"saved": True, "config": cfg}


def trial_config(trial, base, search, stage):
    cfg = dict(base)
    cfg["seed"] = search["seed"]
    arch_field = (
        "pretrain_hidden_options" if stage == "pretrain" else "kinetics_hidden_options"
    )
    arch_index = trial.suggest_categorical(
        "architecture", list(range(len(search[arch_field])))
    )
    dims_field = "hidden_dims" if stage == "pretrain" else "kinetics_hidden_dims"
    cfg[dims_field] = search[arch_field][arch_index]
    cfg["batch_size"] = trial.suggest_categorical("batch_size", search["batch_sizes"])
    cfg["dropout"] = trial.suggest_float("dropout", *search["dropout_range"])
    cfg["weight_decay"] = trial.suggest_float(
        "weight_decay", *search["weight_decay_range"], log=True
    )
    cfg[stage + "_lr"] = trial.suggest_float(
        "learning_rate", *search[stage + "_lr_range"], log=True
    )
    cfg[stage + "_epochs"] = search[stage + "_epochs"]
    return cfg


def run_search(prepared_run, training_cfg, search, progress=lambda _: None):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("贝叶斯优化需要安装 requirements-optimization.txt。") from exc
    from science.reference_encoder import build_references
    from science.training import train_mechanism_encoder, train_kinetics_model
    from app.training_workflow import activate
    from threadpoolctl import threadpool_limits

    search = validate_config(search)
    source = Path(prepared_run)
    master = pd.read_csv(source / "dataset/master.csv")
    val = master.split.eq("val")
    if not val.any():
        raise ValueError(
            "贝叶斯优化需要明确验证集。两集合仍可固定训练；请重新准备带验证集的数据，禁止用测试集调参。"
        )
    if not np.isfinite(pd.to_numeric(master.loc[val, "lgkp"], errors="coerce")).any():
        raise ValueError("验证集合没有有效 lgkp，不能优化动力学。")
    if search["target"] != "kinetics":
        if any(
            not master.loc[val, f"ch{ch}_valid"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            .any()
            for ch in range(1, 7)
        ):
            raise ValueError("机制优化的均值与最差通道目标需要六通道都有有效验证机制。")
    run = (
        core.ROOT
        / "outputs"
        / ("optimized_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    )
    run.mkdir(parents=True)
    shutil.copytree(source / "dataset", run / "dataset")
    if (source / "references/reference_summary.json").is_file():
        shutil.copytree(source / "references", run / "references")
    else:
        progress("优化前构建参考表示")
        build_references(run / "dataset/master.csv", run / "references", training_cfg)
    write_json(run / "training_config.json", training_cfg)
    write_json(run / "optimization_config.json", search)
    report = {
        "run_dir": str(run),
        "algorithm": "Optuna TPE",
        "selection_scope": "validation_only",
        "test_evaluation": "only_after_all_searches",
        "studies": {},
    }
    current = dict(training_cfg)
    selected_pretrain = None

    def search_stage(stage):
        nonlocal current, selected_pretrain
        folder = run / "optimization" / stage
        folder.mkdir(parents=True)
        direction = "maximize" if stage == "pretrain" else "minimize"
        sampler = optuna.samplers.TPESampler(
            seed=search["seed"],
            n_startup_trials=min(5, max(1, search[stage + "_trials"] // 3)),
        )
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
            if search["pruning"]
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            study_name=stage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage="sqlite:///" + str(folder / "study.db"),
        )

        def objective(trial):
            cfg = trial_config(trial, current, search, stage)
            trial_dir = folder / f"trial_{trial.number:04d}"
            trial_dir.mkdir()
            shutil.copytree(run / "dataset", trial_dir / "dataset")
            shutil.copytree(run / "references", trial_dir / "references")
            write_json(trial_dir / "trial_config.json", cfg)
            progress(f"TPE {stage}：试验 {trial.number+1}/{search[stage + '_trials']}")

            def callback(epoch, value):
                if value is None or not np.isfinite(value):
                    raise ValueError("试验没有有效验证指标。")
                trial.report(float(value), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            with threadpool_limits(limits=2):
                if stage == "pretrain":
                    metrics = train_mechanism_encoder(
                        trial_dir, cfg, evaluate_test=False, epoch_callback=callback
                    )
                    values = [
                        metrics["validation_channels"][str(ch)] for ch in range(1, 7)
                    ]
                    score = 0.7 * float(np.mean(values)) + 0.3 * float(np.min(values))
                else:
                    shutil.copytree(run / "pretrained", trial_dir / "pretrained")
                    metrics = train_kinetics_model(
                        trial_dir, cfg, evaluate_test=False, epoch_callback=callback
                    )
                    score = metrics["val"]["RMSE"]
            if score is None or not np.isfinite(score):
                raise ValueError("试验目标不是有限数值。")
            trial.set_user_attr("configuration", cfg)
            trial.set_user_attr("best_epoch", metrics["best_epoch"])
            return float(score)

        try:
            study.optimize(
                objective, n_trials=search[stage + "_trials"], catch=(ValueError,)
            )
        finally:
            study.trials_dataframe().to_csv(
                folder / "trials.csv", index=False, encoding="utf-8-sig"
            )
        try:
            best = study.best_trial
        except ValueError as exc:
            raise ValueError(
                "所有试验失败或被剪枝，未激活优化模型。请查看日志。"
            ) from exc
        selected = best.user_attrs["configuration"]
        current = {**current, **selected}
        item = {
            "best_trial": best.number,
            "best_value": best.value,
            "direction": direction,
            "objective": (
                "0.7*mean(validation_channel_cosine)+0.3*min(validation_channel_cosine)"
                if stage == "pretrain"
                else "validation_RMSE"
            ),
            "best_parameters": best.params,
            "best_configuration": selected,
        }
        write_json(folder / "best_trial.json", item)
        report["studies"][stage] = item
        # Preserve artifacts for FT search, without evaluating test mechanism yet.
        if stage == "pretrain":
            selected_pretrain = dict(selected)
            shutil.copytree(
                folder / f"trial_{best.number:04d}" / "pretrained", run / "pretrained"
            )

    if search["target"] in ["pretrain", "both"]:
        search_stage("pretrain")
    else:
        required = source / "pretrained/best_pretrain_model.npz"
        if not required.is_file():
            raise ValueError("仅动力学优化需先完成本软件的机制预训练。")
        shutil.copytree(source / "pretrained", run / "pretrained")
    if search["target"] in ["kinetics", "both"]:
        search_stage("kinetics")

    progress("搜索完成：按选定配置最终训练与测试评估")
    with threadpool_limits(limits=2):
        if selected_pretrain is not None:
            train_mechanism_encoder(run, selected_pretrain, evaluate_test=True)
        train_kinetics_model(run, current, evaluate_test=True)
    activate(run)
    write_json(run / "optimization_summary.json", report)
    write_json(
        core.ROOT / "config/training_state.json",
        {"run_dir": run.relative_to(core.ROOT).as_posix(), "stage": "completed"},
    )
    write_json(
        core.ROOT / "config/last_optimization.json",
        {"run_dir": run.relative_to(core.ROOT).as_posix(), "summary": report},
    )
    progress("优化模型已激活")
    return report


def status():
    p = core.ROOT / "config/last_optimization.json"
    value = core.read_json(p) if p.is_file() else None
    if value:
        for stage in value["summary"]["studies"]:
            path = (
                core.path_for(value["run_dir"]) / "optimization" / stage / "trials.csv"
            )
            if path.is_file():
                value["summary"]["studies"][stage]["trials"] = json.loads(
                    pd.read_csv(path).to_json(orient="records")
                )
        base = core.path_for(value["run_dir"])
        value["downloads"] = {
            str(p.relative_to(base)): "/download/"
            + p.relative_to(core.ROOT / "outputs").as_posix()
            for p in base.rglob("*")
            if p.is_file()
            and p.suffix in [".json", ".csv"]
            and ("trial_" not in p.as_posix() or p.name == "best_trial.json")
        }
    return {"config": get_config(), "last": value}
