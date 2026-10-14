"""Configuration, dataset state and staged training orchestration."""

import json, time, uuid
from pathlib import Path
from app.core import ROOT, path_for, read_json

DEFAULTS = {
    "dataset_mode": "auto",
    "split_strategy": "source",
    "seed": 42,
    "joint_json_path": "data/private/joint_six_channel.json",
    "excel_path": "data/private/oxidation_dataset.xlsx",
    "train_path": "data/manual/train.xlsx",
    "val_path": "data/manual/val.xlsx",
    "test_path": "data/manual/test.xlsx",
    "scibert_path": "models/scibert",
    "raw_embeddings_dir": "",
    "embedding_dim": 64,
    "hidden_dims": [128, 128, 64],
    "kinetics_hidden_dims": [64, 32],
    "dropout": 0.1,
    "batch_size": 32,
    "pretrain_epochs": 300,
    "kinetics_epochs": 1000,
    "pretrain_lr": 0.001,
    "kinetics_lr": 0.001,
    "weight_decay": 0.00001,
    "patience": 50,
    "device": "cpu",
}
PATH_FIELDS = [
    "joint_json_path",
    "excel_path",
    "train_path",
    "val_path",
    "test_path",
    "scibert_path",
    "raw_embeddings_dir",
]


def validate_training_config(data):
    if not isinstance(data, dict) or set(data) != set(DEFAULTS):
        raise ValueError("训练配置字段不完整。请重新载入默认模板。")
    if data["dataset_mode"] not in ["auto", "manual"] or data["split_strategy"] not in [
        "source",
        "record",
    ]:
        raise ValueError("数据划分模式无效。")
    for k in PATH_FIELDS:
        if not isinstance(data[k], str):
            raise ValueError("路径字段须为字符串：" + k)
    for k in [
        "seed",
        "embedding_dim",
        "batch_size",
        "pretrain_epochs",
        "kinetics_epochs",
        "patience",
    ]:
        if (
            not isinstance(data[k], int)
            or isinstance(data[k], bool)
            or not (0 <= data[k] <= 100000 if k == "seed" else 1 <= data[k] <= 10000)
        ):
            raise ValueError("整数参数无效：" + k)
    for k in ["pretrain_lr", "kinetics_lr"]:
        if not isinstance(data[k], (int, float)) or not 0 < data[k] <= 0.1:
            raise ValueError("学习率无效。")
    if not 0 <= data["dropout"] < 0.8 or not 0 <= data["weight_decay"] < 1:
        raise ValueError("dropout 或 weight_decay 无效。")
    for k in ["hidden_dims", "kinetics_hidden_dims"]:
        if (
            not isinstance(data[k], list)
            or not data[k]
            or any(not isinstance(n, int) or not 2 <= n <= 1024 for n in data[k])
        ):
            raise ValueError("隐藏层宽度无效。")
    return data


def get_training_config():
    path = ROOT / "config/training.json"
    return (
        validate_training_config(read_json(path)) if path.is_file() else dict(DEFAULTS)
    )


def save_training_config(data):
    data = validate_training_config(data)
    write_json(ROOT / "config/training.json", data)
    return {"saved": True, "config": data}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def resolve_config(data):
    data = dict(data)
    for k in PATH_FIELDS:
        if data[k]:
            data[k] = str(path_for(data[k]))
    return data


def workflow_status():
    state_path = ROOT / "config/training_state.json"
    state = read_json(state_path) if state_path.is_file() else None
    if state:
        run = path_for(state["run_dir"])
        state["dataset_summary"] = (
            read_json(run / "dataset/dataset_summary.json")
            if (run / "dataset/dataset_summary.json").is_file()
            else None
        )
        state["pretraining_metrics"] = (
            read_json(run / "pretrained/metrics.json")
            if (run / "pretrained/metrics.json").is_file()
            else None
        )
        state["kinetics_metrics"] = (
            read_json(run / "kinetics/metrics.json")
            if (run / "kinetics/metrics.json").is_file()
            else None
        )
    return {
        "config": get_training_config(),
        "state": state,
        "backend": "NumPy MLP + Adam；SciBERT 编码使用 PyTorch",
    }


def activate(run):
    run = Path(run)
    required = [
        "pretrained/pretrain_config.json",
        "pretrained/scaler.json",
        "kinetics/best_model_state.npz",
        "kinetics/transfer_config.json",
    ]
    if not all((run / p).is_file() for p in required):
        raise ValueError("训练产物不完整，未切换预测模型。")
    for ch in range(1, 7):
        if not (run / f"references/ch{ch}.csv").is_file():
            raise ValueError("参考库不完整。")
    rel = lambda p: (run / p).relative_to(ROOT).as_posix()
    cfg = {
        "pretrained_dir": rel("pretrained"),
        "kinetics_dir": rel("kinetics"),
        "scaler_json": rel("pretrained/scaler.json"),
        "pretrain_split": rel("dataset/split.csv"),
        "references": {str(ch): rel(f"references/ch{ch}.csv") for ch in range(1, 7)},
    }
    # Keep a backup of the previous active configuration.
    previous = ROOT / "config/workspace.json"
    if previous.is_file():
        (run / "previous_workspace.json").write_bytes(previous.read_bytes())
    write_json(previous, cfg)


def execute(kind, data, job_dir):
    from science.dataset_manager import prepare_dataset
    from science.reference_encoder import build_references
    from science.training import train_mechanism_encoder, train_kinetics_model
    from threadpoolctl import threadpool_limits

    cfg = resolve_config(validate_training_config(data))
    job_dir = Path(job_dir)

    def stage(name):
        write_json(job_dir / "progress.json", {"stage": name})
        print("阶段：" + name, flush=True)

    state_path = ROOT / "config/training_state.json"
    if kind == "optimize":
        if not state_path.is_file():
            raise ValueError("请先准备带验证集的数据，再启动优化。")
        from science.optimization import run_search

        previous = path_for(read_json(state_path)["run_dir"])
        saved = read_json(previous / "training_config.json")
        for key in [
            "joint_json_path",
            "excel_path",
            "train_path",
            "val_path",
            "test_path",
            "dataset_mode",
            "split_strategy",
            "seed",
        ]:
            cfg[key] = saved[key]
        search = read_json(job_dir / "optimization_config.json")
        run_search(previous, cfg, search, stage)
        return
    if kind in ["prepare", "pipeline"]:
        run = (
            ROOT
            / "outputs"
            / (
                "training_"
                + time.strftime("%Y%m%d_%H%M%S")
                + "_"
                + uuid.uuid4().hex[:6]
            )
        )
        run.mkdir(parents=True)
        write_json(run / "training_config.json", cfg)
        stage("数据校验与固定划分")
        prepare_dataset(cfg, run / "dataset")
        write_json(
            state_path,
            {"run_dir": run.relative_to(ROOT).as_posix(), "stage": "prepared"},
        )
        if kind == "prepare":
            return
    else:
        if not state_path.is_file():
            raise ValueError("请先准备数据集。")
        state = read_json(state_path)
        run = path_for(state["run_dir"])
        saved = read_json(run / "training_config.json")
        # A prepared run uses its exact data configuration; live path edits never redirect it.
        for key in [
            "joint_json_path",
            "excel_path",
            "train_path",
            "val_path",
            "test_path",
            "dataset_mode",
            "split_strategy",
            "seed",
        ]:
            cfg[key] = saved[key]
    with threadpool_limits(limits=2):
        if kind in ["pretrain", "pipeline"]:
            stage("训练参考文本编码与训练集 PCA")
            build_references(run / "dataset/master.csv", run / "references", cfg)
            stage("六通道机制预训练")
            train_mechanism_encoder(run, cfg)
            write_json(
                state_path,
                {"run_dir": run.relative_to(ROOT).as_posix(), "stage": "pretrained"},
            )
        if kind in ["transfer", "pipeline"]:
            stage("冻结主干的动力学训练")
            train_kinetics_model(run, cfg)
            stage("验证产物并激活预测模型")
            activate(run)
            write_json(
                state_path,
                {"run_dir": run.relative_to(ROOT).as_posix(), "stage": "completed"},
            )
    stage("完成")
