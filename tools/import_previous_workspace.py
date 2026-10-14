"""Import old data/model paths by reference, preserving the older directory and credentials."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import ROOT, read_json
from app.training_workflow import write_json

p = argparse.ArgumentParser()
p.add_argument("previous_directory")
a = p.parse_args()
old = Path(a.previous_directory).expanduser().resolve()
saved = read_json(old / "config/workspace.json")
result = {}
for key, value in saved.items():
    if key == "references":
        result[key] = {
            ch: str((Path(path) if Path(path).is_absolute() else old / path).resolve())
            for ch, path in value.items()
        }
    else:
        result[key] = str(
            (Path(value) if Path(value).is_absolute() else old / value).resolve()
        )
write_json(
    ROOT / "config/workspace_before_import.json",
    read_json(ROOT / "config/workspace.json"),
)
write_json(ROOT / "config/workspace.json", result)
if (old / "config/llm.json").is_file():
    llm = read_json(old / "config/llm.json")
    env = llm.get("env_file", "")
    if env:
        llm["env_file"] = str(
            (Path(env) if Path(env).is_absolute() else old / env).resolve()
        )
    elif (old / ".env").is_file():
        llm["env_file"] = str(old / ".env")
    write_json(ROOT / "config/llm.json", llm)
print("已导入旧工作区路径。原数据、权重及 .env 保留在旧目录，请重启新服务。")
