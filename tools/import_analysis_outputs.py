"""Discover original research-analysis exports without moving or overwriting them."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from science.visualization import get_config, save_config


def main():
    parser = argparse.ArgumentParser(description="导入研究分析产物的路径配置")
    parser.add_argument("research_root")
    parser.add_argument("--raw-embeddings-dir", default="")
    parser.add_argument("--global-coordinates", default="")
    args = parser.parse_args()
    root = Path(args.research_root).expanduser().resolve()
    if not root.is_dir():
        parser.error("研究产物目录不存在")
    cfg = get_config()
    cfg["research_output_root"] = str(root)
    cfg["reuse_existing"] = True
    if args.raw_embeddings_dir:
        cfg["raw_embeddings_dir"] = str(
            Path(args.raw_embeddings_dir).expanduser().resolve()
        )
    if args.global_coordinates:
        cfg["global_coordinates_path"] = str(
            Path(args.global_coordinates).expanduser().resolve()
        )
    report = {}
    for ch in range(1, 7):
        found = sorted(root.rglob(f"channel{ch}_umap_coordinates_and_clusters.csv"))
        if found:
            cfg["channel_directories"][str(ch)] = str(found[0].parent)
            report[str(ch)] = str(found[0])
    if not cfg["global_coordinates_path"]:
        found = sorted(root.rglob("umap_coordinates_combined.csv"))
        if found:
            cfg["global_coordinates_path"] = str(found[0])
    save_config(cfg)
    print(
        json.dumps(
            {
                "saved": True,
                "channel_coordinates": report,
                "global_coordinates": cfg["global_coordinates_path"],
                "note": "只导入路径，不移动原文件。覆盖范围和四元标识将在运行时核对。",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
