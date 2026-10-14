"""Command-line equivalent of the training page workflow."""

import os, sys, json
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.training_workflow import execute

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python tools/workflow.py prepare|pretrain|transfer|pipeline CONFIG_JSON JOB_OUTPUT_DIRECTORY"
        )
    kind, config_path, job_dir = sys.argv[1:]
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    execute(
        kind, json.loads(Path(config_path).read_text(encoding="utf-8-sig")), job_dir
    )
