"""Print local dependency status without printing keys or environment contents."""

import sys, importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import ROOT

print("Oxidation Mechanism Workbench V1.0")
print("Python:", sys.version.split()[0])
print("Software directory:", ROOT)
for name in [
    "numpy",
    "pandas",
    "sklearn",
    "openpyxl",
    "joblib",
    "threadpoolctl",
    "umap",
    "matplotlib",
    "optuna",
    "hdbscan",
    "torch",
    "transformers",
    "dotenv",
    "langchain_community",
    "jwt",
]:
    print(name, ":", "installed" if importlib.util.find_spec(name) else "missing")
try:
    from app.mechanism_analysis import llm_status

    status = llm_status()
    print("LLM model:", status["config"]["model"])
    print("LLM key present:", status["api_key_present"])
except Exception as e:
    print("LLM configuration could not be checked:", type(e).__name__)
