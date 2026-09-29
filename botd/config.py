"""Load research settings from config.toml and secrets from .env."""

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Operational defaults: they don't change model outputs (or any stage/response version), so
# they live here rather than in config.toml. A key under the same section in config.toml
# overrides one.

# Azure connection shared by the generator and the evaluated models ([azure]). The endpoint
# and key come from .env; each model's own settings are in [generate.model] / [[evaluate.models]].
AZURE_DEFAULTS = {
    "max_output_tokens": 6000,
    "request_timeout_s": 180,
    # Rate-limit handling
    "min_interval_s": 4.0,             # client-side pacing between request starts
    "max_retries": 8,                  # per request, for 429 / 5xx / timeouts (by the OpenAI client)
    "max_consecutive_failures": 5,     # stop the run (resume later) after this many failed cases in a row
    "max_attempts_per_case": 3,        # per stage, counted across runs
    "max_repair_rounds": 2,            # follow-ups asking the model to fix output that failed validation
}
EVALUATE_DEFAULTS = {
    "answer_max_tokens": 5,            # answers are read from the first token's logprobs
    "explain_max_tokens": 400,         # free-text stated-principle explanations
    "min_answer_mass": 0.5,            # probability on valid answer tokens needed to score a response
}
# Per evaluated model. Azure quotas are per deployment (currently 500 requests / 500k tokens per
# minute), so each model runs its own pool of `concurrency` in-flight requests.
EVAL_MODEL_DEFAULTS = {
    "concurrency": 8,
    "min_interval_s": 0.15,            # <= 400 request starts per minute
}
ANALYZE_DEFAULTS = {
    "bootstrap_samples": 10000,
    "seed": 13,
}


def load_config(path: str | Path = ROOT / "config.toml") -> dict:
    load_dotenv(ROOT / ".env")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    for key in ("raw_dir", "work_dir", "benchmark_dir", "results_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p

    az = cfg["azure"] = AZURE_DEFAULTS | cfg.get("azure", {})
    az["endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    az["api_key"] = os.environ.get("AZURE_API_KEY", "")

    ev = cfg["evaluate"] = EVALUATE_DEFAULTS | cfg.get("evaluate", {})
    ev["models"] = [EVAL_MODEL_DEFAULTS | m for m in ev.get("models", [])]
    cfg["analyze"] = ANALYZE_DEFAULTS | cfg.get("analyze", {})

    gm = cfg["generate"]["model"]
    gm["deployment"] = os.environ.get("AZURE_DEPLOYMENT") or gm.get("deployment", "")
    cfg["hf_token"] = os.environ.get("HUGGINGFACE") or os.environ.get("HF_TOKEN")
    return cfg


def generator_az(cfg: dict) -> dict:
    """Azure client settings for the generating model: shared connection plus its own settings."""
    return cfg["azure"] | cfg["generate"]["model"]


def public_config(cfg: dict) -> dict:
    """Config snapshot safe to write into the dataset manifest (no secrets, no local paths)."""
    out = {k: v for k, v in cfg.items() if k not in ("hf_token", "paths")}
    out["azure"] = {k: v for k, v in cfg["azure"].items() if k not in ("api_key", "endpoint")}
    return out
