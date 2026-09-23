"""Load research settings from config.toml and secrets from .env."""

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.toml") -> dict:
    load_dotenv(ROOT / ".env")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    for key in ("raw_dir", "work_dir", "output_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p

    az = cfg["azure"]
    az["deployment"] = os.environ.get("AZURE_DEPLOYMENT") or az.get("deployment", "")
    az["endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    az["api_key"] = os.environ.get("AZURE_API_KEY", "")
    cfg["hf_token"] = os.environ.get("HUGGINGFACE") or os.environ.get("HF_TOKEN")
    return cfg


def public_config(cfg: dict) -> dict:
    """Config snapshot safe to write into the dataset manifest (no secrets, no local paths)."""
    out = {k: v for k, v in cfg.items() if k not in ("hf_token", "paths")}
    out["azure"] = {k: v for k, v in cfg["azure"].items() if k not in ("api_key", "endpoint")}
    return out
