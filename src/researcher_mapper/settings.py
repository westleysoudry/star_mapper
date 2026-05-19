from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent.parent  # repo root in editable/source checkouts


def _load_yaml(rel_path: str) -> dict:
    rel = Path(rel_path)
    checkout_path = _ROOT / rel
    if checkout_path.exists():
        with checkout_path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    resource = files("researcher_mapper").joinpath(*rel.parts)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Required config file not found: {rel_path}. "
            "Install package data or run from a source checkout with config/."
        )
    with resource.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings(BaseSettings):
    openalex_api_key: str | None = None
    openalex_email: str | None = None
    semantic_scholar_api_key: str | None = None
    orcid_client_id: str | None = None
    orcid_client_secret: str | None = None
    crossref_mailto: str | None = None

    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


def load_weights() -> dict:
    return _load_yaml("config/weights.yaml").get("weights", {})


def load_bucket_caps() -> dict:
    return _load_yaml("config/bucket_caps.yaml").get("bucket_caps", {})


def load_policy() -> dict:
    return _load_yaml("config/policy.yaml").get("policy", {})


def load_department_aliases() -> dict[str, list[str]]:
    return _load_yaml("config/department_aliases.yaml").get("department_aliases", {})
