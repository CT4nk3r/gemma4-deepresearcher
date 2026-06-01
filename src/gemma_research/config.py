from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass
class ModelConfig:
    provider: str = "lmstudio"
    model: str = "gemma-4-e4b"
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: int = 120
    temperature: float = 0.2
    max_tokens: int = 2048


@dataclass
class SearchConfig:
    provider: str = "duckduckgo"
    max_results: int = 5
    timeout_seconds: int = 20
    user_agent: str = "gemma-deepresearch-e4b/0.1"


@dataclass
class AgentConfig:
    max_iterations: int = 2
    min_sources: int = 2
    max_notes: int = 30
    trace_dir: str = ".gemma-research/traces"
    use_llm_report: bool = True


@dataclass
class RepositoryConfig:
    max_file_bytes: int = 60_000
    ignore_dirs: tuple[str, ...] = (
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
    )


@dataclass
class Config:
    model: ModelConfig
    search: SearchConfig
    agent: AgentConfig
    repository: RepositoryConfig

    @classmethod
    def default(cls) -> "Config":
        return cls(ModelConfig(), SearchConfig(), AgentConfig(), RepositoryConfig())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        allowed_sections = {"model", "search", "agent", "repository"}
        unknown = set(data) - allowed_sections
        if unknown:
            raise ConfigError(f"Unknown config section(s): {', '.join(sorted(unknown))}")

        default = cls.default()
        return cls(
            model=_coerce_dataclass(ModelConfig, data.get("model", {}), default.model),
            search=_coerce_dataclass(SearchConfig, data.get("search", {}), default.search),
            agent=_coerce_dataclass(AgentConfig, data.get("agent", {}), default.agent),
            repository=_coerce_dataclass(
                RepositoryConfig, data.get("repository", {}), default.repository
            ),
        )


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], values: dict[str, Any], default: T) -> T:
    if not isinstance(values, dict):
        raise ConfigError(f"{cls.__name__} must be a table/object")

    names = {field.name for field in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ConfigError(f"Unknown {cls.__name__} key(s): {', '.join(sorted(unknown))}")

    merged = {field.name: getattr(default, field.name) for field in fields(cls)}
    merged.update(values)
    if cls is RepositoryConfig and isinstance(merged.get("ignore_dirs"), list):
        merged["ignore_dirs"] = tuple(merged["ignore_dirs"])
    return cls(**merged)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    config_path = Path(path) if path else _default_config_path()
    if config_path is None:
        return Config.default()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return Config.from_dict(data)


def _default_config_path() -> Path | None:
    for name in ("gemma-research.toml", ".gemma-research.toml"):
        candidate = Path(name)
        if candidate.exists():
            return candidate
    return None


def apply_overrides(
    config: Config,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    search_provider: str | None = None,
    max_iterations: int | None = None,
    trace_dir: str | None = None,
) -> Config:
    if provider:
        config.model.provider = provider
    if model:
        config.model.model = model
    if base_url:
        config.model.base_url = base_url
    if search_provider:
        config.search.provider = search_provider
    if max_iterations is not None:
        if max_iterations < 1:
            raise ConfigError("--max-iterations must be at least 1")
        config.agent.max_iterations = max_iterations
    if trace_dir:
        config.agent.trace_dir = trace_dir
    return config
