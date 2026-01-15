from __future__ import annotations

from dataclasses import dataclass
import os

from .errors import ConfigError

DEFAULT_LOCAL_HOST = "http://localhost:11434"
DEFAULT_CLOUD_HOST = "https://ollama.com"
DEFAULT_MODEL = "llama3.1"
DEFAULT_TIMEOUT = 60


@dataclass(frozen=True)
class Settings:
    host: str
    model: str
    timeout: int
    api_key: str | None


def _config_path() -> str:
    return os.path.join(
        os.path.expanduser("~"), ".config", "estudely-askai", "config.toml"
    )


def config_path() -> str:
    return _config_path()


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_config(host: str, model: str, timeout: int) -> str:
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"host = {_toml_quote(host)}\n")
        handle.write(f"model = {_toml_quote(model)}\n")
        handle.write(f"timeout = {timeout}\n")
    return path


def _load_toml(path: str) -> dict[str, object]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python <3.11
        import tomli as tomllib
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _load_config() -> tuple[dict[str, object], str]:
    path = _config_path()
    if not os.path.exists(path):
        return {}, path
    try:
        data = _load_toml(path)
    except Exception as exc:  # pragma: no cover - depends on toml errors
        raise ConfigError(f"Invalid config file at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Invalid config file at {path}: expected a table at top level."
        )
    return data, path


def _get_config_str(config: dict[str, object], key: str, path: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Invalid config file at {path}: '{key}' must be a string.")
    return value


def _get_config_int(config: dict[str, object], key: str, path: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ConfigError(f"Invalid config file at {path}: '{key}' must be an integer.")
    return value


def resolve_settings(
    *,
    host: str | None,
    model: str | None,
    timeout: int | None,
    cloud: bool,
    local: bool,
) -> Settings:
    api_key = os.getenv("OLLAMA_API_KEY")
    env_host = os.getenv("OLLAMA_HOST")
    env_model = os.getenv("ASKAI_MODEL")
    config, config_path = _load_config()
    config_host = _get_config_str(config, "host", config_path)
    config_model = _get_config_str(config, "model", config_path)
    config_timeout = _get_config_int(config, "timeout", config_path)

    if cloud and not api_key:
        raise ConfigError(
            "OLLAMA_API_KEY is required for --cloud. Set the environment variable and try again."
        )

    if cloud:
        default_host = DEFAULT_CLOUD_HOST
    elif local:
        default_host = DEFAULT_LOCAL_HOST
    else:
        default_host = env_host or config_host or DEFAULT_LOCAL_HOST

    resolved_host = host or default_host
    resolved_model = model or env_model or config_model or DEFAULT_MODEL
    resolved_timeout = (
        timeout
        if timeout is not None
        else config_timeout
        if config_timeout is not None
        else DEFAULT_TIMEOUT
    )

    return Settings(
        host=resolved_host,
        model=resolved_model,
        timeout=resolved_timeout,
        api_key=api_key,
    )
