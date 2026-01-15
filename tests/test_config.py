import pytest

from estudely_askai import config
from estudely_askai.config import resolve_settings, write_config
from estudely_askai.errors import ConfigError


def test_resolve_settings_uses_config_file(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "estudely-askai"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"
    config_path.write_text(
        'host = "http://example.com"\nmodel = "mistral"\ntimeout = 42\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("ASKAI_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    settings = resolve_settings(
        host=None, model=None, timeout=None, cloud=False, local=False
    )

    assert settings.host == "http://example.com"
    assert settings.model == "mistral"
    assert settings.timeout == 42


def test_write_config_handles_os_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def boom(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(config.os, "makedirs", boom)
    with pytest.raises(ConfigError) as exc:
        write_config("http://example.com", "llama3.1", 60)
    assert "Unable to write config file" in str(exc.value)
