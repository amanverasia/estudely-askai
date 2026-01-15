from unittest.mock import patch

from estudely_askai.cli import app


def test_version(capsys) -> None:
    code = app(["--version"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "0.2.1"
    assert code == 0


def test_models_flag_prints_list(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("ASKAI_MODEL", raising=False)
    with patch("estudely_askai.cli.OllamaClient") as mock_client:
        instance = mock_client.return_value
        instance.list_models.return_value = ["m1", "m2"]
        code = app(["--models"])
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["m1", "m2"]
    assert code == 0


def test_no_args_creates_config(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("ASKAI_MODEL", raising=False)
    with patch("estudely_askai.cli.OllamaClient") as mock_client:
        with patch("builtins.input", return_value="1"):
            instance = mock_client.return_value
            instance.list_models.return_value = ["model-a"]
            code = app([])
    config_path = tmp_path / ".config" / "estudely-askai" / "config.toml"
    assert config_path.exists()
    content = config_path.read_text(encoding="utf-8")
    assert 'model = "model-a"' in content
    assert code == 0


def test_no_args_skips_prompt_when_config_exists(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / ".config" / "estudely-askai"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"
    config_path.write_text(
        'host = "http://localhost:11434"\nmodel = "model-a"\ntimeout = 60\n',
        encoding="utf-8",
    )
    with patch("builtins.input") as mock_input:
        with patch("estudely_askai.cli.OllamaClient") as mock_client:
            code = app([])
    assert code == 0
    assert mock_input.call_count == 0
    assert mock_client.call_count == 0
