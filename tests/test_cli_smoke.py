from unittest.mock import patch

from estudely_askai.cli import app


def test_version(capsys) -> None:
    code = app(["--version"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "0.1.0"
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
