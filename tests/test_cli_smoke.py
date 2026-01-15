from estudely_askai.cli import app


def test_version(capsys) -> None:
    code = app(["--version"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "0.1.0"
    assert code == 0
