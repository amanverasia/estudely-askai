from unittest.mock import patch

import pytest

from estudely_askai.errors import OllamaAPIError
from estudely_askai.ollama_client import OllamaClient


class FakeStreamResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._lines)


class FakeReadResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._payload


def test_generate_stream_yields_tokens() -> None:
    client = OllamaClient("http://localhost:11434", 5, None)
    lines = [
        b'{"response":"hel"}\n',
        b'{"response":"lo"}\n',
        b'{"done":true}\n',
    ]
    with patch(
        "estudely_askai.ollama_client.request.urlopen",
        return_value=FakeStreamResponse(lines),
    ):
        chunks = list(client.generate_stream("llama3.1", "hi"))
    assert chunks == ["hel", "lo"]


def test_request_json_raises_on_error_payload() -> None:
    client = OllamaClient("http://localhost:11434", 5, None)
    body = b'{"error":"model not found"}'
    with patch(
        "estudely_askai.ollama_client.request.urlopen",
        return_value=FakeReadResponse(body),
    ):
        with pytest.raises(OllamaAPIError) as exc:
            client.generate("missing", "hi")
    assert "model not found" in str(exc.value)
