from __future__ import annotations

import json
from typing import Any, Iterable
from urllib import error, request

from .errors import ConnectionError, OllamaAPIError


class OllamaClient:
    def __init__(self, host: str, timeout: int, api_key: str | None) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def generate(self, model: str, prompt: str) -> str:
        data = self._request_json(
            "POST",
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": False},
        )
        response = data.get("response")
        if not response:
            raise OllamaAPIError("No response returned from the model.")
        if not isinstance(response, str):
            raise OllamaAPIError("Unexpected response format from the model.")
        return response

    def generate_stream(self, model: str, prompt: str) -> Iterable[str]:
        emitted = False
        for data in self._request_stream(
            "POST",
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": True},
        ):
            if "error" in data:
                raise OllamaAPIError(str(data["error"]))
            if "response" in data:
                response = data["response"]
                if not isinstance(response, str):
                    raise OllamaAPIError("Unexpected response format from the model.")
                emitted = True
                yield response
        if not emitted:
            raise OllamaAPIError("No response returned from the model.")

    def list_models(self) -> list[str]:
        data = self._request_json("GET", "/api/tags", None)
        models = data.get("models", [])
        names: list[str] = []
        for item in models:
            if isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
        return names

    def _request_json(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        url = f"{self.host}{path}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = request.Request(url, data=data, method=method, headers=self.headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            message = _parse_error_message(exc.read().decode("utf-8", errors="replace"))
            raise OllamaAPIError(message) from exc
        except error.URLError as exc:
            raise ConnectionError(f"Cannot reach Ollama host: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ConnectionError("Request timed out.") from exc

        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaAPIError("Invalid JSON response from Ollama.") from exc
        if not isinstance(parsed, dict):
            raise OllamaAPIError("Unexpected response structure from Ollama.")
        if "error" in parsed:
            raise OllamaAPIError(str(parsed["error"]))
        return parsed

    def _request_stream(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> Iterable[dict[str, Any]]:
        url = f"{self.host}{path}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = request.Request(url, data=data, method=method, headers=self.headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                for item in _iter_stream(resp):
                    yield item
        except error.HTTPError as exc:
            message = _parse_error_message(exc.read().decode("utf-8", errors="replace"))
            raise OllamaAPIError(message) from exc
        except error.URLError as exc:
            raise ConnectionError(f"Cannot reach Ollama host: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ConnectionError("Request timed out.") from exc


def _iter_stream(resp: Any) -> Iterable[dict[str, Any]]:
    for line in resp:
        if not line:
            continue
        if isinstance(line, bytes):
            decoded = line.decode("utf-8").strip()
        else:
            decoded = str(line).strip()
        if not decoded:
            continue
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise OllamaAPIError("Invalid JSON response from Ollama.") from exc
        if not isinstance(parsed, dict):
            raise OllamaAPIError("Unexpected response structure from Ollama.")
        yield parsed


def _parse_error_message(raw: str) -> str:
    if not raw:
        return "Ollama request failed."
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() or "Ollama request failed."
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    return raw.strip() or "Ollama request failed."
