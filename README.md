# estudely-askai

Terminal CLI for querying Ollama models via the HTTP API.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
askai --version
askai "what is the capital of france"
askai models
askai --host http://localhost:11434 --model llama3.1 "hello"
askai --stream "stream me"
```

### Options

```bash
--host      Ollama host (default: env OLLAMA_HOST or http://localhost:11434)
--model     Model name (default: env ASKAI_MODEL or llama3.1)
--timeout   Request timeout in seconds (default: 60)
--json      Print models as JSON (models command only)
--stream    Stream response tokens
--cloud     Use Ollama cloud host (requires OLLAMA_API_KEY)
--local     Force localhost host default
```

### Cloud auth

Set `OLLAMA_API_KEY` to send `Authorization: Bearer <key>` on all requests.

### Config file

Defaults can be set in `~/.config/estudely-askai/config.toml`:

```toml
host = "http://localhost:11434"
model = "llama3.1"
timeout = 60
```
