from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .config import config_path, resolve_settings, write_config
from .errors import AppError
from .ollama_client import OllamaClient


def app(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except AppError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code


def _run(argv: list[str] | None) -> int:
    raw_args = argv if argv is not None else sys.argv[1:]
    no_args = len(raw_args) == 0

    parser = argparse.ArgumentParser(prog="askai", description="Query Ollama models.")
    parser.add_argument("--host", help="Ollama host URL.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print models as JSON.")
    parser.add_argument("--stream", action="store_true", help="Stream response tokens.")
    parser.add_argument("--models", action="store_true", help="List available models.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cloud", action="store_true", help="Use Ollama cloud host.")
    group.add_argument("--local", action="store_true", help="Force localhost host default.")

    parser.add_argument("--version", action="store_true", help="Print version.")
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help="Prompt text.")

    args = parser.parse_args(raw_args)

    if args.version:
        print(__version__)
        return 0

    if args.models:
        if args.prompt:
            print("--models does not accept a prompt.", file=sys.stderr)
            return 1
        if args.stream:
            print("--stream cannot be used with --models.", file=sys.stderr)
            return 1

        settings = resolve_settings(
            host=args.host,
            model=args.model,
            timeout=args.timeout,
            cloud=args.cloud,
            local=args.local,
        )

        client = OllamaClient(settings.host, settings.timeout, settings.api_key)
        models = client.list_models()
        if args.json:
            print(json.dumps(models))
        else:
            for name in models:
                print(name)
        return 0

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        if no_args:
            parser.print_help()
            if not os.path.exists(config_path()):
                return _init_config_interactive(args)
            return 0
        print("No prompt provided.", file=sys.stderr)
        return 1

    settings = resolve_settings(
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        cloud=args.cloud,
        local=args.local,
    )

    client = OllamaClient(settings.host, settings.timeout, settings.api_key)

    if args.stream:
        last_chunk = ""
        printed = False
        for chunk in client.generate_stream(settings.model, prompt):
            printed = True
            last_chunk = chunk
            print(chunk, end="", flush=True)
        if printed and not last_chunk.endswith("\n"):
            print()
        return 0

    response = client.generate(settings.model, prompt)
    print(response)
    return 0


def _init_config_interactive(args: argparse.Namespace) -> int:
    settings = resolve_settings(
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        cloud=args.cloud,
        local=args.local,
    )
    client = OllamaClient(settings.host, settings.timeout, settings.api_key)
    models = client.list_models()
    if not models:
        print("No models available to select.", file=sys.stderr)
        return 1

    print("Available models:")
    for index, name in enumerate(models, start=1):
        print(f"{index}. {name}")

    selection = _prompt_model_choice(len(models))
    if selection is None:
        print("No model selected.", file=sys.stderr)
        return 1

    chosen = models[selection - 1]
    path = write_config(settings.host, chosen, settings.timeout)
    print(f"Saved default model '{chosen}' to {path}.")
    return 0


def _prompt_model_choice(count: int) -> int | None:
    while True:
        try:
            raw = input(f"Choose a default model [1-{count}]: ").strip()
        except EOFError:
            return None
        if not raw:
            print("Please enter a number.", file=sys.stderr)
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a number.", file=sys.stderr)
            continue
        if 1 <= choice <= count:
            return choice
        print(f"Choose a number between 1 and {count}.", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(app())
