"""Command-line entry point for ConvXAI."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ConvXAI reference implementation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the web application.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    subparsers.add_parser("chat", help="Start a terminal conversation.")
    return parser


def terminal_chat() -> None:
    from convxai.dialogue import session_store

    session, response = session_store.create()
    print(response.message)
    print("Options:", ", ".join(response.options))
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"quit", "exit"}:
            return
        response = session_store.respond(session.identifier, text)
        print(response.message)
        if response.options:
            print("Options:", ", ".join(response.options))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "chat":
        terminal_chat()
        return

    import uvicorn

    uvicorn.run("convxai.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
