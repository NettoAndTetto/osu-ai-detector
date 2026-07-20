from __future__ import annotations

import argparse
import os
import threading
import webbrowser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the osu! AI fingerprint detector web service.")
    parser.add_argument("--host", default=os.getenv("OSU_AI_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OSU_AI_WEB_PORT", "8000")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("OSU_AI_WEB_WORKERS", "1")))
    parser.add_argument("--reload", action="store_true", help="development only")
    parser.add_argument("--log-level", default=os.getenv("OSU_AI_WEB_LOG_LEVEL", "info"))
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open the local page shortly after the server starts",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.reload and args.workers != 1:
        parser.error("--reload cannot be combined with multiple workers")

    try:
        import uvicorn
        from . import web as _web  # noqa: F401 - validate optional web dependencies early
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise SystemExit('Web dependencies are missing. Run: python -m pip install -e ".[web]"') from exc

    if args.open_browser:
        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        url = f"http://{browser_host}:{args.port}/"
        timer = threading.Timer(1.0, lambda: webbrowser.open(url, new=2))
        timer.daemon = True
        timer.start()

    uvicorn.run(
        "osu_ai_detector.web:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
