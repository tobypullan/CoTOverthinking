#!/usr/bin/env python3
"""Serve the repository over HTTP so visualisations can load project files.

This is the reliable path when the HTML files are opened from a WSL-backed
filesystem, where browsers block `file://` fetches for security reasons.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the repository over HTTP for the visualisations."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind to. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to. Defaults to 8000.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent

    handler = partial(SimpleHTTPRequestHandler, directory=os.fspath(repo_root))
    server = ThreadingHTTPServer((args.host, args.port), handler)

    base_url = f"http://{args.host}:{args.port}"
    localhost_base_url = f"http://localhost:{args.port}"

    print(f"Serving {repo_root} at {base_url}")
    print("Open one of these URLs in your browser:")
    print(f"  {localhost_base_url}/visualisations/resample_dashboard.html")
    print(f"  {localhost_base_url}/visualisations/open_resample_matrix.html (redirects to the dashboard)")
    print(f"  {localhost_base_url}/visualisations/options_resample_matrix.html (redirects to the dashboard)")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
