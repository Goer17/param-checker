#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from param_checker.loader import flatten_tensors, load_checkpoint
from param_checker.server import run_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PyTorch checkpoint parameters in a browser."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to .pth checkpoint file",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server (default: 8000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        checkpoint = load_checkpoint(args.input)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to load checkpoint: {exc}", file=sys.stderr)
        return 1

    tensors = flatten_tensors(checkpoint)
    if not tensors:
        print("No tensors found in checkpoint.", file=sys.stderr)
        return 1

    run_server(tensors, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
