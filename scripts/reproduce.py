"""Run the timed ConvXAI reference experiment."""

import argparse
import logging
from pathlib import Path

from convxai.reproduce import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduction.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.output)


if __name__ == "__main__":
    main()
