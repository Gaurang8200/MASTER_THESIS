# MO_Changes
from __future__ import annotations

import argparse

from .demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated Franky and MuJoCo prototype"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run physics without opening the three dimensional viewer",
    )
    arguments = parser.parse_args()
    run_demo(render=not arguments.headless)


if __name__ == "__main__":
    main()
