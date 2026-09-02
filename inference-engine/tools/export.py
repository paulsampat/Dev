"""Write a demo model to a ``.iem`` file.

    python -m tools.export mlp models/mlp.iem
    python -m tools.export transformer models/block.iem
"""

from __future__ import annotations

import argparse
import os
import sys

from engine.loader import save_graph
from tools.models import BUILDERS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(BUILDERS))
    parser.add_argument("path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    demo = BUILDERS[args.model](seed=args.seed)
    directory = os.path.dirname(os.path.abspath(args.path))
    os.makedirs(directory, exist_ok=True)
    save_graph(args.path, demo.graph)

    size_mb = os.path.getsize(args.path) / 1e6
    print(f"wrote {args.path} ({size_mb:.2f} MB)")
    print(demo.graph.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
