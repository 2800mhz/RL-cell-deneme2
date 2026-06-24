"""Legacy visualization entrypoint."""

import sys

from cell_lab.plotting import run_and_plot


def main() -> None:
    steps = 300
    mode = "heuristic"

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        steps = int(sys.argv[2])

    run_and_plot(mode=mode, steps=steps)


if __name__ == "__main__":
    main()
