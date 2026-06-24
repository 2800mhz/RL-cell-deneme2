"""Legacy training entrypoint."""

import sys

from cell_lab.training import train_agent


def main() -> None:
    algorithm = sys.argv[1] if len(sys.argv) > 1 else "ppo"
    timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    train_agent(algorithm=algorithm, total_timesteps=timesteps)


if __name__ == "__main__":
    main()
