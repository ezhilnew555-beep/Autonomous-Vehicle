"""
main.py
=======
Entry point for the Autonomous Vehicle Simulation.
Run with:  python main.py
"""

from src.simulation import Simulation


def main() -> None:
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
