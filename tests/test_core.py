import unittest

from cell_lab.core import CellSimulator
from cell_lab.policies import heuristic_action


class CellSimulatorTests(unittest.TestCase):
    def test_reset_returns_expected_observation_shape(self) -> None:
        simulator = CellSimulator(max_steps=50, seed=1)
        observation = simulator.reset()
        self.assertEqual(len(observation), CellSimulator.observation_size)
        self.assertTrue(all(value >= 0.0 for value in observation))
        self.assertTrue(all(value <= 1.0 for value in observation))

    def test_step_advances_age_and_history(self) -> None:
        simulator = CellSimulator(max_steps=50, seed=1)
        simulator.reset()
        result = simulator.step([0.5] * 8)
        self.assertEqual(simulator.state.age, 1)
        self.assertEqual(simulator.steps, 1)
        self.assertGreaterEqual(len(simulator.history), 2)
        self.assertEqual(len(result.observation), CellSimulator.observation_size)
        metrics = simulator.get_metrics()
        self.assertIn("mitochondria_efficiency", metrics)
        self.assertIn("golgi_capacity", metrics)

    def test_heuristic_action_shape(self) -> None:
        simulator = CellSimulator(max_steps=50, seed=1)
        simulator.reset()
        action = heuristic_action(simulator)
        self.assertEqual(len(action), 8)
        self.assertTrue(all(value >= 0.0 for value in action))
        self.assertTrue(all(value <= 1.0 for value in action))


if __name__ == "__main__":
    unittest.main()
