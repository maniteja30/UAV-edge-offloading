import unittest
from src.tasksim.simulator import Config, OffloadingSimulator
from src.baselines.policies import enumerate_actions, encode_action, decode_action


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.sim = OffloadingSimulator(Config(task_count=4, seed=3))

    def test_base_candidate_count(self):
        self.assertEqual(len(self.sim.candidates("U0")), 7)

    def test_action_count_and_round_trip(self):
        candidates = self.sim.candidates("U0")
        actions = enumerate_actions(candidates, 2)
        self.assertEqual(len(actions), 28)
        for i, pair in enumerate(actions):
            self.assertEqual(decode_action(i, candidates), pair)
            self.assertEqual(encode_action(pair, candidates), i)

    def test_transmission_delay(self):
        a, b = self.sim.nodes["U0"], self.sim.nodes["E0"]
        self.assertGreater(self.sim.transmission_time(1.0, a, b), 0)

    def test_local_transmission_is_zero(self):
        a = self.sim.nodes["U0"]
        self.assertEqual(self.sim.transmission_time(1.0, a, a), 0)

    def test_end_to_end_policy(self):
        results = self.sim.run_policy("greedy")
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.latency_s >= 0 for r in results))
        self.assertTrue(all(r.energy_j >= 0 for r in results))


if __name__ == "__main__":
    unittest.main()
