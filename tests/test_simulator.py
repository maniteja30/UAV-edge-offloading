import unittest
from src.tasksim.simulator import Config, OffloadingSimulator
from src.baselines.policies import enumerate_actions, encode_action, decode_action


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.sim = OffloadingSimulator(Config(task_count=4, seed=3))

    def test_report_base_configuration(self):
        cfg = Config()
        self.assertEqual((cfg.n_users, cfg.n_edges, cfg.helpers_per_source, cfg.subtasks), (10, 3, 2, 2))
        self.assertEqual(cfg.user_cpu_hz, 4e9)
        self.assertEqual(cfg.edge_cpu_hz, 10e9)
        self.assertEqual(cfg.cloud_cpu_hz, 100e9)
        self.assertEqual(cfg.bandwidth_hz, 20e6)
        self.assertEqual(cfg.episode_duration_s, 60.0)
        self.assertEqual(cfg.arrival_rate_per_user, 0.5)

    def test_base_candidate_count(self):
        self.assertEqual(len(self.sim.candidates("U0")), 7)

    def test_action_count_and_round_trip(self):
        candidates = self.sim.candidates("U0")
        actions = enumerate_actions(candidates, 2)
        self.assertEqual(len(actions), 28)
        for i, pair in enumerate(actions):
            self.assertEqual(decode_action(i, candidates), pair)
            self.assertEqual(encode_action(pair, candidates), i)

    def test_network_has_required_link_types(self):
        kinds = {link["kind"] for edges in self.sim.network.values() for link in edges.values()}
        self.assertIn("air", kinds)
        self.assertIn("backhaul", kinds)
        # Every air link is symmetric and has a positive Shannon rate.
        for u, neighbors in self.sim.network.items():
            for v, link in neighbors.items():
                self.assertIn(u, self.sim.network[v])
                self.assertGreater(link["rate_bps"], 0)

    def test_routes_use_connected_hops_and_reach_cloud(self):
        for destination in self.sim.candidates("U0"):
            path = self.sim._path("U0", destination)
            self.assertTrue(path, f"No route from U0 to {destination}")
            self.assertEqual(path[0], "U0")
            self.assertEqual(path[-1], destination)
            for u, v in zip(path, path[1:]):
                self.assertIn(v, self.sim.network[u])
        cloud_path = self.sim._path("U0", "CLOUD")
        self.assertEqual(self.sim.nodes[cloud_path[-2]].kind, "edge")

    def test_unreachable_destination_is_marked_infeasible(self):
        isolated = OffloadingSimulator(Config(task_count=1, air_link_range_m=0.1))
        self.assertEqual(isolated._path("U0", "E0"), [])
        branch = isolated._branch(isolated.tasks[0], "E0", isolated.nodes)
        self.assertFalse(branch["feasible"])

    def test_transmission_delay(self):
        a, b = self.sim.nodes["U0"], self.sim.nodes["E0"]
        self.assertGreater(self.sim.transmission_time(1.0, a, b), 0)

    def test_local_transmission_is_zero(self):
        a = self.sim.nodes["U0"]
        self.assertEqual(self.sim.transmission_time(1.0, a, a), 0)

    def test_poisson_arrivals_are_reproducible_and_within_episode(self):
        cfg = Config(seed=17)
        first = OffloadingSimulator(cfg).tasks
        second = OffloadingSimulator(cfg).tasks
        self.assertEqual(
            [(t.source_id, t.arrival_s) for t in first],
            [(t.source_id, t.arrival_s) for t in second],
        )
        self.assertGreater(len(first), 0)
        self.assertTrue(all(0 <= t.arrival_s < cfg.episode_duration_s for t in first))
        self.assertEqual([t.arrival_s for t in first], sorted(t.arrival_s for t in first))

    def test_poisson_task_count_is_reasonable_for_seeded_base_case(self):
        tasks = OffloadingSimulator(Config(seed=7)).tasks
        # E[N] = 10 users * 0.5 tasks/s * 60 s = 300 tasks.
        self.assertGreater(len(tasks), 220)
        self.assertLess(len(tasks), 380)

    def test_end_to_end_policy(self):
        results = self.sim.run_policy("greedy")
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.latency_s >= 0 for r in results))
        self.assertTrue(all(r.energy_j >= 0 for r in results))


if __name__ == "__main__":
    unittest.main()
