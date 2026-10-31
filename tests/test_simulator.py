import unittest

from tasksim.simulator import Config, OffloadingSimulator
from baselines.policies import (
    enumerate_actions,
    encode_action,
    decode_action,
)


class SimulatorTests(unittest.TestCase):

    def setUp(self):
        self.sim = OffloadingSimulator(
            Config(task_count=4, seed=3)
        )

    def test_report_base_configuration(self):
        cfg = Config()

        self.assertEqual(
            (
                cfg.n_users,
                cfg.n_edges,
                cfg.helpers_per_source,
                cfg.subtasks,
            ),
            (10, 3, 2, 2),
        )

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
            self.assertEqual(
                decode_action(i, candidates), pair
            )
            self.assertEqual(
                encode_action(pair, candidates), i
            )

    def test_network_has_required_link_types(self):
        kinds = {
            link["kind"]
            for edges in self.sim.network.values()
            for link in edges.values()
        }

        self.assertIn("air", kinds)
        self.assertIn("backhaul", kinds)

        for u, neighbors in self.sim.network.items():
            for v, link in neighbors.items():
                self.assertIn(u, self.sim.network[v])
                self.assertGreater(link["rate_bps"], 0)

    def test_routes_use_connected_hops_and_reach_cloud(self):
        for destination in self.sim.candidates("U0"):
            path = self.sim._path("U0", destination)

            self.assertTrue(path)
            self.assertEqual(path[0], "U0")
            self.assertEqual(path[-1], destination)

            for u, v in zip(path, path[1:]):
                self.assertIn(v, self.sim.network[u])

        cloud_path = self.sim._path("U0", "CLOUD")

        self.assertEqual(
            self.sim.nodes[cloud_path[-2]].kind,
            "edge",
        )

    def test_unreachable_destination_is_marked_infeasible(self):
        isolated = OffloadingSimulator(
            Config(task_count=1, air_link_range_m=0.1)
        )

        self.assertEqual(
            isolated._path("U0", "E0"), []
        )

        branch = isolated._branch(
            isolated.tasks[0],
            "E0",
            isolated.nodes,
        )

        self.assertFalse(branch["feasible"])

    def test_transmission_delay(self):
        a = self.sim.nodes["U0"]
        b = self.sim.nodes["E0"]

        self.assertGreater(
            self.sim.transmission_time(1.0, a, b),
            0,
        )

    def test_local_transmission_is_zero(self):
        a = self.sim.nodes["U0"]

        self.assertEqual(
            self.sim.transmission_time(1.0, a, a),
            0,
        )

    def test_poisson_arrivals_are_reproducible_and_within_episode(self):
        cfg = Config(seed=17)

        first = OffloadingSimulator(cfg).tasks
        second = OffloadingSimulator(cfg).tasks

        self.assertEqual(
            [(t.source_id, t.arrival_s) for t in first],
            [(t.source_id, t.arrival_s) for t in second],
        )

        self.assertGreater(len(first), 0)

        self.assertTrue(
            all(
                0 <= t.arrival_s < cfg.episode_duration_s
                for t in first
            )
        )

        self.assertEqual(
            [t.arrival_s for t in first],
            sorted(t.arrival_s for t in first),
        )

    def test_poisson_task_count_is_reasonable_for_seeded_base_case(self):
        tasks = OffloadingSimulator(
            Config(seed=7)
        ).tasks

        self.assertGreater(len(tasks), 220)
        self.assertLess(len(tasks), 380)

    def test_end_to_end_policy(self):
        results = self.sim.run_policy("greedy")

        self.assertEqual(len(results), 4)

        self.assertTrue(
            all(r.latency_s >= 0 for r in results)
        )

        self.assertTrue(
            all(r.energy_j >= 0 for r in results)
        )

    # -----------------------------------------------------
    # New 2B.2 tests
    # -----------------------------------------------------

    def test_network_links_have_queue_state(self):
        for neighbors in self.sim.network.values():
            for link in neighbors.values():
                self.assertIn("available_at", link)
                self.assertIn("bits_sent", link)
                self.assertGreaterEqual(
                    link["available_at"], 0.0
                )

    def test_cpu_nodes_have_memory_reservations(self):
        for node in self.sim.nodes.values():
            self.assertIsInstance(
                node.memory_reservations, list
            )

    def test_action_mask_has_28_entries(self):
        task = self.sim.tasks[0]

        mask = self.sim.action_mask(task)

        self.assertEqual(len(mask), 28)
        self.assertTrue(
            all(isinstance(value, bool) for value in mask)
        )

    def test_zero_battery_makes_local_action_infeasible(self):
        sim = OffloadingSimulator(
            Config(
                task_count=1,
                user_battery_j=0.0,
            )
        )

        task = sim.tasks[0]
        local_pair = (task.source_id, task.source_id)

        result = sim.evaluate_pair(task, local_pair)

        self.assertEqual(result.latency_s, float("inf"))

    def test_insufficient_memory_makes_local_branch_infeasible(self):
        sim = OffloadingSimulator(
            Config(
                task_count=1,
                user_memory_capacity_mb=10.0,
            )
        )

        task = sim.tasks[0]

        branch = sim._branch(
            task,
            task.source_id,
            sim.nodes,
        )

        self.assertFalse(branch["feasible"])

    def test_evaluation_does_not_modify_live_resources(self):
        task = self.sim.tasks[0]

        original_battery = {
            node_id: node.battery_j
            for node_id, node in self.sim.nodes.items()
        }

        original_cpu_times = {
            node_id: node.available_at
            for node_id, node in self.sim.nodes.items()
        }

        original_link_times = {
            (u, v): link["available_at"]
            for u, neighbors in self.sim.network.items()
            for v, link in neighbors.items()
        }

        self.sim.evaluate_pair(
            task,
            (task.source_id, task.source_id),
        )

        for node_id, node in self.sim.nodes.items():
            self.assertEqual(
                node.battery_j,
                original_battery[node_id],
            )

            self.assertEqual(
                node.available_at,
                original_cpu_times[node_id],
            )

        for u, neighbors in self.sim.network.items():
            for v, link in neighbors.items():
                self.assertEqual(
                    link["available_at"],
                    original_link_times[(u, v)],
                )

    def test_policy_results_have_expected_count(self):
        results = self.sim.run_policy("greedy")

        self.assertEqual(len(results), 4)

        for result in results:
            self.assertIsNotNone(result.destinations)
            self.assertGreaterEqual(
                result.load_imbalance, 0.0
            )


if __name__ == "__main__":
    unittest.main()