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
        self.assertEqual(cfg.arrival_rate_per_user, 0.8)
        self.assertEqual(cfg.cpu_cycles, 1e9)

    def test_original_objective_formula_is_preserved(self):
        sim = OffloadingSimulator(
            Config(task_count=1, seed=11, arrival_rate_per_user=0.8, cpu_cycles=1e9)
        )
        task = sim.tasks[0]
        feasible_results = [
            sim.evaluate_pair(task, pair)
            for pair in sim.action_pairs(task.source_id)
        ]
        result = next(r for r in feasible_results if r.latency_s < float("inf"))

        expected = (
            0.5 * result.latency_s / task.deadline_s
            + 0.3 * result.energy_j / 0.1
            + 0.2 * result.load_imbalance / 0.25
            + max(0.0, (result.latency_s - task.deadline_s) / task.deadline_s)
        )
        self.assertAlmostEqual(result.objective, expected, places=10)

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

    def test_poisson_task_count_is_reasonable_for_experiment2(self):
        tasks = OffloadingSimulator(
            Config(seed=7)
        ).tasks

        # Experiment 2: E[N] = 10 users * 0.8 tasks/s/user * 60 s = 480.
        self.assertGreater(len(tasks), 400)
        self.assertLess(len(tasks), 560)

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



    def test_configuration_rejects_invalid_uav_counts(self):
        invalid_configs = [
            Config(n_users=0),
            Config(n_users=-1),
            Config(n_edges=-1),
            Config(helpers_per_source=-1),
        ]

        for cfg in invalid_configs:
            with self.subTest(config=cfg):
                with self.assertRaises(ValueError):
                    OffloadingSimulator(cfg)

    def test_configuration_rejects_invalid_subtasks_and_area(self):
        invalid_configs = [
            Config(subtasks=0),
            Config(area_width_m=0),
            Config(area_height_m=-10),
        ]

        for cfg in invalid_configs:
            with self.subTest(config=cfg):
                with self.assertRaises(ValueError):
                    OffloadingSimulator(cfg)

    def test_configuration_rejects_invalid_episode_and_task_count(self):
        with self.assertRaises(ValueError):
            OffloadingSimulator(Config(episode_duration_s=0))

        with self.assertRaises(ValueError):
            OffloadingSimulator(Config(task_count=-1))

    def test_edge_positions_preserve_original_topology(self):
        sim = OffloadingSimulator(
            Config(n_edges=4, task_count=1)
        )

        expected = [
            (100.0, 100.0),
            (400.0, 100.0),
            (250.0, 400.0),
            (425.0, 375.0),
        ]

        actual = [
            (sim.nodes[f"E{i}"].x, sim.nodes[f"E{i}"].y)
            for i in range(4)
        ]

        self.assertEqual(actual, expected)

    def test_additional_edge_positions_are_distinct_and_deterministic(self):
        cfg = Config(n_edges=8, task_count=1)

        first = OffloadingSimulator(cfg)
        second = OffloadingSimulator(cfg)

        positions_first = [
            (first.nodes[f"E{i}"].x, first.nodes[f"E{i}"].y)
            for i in range(cfg.n_edges)
        ]

        positions_second = [
            (second.nodes[f"E{i}"].x, second.nodes[f"E{i}"].y)
            for i in range(cfg.n_edges)
        ]

        self.assertEqual(positions_first, positions_second)
        self.assertEqual(len(set(positions_first)), cfg.n_edges)

        for x, y in positions_first:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, cfg.area_width_m)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, cfg.area_height_m)

    def test_zero_edge_uav_configuration_is_supported(self):
        sim = OffloadingSimulator(
            Config(n_edges=0, task_count=1)
        )

        self.assertFalse(
            any(node.kind == "edge" for node in sim.nodes.values())
        )


    def test_energy_events_are_recorded_at_simulated_completion(self):
        sim = OffloadingSimulator(Config(task_count=1))
        task = sim.tasks[0]
        pair = (task.source_id, task.source_id)

        result, updated_nodes, _ = sim._simulate_pair(
            task, pair, sim.nodes, sim.network
        )

        self.assertTrue(result.latency_s < float("inf"))
        source = updated_nodes[task.source_id]
        self.assertTrue(source.energy_events)
        self.assertTrue(
            all(time_s >= 0.0 and energy_j > 0.0
                for time_s, energy_j, _ in source.energy_events)
        )
        self.assertTrue(
            all(operation in ("computation", "transmission")
                for _, _, operation in source.energy_events)
        )

    def test_transmission_energy_event_uses_transmission_completion_time(self):
        sim = OffloadingSimulator(Config(task_count=1))
        source = "U0"
        destination = next(
            node_id for node_id in sim.network[source]
            if sim.network[source][node_id]["kind"] == "air"
        )
        link = sim.network[source][destination]
        rate = link["rate_bps"]
        data_mb = 0.25
        transmission_s = data_mb * 8e6 / rate
        start_s = 2.0

        hop = sim._transmit_hop(
            source, destination, data_mb, start_s,
            sim.nodes, sim.network,
        )

        self.assertTrue(hop["feasible"])
        event = sim.nodes[source].energy_events[-1]
        self.assertAlmostEqual(event[0], start_s + transmission_s)
        self.assertAlmostEqual(event[1], sim.cfg.tx_power_w * transmission_s)
        self.assertEqual(event[2], "transmission")

    def test_trial_evaluation_does_not_add_energy_events_to_live_nodes(self):
        task = self.sim.tasks[0]
        before = {
            node_id: list(node.energy_events)
            for node_id, node in self.sim.nodes.items()
        }

        self.sim.evaluate_pair(task, (task.source_id, task.source_id))

        after = {
            node_id: list(node.energy_events)
            for node_id, node in self.sim.nodes.items()
        }
        self.assertEqual(before, after)

    def test_lifetime_summary_reports_residual_energy_and_fnd(self):
        sim = OffloadingSimulator(
            Config(task_count=1, user_battery_j=10.0,
                   edge_battery_j=20.0,
                   lifetime_death_threshold_fraction=0.01)
        )
        nodes = sim.nodes
        user = nodes["U0"]
        user.battery_j = 0.05
        user.energy_events.append((3.5, 9.95, "computation"))

        summary = sim.lifetime_summary(nodes, 0.01)

        self.assertEqual(summary["uav_count"], 13)
        self.assertEqual(summary["first_node_death_s"], 3.5)
        self.assertEqual(summary["first_node_death_uav"], "U0")
        self.assertEqual(summary["active_uav_count"], 12)
        self.assertAlmostEqual(
            summary["total_residual_energy_j"],
            sum(node.battery_j for node in nodes.values()
                if node.kind in ("user", "edge")),
        )
        self.assertEqual(summary["energy_timeline"][-1]["time_s"], 3.5)

    def test_lifetime_threshold_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            OffloadingSimulator(
                Config(lifetime_death_threshold_fraction=1.0)
            )
        with self.assertRaises(ValueError):
            OffloadingSimulator(
                Config(lifetime_death_threshold_fraction=-0.01)
            )

    def test_run_policy_can_return_committed_resources_for_lifetime(self):
        results, nodes = self.sim.run_policy(
            "local", return_resources=True
        )
        self.assertEqual(len(results), len(self.sim.tasks))
        self.assertTrue(
            any(node.energy_events for node in nodes.values())
        )
        summary = self.sim.summarize(results, nodes)
        self.assertIn("first_node_death_s", summary)


if __name__ == "__main__":
    unittest.main()