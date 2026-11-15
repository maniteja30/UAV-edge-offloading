"""Regression tests for common Experiment 2 defaults across scripts."""

import sys
import unittest
from unittest.mock import patch

import evaluate_dqn
import evaluate_ppo
import train_dqn
import train_ppo


class Experiment2ScriptDefaultsTests(unittest.TestCase):
    def parse_with_argv(self, module, argv):
        with patch.object(sys, "argv", argv):
            return module.parse_args()

    def assert_workload(self, args):
        self.assertEqual(args.n_users, 10)
        self.assertEqual(args.n_edges, 3)
        self.assertEqual(args.helpers_per_source, 2)
        self.assertEqual(args.episode_duration, 60.0)
        self.assertEqual(args.arrival_rate, 0.8)
        self.assertEqual(args.cpu_cycles, 1e9)
        self.assertEqual(args.deadline, 2.0)
        self.assertEqual(args.energy_coefficient, 1e-28)

    def test_dqn_training_defaults_match_experiment2(self):
        args = self.parse_with_argv(train_dqn, ["train_dqn.py"])
        self.assertEqual(args.episodes, 100)
        self.assertEqual(args.seed, 7)
        self.assert_workload(args)

    def test_ppo_training_defaults_match_experiment2(self):
        args = self.parse_with_argv(train_ppo, ["train_ppo.py"])
        self.assertEqual(args.episodes, 100)
        self.assertEqual(args.seed, 7)
        self.assert_workload(args)

    def test_dqn_evaluation_defaults_match_experiment2_test_split(self):
        args = self.parse_with_argv(
            evaluate_dqn,
            ["evaluate_dqn.py", "--model", "checkpoint.pt"],
        )
        self.assertEqual(args.episodes, 30)
        self.assertEqual(args.seed, 1000)
        self.assert_workload(args)

    def test_ppo_evaluation_defaults_match_experiment2_test_split(self):
        args = self.parse_with_argv(
            evaluate_ppo,
            ["evaluate_ppo.py", "--model", "checkpoint.pt"],
        )
        self.assertEqual(args.episodes, 30)
        self.assertEqual(args.seed, 1000)
        self.assert_workload(args)


if __name__ == "__main__":
    unittest.main()
