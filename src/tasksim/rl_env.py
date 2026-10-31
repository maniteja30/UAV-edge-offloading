
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Optional

from .simulator import Config, OffloadingSimulator, Task, TaskResult


class OffloadingRLEnv:
    """
    RL environment wrapping the existing OffloadingSimulator.

    Observation:
        5 task features
        9 features per candidate destination
        2 system features

    Default observation size: 70
    Default action size: 28

    Actions represent unordered destination pairs.

    The action mask considers physical feasibility only.
    Deadline violations are handled through the reward.
    """

    TASK_FEATURES = 5
    NODE_FEATURES = 9
    SYSTEM_FEATURES = 2

    INFEASIBLE_PENALTY = 10.0

    def __init__(
        self,
        config: Optional[Config] = None,
        infeasible_penalty: float = INFEASIBLE_PENALTY,
    ):
        self.config = deepcopy(config or Config())
        self.infeasible_penalty = float(infeasible_penalty)

        if self.infeasible_penalty <= 0:
            raise ValueError("infeasible_penalty must be positive")

        self.sim: OffloadingSimulator
        self.nodes = None
        self.network = None

        self.tasks = []
        self.results: list[TaskResult] = []

        self.current_index = 0
        self.done = False

        self.state_size = 0
        self.action_size = 0

        # Cache the action mask until the environment state changes.
        self._action_mask_cache = None

        self.reset()

    # -----------------------------------------------------
    # Environment reset
    # -----------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        """Start a new episode."""

        cfg = deepcopy(self.config)

        if seed is not None:
            cfg.seed = seed

        self.sim = OffloadingSimulator(cfg)

        # Maintain separate mutable resources for this episode.
        self.nodes, self.network = deepcopy(
            (self.sim.nodes, self.sim.network)
        )

        self.tasks = self.sim.tasks
        self.results = []

        self.current_index = 0
        self.done = len(self.tasks) == 0

        self._action_mask_cache = None

        if self.tasks:
            candidates = self.sim.candidates(
                self.tasks[0].source_id
            )
        else:
            candidates = self.sim.candidates("U0")

        self.state_size = (
            self.TASK_FEATURES
            + len(candidates) * self.NODE_FEATURES
            + self.SYSTEM_FEATURES
        )

        self.action_size = len(
            self.sim.action_pairs(
                self.tasks[0].source_id
                if self.tasks else "U0"
            )
        )

        state = self.get_state()

        info = {
            "tasks": len(self.tasks),
            "state_size": self.state_size,
            "action_size": self.action_size,
            "action_mask": self.get_action_mask(),
        }

        return state, info

    # -----------------------------------------------------
    # Current task
    # -----------------------------------------------------

    @property
    def current_task(self) -> Optional[Task]:
        if self.done:
            return None

        return self.tasks[self.current_index]

    # -----------------------------------------------------
    # Feature normalization helpers
    # -----------------------------------------------------

    @staticmethod
    def _clip(
        value: float,
        low: float = 0.0,
        high: float = 1.0,
    ):
        return max(low, min(high, float(value)))

    def _available_memory_fraction(
        self,
        node,
        current_time: float,
    ) -> float:

        if node.memory_mb <= 0:
            return 0.0

        used = sum(
            amount
            for start, finish, amount in node.memory_reservations
            if start <= current_time < finish
        )

        available = max(0.0, node.memory_mb - used)

        return self._clip(
            available / node.memory_mb
        )

    def _battery_fraction(self, node) -> float:

        if node.kind == "cloud":
            return 1.0

        if node.kind == "user":
            capacity = self.config.user_battery_j
        else:
            capacity = self.config.edge_battery_j

        if capacity <= 0:
            return 0.0

        return self._clip(
            node.battery_j / capacity
        )

    def _estimate_route_delay(
        self,
        source_id: str,
        destination_id: str,
        data_mb: float,
    ) -> float:
        """
        Estimate transmission time using the current network topology.

        This is a base route estimate. It does not predict future
        link queue waits.
        """

        path = self.sim._path(
            source_id,
            destination_id,
            self.network,
        )

        if not path:
            return 1.0

        total_time = 0.0

        for u, v in zip(path, path[1:]):
            link = self.network[u][v]

            rate = link["rate_bps"]

            if rate <= 0:
                return 1.0

            total_time += (
                data_mb * 8e6 / rate
                + link.get("delay_s", 0.0)
            )

        deadline = max(
            self.config.deadline_s,
            1e-6,
        )

        return self._clip(
            total_time / deadline
        )

    # -----------------------------------------------------
    # State representation
    # -----------------------------------------------------

    def get_state(self):
        """Return the current fixed-size normalized observation."""

        task = self.current_task

        if task is None:
            return [0.0] * self.state_size

        cfg = self.config

        task_features = [
            self._clip(task.input_mb / 10.0),
            self._clip(task.output_mb / 1.0),
            self._clip(task.cpu_cycles / 5e9),
            self._clip(task.memory_mb / 1024.0),
            self._clip(task.deadline_s / 5.0),
        ]

        candidates = self.sim.candidates(
            task.source_id
        )

        node_features = []

        for node_id in candidates:
            node = self.nodes[node_id]

            cpu = self._clip(
                node.cpu_hz /
                max(cfg.cloud_cpu_hz, 1e-6)
            )

            queue_wait = max(
                0.0,
                node.available_at - task.arrival_s,
            )

            queue = self._clip(
                queue_wait /
                max(cfg.episode_duration_s, 1e-6)
            )

            memory = self._available_memory_fraction(
                node,
                task.arrival_s,
            )

            battery = self._battery_fraction(node)

            forward_delay = self._estimate_route_delay(
                task.source_id,
                node_id,
                task.input_mb / cfg.subtasks,
            )

            return_delay = self._estimate_route_delay(
                node_id,
                task.source_id,
                task.output_mb / cfg.subtasks,
            )

            is_user = float(node.kind == "user")
            is_edge = float(node.kind == "edge")
            is_cloud = float(node.kind == "cloud")

            node_features.extend([
                cpu,
                queue,
                memory,
                battery,
                forward_delay,
                return_delay,
                is_user,
                is_edge,
                is_cloud,
            ])

        active_nodes = [
            node for node in self.nodes.values()
            if node.kind in ("user", "edge")
        ]

        if active_nodes:
            horizon = max(
                cfg.episode_duration_s,
                1e-6,
            )

            utilization = sum(
                node.busy_cycles /
                max(node.cpu_hz * horizon, 1e-6)
                for node in active_nodes
            ) / len(active_nodes)
        else:
            utilization = 0.0

        system_utilization = self._clip(utilization)

        remaining_time = self._clip(
            1.0 - (
                task.arrival_s /
                max(cfg.episode_duration_s, 1e-6)
            )
        )

        return (
            task_features
            + node_features
            + [
                system_utilization,
                remaining_time,
            ]
        )

    # -----------------------------------------------------
    # Action mapping and masking
    # -----------------------------------------------------

    def get_action_pairs(self):
        """Return the ordered action-to-destination mapping."""

        task = self.current_task

        if task is None:
            return []

        return self.sim.action_pairs(task.source_id)

    def get_action_mask(self):
        """
        Return one Boolean per action.

        True means physically feasible.

        The result is cached because the environment state does
        not change between consecutive calls.
        """

        if self._action_mask_cache is not None:
            return self._action_mask_cache.copy()

        task = self.current_task

        if task is None:
            return [False] * self.action_size

        pairs = self.get_action_pairs()
        mask = []

        for pair in pairs:
            result = self.sim.evaluate_pair(
                task,
                pair,
                self.nodes,
                self.network,
            )

            mask.append(
                isfinite(result.latency_s)
                and isfinite(result.objective)
            )

        self._action_mask_cache = mask

        return mask.copy()

    # -----------------------------------------------------
    # Environment step
    # -----------------------------------------------------

    def step(self, action: Optional[int]):
        """Execute one task."""

        if self.done:
            raise RuntimeError(
                "Episode has ended. Call reset() before step()."
            )

        task = self.current_task

        pairs = self.get_action_pairs()
        mask = self.get_action_mask()

        if action is None:

            if any(mask):
                raise ValueError(
                    "A feasible action exists. Select a valid action."
                )

            failed_pair = (
                task.source_id,
                task.source_id,
            )

            result = self.sim._infeasible_result(
                task,
                failed_pair,
            )

            reward = -self.infeasible_penalty

            self.results.append(result)

            no_feasible_action = True

        else:

            if not isinstance(action, int) or isinstance(action, bool):
                raise TypeError(
                    "Action must be an integer or None"
                )

            if action < 0 or action >= self.action_size:
                raise ValueError(
                    f"Action must be between 0 and {self.action_size - 1}"
                )

            if not mask[action]:
                raise ValueError(
                    f"Action {action} is physically infeasible."
                )

            chosen_pair = pairs[action]

            result, updated_nodes, updated_network = (
                self.sim._simulate_pair(
                    task,
                    chosen_pair,
                    self.nodes,
                    self.network,
                )
            )

            if not isfinite(result.objective):
                raise RuntimeError(
                    "Selected action became infeasible during execution."
                )

            self.nodes = updated_nodes
            self.network = updated_network

            reward = -result.objective

            self.results.append(result)

            no_feasible_action = False

        self.current_index += 1

        # The task and resource state have changed.
        self._action_mask_cache = None

        self.done = (
            self.current_index >= len(self.tasks)
        )

        terminated = self.done
        truncated = False

        next_state = self.get_state()

        info = {
            "task_id": task.task_id,
            "source_id": task.source_id,
            "destinations": result.destinations,
            "latency_s": result.latency_s,
            "energy_j": result.energy_j,
            "load_imbalance": result.load_imbalance,
            "deadline_met": result.deadline_met,
            "objective": result.objective,
            "no_feasible_action": no_feasible_action,
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    # -----------------------------------------------------
    # Episode summary
    # -----------------------------------------------------

    def get_episode_summary(self):
        return self.sim.summarize(self.results)
