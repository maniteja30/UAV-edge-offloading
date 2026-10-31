from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from math import ceil, log2, sqrt
from typing import Dict, List, Tuple
import copy
import heapq
import random
import statistics


@dataclass
class Config:
    """Base configuration and explicitly labelled simulation assumptions."""

    n_users: int = 10
    n_edges: int = 3
    helpers_per_source: int = 2

    input_mb: float = 2.0
    output_mb: float = 0.1
    cpu_cycles: float = 5e8
    memory_mb: float = 256.0
    deadline_s: float = 2.0
    subtasks: int = 2

    episode_duration_s: float = 60.0
    arrival_rate_per_user: float = 0.5
    task_count: int | None = None
    seed: int = 7

    area_width_m: float = 500.0
    area_height_m: float = 500.0
    uav_altitude_m: float = 100.0

    user_cpu_hz: float = 4e9
    edge_cpu_hz: float = 10e9
    cloud_cpu_hz: float = 100e9
    bandwidth_hz: float = 20e6

    # Assumed radio and backhaul parameters.
    tx_power_w: float = 0.1
    reference_gain: float = 1e-3
    path_loss_exponent: float = 2.2
    noise_density_w_hz: float = 1e-17
    edge_cloud_rate_bps: float = 100e6
    edge_cloud_delay_s: float = 0.035
    air_link_range_m: float = 700.0

    # Assumed memory and battery capacities.
    user_memory_capacity_mb: float = 1024.0
    edge_memory_capacity_mb: float = 8192.0
    cloud_memory_capacity_mb: float = 1_000_000.0

    user_battery_j: float = 500_000.0
    edge_battery_j: float = 2_000_000.0

    split_cycles: float = 1e6
    aggregation_cycles: float = 1e6
    cpu_energy_coefficient: float = 1e-28

    energy_reference_j: float = 0.1
    load_reference: float = 0.25
    deadline_penalty: float = 1.0

    w_latency: float = 0.5
    w_energy: float = 0.3
    w_load: float = 0.2


@dataclass
class Node:
    node_id: str
    kind: str
    x: float
    y: float
    cpu_hz: float
    memory_mb: float
    z: float = 100.0
    battery_j: float = float("inf")

    # CPU scheduling state.
    available_at: float = 0.0
    busy_cycles: float = 0.0

    # Entries are (start_time, finish_time, reserved_memory_mb).
    memory_reservations: List[Tuple[float, float, float]] = field(
        default_factory=list
    )


@dataclass
class Task:
    task_id: int
    source_id: str
    arrival_s: float
    input_mb: float
    output_mb: float
    cpu_cycles: float
    memory_mb: float
    deadline_s: float


@dataclass
class TaskResult:
    task_id: int
    source_id: str
    destinations: Tuple[str, str]
    latency_s: float
    energy_j: float
    load_imbalance: float
    deadline_met: bool
    objective: float
    branch_details: List[dict] = field(default_factory=list)


class OffloadingSimulator:

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.rng = random.Random(self.cfg.seed)

        self.nodes = self._make_nodes()
        self.network = self._build_network()
        self.tasks = self._make_tasks()

    # ---------------------------------------------------------
    # Node and task creation
    # ---------------------------------------------------------

    def _make_nodes(self) -> Dict[str, Node]:
        cfg = self.cfg
        nodes: Dict[str, Node] = {}

        cx = cfg.area_width_m / 2.0
        cy = cfg.area_height_m / 2.0
        radius = 0.36 * min(cfg.area_width_m, cfg.area_height_m)

        import math

        for i in range(cfg.n_users):
            angle = 2 * math.pi * i / max(1, cfg.n_users)

            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)

            nodes[f"U{i}"] = Node(
                node_id=f"U{i}",
                kind="user",
                x=x,
                y=y,
                cpu_hz=cfg.user_cpu_hz,
                memory_mb=cfg.user_memory_capacity_mb,
                z=cfg.uav_altitude_m,
                battery_j=cfg.user_battery_j,
            )

        edge_positions = [
            (0.20 * cfg.area_width_m, 0.20 * cfg.area_height_m),
            (0.80 * cfg.area_width_m, 0.20 * cfg.area_height_m),
            (0.50 * cfg.area_width_m, 0.80 * cfg.area_height_m),
            (0.85 * cfg.area_width_m, 0.75 * cfg.area_height_m),
        ]

        for i in range(cfg.n_edges):
            x, y = edge_positions[i % len(edge_positions)]

            nodes[f"E{i}"] = Node(
                node_id=f"E{i}",
                kind="edge",
                x=x,
                y=y,
                cpu_hz=cfg.edge_cpu_hz,
                memory_mb=cfg.edge_memory_capacity_mb,
                z=cfg.uav_altitude_m,
                battery_j=cfg.edge_battery_j,
            )

        nodes["CLOUD"] = Node(
            node_id="CLOUD",
            kind="cloud",
            x=cx,
            y=cy,
            cpu_hz=cfg.cloud_cpu_hz,
            memory_mb=cfg.cloud_memory_capacity_mb,
            z=0.0,
            battery_j=float("inf"),
        )

        return nodes

    def _make_tasks(self) -> List[Task]:
        arrivals: List[Tuple[float, str]] = []

        if self.cfg.task_count is not None:
            for i in range(self.cfg.task_count):
                source = f"U{i % self.cfg.n_users}"
                arrivals.append((i * 0.05, source))

        else:
            rate = self.cfg.arrival_rate_per_user

            if rate <= 0:
                raise ValueError("arrival_rate_per_user must be positive")

            if self.cfg.episode_duration_s <= 0:
                raise ValueError("episode_duration_s must be positive")

            for user_index in range(self.cfg.n_users):
                source = f"U{user_index}"
                arrival = self.rng.expovariate(rate)

                while arrival < self.cfg.episode_duration_s:
                    arrivals.append((arrival, source))
                    arrival += self.rng.expovariate(rate)

        arrivals.sort(key=lambda item: (item[0], item[1]))

        return [
            Task(
                task_id=i,
                source_id=source,
                arrival_s=arrival,
                input_mb=self.cfg.input_mb,
                output_mb=self.cfg.output_mb,
                cpu_cycles=self.cfg.cpu_cycles,
                memory_mb=self.cfg.memory_mb,
                deadline_s=self.cfg.deadline_s,
            )
            for i, (arrival, source) in enumerate(arrivals)
        ]

    # ---------------------------------------------------------
    # Network creation and wireless rate
    # ---------------------------------------------------------

    @staticmethod
    def distance(a: Node, b: Node) -> float:
        return sqrt(
            (a.x - b.x) ** 2
            + (a.y - b.y) ** 2
            + (a.z - b.z) ** 2
        )

    def _build_network(self) -> Dict[str, Dict[str, dict]]:
        """
        Build an undirected network.

        Each physical link has one shared queue in both directions.
        """

        graph: Dict[str, Dict[str, dict]] = {
            node_id: {} for node_id in self.nodes
        }

        aerial = [
            n for n in self.nodes.values()
            if n.kind in ("user", "edge")
        ]

        for i, a in enumerate(aerial):
            for b in aerial[i + 1:]:
                distance = self.distance(a, b)

                if distance <= self.cfg.air_link_range_m:
                    rate = self._shannon_rate(a, b, distance)

                    link = {
                        "rate_bps": rate,
                        "distance_m": distance,
                        "kind": "air",
                        "delay_s": 0.0,
                        "available_at": 0.0,
                        "bits_sent": 0.0,
                    }

                    graph[a.node_id][b.node_id] = link
                    graph[b.node_id][a.node_id] = link

        for edge in (
            n for n in self.nodes.values() if n.kind == "edge"
        ):
            link = {
                "rate_bps": self.cfg.edge_cloud_rate_bps,
                "distance_m": None,
                "kind": "backhaul",
                "delay_s": self.cfg.edge_cloud_delay_s,
                "available_at": 0.0,
                "bits_sent": 0.0,
            }

            graph[edge.node_id]["CLOUD"] = link
            graph["CLOUD"][edge.node_id] = link

        return graph

    def _shannon_rate(
        self, a: Node, b: Node, distance: float
    ) -> float:
        d = max(1.0, distance)

        snr = (
            self.cfg.tx_power_w
            * self.cfg.reference_gain
            * d ** (-self.cfg.path_loss_exponent)
            / (
                self.cfg.noise_density_w_hz
                * self.cfg.bandwidth_hz
            )
        )

        return self.cfg.bandwidth_hz * log2(1.0 + snr)

    def transmission_rate(self, a: Node, b: Node) -> float:
        if a.node_id == b.node_id:
            return float("inf")

        link = self.network.get(a.node_id, {}).get(b.node_id)

        return link["rate_bps"] if link else 0.0

    def transmission_time(
        self, data_mb: float, a: Node, b: Node
    ) -> float:
        if a.node_id == b.node_id:
            return 0.0

        rate = self.transmission_rate(a, b)

        if rate <= 0:
            return float("inf")

        return data_mb * 8e6 / rate

    # ---------------------------------------------------------
    # Candidate destinations and actions
    # ---------------------------------------------------------

    def candidates(self, source_id: str) -> List[str]:
        source = self.nodes[source_id]

        helpers = sorted(
            (
                n for n in self.nodes.values()
                if n.kind == "user" and n.node_id != source_id
            ),
            key=lambda n: self.distance(source, n),
        )[:self.cfg.helpers_per_source]

        edges = sorted(
            n.node_id for n in self.nodes.values()
            if n.kind == "edge"
        )

        return (
            [source_id]
            + [n.node_id for n in helpers]
            + edges
            + ["CLOUD"]
        )

    def action_pairs(self, source_id: str) -> List[Tuple[str, str]]:
        return list(
            combinations_with_replacement(
                self.candidates(source_id),
                self.cfg.subtasks,
            )
        )

    # ---------------------------------------------------------
    # Shortest-path routing
    # ---------------------------------------------------------

    def _path(
        self,
        source_id: str,
        destination_id: str,
        network: Dict[str, Dict[str, dict]] | None = None,
    ) -> List[str]:

        graph = network if network is not None else self.network

        if source_id == destination_id:
            return [source_id]

        if source_id not in graph or destination_id not in graph:
            return []

        distances = {
            node_id: float("inf") for node_id in graph
        }
        previous: Dict[str, str] = {}

        distances[source_id] = 0.0
        queue = [(0.0, source_id)]

        while queue:
            cost, current = heapq.heappop(queue)

            if cost != distances[current]:
                continue

            if current == destination_id:
                break

            for neighbor, link in graph[current].items():
                rate = link["rate_bps"]

                hop_cost = (
                    8e6 / max(1.0, rate)
                    + link.get("delay_s", 0.0)
                )

                candidate_cost = cost + hop_cost

                if candidate_cost < distances[neighbor]:
                    distances[neighbor] = candidate_cost
                    previous[neighbor] = current

                    heapq.heappush(
                        queue,
                        (candidate_cost, neighbor),
                    )

        if distances[destination_id] == float("inf"):
            return []

        path = [destination_id]

        while path[-1] != source_id:
            parent = previous.get(path[-1])

            if parent is None:
                return []

            path.append(parent)

        return list(reversed(path))

    # ---------------------------------------------------------
    # Network queue and transmission scheduling
    # ---------------------------------------------------------

    def _transmit_hop(
        self,
        source_id: str,
        destination_id: str,
        data_mb: float,
        current_time: float,
        nodes: Dict[str, Node],
        network: Dict[str, Dict[str, dict]],
    ) -> dict:

        if source_id == destination_id:
            return {
                "feasible": True,
                "finish": current_time,
                "queue": 0.0,
                "transmission": 0.0,
                "energy": 0.0,
            }

        link = network.get(source_id, {}).get(destination_id)

        if link is None or link["rate_bps"] <= 0:
            return {"feasible": False}

        rate = link["rate_bps"]
        transmission = data_mb * 8e6 / rate

        # Shared FIFO queue for this physical link.
        start = max(current_time, link["available_at"])
        queue_wait = start - current_time

        finish_transmission = start + transmission
        finish = finish_transmission + link.get("delay_s", 0.0)

        sender = nodes[source_id]
        energy = self.cfg.tx_power_w * transmission

        if sender.kind != "cloud":
            if sender.battery_j < energy:
                return {"feasible": False}

            sender.battery_j -= energy

        # The link is occupied during transmission, not propagation.
        link["available_at"] = finish_transmission
        link["bits_sent"] += data_mb * 8e6

        return {
            "feasible": True,
            "finish": finish,
            "queue": queue_wait,
            "transmission": transmission,
            "energy": energy,
        }

    # ---------------------------------------------------------
    # CPU memory reservation and computation scheduling
    # ---------------------------------------------------------

    @staticmethod
    def _reserve_memory(
        node: Node,
        start: float,
        finish: float,
        memory_mb: float,
    ) -> bool:

        node.memory_reservations = [
            reservation
            for reservation in node.memory_reservations
            if reservation[1] > start
        ]

        if memory_mb > node.memory_mb:
            return False

        used_memory = sum(
            amount
            for begin, end, amount in node.memory_reservations
            if begin <= start < end
        )

        if used_memory + memory_mb > node.memory_mb:
            return False

        node.memory_reservations.append(
            (start, finish, memory_mb)
        )

        return True

    def _compute(
        self,
        node: Node,
        ready_time: float,
        cycles: float,
        memory_mb: float = 0.0,
    ) -> dict:

        if node.cpu_hz <= 0:
            return {"feasible": False}

        if memory_mb > node.memory_mb:
            return {"feasible": False}

        start = max(ready_time, node.available_at)
        queue_wait = start - ready_time

        duration = cycles / node.cpu_hz
        finish = start + duration

        if not self._reserve_memory(
            node, start, finish, memory_mb
        ):
            return {"feasible": False}

        energy = (
            self.cfg.cpu_energy_coefficient
            * (node.cpu_hz ** 2)
            * cycles
        )

        if node.kind != "cloud":
            if node.battery_j < energy:
                return {"feasible": False}

            node.battery_j -= energy

        node.available_at = finish
        node.busy_cycles += cycles

        return {
            "feasible": True,
            "start": start,
            "finish": finish,
            "queue": queue_wait,
            "duration": duration,
            "energy": energy,
        }

    # ---------------------------------------------------------
    # One subtask branch
    # ---------------------------------------------------------

    def _branch(
        self,
        task: Task,
        destination_id: str,
        nodes: Dict[str, Node],
        network: Dict[str, Dict[str, dict]] | None = None,
        start_time: float | None = None,
    ) -> dict:

        graph = network if network is not None else self.network

        p = self.cfg.subtasks

        input_mb = task.input_mb / p
        output_mb = task.output_mb / p
        cycles = task.cpu_cycles / p
        memory = task.memory_mb / p

        path = self._path(
            task.source_id,
            destination_id,
            graph,
        )

        if not path:
            return {
                "destination": destination_id,
                "completion": float("inf"),
                "energy": float("inf"),
                "forward": float("inf"),
                "queue": 0.0,
                "compute": 0.0,
                "return": float("inf"),
                "path": [],
                "feasible": False,
            }

        now = (
            task.arrival_s
            if start_time is None
            else start_time
        )

        communication_energy = 0.0
        network_queue = 0.0
        forward_start = now

        # Forward input transmission.
        for u, v in zip(path, path[1:]):
            hop = self._transmit_hop(
                u, v, input_mb, now, nodes, graph
            )

            if not hop["feasible"]:
                return {
                    "destination": destination_id,
                    "completion": float("inf"),
                    "energy": float("inf"),
                    "forward": float("inf"),
                    "queue": network_queue,
                    "compute": 0.0,
                    "return": float("inf"),
                    "path": path,
                    "feasible": False,
                }

            now = hop["finish"]
            network_queue += hop["queue"]
            communication_energy += hop["energy"]

        forward_time = now - forward_start

        # Computation at destination.
        destination = nodes[destination_id]

        computation = self._compute(
            destination,
            now,
            cycles,
            memory,
        )

        if not computation["feasible"]:
            return {
                "destination": destination_id,
                "completion": float("inf"),
                "energy": float("inf"),
                "forward": forward_time,
                "queue": network_queue,
                "compute": 0.0,
                "return": float("inf"),
                "path": path,
                "feasible": False,
            }

        network_queue += computation["queue"]

        # Return output along the reverse route.
        now = computation["finish"]
        return_start = now

        reverse_path = list(reversed(path))

        for u, v in zip(reverse_path, reverse_path[1:]):
            hop = self._transmit_hop(
                u, v, output_mb, now, nodes, graph
            )

            if not hop["feasible"]:
                return {
                    "destination": destination_id,
                    "completion": float("inf"),
                    "energy": float("inf"),
                    "forward": forward_time,
                    "queue": network_queue,
                    "compute": computation["duration"],
                    "return": float("inf"),
                    "path": path,
                    "feasible": False,
                }

            now = hop["finish"]
            network_queue += hop["queue"]
            communication_energy += hop["energy"]

        return_time = now - return_start

        return {
            "destination": destination_id,
            "completion": now,
            "energy": communication_energy + computation["energy"],
            "forward": forward_time,
            "queue": network_queue,
            "compute": computation["duration"],
            "return": return_time,
            "path": path,
            "feasible": True,
        }

    # ---------------------------------------------------------
    # Load imbalance
    # ---------------------------------------------------------

    def _load_imbalance(
        self, nodes: Dict[str, Node]
    ) -> float:

        active = [
            n for n in nodes.values()
            if n.kind in ("user", "edge")
        ]

        if not active:
            return 0.0

        horizon = max(1.0, self.cfg.episode_duration_s)

        loads = [
            n.busy_cycles / (n.cpu_hz * horizon)
            for n in active
        ]

        mean_load = sum(loads) / len(loads)

        return sum(
            (load - mean_load) ** 2 for load in loads
        ) / len(loads)

    # ---------------------------------------------------------
    # Transactional evaluation of a destination pair
    # ---------------------------------------------------------

    def _simulate_pair(
        self,
        task: Task,
        pair: Tuple[str, str],
        base_nodes: Dict[str, Node] | None = None,
        base_network: Dict[str, Dict[str, dict]] | None = None,
    ):

        # Copy nodes and network together to preserve shared link objects.
        nodes, network = copy.deepcopy((
            base_nodes if base_nodes is not None else self.nodes,
            base_network if base_network is not None else self.network,
        ))

        source = nodes[task.source_id]

        # Task splitting consumes source CPU and energy.
        split = self._compute(
            source,
            task.arrival_s,
            self.cfg.split_cycles,
        )

        if not split["feasible"]:
            return self._infeasible_result(task, pair), nodes, network

        branches = [
            self._branch(
                task,
                destination,
                nodes,
                network,
                start_time=split["finish"],
            )
            for destination in pair
        ]

        if not all(branch["feasible"] for branch in branches):
            return self._infeasible_result(
                task, pair, branches
            ), nodes, network

        # Aggregation starts after both branches return.
        branch_finish = max(
            branch["completion"] for branch in branches
        )

        aggregation = self._compute(
            source,
            branch_finish,
            self.cfg.aggregation_cycles,
        )

        if not aggregation["feasible"]:
            return self._infeasible_result(
                task, pair, branches
            ), nodes, network

        latency = aggregation["finish"] - task.arrival_s

        energy = (
            split["energy"]
            + aggregation["energy"]
            + sum(branch["energy"] for branch in branches)
        )

        load = self._load_imbalance(nodes)

        penalty = max(
            0.0,
            (latency - task.deadline_s) / task.deadline_s,
        )

        objective = (
            self.cfg.w_latency * latency / task.deadline_s
            + self.cfg.w_energy * energy / self.cfg.energy_reference_j
            + self.cfg.w_load * load / self.cfg.load_reference
            + self.cfg.deadline_penalty * penalty
        )

        result = TaskResult(
            task.task_id,
            task.source_id,
            pair,
            latency,
            energy,
            load,
            latency <= task.deadline_s,
            objective,
            branches,
        )

        return result, nodes, network

    def _infeasible_result(
        self,
        task: Task,
        pair: Tuple[str, str],
        branches: List[dict] | None = None,
    ) -> TaskResult:

        return TaskResult(
            task_id=task.task_id,
            source_id=task.source_id,
            destinations=pair,
            latency_s=float("inf"),
            energy_j=float("inf"),
            load_imbalance=0.0,
            deadline_met=False,
            objective=float("inf"),
            branch_details=branches or [],
        )

    def evaluate_pair(
        self,
        task: Task,
        pair: Tuple[str, str],
        base_nodes: Dict[str, Node] | None = None,
        base_network: Dict[str, Dict[str, dict]] | None = None,
    ) -> TaskResult:

        result, _, _ = self._simulate_pair(
            task,
            pair,
            base_nodes,
            base_network,
        )

        # Evaluation does not modify the simulator's live resources.
        return result

    # ---------------------------------------------------------
    # Action masking
    # ---------------------------------------------------------

    def action_mask(
        self,
        task: Task,
        base_nodes: Dict[str, Node] | None = None,
        base_network: Dict[str, Dict[str, dict]] | None = None,
    ) -> List[bool]:
        """
        Return one Boolean per action.

        True means the action is physically feasible and meets the deadline.
        The output length remains 28 for the base configuration.
        """

        pairs = self.action_pairs(task.source_id)
        mask = []

        for pair in pairs:
            result = self.evaluate_pair(
                task,
                pair,
                base_nodes,
                base_network,
            )

            mask.append(
                result.latency_s < float("inf")
                and result.deadline_met
            )

        return mask

    def feasible_action_pairs(
        self,
        task: Task,
        base_nodes: Dict[str, Node] | None = None,
        base_network: Dict[str, Dict[str, dict]] | None = None,
    ) -> List[Tuple[str, str]]:

        pairs = self.action_pairs(task.source_id)
        mask = self.action_mask(
            task, base_nodes, base_network
        )

        return [
            pair
            for pair, valid in zip(pairs, mask)
            if valid
        ]

    # ---------------------------------------------------------
    # Policies and resource commitment
    # ---------------------------------------------------------

    def run_policy(self, policy: str) -> List[TaskResult]:

        nodes, network = copy.deepcopy(
            (self.nodes, self.network)
        )

        results = []

        for task in self.tasks:
            candidates = self.candidates(task.source_id)
            pairs = self.action_pairs(task.source_id)

            if policy == "local":
                chosen = (task.source_id, task.source_id)

            elif policy == "nearest":
                source = nodes[task.source_id]

                remote = [
                    nodes[node_id]
                    for node_id in candidates
                    if node_id != task.source_id
                    and nodes[node_id].kind != "cloud"
                ]

                nearest = min(
                    remote,
                    key=lambda n: self.distance(source, n),
                ).node_id

                chosen = (nearest, nearest)

            elif policy == "greedy":
                evaluations = [
                    (
                        self.evaluate_pair(
                            task, pair, nodes, network
                        ),
                        pair,
                    )
                    for pair in pairs
                ]

                # Prefer actions that meet the deadline.
                valid = [
                    (result, pair)
                    for result, pair in evaluations
                    if result.deadline_met
                    and result.latency_s < float("inf")
                ]

                # If none meet the deadline, use the best
                # physically feasible action.
                if not valid:
                    valid = [
                        (result, pair)
                        for result, pair in evaluations
                        if result.latency_s < float("inf")
                    ]

                if valid:
                    _, chosen = min(
                        valid,
                        key=lambda item: item[0].objective,
                    )
                else:
                    chosen = (task.source_id, task.source_id)

            else:
                raise ValueError(f"Unknown policy: {policy}")

            result, updated_nodes, updated_network = (
                self._simulate_pair(
                    task,
                    chosen,
                    nodes,
                    network,
                )
            )

            # Commit only physically feasible executions.
            # Failed trial evaluations do not consume resources.
            if result.latency_s < float("inf"):
                nodes = updated_nodes
                network = updated_network

            results.append(result)

        return results

    # ---------------------------------------------------------
    # Result summary
    # ---------------------------------------------------------

    @staticmethod
    def summarize(results: List[TaskResult]) -> dict:

        latencies = [r.latency_s for r in results]
        energies = [r.energy_j for r in results]

        if latencies:
            ordered = sorted(latencies)
            p95 = ordered[
                max(0, ceil(0.95 * len(ordered)) - 1)
            ]
        else:
            p95 = 0.0

        return {
            "tasks": len(results),
            "mean_latency_s": (
                statistics.mean(latencies) if latencies else 0.0
            ),
            "p95_latency_s": p95,
            "mean_energy_j": (
                statistics.mean(energies) if energies else 0.0
            ),
            "deadline_violation_rate": (
                sum(not r.deadline_met for r in results) / len(results)
                if results else 0.0
            ),
            "success_rate": (
                sum(r.deadline_met for r in results) / len(results)
                if results else 0.0
            ),
            "mean_load_imbalance": (
                statistics.mean(
                    r.load_imbalance for r in results
                )
                if results else 0.0
            ),
        }