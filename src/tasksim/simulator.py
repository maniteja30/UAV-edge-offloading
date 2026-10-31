from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from math import log2, hypot
from typing import Dict, List, Tuple
import copy
import random
import statistics


@dataclass
class Config:
    n_users: int = 10
    n_edges: int = 3
    helpers_per_source: int = 2
    input_mb: float = 2.0
    output_mb: float = 0.1
    cpu_cycles: float = 5e8
    memory_mb: float = 256.0
    deadline_s: float = 2.0
    subtasks: int = 2
    task_count: int = 100
    seed: int = 7

    # Configurable illustrative radio assumptions; not calibrated measurements.
    bandwidth_hz: float = 10e6
    tx_power_w: float = 0.1
    reference_gain: float = 1e-3
    path_loss_exponent: float = 2.2
    noise_density_w_hz: float = 1e-17
    edge_cloud_rate_bps: float = 100e6
    edge_cloud_delay_s: float = 0.035

    split_cycles: float = 1e6
    aggregation_cycles: float = 1e6
    user_cpu_hz: float = 1.5e9
    edge_cpu_hz: float = 5e9
    cloud_cpu_hz: float = 20e9
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
    battery_j: float = float("inf")
    available_at: float = 0.0
    busy_cycles: float = 0.0


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
        self.tasks = self._make_tasks()

    def _make_nodes(self) -> Dict[str, Node]:
        nodes: Dict[str, Node] = {}
        for i in range(self.cfg.n_users):
            angle = 2 * 3.141592653589793 * i / max(1, self.cfg.n_users)
            nodes[f"U{i}"] = Node(f"U{i}", "user",
                500 + 400 * __import__("math").cos(angle),
                500 + 400 * __import__("math").sin(angle),
                self.cfg.user_cpu_hz, 1024, 500_000.0)
        edge_positions = [(300, 300), (700, 300), (500, 750), (900, 600)]
        for i in range(self.cfg.n_edges):
            x, y = edge_positions[i % len(edge_positions)]
            nodes[f"E{i}"] = Node(f"E{i}", "edge", x, y,
                self.cfg.edge_cpu_hz, 8192, 2_000_000.0)
        nodes["CLOUD"] = Node("CLOUD", "cloud", 500, 500,
            self.cfg.cloud_cpu_hz, 1_000_000, float("inf"))
        return nodes

    def _make_tasks(self) -> List[Task]:
        tasks = []
        for i in range(self.cfg.task_count):
            source = f"U{i % self.cfg.n_users}"
            tasks.append(Task(i, source, i * 0.05,
                self.cfg.input_mb, self.cfg.output_mb,
                self.cfg.cpu_cycles, self.cfg.memory_mb,
                self.cfg.deadline_s))
        return tasks

    @staticmethod
    def distance(a: Node, b: Node) -> float:
        return hypot(a.x - b.x, a.y - b.y)

    def transmission_rate(self, a: Node, b: Node) -> float:
        """Shannon rate in bit/s using configurable path-loss assumptions."""
        if {a.kind, b.kind} == {"edge", "cloud"}:
            return self.cfg.edge_cloud_rate_bps
        d = max(1.0, self.distance(a, b))
        snr = (self.cfg.tx_power_w * self.cfg.reference_gain * d ** (-self.cfg.path_loss_exponent)
               / (self.cfg.noise_density_w_hz * self.cfg.bandwidth_hz))
        return self.cfg.bandwidth_hz * log2(1.0 + snr)

    def transmission_time(self, data_mb: float, a: Node, b: Node) -> float:
        if a.node_id == b.node_id:
            return 0.0
        return data_mb * 8e6 / max(1.0, self.transmission_rate(a, b))

    def candidates(self, source_id: str) -> List[str]:
        source = self.nodes[source_id]
        helpers = sorted(
            (n for n in self.nodes.values() if n.kind == "user" and n.node_id != source_id),
            key=lambda n: self.distance(source, n)
        )[:self.cfg.helpers_per_source]
        edges = sorted(n.node_id for n in self.nodes.values() if n.kind == "edge")
        return [source_id] + [n.node_id for n in helpers] + edges + ["CLOUD"]

    def action_pairs(self, source_id: str) -> List[Tuple[str, str]]:
        return list(combinations_with_replacement(self.candidates(source_id), self.cfg.subtasks))

    def _path(self, source_id: str, destination_id: str) -> List[str]:
        if source_id == destination_id:
            return [source_id]
        dest = self.nodes[destination_id]
        if dest.kind in ("user", "edge"):
            return [source_id, destination_id]
        # Cloud is reached through the edge UAV with the lowest source-to-edge
        # transmission delay plus fixed edge-cloud backhaul delay.
        edges = [n for n in self.nodes.values() if n.kind == "edge"]
        gateway = min(edges, key=lambda e: self.transmission_time(
            self.cfg.input_mb / self.cfg.subtasks, self.nodes[source_id], e)
            + self.cfg.edge_cloud_delay_s)
        return [source_id, gateway.node_id, "CLOUD"]

    def _branch(self, task: Task, destination_id: str, nodes: Dict[str, Node]) -> dict:
        p = self.cfg.subtasks
        in_mb, out_mb = task.input_mb / p, task.output_mb / p
        cycles, memory = task.cpu_cycles / p, task.memory_mb / p
        path = self._path(task.source_id, destination_id)
        now = task.arrival_s
        comm_energy = 0.0
        forward = 0.0
        for u, v in zip(path, path[1:]):
            a, b = nodes[u], nodes[v]
            tx = self.transmission_time(in_mb, a, b)
            if a.kind == "edge" and b.kind == "cloud":
                tx += self.cfg.edge_cloud_delay_s
            forward += tx
            comm_energy += self.cfg.tx_power_w * tx
        dest = nodes[destination_id]
        if memory > dest.memory_mb:
            return {"destination": destination_id, "completion": float("inf"),
                    "energy": float("inf"), "forward": forward, "queue": 0,
                    "compute": 0, "return": 0, "path": path, "feasible": False}
        arrival_at_cpu = now + forward
        queue = max(0.0, dest.available_at - arrival_at_cpu)
        compute = cycles / dest.cpu_hz
        start = arrival_at_cpu + queue
        finish_cpu = start + compute
        dest.available_at = finish_cpu
        dest.busy_cycles += cycles
        comp_energy = self.cfg.cpu_energy_coefficient * (dest.cpu_hz ** 2) * cycles
        # Return along reverse path; output transmission is charged per hop.
        ret = 0.0
        for u, v in zip(reversed(path[1:]), reversed(path[:-1])):
            a, b = nodes[u], nodes[v]
            tx = self.transmission_time(out_mb, a, b)
            if a.kind == "cloud" and b.kind == "edge":
                tx += self.cfg.edge_cloud_delay_s
            ret += tx
            comm_energy += self.cfg.tx_power_w * tx
        return {"destination": destination_id, "completion": finish_cpu + ret,
                "energy": comm_energy + comp_energy, "forward": forward,
                "queue": queue, "compute": compute, "return": ret,
                "path": path, "feasible": True}

    def _load_imbalance(self, nodes: Dict[str, Node]) -> float:
        # Relative busy-cycle utilization over the configured demo horizon.
        active = [n for n in nodes.values() if n.kind in ("user", "edge")]
        horizon = max(1.0, self.cfg.task_count * 0.05 + 1.0)
        loads = [n.busy_cycles / (n.cpu_hz * horizon) for n in active]
        mean = sum(loads) / len(loads)
        return sum((x - mean) ** 2 for x in loads) / len(loads)

    def evaluate_pair(self, task: Task, pair: Tuple[str, str],
                      base_nodes: Dict[str, Node] | None = None) -> TaskResult:
        nodes = copy.deepcopy(base_nodes if base_nodes is not None else self.nodes)
        split = self.cfg.split_cycles / nodes[task.source_id].cpu_hz
        branches = [self._branch(task, d, nodes) for d in pair]
        if not all(b["feasible"] for b in branches):
            latency = float("inf")
            energy = float("inf")
        else:
            aggregation = self.cfg.aggregation_cycles / nodes[task.source_id].cpu_hz
            latency = split + max(b["completion"] - task.arrival_s for b in branches) + aggregation
            energy = sum(b["energy"] for b in branches)
        load = self._load_imbalance(nodes)
        penalty = max(0.0, (latency - task.deadline_s) / task.deadline_s) if latency < float("inf") else 1.0
        objective = (self.cfg.w_latency * latency / task.deadline_s
                     + self.cfg.w_energy * energy / self.cfg.energy_reference_j
                     + self.cfg.w_load * load / self.cfg.load_reference
                     + self.cfg.deadline_penalty * penalty)
        return TaskResult(task.task_id, task.source_id, pair, latency, energy,
                          load, latency <= task.deadline_s, objective, branches)

    def run_policy(self, policy: str) -> List[TaskResult]:
        nodes = copy.deepcopy(self.nodes)
        results = []
        for task in self.tasks:
            candidates = self.candidates(task.source_id)
            pairs = list(combinations_with_replacement(candidates, self.cfg.subtasks))
            if policy == "local":
                chosen = (task.source_id, task.source_id)
            elif policy == "nearest":
                source = nodes[task.source_id]
                remote = [nodes[x] for x in candidates if x != task.source_id and nodes[x].kind != "cloud"]
                nearest = min(remote, key=lambda n: self.distance(source, n)).node_id
                chosen = (nearest, nearest)
            elif policy == "greedy":
                chosen = min(pairs, key=lambda p: self.evaluate_pair(task, p, nodes).objective)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            result = self.evaluate_pair(task, chosen, nodes)
            # Commit the selected branch queue/busy-work changes to shared nodes.
            for branch in result.branch_details:
                if branch["feasible"]:
                    d = nodes[branch["destination"]]
                    d.available_at = max(d.available_at, task.arrival_s + branch["forward"]
                                         + branch["queue"] + branch["compute"])
                    d.busy_cycles += task.cpu_cycles / self.cfg.subtasks
            results.append(result)
        return results

    @staticmethod
    def summarize(results: List[TaskResult]) -> dict:
        latencies = [r.latency_s for r in results]
        energies = [r.energy_j for r in results]
        return {
            "tasks": len(results),
            "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
            "p95_latency_s": sorted(latencies)[min(len(latencies)-1, int(0.95*len(latencies)))] if latencies else 0.0,
            "mean_energy_j": statistics.mean(energies) if energies else 0.0,
            "deadline_violation_rate": sum(not r.deadline_met for r in results) / len(results) if results else 0.0,
            "success_rate": sum(r.deadline_met for r in results) / len(results) if results else 0.0,
            "mean_load_imbalance": statistics.mean(r.load_imbalance for r in results) if results else 0.0,
        }
