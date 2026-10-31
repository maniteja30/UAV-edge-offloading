"""Demonstrate a direct User-UAV / Edge-UAV / Cloud topology in EdgeSimPy.

This is a network-topology and mobility demonstration. It does not yet model
application/task transfers or DQN decisions.

Install:
    pip install edge-sim-py networkx

Run from the folder containing direct_uav_network.json:
    python demonstrate_direct_uav_network.py
"""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
from edge_sim_py import EdgeServer, NetworkLink, Simulator, User

DATASET_FILE = "direct_uav_network.json"
SIMULATION_STEPS = 12
TICK_SECONDS = 1.0


def hold_position(user: User) -> None:
    """Keep the native EdgeSimPy User at its current coordinates.

    The direct-radio demonstration handles link distance/rate itself. We do
    not use EdgeSimPy's built-in BaseStation-based user mobility workflow here.
    """
    user.coordinates_trace.append(list(user.coordinates))


def distance_2d(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def advance_waypoint_mobility(node: EdgeServer, tick_seconds: float) -> None:
    """Move one Edge UAV towards its next waypoint at the configured speed."""
    if not getattr(node, "mobility_enabled", False):
        return

    waypoints = getattr(node, "mobility_waypoints", [])
    if not waypoints:
        return

    index = int(getattr(node, "mobility_waypoint_index", 0)) % len(waypoints)
    target = waypoints[index]
    current = node.coordinates
    remaining = distance_2d(current, target)
    max_step = float(node.mobility_speed_mps) * tick_seconds

    if remaining <= max_step or remaining == 0:
        node.coordinates = [float(target[0]), float(target[1])]
        node.mobility_waypoint_index = (index + 1) % len(waypoints)
    else:
        ratio = max_step / remaining
        node.coordinates = [
            float(current[0]) + (float(target[0]) - float(current[0])) * ratio,
            float(current[1]) + (float(target[1]) - float(current[1])) * ratio,
        ]


def update_direct_links() -> None:
    """Recompute direct User-to-Edge wireless links after UAV movement.

    Illustrative link model:
      bandwidth = reference_bandwidth / (1 + distance / 500)
      delay = base_delay + distance / 100

    This is a simple demonstration model, not a calibrated radio-propagation
    model. Edge-to-Cloud backhaul values remain fixed in this example.
    """
    for link in NetworkLink.all():
        node_a, node_b = link.nodes

        if getattr(link, "link_role", "") == "access":
            d = distance_2d(node_a.coordinates, node_b.coordinates)
            link.distance_m = d
            link.active = d <= float(link.max_range_m)

            if link.active:
                link.bandwidth = float(link.reference_bandwidth_mbps) / (1.0 + d / 500.0)
                link.delay = float(link.base_delay_ms) + d / 100.0
            else:
                # Inactive links have no usable bandwidth.
                link.bandwidth = 0.0
                link.delay = float("inf")

        elif getattr(link, "link_role", "") == "backhaul":
            link.active = True
            link.bandwidth = float(link.reference_bandwidth_mbps)
            link.delay = float(link.fixed_delay_ms)


def update_network_each_step(parameters: dict) -> None:
    """EdgeSimPy resource-management callback, called once per simulation tick."""
    for edge_server in EdgeServer.all():
        advance_waypoint_mobility(edge_server, TICK_SECONDS)

    update_direct_links()

    step = parameters.get("current_step", 0)
    print(f"\n--- Simulation step {step} ---")
    for edge_server in EdgeServer.all():
        if getattr(edge_server, "role", "") == "edge_uav":
            x, y = edge_server.coordinates
            print(f"{edge_server.model_name}: position=({x:.1f}, {y:.1f})")

    for link in NetworkLink.all():
        if getattr(link, "link_role", "") == "access":
            uav_name = link.nodes[1].model_name
            status = "UP" if link.active else "DOWN"
            rate = f"{link.bandwidth:.2f} Mbps" if link.active else "0 Mbps"
            delay = f"{link.delay:.2f} ms" if link.active else "unavailable"
            print(f"User -> {uav_name}: {status}, distance={link.distance_m:.1f} m, "
                  f"rate={rate}, delay={delay}")


def stop_after_steps(model: Simulator) -> bool:
    return model.schedule.steps >= SIMULATION_STEPS


def main() -> None:
    simulator = Simulator(
        stopping_criterion=stop_after_steps,
        resource_management_algorithm=update_network_each_step,
        user_defined_functions=[hold_position],
        tick_duration=1,
        tick_unit="seconds",
        dump_interval=float("inf"),
    )

    simulator.initialize(input_file=DATASET_FILE)

    print("Loaded EdgeSimPy components:")
    print(f"  Users: {User.count()}")
    print(f"  EdgeServer objects: {EdgeServer.count()} "
          "(3 Edge UAVs + 1 Cloud)")
    print(f"  NetworkLinks: {NetworkLink.count()}")
    print(f"  Topology nodes: {simulator.topology.number_of_nodes()}")
    print(f"  Topology edges: {simulator.topology.number_of_edges()}")

    # Check that the six links are truly direct: User/EdgeServer nodes only.
    print("\nDirect topology connections:")
    for a, b in simulator.topology.edges():
        link = simulator.topology[a][b]
        print(f"  {a} <-> {b}: {link.bandwidth} Mbps, {link.delay} ms")

    # Demonstrate shortest path from the User UAV to Cloud through the
    # selected Edge UAV/backhaul, using current link-delay values.
    simulator.run_model()

    user = User.find_by_id(1)
    cloud = EdgeServer.find_by_id(4)
    usable_graph = nx.Graph()
    for link in NetworkLink.all():
        if link.active:
            a, b = link.nodes
            usable_graph.add_edge(a, b, delay=link.delay, bandwidth=link.bandwidth)

    print("\nFinal User-to-Cloud route (minimum total link delay):")
    try:
        path = nx.shortest_path(
            usable_graph,
            source=user,
            target=cloud,
            weight="delay",
        )
        total_delay = sum(
            usable_graph[a][b]["delay"] for a, b in zip(path, path[1:])
        )
        print("  " + " -> ".join(str(node) for node in path))
        print(f"  Approximate link delay: {total_delay:.2f} ms")
    except nx.NetworkXNoPath:
        print("  No active direct route to the Cloud at the final step.")

    print("\nDemo finished. This tests topology, waypoint movement, and link updates only.")


if __name__ == "__main__":
    main()
