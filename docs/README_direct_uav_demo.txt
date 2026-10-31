DIRECT UAV NETWORK DEMO (EdgeSimPy)
==================================

Files
-----
1. direct_uav_network.json
   EdgeSimPy-style dataset: 1 User, 3 mobile EdgeServer nodes (Edge UAVs),
   1 stationary EdgeServer node (Cloud), and 6 direct NetworkLinks.

2. demonstrate_direct_uav_network.py
   Loads the dataset with Simulator.initialize(), moves the three Edge UAVs
   along waypoints, recalculates direct User-to-Edge link bandwidth/delay,
   and prints the topology and final User-to-Cloud route.

Install
-------
python -m pip install edge-sim-py networkx

Run
---
Put both files in the same folder, activate your Python environment, then:

python demonstrate_direct_uav_network.py

Important modelling notes
-------------------------
- There are no BaseStation or NetworkSwitch components in this dataset.
- Direct NetworkLink objects connect the native EdgeSimPy User and EdgeServer
  objects. EdgeSimPy builds its Topology graph from those links.
- The native EdgeSimPy User normally uses BaseStation-based access/mobility
  features. Here, base_station is null and a custom hold_position callback is
  used. The demo therefore uses a custom direct-link model, not EdgeSimPy's
  built-in application communication path workflow.
- Edge UAV movement, link range, bandwidth, and delay calculations are custom
  demonstration logic in the Python file. The sample radio formula is
  illustrative, not a calibrated wireless propagation model.
- Cloud backhaul links are kept fixed and are modelled as abstract direct
  backhaul links.
- This demonstrates topology, UAV movement, and dynamic link updates only.
  It does not yet create task NetworkFlows, process tasks, or run a DQN agent.

To adapt this model for research, replace the illustrative link formula with
an appropriate channel/rate model and add the task execution/offloading layer.
