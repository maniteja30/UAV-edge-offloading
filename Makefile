
PYTHON = python
PYTHONPATH = src

EPISODES = 30
SEED = 200
ARRIVAL_RATE = 1.5
CPU_CYCLES = 2e9
DEADLINE = 2
ENERGY_COEFFICIENT = 1e-28

MODEL = checkpoints/dqn_deadline_aware.pt

.PHONY: help baselines dqn demo

help:
	@echo "Available commands:"
	@echo "  make baselines  - Run baseline policies"
	@echo "  make dqn        - Evaluate trained DQN model"
	@echo "  make demo       - Run baselines followed by DQN"

baselines:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run_demo.py

dqn:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) evaluate_dqn.py \
		--episodes $(EPISODES) \
		--arrival-rate $(ARRIVAL_RATE) \
		--cpu-cycles $(CPU_CYCLES) \
		--deadline $(DEADLINE) \
		--energy-coefficient $(ENERGY_COEFFICIENT) \
		--seed $(SEED) \
		--model $(MODEL) \
		--tag presentation_demo

demo: baselines dqn
