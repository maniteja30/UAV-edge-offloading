
import os
import matplotlib.pyplot as plt
import numpy as np

# Evaluation results: 30 episodes
policies = ["LOCAL", "NEAREST", "GREEDY", "DQN"]

results = {
    "Mean Latency": {
        "values": [1.2089, 18.9182, 15.4393, 1.3466],
        "ylabel": "Latency (seconds)",
        "filename": "01_mean_latency.png"
    },
    "P95 Latency": {
        "values": [2.9390, 51.6229, 30.6924, 3.1215],
        "ylabel": "Latency (seconds)",
        "filename": "02_p95_latency.png"
    },
    "Mean Energy Consumption": {
        "values": [3.2032, 13.2421, 3.2510, 3.4960],
        "ylabel": "Energy (Joules)",
        "filename": "03_mean_energy.png"
    },
    "Task Success Rate": {
        "values": [85.00, 16.94, 6.87, 84.86],
        "ylabel": "Success Rate (%)",
        "filename": "04_success_rate.png"
    },
    "Overall Objective": {
        "values": [10.0013, 53.0310, 20.3873, 10.9274],
        "ylabel": "Objective Value",
        "filename": "05_objective.png"
    }
}

output_dir = "results/policy_comparison"
os.makedirs(output_dir, exist_ok=True)

for title, data in results.items():

    values = data["values"]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(policies, values)

    plt.title(title + " Comparison", fontsize=14)
    plt.xlabel("Offloading Policy")
    plt.ylabel(data["ylabel"])

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    save_path = os.path.join(output_dir, data["filename"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    print("Saved:", save_path)

print("\nAll comparison graphs generated successfully.")
