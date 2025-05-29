import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from constants import MODEL_NAMES, PERMANENT_PATH

"""
This script creates the figure with the count of used knowledge sources for all the LLMs.
"""

all_prompting_results = []

for model_name in MODEL_NAMES:
    prompting_results = pd.read_pickle(
        f"{PERMANENT_PATH}/paper_data/prompting_results/{model_name}_prompting_results_without_activations.pkl"
    )
    prompting_results["model_name"] = model_name
    all_prompting_results.append(prompting_results)

all_prompting_results = pd.concat(all_prompting_results)

# Set up the colors and hatches for each knowledge source
color_map = {"CK": "#d95f0e", "ND": "#1c69a7", "PK": "#4daf4a"}
hatch_map = {"CK": "--", "ND": "\\", "PK": "//"}

# Count the occurrences of each knowledge_source per model_name
data = all_prompting_results.groupby(["model_name", "knowledge_source"]).size().unstack(fill_value=0)

# Define the number of bars per group and set the width of each bar
n_bars = len(color_map)  # Number of knowledge sources
bar_width = 0.3  # Width of each bar

# Create an array of positions for each model to adjust for multiple bars per model
indices = np.arange(len(data.index))

# Plot each knowledge source as its own bar in the grouped bar chart
fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size for readability

for i, (knowledge_source, color) in enumerate(color_map.items()):
    ax.bar(
        indices + i * bar_width,
        data[knowledge_source],
        width=bar_width,
        color=color,
        label=knowledge_source,
        edgecolor="black",
        hatch=hatch_map[knowledge_source],
    )

# Set the x-axis with model names centered for each group of bars
ax.set_xticks(indices + bar_width * (n_bars - 1) / 2)
ax.set_xticklabels(
    [MODEL_NAMES[model_name] for model_name in data.index], fontsize=32
)  # Increased font size for x-tick labels

# Add labels and title with increased font size
ax.set_xlabel("Model", fontsize=22)
ax.set_ylabel("Count of Used Knowledge Source", fontsize=18)

# Customize legend with a larger font size
ax.legend(title="Knowledge Source", loc="upper right", frameon=True, title_fontsize=24, prop={"size": 24})

# Increase the font size of the tick parameters for both axes
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=14)

plt.tight_layout()

plt.savefig(f"{PERMANENT_PATH}/paper_figures/knowledge_source_count.pdf", bbox_inches="tight")

# Show the plot
plt.show()
