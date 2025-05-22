import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from constants import PERMANENT_PATH, MODEL_NAMES

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


# Group by 'relation' and 'knowledge_source', and count occurrences
df_counts = all_prompting_results.groupby(['rel_lemma', 'knowledge_source']).size().unstack(fill_value=0)

# Ensure that all knowledge sources ('PK', 'CK', 'ND') are present even if some are missing
df_counts = df_counts.reindex(columns=['PK', 'CK', 'ND'], fill_value=0)

# Sort relations by total count across all knowledge sources
df_counts['Total'] = df_counts.sum(axis=1)  # Create a 'Total' column to sort by
df_counts = df_counts.sort_values(by='Total', ascending=False).drop(columns='Total')  # Sort and drop 'Total'

# Plotting
ax = df_counts.plot(kind='barh', stacked=True, color=['#2ca02c', '#ff7f0e', '#1f77b4'], figsize=(16, 14), width=.8)

# Remove the top and right spines (borders)
# Add labels and title with increased font sizes
plt.xlabel('Count', fontsize=14)  # Increased fontsize for x-label
plt.ylabel('Relation', fontsize=14)  # Increased fontsize for y-label

# Increase fontsize for y-axis tick labels
ax.tick_params(axis='y', labelsize=14)  # Set y-axis label size

# Increase fontsize for the legend
ax.legend(fontsize=22, title="Knowledge Source", title_fontsize='22')  # Adjust fontsize of the legend

# Show plot
plt.tight_layout()

# Save the figure as a PDF file
plt.savefig(f"{PERMANENT_PATH}/paper_figures/relation_knowledge_source.pdf")

plt.show()