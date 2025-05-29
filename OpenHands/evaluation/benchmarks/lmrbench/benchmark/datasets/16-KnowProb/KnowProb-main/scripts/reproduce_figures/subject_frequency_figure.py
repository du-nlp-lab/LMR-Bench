import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from constants import MODEL_NAMES, PERMANENT_PATH

"""
This script creates the figure representing the subject frequency in The Pile corpus for all the LLMs and by knowledge source.
"""

all_prompting_results = []

for model_name in MODEL_NAMES:
    prompting_results = pd.read_pickle(
        f"{PERMANENT_PATH}/paper_data/prompting_results/{model_name}_prompting_results_without_activations.pkl"
    )
    prompting_results["model_name"] = MODEL_NAMES[model_name]
    all_prompting_results.append(prompting_results)

raw_pararel_dataset_with_statements = pd.read_csv(
    f"{PERMANENT_PATH}/paper_data/raw_pararel_with_correctly_reformatted_statements.csv"
)
all_prompting_results = pd.concat(all_prompting_results)
entity_counts = pd.read_pickle(f"{PERMANENT_PATH}/paper_data/entity_counts.pkl")

# get the subject column to count the frequencies in The Pile corpus
all_prompting_results = all_prompting_results.merge(
    raw_pararel_dataset_with_statements[["statement_subject", "subject"]], how="left", on="statement_subject"
)

all_prompting_results["subject_count_in_thepile"] = all_prompting_results.subject.apply(
    lambda subject: entity_counts[subject]
)

# Create a boxplot
plt.figure(figsize=(12, 6))

# Set the order of knowledge_source categories
knowledge_source_order = ["CK", "ND", "PK"]
# Define custom palette using default colors from matplotlib
custom_palette = {
    "CK": plt.cm.tab10(1),  # Orange (second color)
    "ND": plt.cm.tab10(0),  # Blue (first color)
    "PK": plt.cm.tab10(2),  # Green (third color)
}

# Create the boxplot
sns.boxplot(
    data=all_prompting_results,
    x="model_name",
    y="subject_count_in_thepile",
    hue="knowledge_source",
    hue_order=knowledge_source_order,
    palette=custom_palette,
)

# Set y-axis to log scale
plt.yscale("log")

# Add titles and labels
plt.xlabel("Model Name", fontsize=16)
plt.ylabel("Subject Count in Pile (Log Scale)", fontsize=16)

# Show the plot
plt.legend(title="Knowledge Source", loc="lower right", frameon=True, title_fontsize=16, prop={"size": 17})

# Increase the font size of the tick parameters for both axes
plt.tick_params(axis="x", labelsize=22)
plt.tick_params(axis="y", labelsize=14)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()

plt.savefig(f"{PERMANENT_PATH}/paper_figures/subject_count_in_thepile_by_knowledge_source.pdf", bbox_inches="tight")

plt.show()
