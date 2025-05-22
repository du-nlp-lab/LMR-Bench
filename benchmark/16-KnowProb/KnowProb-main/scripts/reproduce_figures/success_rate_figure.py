import seaborn as sns
import matplotlib.pyplot as plt

from src.classification.eval import load_and_process_data_with_ci

from constants import MODEL_NAMES, PERMANENT_PATH

"""
This script create the main figure of the success rates for all the LLMs, modules, and tokens.
"""

ticks_32 = [1, 3, 6, 9, 12, 15, 18, 21, 24, 28, 32]
ticks_24 = [1, 3, 6, 9, 12, 15, 18, 21, 24]

# Increase default font size for all plot elements
plt.rcParams.update({"font.size": 16})

specific_relation = None

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
markers = ["o", "s", "d"]
token_labels_to_filepath = {
    "First (control)": "first",
    "Object": "object",
    "Subject": "subject_query",
    "Relation": "relation_query",
}

fig, axes = plt.subplots(4, 4, figsize=(32, 32), sharey="row")

for model_idx, model_name in enumerate(MODEL_NAMES):
    for token_plot_position_in_plot, token_label in enumerate(token_labels_to_filepath.keys()):
        ax = axes[model_idx, token_plot_position_in_plot]

        # Load data for each model
        mlp_data = load_and_process_data_with_ci(
            model_name=model_name,
            module_type="mlps",
            token_type=token_labels_to_filepath[token_label],
            specific_relation=specific_relation,
        )
        mhsa_data = load_and_process_data_with_ci(
            model_name=model_name,
            module_type="mhsa",
            token_type=token_labels_to_filepath[token_label],
            specific_relation=specific_relation,
        )
        mlp_l1_data = load_and_process_data_with_ci(
            model_name=model_name,
            module_type="mlps_l1",
            token_type=token_labels_to_filepath[token_label],
            specific_relation=specific_relation,
        )

        for data, label, color, marker in zip(
            [mlp_data, mhsa_data, mlp_l1_data], ["MLP-L2", "MHSA", "MLP-L1"], colors, markers
        ):
            # ax.set_title(f"{token_label}", fontsize=25)
            ax.plot(data["layer"] + 1, data["P"], label=label, color=color, marker=marker, markersize=10, linewidth=6)
            ax.fill_between(data["layer"] + 1, data["P_ci_lower"], data["P_ci_upper"], color=color, alpha=0.1)

        if token_plot_position_in_plot == 1 and model_idx == 0:
            ax.set_title(
                f"                                                          {MODEL_NAMES[model_name]}\n\n{token_label}",
                fontsize=30,
            )  # Only top row gets token_label titles
        elif token_plot_position_in_plot != 1 and model_idx == 0:
            ax.set_title(f"{token_label}", fontsize=30)
        elif token_plot_position_in_plot == 1 and model_idx != 0:
            ax.set_title(
                f"                                                          {MODEL_NAMES[model_name]}",
                fontsize=30,
            )  # Only top row gets token_label titles

        if model_idx == len(MODEL_NAMES) - 1:
            ax.set_xlabel("Layer", fontsize=30)

        if max(data["layer"]) == 31:
            ticks = ticks_32
        elif max(data["layer"]) == 23:
            ticks = ticks_24

        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlim(xmin=1)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.tick_params(axis="y", labelsize=23)
        handles, labels = ax.get_legend_handles_labels()

        if token_plot_position_in_plot == 0:
            ax.set_ylabel("Success Rate", fontsize=30)

# Configure legend
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=31, frameon=False)
sns.despine()

plt.tight_layout()

plt.savefig(f"{PERMANENT_PATH}/paper_figures/all_models_avg_success_rate.pdf", bbox_inches="tight")

plt.show()
