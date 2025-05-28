from pathlib import Path

import pandas as pd
import numpy as np

DEFAULT_BASE_PATH = "."


def load_and_process_data_with_ci(
    model_name: str, module_type: str, token_type: str, specific_relation: str, base_path: str = DEFAULT_BASE_PATH
) -> pd.DataFrame:
    classification_metrics = pd.read_csv(
        Path(base_path) / f"classification_metrics/{model_name}/{model_name}_classification_metrics_by_relation_groups_{token_type}_{module_type}.csv"
    )

    if specific_relation is not None:
        classification_metrics = classification_metrics[classification_metrics.relation_group_id == specific_relation]

    classification_metrics = classification_metrics.rename(columns={"success_rate": "success_rate_layer"})

    classification_metrics["se_k"] = np.sqrt(
        (classification_metrics["success_rate_layer"] * (1 - classification_metrics["success_rate_layer"]))
        / classification_metrics["nb_test_examples"]
    )
    classification_metrics["nb_success_trials_layer"] = (
        classification_metrics["success_rate_layer"] * classification_metrics["nb_test_examples"]
    )

    grouped_classification_metrics = (
        classification_metrics.groupby(["layer"])
        .agg(
            acc_mean=("success_rate_layer", "mean"),
            total_trials=("nb_success_trials_layer", "sum"),
        )
        .reset_index()
    )

    all_data_nb_trials = pd.merge(classification_metrics, grouped_classification_metrics, on="layer")

    all_data_nb_trials["weighted_se_k"] = (
        all_data_nb_trials["nb_test_examples"] / all_data_nb_trials["total_trials"]
    ) ** 2 * all_data_nb_trials["se_k"] ** 2

    global_se = (
        all_data_nb_trials.groupby(["layer"])
        .agg(
            se=("weighted_se_k", "sum"),
            nb_success_total=("nb_success_trials_layer", "sum"),
            nb_trials_total=("nb_test_examples", "sum"),
        )
        .reset_index()
    )

    global_se["se"] = np.sqrt(global_se["se"])
    global_se["P"] = global_se["nb_success_total"] / global_se["nb_trials_total"]

    # Calculate the confidence intervals using the Z-score for 95% CI
    z = 1.96
    global_se["P_ci_lower"] = global_se["P"] - z * global_se["se"]
    global_se["P_ci_upper"] = global_se["P"] + z * global_se["se"]

    return global_se
