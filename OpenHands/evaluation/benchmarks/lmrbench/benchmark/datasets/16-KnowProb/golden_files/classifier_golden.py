import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed

from src.classification.data import get_features_and_targets_for_dataset, undersample_dataset


def _fit_regression(train_activations, train_targets, test_activations, test_targets, layer, relation_group_id):
    if test_activations.shape[0] > 0:
        predictive_model = LogisticRegression(random_state=16, max_iter=10000, n_jobs=1)
        predictive_model.fit(train_activations[:, layer, 0, :], train_targets)

        predictions = predictive_model.predict(test_activations[:, layer, 0, :])
        return {
            "layer": layer,
            "relation_group_id": relation_group_id,
            "success_rate": accuracy_score(test_targets, predictions),
            "nb_test_examples": test_activations.shape[0],
            "nb_train_examples": train_activations.shape[0],
        }

    else:
        return {
            "layer": None,
            "relation_group_id": relation_group_id,
            "success_rate": None,
            "nb_test_examples": None,
            "nb_train_examples": None,
        }


def perform_classification_by_relation_group(
    prompting_results: pd.DataFrame,
    include_mlps: bool,
    include_mlps_l1: bool,
    include_mhsa: bool,
    vertical: bool,
    position: str,
) -> None:
    """this function performs classification over the saved activations, the classification can be module-wise and token-wise (i.e. vertical)

    Args:
        prompting_results (pd.DataFrame): the dataframe with all the activations resulting from the prompting
        include_mlps (bool): whether to include the MLPs or not
        include_mlps_l1 (bool): whether to include the first layer of the MLPs or not
        include_mhsa (bool): whether to include the MHSAs or not
        vertical (bool): whether to train the classifier vertically (by token) or all at once
        position (str): the token where the classification is performed (subject_query, object, or relation_query)
    """

    label_encoder = {"CK": 0, "PK": 1}

    prompting_results = prompting_results.reset_index()

    classification_metrics = []

    for relation_group_id in prompting_results.relation_group_id.unique():
        print(f"Evaluating the {relation_group_id} relation group.")

        train_dataset = undersample_dataset(prompting_results[prompting_results.relation_group_id != relation_group_id])
        test_dataset = undersample_dataset(prompting_results[prompting_results.relation_group_id == relation_group_id])

        train_activations, train_targets = get_features_and_targets_for_dataset(
            dataset=train_dataset,
            vertical=vertical,
            position=position,
            include_mlps=include_mlps,
            include_mlps_l1=include_mlps_l1,
            include_mhsa=include_mhsa,
            label_encoder=label_encoder,
        )

        test_activations, test_targets = get_features_and_targets_for_dataset(
            dataset=test_dataset,
            vertical=vertical,
            position=position,
            include_mlps=include_mlps,
            include_mlps_l1=include_mlps_l1,
            include_mhsa=include_mhsa,
            label_encoder=label_encoder,
        )

        classification_metrics += Parallel(n_jobs=12, backend="loky", verbose=10)(
            delayed(_fit_regression)(
                train_activations=train_activations,
                train_targets=train_targets,
                test_activations=test_activations,
                test_targets=test_targets,
                layer=layer,
                relation_group_id=relation_group_id,
            )
            for layer in list(range(train_activations.shape[1]))
        )

    return classification_metrics


def save_classification_metrics(
    classification_metrics: pd.DataFrame,
    model_name: str,
    permanent_path: str,
    include_mlps: bool,
    include_mlps_l1: bool,
    include_mhsa: bool,
    vertical: bool,
    position: str,
    test_on_head: bool,
    n_head: int,
) -> None:
    """this function saves the classification results into a csv file.

    Args:
        classification_metrics (pd.DataFrame): the classification metrics to save
        model_name (str): the name of the current LLM
        permanent_path (str): the path where to store the results
        include_mlps (bool): whether to include the MLPs or not
        include_mlps_l1 (bool): whether to include the first layer of the MLPs or not
        include_mhsa (bool): whether to include the MHSAs or not
        vertical (bool): whether to train the classifier vertically (by token) or all at once
        position (str): the token where the classification is performed (subject_query, object, or relation_query)
        test_on_head (bool): whether the current experiment is applied on the header of the reformatted ParaRel dataset
        n_head (int): the number of rows to consider on the header (if test_on_head is True)
    """
    # save the classification metrics
    save_path = f"{permanent_path}/classification_metrics/{model_name}_classification_metrics_by_relation_groups"

    if vertical:
        save_path += "_" + position
    else:
        save_path += "_all_positions"

    if test_on_head:
        save_path += f"_n_head={n_head}"

    # Append the appropriate suffix based on the included modules
    if include_mhsa:
        save_path += "_mhsa"
    if include_mlps:
        save_path += "_mlps"
    if include_mlps_l1:
        save_path += "_mlps_l1"
    save_path += ".csv"

    pd.DataFrame(classification_metrics).to_csv(save_path)
