import pandas as pd
import numpy as np


def combine_selected_modules_activations(
    dataset: pd.DataFrame, vertical: bool, position: str, include_mhsa: bool, include_mlps: bool, include_mlps_l1: bool
) -> np.array:
    """this function combines the activations of all the specified modules into one single activation vector (np.array).
    If vertical is False, it combines all the activations of all the tokens, otherwise, it will consider the specified
    token activations.

    Args:
        dataset (pd.DataFrame): the dataframe containing all the activations resulting from the prompting
        vertical (bool): whether the classification will be applied by token.
        position (str): the token to consider for classification (only valid if vertical is True)

    Returns:
        np.array: an array containing all the features (activations) to use for training.
    """
    selected_modules = {"mhsa": include_mhsa, "mlp": include_mlps, "mlp-l1": include_mlps_l1}
    if vertical:
        # List of length equal to the number of selected modules.
        # Each element of `combined_activations` is a list of length of n_examples_in_dataset
        # Each element of this list is a numpy array of shape
        # (n_layers, 1, module_dim) (this last dimension is different for each module)
        # -> the shape of this list is (n_selected_modules, n_examples_in_dataset, n_layers, 1, module_dim)
        combined_activations: list[list[np.array]] = [
            dataset[module_name].apply(lambda act: np.array(act[position])).tolist()
            for module_name, module_is_selected in selected_modules.items()
            if module_is_selected
        ]
    else:
        combined_activations: list[list[np.array]] = [
            dataset[module_name].apply(lambda act: np.array(act)).tolist()
            for module_name, module_is_selected in selected_modules.items()
            if module_is_selected
        ]

    # zip(*combined_activations) is of "shape" (n_examples_in_dataset, n_selected_modules, n_layers, 1, module_dim)
    # List of shape (n_examples_in_dataset, n_layers, 1, module_dim)
    X: list[np.array] = [np.concatenate(activations) for activations in zip(*combined_activations)]

    # List of shape (n_examples_in_dataset, n_layers, 1, module_dim)
    return np.array(X)


def get_features_and_targets_for_dataset(
    dataset: pd.DataFrame,
    vertical: bool,
    position: str,
    include_mlps: bool,
    include_mlps_l1: bool,
    include_mhsa: bool,
    label_encoder: dict,
) -> tuple:
    """this function returns the features (activations) and targets in the form of numpy arrays for the
    specified modules (MLP, MLP-L1, or MHSA) and tokens (subject_query, relation_query, or object_context)
    or combines multiple tokens/modules.

    Args:
        dataset (pd.DataFrame): the dataset dataframe
        vertical (bool): whether to classify vertically (by token)
        position (str): the token on which to perform classification (subject_q, object, or rel_q)
        include_mlps (bool): whether to include MLPs or not
        include_mlps_l1 (bool): whether to include the first layer of the MLPs or not
        include_mhsa (bool): whether to include the MHSAs or not
        label_encoder (dict): the main label encoder to convert the str labels to ids

    Returns:
        tuple: a tuple containing the train features, test features, train labels, test labels respectively.
    """

    activations = combine_selected_modules_activations(
        dataset=dataset,
        vertical=vertical,
        position=position,
        include_mhsa=include_mhsa,
        include_mlps=include_mlps,
        include_mlps_l1=include_mlps_l1,
    )
    targets = np.array([label_encoder[knowledge_source] for knowledge_source in dataset.knowledge_source])

    return activations, targets


def undersample_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """this function undersamples the given dataset containing CK and PK classes

    Args:
        dataset (pd.DataFrame): the input dataset to undersample to get balanced classes
        prompting_results (pd.DataFrame): the whole dataset

    Returns:
        pd.DataFrame: the balanced dataset
    """

    dataset_minority_class_count = min(
        dataset[dataset.knowledge_source == "CK"].shape[0], dataset[dataset.knowledge_source == "PK"].shape[0]
    )

    dataset_context_knowledge_labels_only = dataset[dataset.knowledge_source == "CK"].sample(
        n=dataset_minority_class_count, random_state=0
    )

    dataset_parametric_knowledge_labels_only = dataset[dataset.knowledge_source == "PK"].sample(
        n=dataset_minority_class_count, random_state=0
    )

    balanced_dataset = pd.concat([dataset_context_knowledge_labels_only, dataset_parametric_knowledge_labels_only])

    return balanced_dataset
