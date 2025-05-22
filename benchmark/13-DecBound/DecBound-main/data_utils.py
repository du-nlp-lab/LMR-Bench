from sklearn.datasets import make_circles, make_moons
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def generate_x_y(
    num_samples,
    num_dimensions,
    seed,
    data_type="linear",
    factor=0.5,
    class_sep=1,
    noise_moon=0.05,
    num_classes=2,
):
    """Generate X and y data based on the specified data type."""
    if data_type == "linear":
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_dimensions,
            n_informative=num_dimensions,
            n_redundant=0,  # no redundant features
            n_clusters_per_class=1,  # each class is a single cluster
            flip_y=0,  # no noise
            shuffle=True,
            random_state=seed,
            n_classes=num_classes,
            class_sep=class_sep,  # make classes clearly separable
        )
    elif data_type == "circle":
        X, y = make_circles(n_samples=num_samples, shuffle=True, noise=0.05, random_state=seed, factor=factor)
    elif data_type == "moon":
        X, y = make_moons(n_samples=num_samples, shuffle=True, noise=noise_moon, random_state=seed)

    # Normalize X to [0, 1] and then scale to [0, 100]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 100 * (X - X_min) / (X_max - X_min)

    return X, y


def generate_tasks(
    num_tasks, num_samples_per_task, num_dimensions, seed, data_type="linear", factor=0.5, class_sep=2
):
    """
    Generate multiple machine learning tasks, each represented as a separate dataset.

    Args:
        num_tasks (int): Number of tasks to generate.
        num_samples_per_task (int): Number of samples per task.
        num_dimensions (int): Number of features (dimensions) for each sample.
        seed (int): Random seed for reproducibility.
        data_type (str, optional): Type of data distribution. Options are 'linear', 'moon', 'circle'. Default is 'linear'.
        factor (float, optional): Factor controlling the radius of the inner circle for 'circle' data. Default is 0.5.
        class_sep (float, optional): Separation factor for the classes in the generated data. Default is 2.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - X_data (ndarray): Array of shape (num_tasks, num_samples_per_task, num_dimensions) containing feature data.
            - Y_data (ndarray): Array of shape (num_tasks, num_samples_per_task) containing corresponding labels.
    """

    return X_data, Y_data


def generate_context_prompt(X, y, class_names):
    y_named = [class_names[int(label)] for label in y]

    prompt = ""
    for features, label in zip(X, y_named):
        features_str = " ".join(f"{int(num)}" for num in np.round(features))
        prompt += f"Input: {features_str}\nLabel: {label}\n"
    return prompt


def generate_dataset(args, meta_train_X, meta_train_y):
    """
    Generate context and query datasets for training and testing in few-shot learning tasks.

    This function creates context (in-context learning) and query (evaluation) datasets
    for each task based on the provided training data. It ensures that the context and query 
    samples are distinct, preventing data leakage during model evaluation.

    Args:
        args (Namespace): The argument parser namespace containing configuration options.
            - num_in_context (int): Number of samples to include in the context set for each task.
            - num_test_samples (int): Number of samples to include in the query set for each task.
        meta_train_X (numpy.ndarray): The training data for each task, with shape (num_tasks, num_samples_per_task, num_dimensions).
        meta_train_y (numpy.ndarray): The corresponding labels for the training data, with shape (num_tasks, num_samples_per_task).

    Returns:
        context_x (numpy.ndarray): The context inputs for each task, with shape (num_tasks, num_in_context, num_dimensions).
        context_y (numpy.ndarray): The context labels for each task, with shape (num_tasks, num_in_context).
        query_x (numpy.ndarray): The query inputs for each task, with shape (num_tasks, num_test_samples, num_dimensions).
        query_y (numpy.ndarray): The query labels for each task, with shape (num_tasks, num_test_samples).
    
    Function Details:
        - Splits each task into context and query sets.
        - Balances classes within each task to ensure equal representation.
        - Ensures no overlap between context and query samples within each task.
        - Shuffles context samples to reduce order bias during in-context learning."""
    
    context_x = []
    context_y = []
    query_x = []
    query_y = []

    for task_idx, (task_x, task_y) in enumerate(zip(meta_train_X, meta_train_y)):
        num_per_class = args.num_in_context // 2 + args.num_test_samples // 2
        class_0_indices = np.where(task_y == 0)[0][:num_per_class]
        class_1_indices = np.where(task_y == 1)[0][:num_per_class]
        context_0_indices = class_0_indices[: args.num_in_context // 2]
        context_1_indices = class_1_indices[: args.num_in_context // 2]
        test_0_indices = class_0_indices[args.num_in_context // 2 :]
        test_1_indices = class_1_indices[args.num_in_context // 2 :]
        context_indices = np.concatenate([context_0_indices, context_1_indices])
        test_indices = np.concatenate([test_0_indices, test_1_indices])
        np.random.shuffle(context_indices)

        context_x.append(task_x[context_indices])
        context_y.append(task_y[context_indices])
        query_x.append(task_x[test_indices])
        query_y.append(task_y[test_indices])

        # Ensure no overlap between context and query sets
        assert len(set(context_indices) & set(test_indices)) == 0

    print("Generated context and query datasets.")
    return np.array(context_x), np.array(context_y), np.array(query_x), np.array(query_y)
