from sklearn.datasets import make_classification
import numpy as np

def generate_experiment_datasets():
    """
    Generates a dictionary of binary classification datasets based on the assignment rules.
    Returns:
        datasets: A dictionary where keys are (N, p) tuples and values are (X, y) tuples.
    """
    # Parameters from the assignment
    N_values = [500, 2000, 5000]
    p_values = [0.1, 0.01, 0.005]
    n_features = 16

    datasets = {}

    for N in N_values:
        for p in p_values:
            X, y = make_classification(
                n_samples=N,
                n_features=n_features,
                n_informative=10,  # Number of informative features for the problem
                n_redundant=2,  # noise/redundancy to make optimization realistic
                n_repeated=0,
                n_classes=2,
                weights=[1.0 - p, p],
                random_state=42  # Deterministic for reproducibility
            )

            datasets[(N, p)] = (X, y)

            # Sanity check to verify the actual imbalance
            actual_p = np.mean(y)
            print(
                f"Dataset [N={N:4d}, p={p:5.3f}] -> Actual positive ratio: {actual_p:.4f} (Positives: {int(np.sum(y))})")

    return datasets