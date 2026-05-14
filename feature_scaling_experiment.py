import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from models.logistic_regression_model import LogisticRegressionModel
from main import run_single_experiment, DualLogger
from models.optimizers import Optimizers

if __name__ == "__main__":
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(results_dir, "feature_scaling_log.txt"))

    # Hardcoded Configuration
    N = 5000
    p = 0.01
    epochs = 300

    # Feature grid to explore
    feature_counts = [16, 50, 100, 250, 500, 1000]

    adam_acc_c1 = []
    sgd_acc_c1 = []

    print("=" * 60)
    print(f"🚀 Running Feature Scaling Experiment")
    print(f"Dataset: N={N}, p={p}")
    print("=" * 60)

    for n_features in feature_counts:
        print(f"\nGenerating Dataset with {n_features} features...")

        n_inf = int(n_features * 0.8)  # 80% of features are informative, rest are redundant (noise)
        X, y = make_classification(
            n_samples=N,
            n_features=n_features,
            n_informative=n_inf,
            n_redundant=int(n_features * 0.1),
            weights=[1 - p, p],
            random_state=42
        )

        model = LogisticRegressionModel(X, y)
        w0 = np.zeros(model.n)

        # SGD Polyak (Minibatch)
        print(f"  ▶ Training SGD Polyak (MB) on {n_features} features...")
        res_sgd = run_single_experiment(model, Optimizers.sgd_polyak, w0, is_minibatch=True, epochs=epochs)
        sgd_acc_c1.append(res_sgd['acc_c1'] * 100)

        # Adam (Minibatch)
        print(f"  ▶ Training Adam (MB) on {n_features} features...")
        res_adam = run_single_experiment(model, Optimizers.adam, w0, is_minibatch=True, epochs=epochs)
        adam_acc_c1.append(res_adam['acc_c1'] * 100)

        print(
            f"    Results at {n_features} features -> SGD C1 Acc: {res_sgd['acc_c1'] * 100:.2f}% | Adam C1 Acc: {res_adam['acc_c1'] * 100:.2f}%")

    # ======
    # PLOT
    # ======
    plt.figure(figsize=(10, 6))

    plt.plot(feature_counts, adam_acc_c1, label='Adam (Minibatch)', color='dodgerblue', marker='o', linewidth=2.5)
    plt.plot(feature_counts, sgd_acc_c1, label='SGD Polyak (Minibatch)', color='darkorange', marker='s', linewidth=2.5)

    # English Formatting
    plt.title(
        f"Phase Transition: Dimensionality vs Minority Accuracy\n(N={N}, p={p}, Epochs={epochs})",
        fontsize=14)
    plt.xlabel("Number of Features (n)", fontsize=12)
    plt.ylabel("Minority Class (C1) Accuracy (%)", fontsize=12)
    plt.xscale('log')

    plt.xticks(feature_counts, labels=[str(f) for f in feature_counts])

    plt.ylim(-5, 105)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=12, loc='lower right')

    save_path = os.path.join(results_dir, f"Features_Scaling_N_{N}_p_{p}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Experiment completed successfully. Plot saved at: {save_path}")