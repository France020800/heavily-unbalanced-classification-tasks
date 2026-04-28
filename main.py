import numpy as np
from matplotlib import pyplot as plt

from models.logistic_regression_model import LogisticRegressionModel
from models.optimizers import Optimizers
from utils.dgp import generate_experiment_datasets
from utils.metrics_and_baseline import get_minibatches, compute_class_metrics, run_lbfgs_b


def run_single_experiment(model, optimizer_func, w0, is_minibatch=False, epochs=100, **kwargs):
    """Runs an optimizer and tracks full and per-class losses over epochs."""

    dataloader = get_minibatches(model.N, batch_size=32) if is_minibatch else None

    w_final, full_losses, w_history = optimizer_func(model, w0, dataloader=dataloader, epochs=epochs, **kwargs)

    loss_history_c0 = []
    loss_history_c1 = []

    for w in w_history:
        (l0, l1), _ = compute_class_metrics(model, w)
        loss_history_c0.append(l0)
        loss_history_c1.append(l1)

    _, (acc_c0, acc_c1) = compute_class_metrics(model, w_final)

    return {
        'w_final': w_final,
        'full_losses': full_losses,
        'loss_c0': loss_history_c0,
        'loss_c1': loss_history_c1,
        'acc_c0': acc_c0,
        'acc_c1': acc_c1
    }


def plot_results(results_dict, lbfgs_opt_loss, title_suffix):
    """Plots the overall loss, per-class losses, and an overlapping view."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Optimization Performance: {title_suffix}", fontsize=18)

    ax_full = axes[0, 0]
    ax_overlap = axes[0, 1]
    ax_c0 = axes[1, 0]
    ax_c1 = axes[1, 1]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (method_name, metrics) in enumerate(results_dict.items()):
        epochs = range(len(metrics['full_losses']))
        color = colors[i % len(colors)]  # Cycle through colors

        # Plot 1: Full Loss
        ax_full.plot(epochs, metrics['full_losses'], color=color,
                     label=f"{method_name} (Avg Acc: {np.mean([metrics['acc_c0'], metrics['acc_c1']]):.2f})")

        # Plot 2: Overlap of Class 0 and Class 1
        ax_overlap.plot(epochs, metrics['loss_c0'], color=color, linestyle='-', label=f"{method_name} (Class 0)")
        ax_overlap.plot(epochs, metrics['loss_c1'], color=color, linestyle='--', label=f"{method_name} (Class 1)")

        # Plot 3: Class 0 Loss (Majority)
        ax_c0.plot(epochs, metrics['loss_c0'], color=color, label=method_name)

        # Plot 4: Class 1 Loss (Minority)
        ax_c1.plot(epochs, metrics['loss_c1'], color=color, label=method_name)

    # --- Plot 1: Full Loss ---
    ax_full.axhline(y=lbfgs_opt_loss, color='r', linestyle=':', label='L-BFGS-B (Global Optimum)')
    ax_full.set_title("Full Training Loss")
    ax_full.set_xlabel("Epochs")
    ax_full.set_ylabel("Loss")
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)

    # --- Plot 2: Overlapping Classes ---
    ax_overlap.set_title("Overlapping Class Losses (Solid: C0, Dashed: C1)")
    ax_overlap.set_xlabel("Epochs")
    ax_overlap.set_ylabel("Loss")
    ax_overlap.legend()
    ax_overlap.grid(True, alpha=0.3)

    # --- Plot 3: Class 0 ---
    ax_c0.set_title("Loss - Class 0 (Majority)")
    ax_c0.set_xlabel("Epochs")
    ax_c0.set_ylabel("Loss")
    ax_c0.legend()
    ax_c0.grid(True, alpha=0.3)

    # --- Plot 4: Class 1 ---
    ax_c1.set_title("Loss - Class 1 (Minority)")
    ax_c1.set_xlabel("Epochs")
    ax_c1.set_ylabel("Loss")
    ax_c1.legend()
    ax_c1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --- Example Execution on a Single Dataset ---
if __name__ == "__main__":
    # let's pick one of the highly unbalanced ones (N=5000, p=0.01)
    datasets_dict = generate_experiment_datasets()
    X, y = datasets_dict[(5000, 0.01)]
    model = LogisticRegressionModel(X, y)

    # Initialize weights to zero
    w0 = np.zeros(model.n)

    # 1. Get Global Optimum baseline
    w_opt, lbfgs_loss = run_lbfgs_b(model, w0)
    print(f"L-BFGS-B Global Minimum Loss: {lbfgs_loss:.6f}")

    # 2. Run experiments
    results = {}
    epochs = 150

    results['Adam'] = run_single_experiment(model, Optimizers.adam, w0, is_minibatch=True, epochs=epochs, lr=0.01)
    # results['Adagrad'] = run_single_experiment(model, Optimizers.adagrad, w0, is_minibatch=True, epochs=epochs)
    # results['SGD Armijo'] = run_single_experiment(model, Optimizers.gradient_descent_armijo, w0, is_minibatch=False, epochs=epochs)

    # 3. Print Final Accuracies
    for method, res in results.items():
        print(f"\n{method}:")
        print(f"  Class 0 (Majority) Accuracy: {res['acc_c0'] * 100:.2f}%")
        print(f"  Class 1 (Minority) Accuracy: {res['acc_c1'] * 100:.2f}%")

    # 4. Plot [cite: 46]
    plot_results(results, lbfgs_loss, "Minibatch (N=5000, p=0.01)")