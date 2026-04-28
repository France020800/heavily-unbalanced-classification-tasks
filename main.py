import datetime
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from models.logistic_regression_model import LogisticRegressionModel
from models.optimizers import Optimizers
from utils.dgp import generate_experiment_datasets
from utils.logger import DualLogger
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


def plot_results(results_dict, lbfgs_opt_loss, title_suffix, save_path=None):
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()


# --- Example Execution on a Single Dataset ---
if __name__ == "__main__":
    # 1. Setup Results Directory and Logging
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "experiment_log.txt")
    sys.stdout = DualLogger(log_path)

    print("=" * 60)
    print(f"🚀 Starting Optimization Experiments Pipeline")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 2. Define the optimizer configurations
    fb_optimizers = {
        'SGD_Armijo': Optimizers.gradient_descent_armijo,
        'CG_Armijo': Optimizers.cg_armijo,
        'Adagrad': Optimizers.adagrad,
        'RMSProp': Optimizers.rmsprop,
        'AdaDelta': Optimizers.adadelta,
        'Adam': Optimizers.adam
    }

    mb_optimizers = {
        'SGD_Polyak': Optimizers.sgd_polyak,
        'Adagrad': Optimizers.adagrad,
        'RMSProp': Optimizers.rmsprop,
        'AdaDelta': Optimizers.adadelta,
        'Adam': Optimizers.adam
    }

    # 3. Generate Datasets
    print("\nGenerating datasets...")
    datasets_dict = generate_experiment_datasets()
    epochs = 150

    # 4. Main Experiment Loop
    for (N, p), (X, y) in datasets_dict.items():
        print(f"\n{'=' * 60}")
        print(f"📊 DATASET: Size (N) = {N} | Minority Class Ratio (p) = {p}")
        print(f"{'=' * 60}")

        model = LogisticRegressionModel(X, y)
        w0 = np.zeros(model.n)

        # Baseline Calculation
        w_opt, lbfgs_loss = run_lbfgs_b(model, w0)
        print(f"🟢 L-BFGS-B Global Minimum Loss: {lbfgs_loss:.6f}\n")


        # --- Helper Function for Execution ---
        def run_suite(suite_dict, is_mb, regime_name):
            print(f"\n--- Running {regime_name} Regime ---")
            for opt_name, opt_func in suite_dict.items():
                print(f"  ▶ Optimizer: {opt_name}")

                try:
                    # Run the algorithm
                    res_dict = {}
                    res_dict[opt_name] = run_single_experiment(
                        model, opt_func, w0, is_minibatch=is_mb, epochs=epochs
                    )

                    # Print metrics
                    res = res_dict[opt_name]
                    print(f"    Class 0 (Majority) Accuracy: {res['acc_c0'] * 100:.2f}%")
                    print(f"    Class 1 (Minority) Accuracy: {res['acc_c1'] * 100:.2f}%")

                    # Format human-readable filename
                    # Format: regime - optimizer - size - class probability.png
                    filename = f"{regime_name} - {opt_name} - N_{N} - p_{p}.png"
                    save_path = os.path.join(results_dir, filename)
                    title = f"{regime_name} | {opt_name} | N={N}, p={p}"

                    # Plot and Save
                    plot_results(res_dict, lbfgs_loss, title, save_path=save_path)
                    print(f"    💾 Plot saved -> {filename}")

                except Exception as e:
                    print(f"    ❌ Error running {opt_name}: {e}")


        # Run Full Batch models
        run_suite(fb_optimizers, is_mb=False, regime_name="FullBatch")

        # Run Minibatch models
        run_suite(mb_optimizers, is_mb=True, regime_name="Minibatch")

    print("\n" + "=" * 60)
    print("✅ All experiments completed successfully! Check the 'results' folder.")
    print("=" * 60)