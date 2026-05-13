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
    """Esegue un ottimizzatore e traccia loss globali, per classe e batch-level."""

    dataloader = get_minibatches(model.N, batch_size=32) if is_minibatch else None

    w_final, epoch_losses, batch_losses, w_history = optimizer_func(
        model, w0, dataloader=dataloader, epochs=epochs, **kwargs
    )

    loss_history_c0 = []
    loss_history_c1 = []

    for w in w_history:
        (l0, l1), _ = compute_class_metrics(model, w)
        loss_history_c0.append(l0)
        loss_history_c1.append(l1)

    _, (acc_c0, acc_c1) = compute_class_metrics(model, w_final)

    return {
        'w_final': w_final,
        'full_losses': epoch_losses,
        'batch_losses': batch_losses,
        'loss_c0': loss_history_c0,
        'loss_c1': loss_history_c1,
        'acc_c0': acc_c0,
        'acc_c1': acc_c1
    }


def plot_results(results_dict, lbfgs_opt_loss, title_suffix, save_path=None):
    """Genera una dashboard 3x2 con scala logaritmica e f - f*."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"Optimization Performance: {title_suffix}", fontsize=20)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f_star = lbfgs_opt_loss

    for i, (method_name, metrics) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(metrics['full_losses']) + 1)
        total_batches = len(metrics['batch_losses'])
        x_batches = np.linspace(0, len(metrics['full_losses']), total_batches)

        # Calcolo di f - f* (con limite inferiore di sicurezza per la scala logaritmica)
        f_diff_epoch = np.maximum(np.array(metrics['full_losses']) - f_star, 1e-10)
        f_diff_batch = np.maximum(np.array(metrics['batch_losses']) - f_star, 1e-10)

        # 1. Full Training Loss (f - f*) [LOG SCALE]
        axes[0, 0].plot(epochs, f_diff_epoch, color=color, label=method_name)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_ylabel("Loss (f - f*) [Log]")

        # 2. Overlap Class Losses [Standard Loss, LOG SCALE]
        axes[0, 1].plot(epochs, metrics['loss_c0'], color=color, linestyle='-', label=f"{method_name} (C0)")
        axes[0, 1].plot(epochs, metrics['loss_c1'], color=color, linestyle='--', label=f"{method_name} (C1)")
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_ylabel("Raw Loss [Log]")

        # 3. STOCHASTIC NOISE: Batch vs Epoch Loss (f - f*) [LOG SCALE]
        axes[1, 0].plot(x_batches, f_diff_batch, color=color, alpha=0.3, linewidth=0.5)
        axes[1, 0].plot(epochs, f_diff_epoch, color=color, linewidth=2, label=f"{method_name} (Smooth)")
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_ylabel("Loss (f - f*) [Log]")

        # 4. Zoom sui primi batch (f - f*) [LOG SCALE]
        n_zoom = min(total_batches, 500)
        axes[1, 1].plot(x_batches[:n_zoom], f_diff_batch[:n_zoom], color=color, alpha=0.6)
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_ylabel("Loss (f - f*) [Log]")

        # 5. Class 0 Loss [LOG SCALE]
        axes[2, 0].plot(epochs, metrics['loss_c0'], color=color, label=method_name)
        axes[2, 0].set_yscale('log')
        axes[2, 0].set_ylabel("Raw Loss [Log]")

        # 6. Class 1 Loss [LOG SCALE]
        axes[2, 1].plot(epochs, metrics['loss_c1'], color=color, label=method_name)
        axes[2, 1].set_yscale('log')
        axes[2, 1].set_ylabel("Raw Loss [Log]")

    # Formattazione Grafici
    titles = [
        "Global Loss (f - f*)", "Overlapping Classes (Solid:C0, Dash:C1)",
        "Stochastic Noise (f - f*)", "Stochastic Noise (Initial Zoom)",
        "Loss - Class 0 (Majority)", "Loss - Class 1 (Minority)"
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epochs")
        ax.grid(True, which="both", ls="--", alpha=0.3)  # Grid migliore per log scale
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "experiment_log.txt")
    sys.stdout = DualLogger(log_path)

    print("=" * 60)
    print(f"🚀 Starting Optimization Experiments Pipeline")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    fb_optimizers = {
        'GD_Armijo': Optimizers.gradient_descent_armijo,
        'CG_Armijo': Optimizers.cg_wolfe,
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

    print("\nGenerating datasets...")
    datasets_dict = generate_experiment_datasets()
    epochs = 1000

    for (N, p), (X, y) in datasets_dict.items():
        print(f"\n{'=' * 60}")
        print(f"📊 DATASET: Size (N) = {N} | Minority Class Ratio (p) = {p}")
        print(f"{'=' * 60}")

        model = LogisticRegressionModel(X, y)
        w0 = np.zeros(model.n)

        w_opt, lbfgs_loss = run_lbfgs_b(model, w0)
        print(f"🟢 L-BFGS-B Global Minimum Loss: {lbfgs_loss:.6f}\n")

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

                    res = res_dict[opt_name]

                    # Extract final losses
                    final_global_loss = res['full_losses'][-1]
                    final_loss_c0 = res['loss_c0'][-1]
                    final_loss_c1 = res['loss_c1'][-1]

                    # Calculate true global accuracy weighted by probability (p)
                    global_acc = res['acc_c0'] * (1 - p) + res['acc_c1'] * p

                    print(f"    ✅ Global Accuracy: {global_acc * 100:.2f}%")
                    print(f"       Class 0 (Majority) Accuracy: {res['acc_c0'] * 100:.2f}%")
                    print(f"       Class 1 (Minority) Accuracy: {res['acc_c1'] * 100:.2f}%")
                    print(f"    📉 Final Global Loss: {final_global_loss:.6f}")
                    print(f"       Final Class 0 Loss: {final_loss_c0:.6f}")
                    print(f"       Final Class 1 Loss: {final_loss_c1:.6f}")

                    # Format human-readable filename
                    filename = f"{regime_name}-{opt_name}-N_{N}-p_{p}.png"
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


def single_run():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 Starting Single Optimization Experiment")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    N = 5000
    p = 0.01
    epochs = 10000
    opt_name = 'SGD_Polyak'
    regime_name = 'Minibatch'

    log_path = os.path.join(results_dir, f"{regime_name}-{opt_name}-N_{N}-p_{p}-e_{epochs}.txt")
    sys.stdout = DualLogger(log_path)

    print("\nGenerating datasets...")
    datasets_dict = generate_experiment_datasets()
    X, y = datasets_dict[(N, p)]

    print(f"\n{'=' * 60}")
    print(f"📊 DATASET: Size (N) = {N} | Minority Class Ratio (p) = {p}")
    print(f"{'=' * 60}")

    model = LogisticRegressionModel(X, y)
    w0 = np.zeros(model.n)

    # Baseline Calculation
    w_opt, lbfgs_loss = run_lbfgs_b(model, w0)
    print(f"🟢 L-BFGS-B Global Minimum Loss: {lbfgs_loss:.6f}\n")

    print(f"--- Running {regime_name} Regime ---")
    print(f"  ▶ Optimizer: {opt_name} (Epochs: {epochs})")

    try:
        res_dict = {}
        res_dict[opt_name] = run_single_experiment(
            model, Optimizers.sgd_polyak, w0, is_minibatch=False, epochs=epochs
        )

        res = res_dict[opt_name]

        final_global_loss = res['full_losses'][-1]
        final_loss_c0 = res['loss_c0'][-1]
        final_loss_c1 = res['loss_c1'][-1]

        global_acc = res['acc_c0'] * (1 - p) + res['acc_c1'] * p

        # Print metrics
        print(f"    ✅ Global Accuracy: {global_acc * 100:.2f}%")
        print(f"       Class 0 (Majority) Accuracy: {res['acc_c0'] * 100:.2f}%")
        print(f"       Class 1 (Minority) Accuracy: {res['acc_c1'] * 100:.2f}%")
        print(f"    📉 Final Global Loss: {final_global_loss:.6f}")
        print(f"       Final Class 0 Loss: {final_loss_c0:.6f}")
        print(f"       Final Class 1 Loss: {final_loss_c1:.6f}")

        filename = f"SingleExp - {regime_name} - {opt_name} - N_{N} - p_{p} - {epochs}ep.png"
        save_path = os.path.join(results_dir, filename)
        title = f"{regime_name} | {opt_name} | N={N}, p={p} | Epochs={epochs}"

        # Plot and Save
        plot_results(res_dict, lbfgs_loss, title, save_path=save_path)
        print(f"    💾 Plot saved -> {filename}")

    except Exception as e:
        print(f"    ❌ Error running {opt_name}: {e}")

    print("\n" + "=" * 60)
    print("✅ Single experiment completed successfully! Check the 'results' folder.")
    print("=" * 60)


if __name__ == "__main__":
    #main()
    single_run()
