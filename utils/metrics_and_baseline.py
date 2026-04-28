import numpy as np
from scipy.optimize import minimize


def get_minibatches(N, batch_size=32):
    """Generates random minibatches for an epoch."""
    indices = np.random.permutation(N)
    return [indices[i:i + batch_size] for i in range(0, N, batch_size)]


def compute_class_metrics(model, w):
    """Computes loss and accuracy separated by class."""
    y = model.y
    preds = model.sigmoid(model.X @ w) >= 0.5

    idx_c0 = np.where(y == 0)[0]
    idx_c1 = np.where(y == 1)[0]

    # Per-class loss
    loss_c0 = model.compute_loss(w, indices=idx_c0) if len(idx_c0) > 0 else 0
    loss_c1 = model.compute_loss(w, indices=idx_c1) if len(idx_c1) > 0 else 0

    # Per-class accuracy
    acc_c0 = np.mean(preds[idx_c0] == y[idx_c0]) if len(idx_c0) > 0 else 0
    acc_c1 = np.mean(preds[idx_c1] == y[idx_c1]) if len(idx_c1) > 0 else 0

    return (loss_c0, loss_c1), (acc_c0, acc_c1)


def run_lbfgs_b(model, w0):
    """Runs L-BFGS-B to find the global optimum as a reference."""

    def objective(w):
        return model.compute_loss(w)

    def gradient(w):
        return model.compute_gradient(w)

    res = minimize(objective, w0, method='L-BFGS-B', jac=gradient)
    return res.x, res.fun