import numpy as np

class LogisticRegressionModel:
    def __init__(self, X, y):
        """
        X: numpy array of shape (N, n)
        y: numpy array of shape (N,) with labels 0 or 1
        """
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y
        self.N, self.n = self.X.shape

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, w, indices=None):
        """Computes the loss, optionally on a minibatch defined by indices."""
        X_batch = self.X[indices] if indices is not None else self.X
        y_batch = self.y[indices] if indices is not None else self.y

        preds = self.sigmoid(X_batch @ w)
        eps = 1e-15
        loss = -np.mean(y_batch * np.log(preds + eps) + (1 - y_batch) * np.log(1 - preds + eps))
        return loss

    def compute_gradient(self, w, indices=None):
        """Computes the gradient, optionally on a minibatch defined by indices."""
        X_batch = self.X[indices] if indices is not None else self.X
        y_batch = self.y[indices] if indices is not None else self.y

        preds = self.sigmoid(X_batch @ w)
        grad = (X_batch.T @ (preds - y_batch)) / len(y_batch)
        return grad