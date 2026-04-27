import numpy as np
from utils.armijo import armijo_line_search

class Optimizers:
    @staticmethod
    def gradient_descent_armijo(model, w0, epochs=100):
        """Gradient descent with Armijo Line Search """
        w = w0.copy()
        losses = []
        for _ in range(epochs):
            loss = model.compute_loss(w)
            grad = model.compute_gradient(w)
            losses.append(loss)

            # Direction is the gradient itself
            alpha = armijo_line_search(model, w, grad, grad, loss)
            w = w - alpha * grad
        return w, losses

    @staticmethod
    def cg_armijo(model, w0, epochs=100):
        """Conjugate Gradient Method with Armijo line search and restart"""
        w = w0.copy()
        losses = []

        grad = model.compute_gradient(w)
        direction = grad.copy()
        grad_prev = grad.copy()

        for i in range(epochs):
            loss = model.compute_loss(w)
            losses.append(loss)

            # Restart condition based on gradient-related conditions
            # using the Polak-Ribiere restart heuristic: |grad^T grad_prev| > 0.2 ||grad||^2
            if i > 0 and abs(np.dot(grad.T, grad_prev)) > 0.2 * np.linalg.norm(grad) ** 2:
                direction = grad.copy()

            alpha = armijo_line_search(model, w, direction, grad, loss)
            w = w - alpha * direction

            grad_prev = grad.copy()
            grad = model.compute_gradient(w)

            # Fletcher-Reeves beta update
            beta = (np.linalg.norm(grad) ** 2) / (np.linalg.norm(grad_prev) ** 2 + 1e-8)
            direction = grad + beta * direction

        return w, losses

    @staticmethod
    def sgd_polyak(model, w0, dataloader, epochs=100):
        """Stochastic gradient descent with Polyak stepsize"""
        w = w0.copy()
        losses = []
        f_star = 0.0  # Lower bound for binary cross-entropy loss

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0
            for indices in dataloader:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                epoch_loss += loss
                batches += 1

                # Polyak step size calculation
                grad_norm_sq = np.linalg.norm(grad) ** 2
                if grad_norm_sq > 1e-8:
                    alpha = (loss - f_star) / grad_norm_sq
                else:
                    alpha = 1e-4

                w = w - alpha * grad
            losses.append(epoch_loss / batches)
        return w, losses

    @staticmethod
    def adam(model, w0, dataloader=None, epochs=100, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam"""
        w = w0.copy()
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        t = 0
        losses = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0

            # If dataloader is None, treat as Full Batch
            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                t += 1
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                epoch_loss += loss
                batches += 1

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

            losses.append(epoch_loss / batches)
        return w, losses