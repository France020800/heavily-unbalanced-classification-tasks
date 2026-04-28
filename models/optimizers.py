import numpy as np
from utils.armijo import armijo_line_search

class Optimizers:
    @staticmethod
    def gradient_descent_armijo(model, w0, dataloader=None, epochs=100, **kwargs):
        """Gradient descent with Armijo Line Search """
        w = w0.copy()
        w_hist = []
        losses = []
        for _ in range(epochs):
            loss = model.compute_loss(w)
            grad = model.compute_gradient(w)
            losses.append(loss)
            w_hist.append(w.copy())

            # Direction is the gradient itself
            alpha = armijo_line_search(model, w, grad, grad, loss)
            w = w - alpha * grad
        return w, losses, w_hist

    @staticmethod
    def cg_armijo(model, w0, dataloader=None, epochs=100, **kwargs):
        """Conjugate Gradient Method with Armijo line search and restart"""
        w = w0.copy()
        losses = []
        w_hist = []

        grad = model.compute_gradient(w)
        direction = grad.copy()
        grad_prev = grad.copy()

        for i in range(epochs):
            loss = model.compute_loss(w)
            losses.append(loss)
            w_hist.append(w.copy())

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

        return w, losses, w_hist

    @staticmethod
    def sgd_polyak(model, w0, dataloader, epochs=100):
        """Stochastic gradient descent with Polyak stepsize"""
        w = w0.copy()
        losses = []
        w_hist = []
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
            w_hist.append(w.copy())
        return w, losses, w_hist

    @staticmethod
    def adagrad(model, w0, dataloader=None, epochs=100, lr=0.01, eps=1e-8):
        """Adagrad"""
        w = w0.copy()
        G = np.zeros_like(w)
        losses = []
        w_hist = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0

            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                epoch_loss += loss
                batches += 1

                G += grad ** 2

                w = w - (lr / (np.sqrt(G) + eps)) * grad

            losses.append(epoch_loss / batches)
            w_hist.append(w.copy())
        return w, losses, w_hist

    @staticmethod
    def rmsprop(model, w0, dataloader=None, epochs=100, lr=0.01, gamma=0.9, eps=1e-8):
        """RMSProp"""
        w = w0.copy()
        Eg2 = np.zeros_like(w)
        losses = []
        w_hist = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0

            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                epoch_loss += loss
                batches += 1

                Eg2 = gamma * Eg2 + (1 - gamma) * (grad ** 2)

                w = w - (lr / (np.sqrt(Eg2) + eps)) * grad

            losses.append(epoch_loss / batches)
            w_hist.append(w.copy())
        return w, losses, w_hist

    @staticmethod
    def adadelta(model, w0, dataloader=None, epochs=100, gamma=0.9, eps=1e-8):
        """AdaDelta (FB and MB)"""
        w = w0.copy()
        Eg2 = np.zeros_like(w)
        Edw2 = np.zeros_like(w)
        losses = []
        w_hist = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0

            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                epoch_loss += loss
                batches += 1

                # Moving average of squared gradients
                Eg2 = gamma * Eg2 + (1 - gamma) * (grad ** 2)

                # Compute update magnitude
                RMS_dw = np.sqrt(Edw2 + eps)
                RMS_g = np.sqrt(Eg2 + eps)
                dw = - (RMS_dw / RMS_g) * grad

                # Moving average of squared updates
                Edw2 = gamma * Edw2 + (1 - gamma) * (dw ** 2)

                # Update weights
                w = w + dw

            losses.append(epoch_loss / batches)
            w_hist.append(w.copy())
        return w, losses, w_hist

    @staticmethod
    def adam(model, w0, dataloader=None, epochs=100, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam"""
        w = w0.copy()
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        t = 0
        losses = []
        w_hist = []

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
            w_hist.append(w.copy())
        return w, losses, w_hist