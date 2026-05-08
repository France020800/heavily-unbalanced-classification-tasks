import numpy as np
from utils.armijo import armijo_line_search, wolfe_line_search


class Optimizers:
    @staticmethod
    def gradient_descent_armijo(model, w0, dataloader=None, epochs=100, **kwargs):
        """Gradient descent with Armijo Line Search (FB)"""
        w = w0.copy()
        epoch_losses = []
        batch_losses = []
        w_hist = []

        for _ in range(epochs):
            loss = model.compute_loss(w)
            grad = model.compute_gradient(w)

            # Save metrics
            epoch_losses.append(loss)
            batch_losses.append(loss)  # In FB, 1 epoch = 1 batch
            w_hist.append(w.copy())

            # Update weights
            alpha = armijo_line_search(model, w, grad, grad, loss)
            w = w - alpha * grad

        return w, epoch_losses, batch_losses, w_hist

    @staticmethod
    def cg_wolfe(model, w0, dataloader=None, epochs=100, **kwargs):
        """Conjugate Gradient Method with Wolfe line search (FB)"""
        w = w0.copy()
        epoch_losses = []
        batch_losses = []
        w_hist = []

        grad = model.compute_gradient(w)
        direction = -grad.copy()  # d_0 = -g_0

        for i in range(epochs):
            loss = model.compute_loss(w)

            # Save metrics
            epoch_losses.append(loss)
            batch_losses.append(loss)  # In FB, 1 epoch = 1 batch
            w_hist.append(w.copy())

            if np.linalg.norm(grad) <= 1e-6:
                break

            alpha = wolfe_line_search(model, w, direction, grad, loss)
            w = w + alpha * direction

            grad_prev = grad.copy()
            grad = model.compute_gradient(w)

            # Fletcher-Reeves beta update
            beta = (np.linalg.norm(grad) ** 2) / (np.linalg.norm(grad_prev) ** 2 + 1e-8)
            direction = -grad + beta * direction

        return w, epoch_losses, batch_losses, w_hist

    @staticmethod
    def sgd_polyak(model, w0, dataloader, epochs=100, **kwargs):
        """Stochastic gradient descent with Polyak stepsize (MB)"""
        w = w0.copy()
        epoch_losses = []
        batch_losses = []
        w_hist = []
        f_star = 0.0  # Lower bound for BCE loss

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0

            for indices in dataloader:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                # Save batch metric
                batch_losses.append(loss)
                epoch_loss += loss
                batches += 1

                grad_norm_sq = np.linalg.norm(grad) ** 2
                if grad_norm_sq > 1e-8:
                    alpha = (loss - f_star) / grad_norm_sq
                else:
                    alpha = 1e-4

                w = w - alpha * grad

            # Save epoch metrics
            epoch_losses.append(epoch_loss / batches)
            w_hist.append(w.copy())

        return w, epoch_losses, batch_losses, w_hist

    @staticmethod
    def adagrad(model, w0, dataloader=None, epochs=100, lr=0.01, eps=1e-8, **kwargs):
        """Adagrad (FB and MB)"""
        w = w0.copy()
        G = np.zeros_like(w)
        epoch_losses = []
        batch_losses = []
        w_hist = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0
            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                batch_losses.append(loss)
                epoch_loss += loss
                batches += 1

                G += grad ** 2
                w = w - (lr / (np.sqrt(G) + eps)) * grad

            epoch_losses.append(epoch_loss / batches)
            w_hist.append(w.copy())

        return w, epoch_losses, batch_losses, w_hist

    @staticmethod
    def rmsprop(model, w0, dataloader=None, epochs=100, lr=0.01, gamma=0.9, eps=1e-8, **kwargs):
        """RMSProp (FB and MB)"""
        w = w0.copy()
        Eg2 = np.zeros_like(w)
        epoch_losses = []
        batch_losses = []
        w_hist = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0
            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                batch_losses.append(loss)
                epoch_loss += loss
                batches += 1

                Eg2 = gamma * Eg2 + (1 - gamma) * (grad ** 2)
                w = w - (lr / (np.sqrt(Eg2) + eps)) * grad

            epoch_losses.append(epoch_loss / batches)
            w_hist.append(w.copy())

        return w, epoch_losses, batch_losses, w_hist

    @staticmethod
    def adadelta(model, w0, dataloader=None, epochs=100, gamma=0.9, eps=1e-8, **kwargs):
        """AdaDelta (FB and MB)"""
        w = w0.copy()
        Eg2 = np.zeros_like(w)
        Edw2 = np.zeros_like(w)
        epoch_losses = []
        batch_losses = []
        w_hist = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0
            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                batch_losses.append(loss)
                epoch_loss += loss
                batches += 1

                Eg2 = gamma * Eg2 + (1 - gamma) * (grad ** 2)
                RMS_dw = np.sqrt(Edw2 + eps)
                RMS_g = np.sqrt(Eg2 + eps)
                dw = - (RMS_dw / RMS_g) * grad
                Edw2 = gamma * Edw2 + (1 - gamma) * (dw ** 2)
                w = w + dw

            epoch_losses.append(epoch_loss / batches)
            w_hist.append(w.copy())

        return w, epoch_losses, batch_losses, w_hist

    @staticmethod
    def adam(model, w0, dataloader=None, epochs=100, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        """Adam (FB and MB)"""
        w = w0.copy()
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        t = 0
        epoch_losses = []
        batch_losses = []
        w_hist = []

        for _ in range(epochs):
            epoch_loss = 0
            batches = 0
            batch_list = dataloader if dataloader is not None else [None]

            for indices in batch_list:
                t += 1
                loss = model.compute_loss(w, indices)
                grad = model.compute_gradient(w, indices)

                batch_losses.append(loss)
                epoch_loss += loss
                batches += 1

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

            epoch_losses.append(epoch_loss / batches)
            w_hist.append(w.copy())

        return w, epoch_losses, batch_losses, w_hist