import numpy as np

def armijo_line_search(model, w, direction, grad, loss_w, alpha_init=1.0, c1=1e-4, tau=0.5):
    alpha = alpha_init
    dir_dot_grad = np.dot(grad.T, direction)

    while True:
        w_new = w - alpha * direction
        loss_new = model.compute_loss(w_new)

        if loss_new <= loss_w - c1 * alpha * dir_dot_grad:
            break

        alpha *= tau

        if alpha < 1e-8:  # Safety break to avoid infinite loops
            break

    return alpha


def wolfe_line_search(model, w, direction, grad, loss_w, alpha_init=1.0, gamma=1e-4, sigma=0.9):
    """
    Ricerca di linea basata sulle Condizioni di Wolfe deboli.
    gamma corrisponde al c1 di Armijo, sigma controlla la curvatura.
    """
    alpha = alpha_init
    dir_dot_grad = np.dot(grad.T, direction)

    if dir_dot_grad >= 0:
        return 1e-4

    while True:
        w_new = w + alpha * direction
        loss_new = model.compute_loss(w_new)

        # Armijo
        if loss_new > loss_w + gamma * alpha * dir_dot_grad:
            alpha *= 0.5
            if alpha < 1e-8: break
            continue

        grad_new = model.compute_gradient(w_new)
        dir_dot_grad_new = np.dot(grad_new.T, direction)

        # 2. Condizione di Curvatura (Wolfe debole)
        if dir_dot_grad_new < sigma * dir_dot_grad:
            alpha *= 2.0
            if alpha > 1e5: break  # Safety break
            continue

        break

    return alpha