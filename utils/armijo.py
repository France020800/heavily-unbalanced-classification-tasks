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