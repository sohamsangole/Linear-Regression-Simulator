def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):
    dJwb_dw = 0
    dJwb_db = 0
    m = x.shape[0]
    for i in range(m):
        fwb = w * x[i] + b
        dJwb_dw = dJwb_dw + (fwb - y[i]) * x[i]
        dJwb_db = dJwb_db + (fwb - y[i])
    dJwb_dw = dJwb_dw / m
    dJwb_db = dJwb_db / m

    return dJwb_dw, dJwb_db


def gradient_descent(x, y, w_input, b_input, alpha, num_iters):
    J = []
    p = []
    w = w_input
    b = b_input

    for i in range(num_iters):
        # Calculating derivative terms
        dJwb_dw, dJwb_db = compute_gradient(x, y, w, b)

        # Updaing w and b
        w = w - alpha * dJwb_dw
        b = b - alpha * dJwb_db

        if i < 1000:
            J.append(compute_cost(x, y, w, b))
            p.append([w, b])

    return w, b, J, p
