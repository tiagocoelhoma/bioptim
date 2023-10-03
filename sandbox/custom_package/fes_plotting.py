import numpy as np
from casadi import MX

def fes_plot_callback(x: MX, model) -> MX:
    result = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        q = x[0, i, np.newaxis]
        dq = x[1, i, np.newaxis]
        f = x[3, i, np.newaxis]
        result[:, i] = model.muscularJointTorque(f, q, dq).to_array()
    return result
