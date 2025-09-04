import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import biorbd as brbd
from biorbd import GeneralizedTorque
from matplotlib.pyplot import legend


matplotlib.use('Qt5Agg')  # Use 'Qt5Agg' for PyQt5 compatibility, 'Qt6Agg' if using PyQt6


def organize_data(data_array, n):
    data_aux = []
    for data_index in range(n):
        data_app = []
        for i, node_data in enumerate(data_array):
            _data = node_data[data_index]
            data_app.append(_data[:-1])
        data_app.append(_data)
        concatenated_samples = np.concatenate(data_app)
        data_aux.append(concatenated_samples)
    return data_aux

def create_vec(vector):
    vec = []
    for item in vector:
        vec += item  # Concatenate lists or add items to the list (works for lists))
    return vec


if __name__ == "__main__":

    model = brbd.Model("../models/leg_model_cyclic_muscle_inversed_root_3.bioMod")
    # activity = 'res_constant_n25_15.0W_41.92_f35_m6'
    activity = 'fes_leg_pedalling/res_constant_n35_15.0W_41.92_f35_m6.pkl'
    activity = 'fes_leg_pedalling/res_constant_n35_10.0W_41.92_f35_m6.pkl'
    activity = 'kinematics/dyn_0.7_50_42.92rpm_10W'
    activity = 'kinematics/dyn_0.5_30_42.92rpm_15W'
    # activity = 'fes_leg_pedalling/friction_n35_15.0W_41.92_f35_m6.pkl'

    with open(activity, "rb") as file:
        data = pickle.load(file)

    q_sol, qdot_sol, f_sol = data["q"], data["qdot"], data["f"]

    n_q, n_mus = model.nbQ(), model.nbMuscles()
    q_data_aux, dq_data_aux, f_data_aux = organize_data(q_sol, n_q), organize_data(qdot_sol, n_q), organize_data(f_sol, n_mus)

    contact_force, joint_torque, torque, pwr, dq_test = [], [], [], [], []
    for t in range(len(f_data_aux[0])):
        _f, _q, _dq = [], [], []
        for m in range(n_mus):
            _f.append(float(f_data_aux[m][t]))
        for q in range(n_q):
            _q.append(float(q_data_aux[q][t]))
            _dq.append(float(dq_data_aux[q][t]))

        f, q, dq = np.array(_f), np.array(_q), np.array(_dq)
        muscles_tau = model.muscularJointTorque(f, q, dq).to_array()
        muscles_tau[0] = muscles_tau[0] + data['res_tau_value']

        ddq = model.ForwardDynamicsConstraintsDirect(q, dq, GeneralizedTorque(muscles_tau))
        # ddq = model.ForwardDynamicsConstraintsDirect(q, dq, muscles_tau)
        tau = model.InverseDynamics(q, dq, ddq).to_array()

        joint_torque.append(muscles_tau)
        torque.append(tau)
        # pwr.append(tau[0] * dq[0])
        # pwr.append((tau[0] - float(torque[0][0])) * dq[0])
        dq_test.append(dq)
        contact_force.append(model.ContactForcesFromForwardDynamicsConstraintsDirect(q, dq, muscles_tau).to_array())

    # joint_0_torques = [-t[0] for t in torque]
    joint_0_torques = np.array([-(t[0] - float(torque[0][0])) for t in torque])
    plt.plot(joint_0_torques, label=f"torque")
    dq_0 = np.array([float(t[0]) for t in dq_test])
    # data['res_tau_value'] = -3.154
    pwr = dq_0 * -joint_0_torques
    plt.plot(-dq_0, label=f"velocity")
    plt.plot(pwr, label=f"power")
    plt.legend(loc='upper center', fontsize='medium', frameon=False, ncol=3)
    plt.show()


    # tau = []
    # nb_nodes = len(q_sol)
    # mjt_nodes = []
    # for i in range(nb_nodes):
    #     f = f_sol[i]
    #     q = q_sol[i]
    #     dq = qdot_sol[i]
    #
    #     mjt = np.zeros((n_q, q.shape[1]))
    #     for j in range(q.shape[1]):
    #         mjt[:, j] = model.muscularJointTorque(f[:, j].flatten(), q[:, j].flatten(), dq[:, j].flatten()).to_array()
    #     mjt_nodes.append(mjt)
    # tau.append(mjt_nodes)
    # plt.plot(mjt_nodes)
    # plt.show()

