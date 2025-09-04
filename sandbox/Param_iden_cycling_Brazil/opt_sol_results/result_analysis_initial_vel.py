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

def main(model_path, activity):

    model = brbd.Model(model_path)
    with open(activity, "rb") as file:
        data = pickle.load(file)

    q_sol, qdot_sol, tau_sol = data["q"], data["qdot"], data["tau_joints"]

    n_q = model.nbQ()
    q_data_aux, dq_data_aux = organize_data(q_sol, n_q), organize_data(qdot_sol, n_q)
    joint_0_torques = np.array([[t[0]] * 5 for t in tau_sol]).flatten()

    # joint_0_torques = joint_0_torques + joint_0_torques[-1]np.array([[t[0]] * 5 for t in tau_sol]).flatten()
    dq_0 = dq_data_aux[0]
    tau_0 = joint_0_torques.flatten()
    pwr = dq_0[:-1] * tau_0
    pwr_friction = dq_0[:-1] ** 2 * 0.78
    # pwr = dq_0[:-1] * (tau_0 + 0.78)
    # pwr = np.abs(dq_0[:-1]) * np.abs(tau_0)

    # potência_total = torque_aplicado * velocidade
    # potência_fricção = torque_fricção * velocidade
    # potência_cinética = potência_total - potência_fricção  # vai para acelerar o sistema

    # Resetar o estilo antes de plotar
    plt.style.use('default')
    # Criar figura e eixos limpos
    fig, ax = plt.subplots()
    # Plotar os dados
    ax.plot(dq_0[:-1], label="velocity")
    ax.plot(tau_0, label="torque")
    ax.plot(pwr, label="power")
    ax.plot(pwr_friction, label="power friction")
    # ax.plot(pwr - pwr_friction, label="power kinetic")
    # Ajustes visuais
    ax.set_facecolor("white")  # Fundo branco para o gráfico
    fig.patch.set_facecolor("white")  # Fundo branco para a figura
    # ax.set_title("Active forces")
    # ax.set_ylabel("Active forces coeff")
    # Legenda
    fig.legend(loc='upper center', fontsize='medium', frameon=False, ncol=4)
    # Mostrar
    plt.show()


if __name__ == "__main__":
    model = "../models/leg_model_cyclic_muscle_inversed_root_3.bioMod"
    # activity = 'kinematics/dyn_0.7_50_42.92rpm_10W'
    activity = 'kinematics/dyn_0.25_15_42.92rpm_10W'
    activity = 'kinematics/dyn_2.5_30_42.92rpm_15W'
    main(model, activity)


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

