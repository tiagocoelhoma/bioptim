import numpy as np
import biorbd as brbd
import pickle

from biorbd_casadi import GeneralizedCoordinates, GeneralizedVelocity
from casadi import MX
from custom_package.fes_dynamics import calculate_friction_torque
import matplotlib.pyplot as plt


def get_data(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data

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

if __name__ == "__main__":


    model_path = "../../models/leg_model_cyclic_muscle_inversed_root_3.bioMod"
    file = 'friction_2.18s_75n_41.92rpm_25.0W_fric_0.05_1.2973'  # only quadriceps
    path = f'../opt_sol_results/fes_leg_pedalling/{file}.pkl'

    friction_coef_mu = 0.05
    friction_coef_b = float(file[-6:])
    model = brbd.Model(model_path)
    model.setGravity(np.array([0, 0, 0]))
    n_q = model.nbDof()
    n_mus = model.nbMuscles()
    muscle_names = tuple(s.to_string() for s in model.muscleNames())
    data = get_data(path)

    offset = 4
    q_sol, qdot_sol = data["q"][offset:], data["qdot"][offset:]
    f_sol, pw_sol, time_sol = data["f"][offset:], data["pw"][offset:], data["time"][offset:]

    time = data["time"]
    t_array = np.concatenate(
        [np.array(t.full()).flatten()[:-1] if i < len(time) - 1 else np.array(t.full()).flatten() for i, t in
         enumerate(time)])

    q_data_aux = organize_data(q_sol, n_q)
    dq_data_aux = organize_data(qdot_sol, n_q)
    f_data_aux = organize_data(f_sol, n_mus)
    pw_data_aux = organize_data(pw_sol, n_mus)

    ddq_gear, dq_gear, q_gear, tau, tau_, tau_knee, tau_inv, jac = [], [], [], [], [], [], [], []
    for t in range(len(f_data_aux[0])):
        _f, _q, _dq = [], [], []
        _q_torque, _dq_torque = [], []
        for m in range(n_mus):
            _f.append(float(f_data_aux[m][t]))
        for q in range(n_q):
            _q.append(float(q_data_aux[q][t]))
            _dq.append(float(dq_data_aux[q][t]))

        f, q, dq = np.array(_f), np.array(_q), np.array(_dq)
        q_torque, dq_torque = np.array(_q_torque), np.array(_dq_torque)

        muscles_tau = model.muscularJointTorque(f, q, dq).to_array()
        friction_torque = calculate_friction_torque(dq[0], mu=friction_coef_mu, B=friction_coef_b)
        muscles_tau[0] = muscles_tau[0] + friction_torque
        ddq = model.ForwardDynamicsConstraintsDirect(q, dq, muscles_tau)
        torque = model.InverseDynamics(q, dq, ddq).to_array() # nao aceita pois inverseDynamics with contact nao foi impolementada

        tau_.append(muscles_tau[2])
        tau_inv.append(torque[2])

        # try to get the jacobian?
        J = model.musclesLengthJacobian(q).to_array()
        torque_from_muscles =  J.T @ f
        tau.append(torque_from_muscles[0])
        tau_knee.append(torque_from_muscles[2])

        print("Jac crank column (DOF 0):", J[:, 0])
        print("Torque at crank:", J.T @ f)


    #['Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ']
    plt.plot(tau, label='at crank with jacobian')
    plt.plot(tau_, label='at knee')
    plt.plot(tau_inv, label='at knee with inverse')
    plt.plot(tau_knee, '--', label='at knee with jacobian')

    plt.legend()
    plt.show()
    print('end...')

#        jac.append(model.musclesLengthJacobian(q).to_array())
# q_biorbd = GeneralizedCoordinates(q)
# J = model.markersJacobian(q)