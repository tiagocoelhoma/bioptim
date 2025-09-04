"""

"""
import numpy as np
import biorbd as brbd
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    InitialGuessList,
    PlotType, PhaseDynamics, Node, VariableScalingList, InterpolationType
)
from biorbd import GeneralizedTorque
from casadi import MX

from custom_package.fes_configure_probem import FesDynamicsFcn
from custom_package.fes_objectives import FesObjective
import pickle, os
import matplotlib.pyplot as plt


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

def get_data(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data

def unpack_sol(list):
    out = []
    for item in list:
        out.append([float(it[0]) for it in item])
    return out

if __name__ == "__main__":

    file = 'friction_2.03s_70n_41.92rpm_10.0W_fric_0.0_0.5189'
    file = 'friction_2.32s_80n_41.92rpm_10.0W_fric_0.0_0.353'
    file = 'friction_2.32s_80n_41.92rpm_10.0W_fric_0.0_0.0706'
    file = 'friction_2.32s_80n_41.92rpm_10.0W_fric_0.0_0.0353'
    file = 'friction_0.58s_20n_41.92rpm_6.0W_fric_0.0_0.3114'
    file = 'friction_2.32s_80n_41.92rpm_10.0W_fric_0.0_0.5189'
    file = 'friction_1.02s_35n_41.92rpm_15.0W_fric_0.0_0.7784'

    model = brbd.Model(f"../models/leg_model_cyclic_muscle_inversed_root_3.bioMod")
    n_mus = model.nbMuscles()
    n_q = model.nbQ()
    muscle_names = tuple(s.to_string() for s in model.muscleNames())
    file_name = f"fes_leg_pedalling/{file}.pkl"
    data = get_data(file_name)
    print(f"{file_name}, iterations:{data["n_iterations"]}, time:{round(data["real_time"], 2)}")
    q_sol, qdot_sol = data["q"], data["qdot"]
    f_sol, pw_sol, time_sol = data["f"], data["pw"], data["time"]

    time = data["time"]
    t_array = np.concatenate(
        [np.array(t.full()).flatten()[:-1] if i < len(time) - 1 else np.array(t.full()).flatten() for i, t in
         enumerate(time)])

    q_data_aux = organize_data(q_sol, n_q)
    dq_data_aux =  organize_data(qdot_sol, n_q)
    f_data_aux = organize_data(f_sol, n_mus)
    pw_data_aux = organize_data(pw_sol, n_mus)

    file_name = f"fes_leg_pedalling/eval_pwr_{file}.pkl"
    data = get_data(file_name)
    tau_control = data["tau_joints"] if data["tau_joints"] else None
    tau_gear = []
    mult_factor = len(q_data_aux[0]) // len(tau_control)
    aux = np.array([[float(tau[0, 0])] * mult_factor for tau in tau_control]).flatten()
    tau_gear.append(aux)


    pwr2, torque, pwr, dq_test = [], [], [], []

    for t in range(len(f_data_aux[0])):
        _f, _q, _dq = [], [], []
        for m in range(n_mus):
            _f.append(float(f_data_aux[m][t]))
        for q in range(n_q):
            _q.append(float(q_data_aux[q][t]))
            _dq.append(float(dq_data_aux[q][t]))

        f, q, dq = np.array(_f), np.array(_q), np.array(_dq)
        # model.setGravity(np.array([0, 0, 0]))
        muscles_tau = model.muscularJointTorque(f, q, dq)
        ddq = model.ForwardDynamicsConstraintsDirect(q, dq, muscles_tau)
        tau = model.InverseDynamics(q, dq, ddq).to_array()
        # tau[0] = tau[0] + 0.0706
        pwr.append(tau[0] * dq[0])
        pwr2.append(0.0706 * np.pow(dq[0], 2))
        torque.append(ddq.to_array())
        dq_test.append(float(dq[0]))



    fig, axes = plt.subplots(1, 3)
    # joint_0_torques = np.array([-(t[0] - float(torque[0][0])) for t in torque])
    joint_0_torques = np.array([t[0] for t in torque])
    # axes[0].plot(np.diff(dq_test) / (t_array[6] - t_array[5]), label="Power [W] (torque * qdot)", color="tab:blue")
    # axes[0].plot(joint_0_torques, label="Power [W] (torque * qdot)", color="tab:green")
    axes[0].plot(tau_gear[0] * dq_test[:-1], label="Power [W] (torque * qdot)", color="tab:green")
    axes[0].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    axes[1].plot(pwr2, label="Power [W] (B * qdot^2)", color="tab:blue")
    axes[1].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    axes[2].plot(tau_gear[0], label="torque [Nm]", color="tab:blue")
    axes[2].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    # plt.show()

    # model.setGravity(np.array([-9.8066499999999994, 0, 0]))
    # model.setGravity(np.array([0, 0, 0]))

    fig, axes = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(16, 6))
    axes[0].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    # axes[0].set_title("Stimulation Pattern", fontsize=14)
    # axes[0].set_xticklabels(['0°', '90°', '180°', '270°'])
    # axes[0].set_yticks([])  # Remove radial ticks
    # axes[0].set_yticklabels([])  # Remove radial tick labels
    # # plt.legend(loc='upper left', bbox_to_anchor=(0, 0), fontsize='small', frameon=True, ncol=1)
    # axes[0].legend(loc='center', fontsize='medium', frameon=False, ncol=1)
    #
    q_radians = q_data_aux[0]
    qdot_radians = dq_data_aux[0]
    axes[1].plot(q_radians, qdot_radians, label="Velocity [rad/s]", color="tab:blue")
    axes[1].set_title("Crank Angular Velocity", fontsize=14)
    axes[1].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes[1].set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes[1].tick_params(labelsize=10)
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    axes[1].plot(q_radians[0], qdot_radians[0], 'ro')  # Red dot at start
    axes[1].text(q_radians[0], qdot_radians[0] + 0.5, 'Start', color='red', fontsize=10, ha='left')
    #
    axes[1].plot(q_radians[-1], qdot_radians[-1], 'ro')  # Red dot at start
    axes[1].text(q_radians[-1], qdot_radians[-1] + 0.5, 'End', color='red', fontsize=10, ha='right')
    #
    # print mean values
    mean_rpm = np.mean(qdot_radians) * (60 / (2 * np.pi))
    std_rpm = np.std(qdot_radians) * (60 / (2 * np.pi))
    print(f'Mean velocity: {mean_rpm:.2f} ± {std_rpm:.2f} RPM')
    # plt.legend(fontsize='small', frameon=False, ncol=4)

    pwr_radians = pwr
    axes[2].plot(q_radians, pwr_radians, label="Power [W]", color="tab:green")
    # axes[2].plot(q_radians, pwr_radians_res, label="Res Power", color="tab:blue")
    axes[2].fill_between(q_radians, 0, pwr_radians, alpha=0.1, color="tab:green")
    # axes[2].fill_between(q_radians, 0, pwr_radians_res, alpha=0.1, color="tab:blue")
    axes[2].set_title("Power Output", fontsize=14)
    axes[2].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes[2].set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes[2].grid(True, linestyle='--', linewidth=0.5)
    axes[2].legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    axes[2].plot(q_radians[0], pwr_radians[0], 'ro')  # Red dot at start
    axes[2].text(q_radians[0], pwr_radians[0] + 0.5, 'Start', color='red', fontsize=10, ha='left')

    axes[2].plot(q_radians[-1], pwr_radians[-1], 'ro')  # Red dot at start
    axes[2].text(q_radians[-1], pwr_radians[-1] + 0.5, 'End', color='red', fontsize=10, ha='right')

    # Smart zoom on power
    # pwr_min = np.min(pwr_radians)
    # pwr_max = np.max(pwr_radians)
    # margin = 0.1 * (pwr_max - pwr_min)
    # axes[2].set_ylim(pwr_min - margin, pwr_max + margin)
    # Optional: add line at 0
    # axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)

    # print mean values
    mean_pwr = np.mean(pwr_radians)
    std_pwr = np.std(pwr_radians)
    print(f'Mean power: {mean_pwr:.2f} ± {std_pwr:.2f} W')

    # plt.savefig('images/stim_pattern.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_power(model_path, activity):
    model = brbd.Model(model_path)
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
        # muscles_tau[0] = muscles_tau[0] + data['res_tau_value']

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
    pwr = dq_0 * -joint_0_torques
    plt.plot(-dq_0, label=f"velocity")
    plt.plot(pwr, label=f"power")
    plt.legend(loc='upper center', fontsize='medium', frameon=False, ncol=3)
    plt.show()
