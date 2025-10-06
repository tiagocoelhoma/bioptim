import os
import pickle
import numpy as np
import biorbd as brbd
import pandas as pd
from matplotlib import pyplot as plt

from sandbox.custom_package.fes_dynamics import calculate_friction_torque
from sandbox.custom_package.fes_plotting import get_data, organize_data


def plot_power(dump_path, model, file, friction_coef_mu,friction_coef_b, set_gravity_to_zero):

    path = dump_path + f'{file}.pkl'

    n_q = model.nbDof()
    n_mus = model.nbMuscles()
    muscle_names = tuple(s.to_string() for s in model.muscleNames())
    data = get_data(path)
    print(f"{path}, iterations:{data["n_iterations"]}, time:{round(data["real_time"], 2)}")

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

    file_name = dump_path + f'eval_pwr_{file}.pkl'
    data = get_data(file_name)
    q_sol_torque, qdot_sol_torque = data["q"][offset:], data["qdot"][offset:]
    tau_control = data["tau_joints"][offset:] if data["tau_joints"] else None
    pwr2, ddq_gear, pwr, dq_gear, tau_gear, q_gear = [], [], [], [], [], []
    dq_gear_torque, q_gear_torque = [], []
    mult_factor = len(q_data_aux[0]) // len(tau_control)
    aux = np.array([[float(tau[0, 0])] * mult_factor for tau in tau_control]).flatten()
    tau_gear.append(aux)
    q_data_aux_torque = organize_data(q_sol_torque, n_q)
    dq_data_aux_torque = organize_data(qdot_sol_torque, n_q)

    if set_gravity_to_zero:
        model.setGravity(np.array([0, 0, 0]))

    jac = []
    for t in range(len(f_data_aux[0])):
        _f, _q, _dq = [], [], []
        _q_torque, _dq_torque = [], []
        for m in range(n_mus):
            _f.append(float(f_data_aux[m][t]))
        for q in range(n_q):
            _q.append(float(q_data_aux[q][t]))
            _dq.append(float(dq_data_aux[q][t]))
            _q_torque.append(float(q_data_aux_torque[q][t]))
            _dq_torque.append(float(dq_data_aux_torque[q][t]))

        f, q, dq = np.array(_f), np.array(_q), np.array(_dq)
        q_torque, dq_torque = np.array(_q_torque), np.array(_dq_torque)
        muscles_tau = model.muscularJointTorque(f, q, dq).to_array()

        jac.append(model.musclesLengthJacobian(q).to_array())

        friction_torque = calculate_friction_torque(dq[0], mu=friction_coef_mu, B=friction_coef_b)
        muscles_tau[0] = muscles_tau[0] + friction_torque

        ddq = model.ForwardDynamicsConstraintsDirect(q, dq, muscles_tau)
        ddq_gear.append(float(ddq.to_array()[0]))
        q_gear.append(float(q[0]))
        dq_gear.append(float(dq[0]))

        # just to compare the two examples
        q_gear_torque.append(float(q_torque[0]))
        dq_gear_torque.append(float(dq_torque[0]))

        pwr2.append(friction_coef_b * np.pow(dq[0], 2))

    pwr = tau_gear[0] * dq_gear[:-1]
    # --------------------------------- First Figure ---------------------------------
    # fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    # ------------------------------------ plot 1 ------------------------------------
    # axes[0].plot(pwr, label="Power [W] (torque * qdot)", color="tab:green")
    # axes[0].plot(pwr2, label=f"B * qdot² (B = {friction_coef_b})", color="tab:blue")
    # axes[0].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    #
    # # ------------------------------------ plot 2 ------------------------------------
    # axes[1].plot(dq_gear, label=f"dq [rad/s]", color="tab:green")       # plot velocity
    # axes[1].plot(dq_gear_torque, label=f"dq_torque [rad/s]")  # plot velocity
    # axes[1].plot(np.diff(q_gear) / (t_array[6] - t_array[5]), "b--", label=f"dq [rad/s]")    # manual calc. of velocity
    # axes[1].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    # axes[1].set_ylabel('dq [rad/s]', color='green')
    #
    # ax2 = axes[1].twinx()
    # ax2.plot(ddq_gear, label=f"dqq [rad/s²]", color="tab:red")                              # plot acceleration
    # ax2.plot(np.diff(dq_gear) / (t_array[6] - t_array[5]), "b--", label=f"dqq [rad/s²]")    # manual calc. of acc.
    # ax2.legend(loc='upper right', fontsize='small', frameon=True, ncol=1)
    # ax2.set_ylabel('dqq [rad/s²]', color='red')
    #
    # # ------------------------------------ plot 3 ------------------------------------
    # axes[2].plot(tau_gear[0], label="torque [Nm]", color="tab:blue")
    # axes[2].legend(loc='upper left', fontsize='small', frameon=True, ncol=1)
    # ax3 = axes[2].twinx()
    # ax3.plot(ddq_gear, label=f"dqq [rad/s²]", color="tab:red")
    # ax3.legend(loc='upper right', fontsize='small', frameon=True, ncol=1)
    # plt.tight_layout()
    # plt.show()

    # --------------------------------- Second Figure ---------------------------------
    fig, axes = plt.subplots(subplot_kw={'polar': True}, figsize=(6, 6))
    # ------------------------------------ plot 1 ------------------------------------
    q_size = len(q_data_aux[0])
    pulse_width = np.array(pw_sol)
    node_count = pulse_width.shape[0]
    repeat_factor = q_size // node_count
    pw_squeezed = np.squeeze(pulse_width, axis=2)
    pw_stretched = np.repeat(pw_squeezed, repeat_factor, axis=0)
    pw_stretched = pw_stretched[:q_size, :]



    radial_offsets = np.linspace(10.5, 20.5, 8)
    # fig, axes = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(16, 6))
    q_radians = q_data_aux[0] % (2 * np.pi) * -1
    markers = {'rect_fem_r': 'o', 'bflh140_r': 's', 'vasmed_r': '^', 'rect_fem_l': 'D', 'bflh140_l': 'v', 'vasmed_l': 'P'}
    colors = {'rect_fem_r': 'tab:blue', 'bflh140_r': 'tab:orange', 'vasmed_r': 'tab:green', 'rect_fem_l': 'tab:red', 'bflh140_l': 'tab:purple',
               'vasmed_l': 'tab:brown'}
    offsets = {'rect_fem_r': radial_offsets[0],
               'bflh140_r': radial_offsets[1],
               'vasmed_r': radial_offsets[2],
               'rect_fem_l': radial_offsets[3],
               'bflh140_l': radial_offsets[4],
               'vasmed_l': radial_offsets[5]}
    # markers = ['o', 's', '^', 'D', 'v', 'P']
    # markers = {['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>'}  # as many as needed
    for idx in range(n_mus):
        # Usa diretamente os dados estendidos por músculo
        pw_data_stretched = pw_stretched[:, idx]
        # Define os pontos em que há ativação
        activated_points = np.where(pw_data_stretched[:-5] > 300e-6)[0]
        if len(activated_points) > 0:
            axes.scatter(q_radians[activated_points],
                         np.full_like(activated_points, offsets[muscle_names[idx]]),
                         marker=markers[muscle_names[idx]],
                         color=colors[muscle_names[idx]],
                         label=f"{muscle_names[idx]}")
    axes.set_xticks(np.linspace(0, 2 * np.pi, 36, endpoint=False))
    axes.set_title("b) stimulation Pattern (only quadriceps)", fontsize=14)
    # axes[0].set_xticklabels(['0°', '90°', '180°', '270°'])
    axes.set_yticks([])  # Remove radial ticks
    axes.set_yticklabels([])  # Remove radial tick labels
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 0), fontsize='small', frameon=True, ncol=1)
    axes.legend(loc='lower center', fontsize='medium', frameon=False, ncol=6, bbox_to_anchor=(0.50, -0.15))
    # axes.legend(loc='center', fontsize='medium', frameon=False, ncol=1)
    plt.savefig('Param_iden_cycling_Brazil/opt_sol_results/images/stim_pattern_only.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    q_radians = q_data_aux[0] * -1
    qdot_radians = dq_data_aux[0]

    fig, axes = plt.subplots(subplot_kw={'polar': True}, figsize=(6, 6))
    axes.plot(q_radians, qdot_radians, label="Velocity [rad/s]", color="tab:blue")
    axes.set_title("b) crank Angular Velocity", fontsize=14)
    axes.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes.tick_params(labelsize=10)
    axes.grid(True, linestyle='--', linewidth=0.5)
    axes.legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    axes.plot(q_radians[0], qdot_radians[0], 'ro')  # Red dot at start
    axes.text(q_radians[0], qdot_radians[0] + 0.1, 'Start', color='red', fontsize=10, ha='left')
    #
    axes.plot(q_radians[-1], qdot_radians[-1], 'ro')  # Red dot at start
    axes.text(q_radians[-1], qdot_radians[-1] + 0.1, 'End', color='red', fontsize=10, ha='right')
    #
    # print mean values
    mean_vel = np.mean(qdot_radians)
    std_rpm = np.std(qdot_radians)
    print(f'Mean velocity: {mean_vel:.2f} ± {std_rpm:.2f} rad/s')
    mean_rpm = np.mean(qdot_radians) * (60 / (2 * np.pi))
    std_rpm = np.std(qdot_radians) * (60 / (2 * np.pi))
    print(f'Mean velocity: {mean_rpm:.2f} ± {std_rpm:.2f} RPM')
    # plt.legend(fontsize='small', frameon=False, ncol=4)
    plt.show()

    pwr_radians = np.array(pwr)
    q_radians = q_radians[:-1]
    fig, axes = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 6))
    axes.plot(q_radians, pwr_radians, label="Power [W]", color="tab:green")
    axes.fill_between(q_radians, 0, pwr_radians, alpha=0.1, color="tab:green")
    # pwr_radians = np.array(pwr2) * -1
    # axes[2].plot(q_radians, pwr_radians[:-1], label="Power [W]", color="tab:blue")
    axes.set_title("c) power Output", fontsize=14)
    axes.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes.grid(True, linestyle='--', linewidth=0.5)
    axes.legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    axes.plot(q_radians[0], pwr_radians[0], 'ro')  # Red dot at start
    axes.text(q_radians[0], pwr_radians[0] + 0.1, 'Start', color='red', fontsize=10, ha='left')

    axes.plot(q_radians[-1], pwr_radians[-1], 'ro')  # Red dot at start
    axes.text(q_radians[-1], pwr_radians[-1] + 0.1, 'End', color='red', fontsize=10, ha='right')
    # plt.savefig('opt_sol_results/images/stim_pattern.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # print mean values
    mean_pwr = np.mean(pwr_radians);
    std_pwr = np.std(pwr_radians)
    print(f'Mean power: {mean_pwr:.2f} ± {std_pwr:.2f} W')


def plot_data():
    # 1. Carregar o arquivo .txt
    df = pd.read_csv('Param_iden_cycling_Brazil/data_exp/rotor_bike/20250703_151705_pedalling_2.txt', sep='\t')  # ajuste o caminho do arquivo aqui

    # Aplicar média móvel com janela de 3
    df['Power_smoothed'] = df['Power(W)'].rolling(window=3, center=True).mean()
    # 2. Aplicar filtro de média móvel na coluna de potência
    df['Power_smoothed'] = df['Power(W)'].rolling(window=3, center=True).mean()
    # 3. Converter ângulo para radianos
    df['Angle(rad)'] = np.deg2rad(df['Angle(deg)'])
    # 4. Preparar os dados
    q_radians = df['Angle(rad)'][1600:1700]
    pwr_radians = df['Power(W)'][1600:1700]
    pwr_smoothed = df['Power_smoothed'][1600:1700]
    # 5. Plotar gráfico polar
    fig, axes = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 6))
    # Potência original (tracejado verde)
    axes.plot(q_radians, pwr_radians, label="Power [W]", color="tab:green", linestyle='dotted')
    # axes.fill_between(q_radians, 0, pwr_radians, alpha=0.05, color="tab:green")
    # Potência suavizada (azul)
    # axes.plot(q_radians, pwr_smoothed, label="Smoothed Power", color="tab:blue")
    # axes.fill_between(q_radians, 0, pwr_smoothed, alpha=0.1, color="tab:blue")
    # Personalizações do gráfico
    axes.set_title("c) Power Output", fontsize=14)
    axes.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes.grid(True, linestyle='--', linewidth=0.5)
    axes.legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    # Marcar início e fim
    axes.plot(q_radians.iloc[0], pwr_radians.iloc[0], 'ro')
    axes.text(q_radians.iloc[0], pwr_radians.iloc[0] + 0.1, 'Start', color='red', fontsize=10, ha='left')

    axes.plot(q_radians.iloc[-1], pwr_radians.iloc[-1], 'ro')
    axes.text(q_radians.iloc[-1], pwr_radians.iloc[-1] + 0.1, 'End', color='red', fontsize=10, ha='right')

    plt.show()


if __name__ == "__main__":

    print('test')
    dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/fes_leg_pedalling/'
    model_path = "models/legcyc.bioMod"
    # file = 'friction_3.05s_105n_41.92rpm_20.0W_fric_0.2_1.0378'
    # filename = f'friction_{round(duration, 2)}s_{n_nodes}n_{target_velocity_rpm}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}_opt_crank'
    # file = 'friction_2.09s_72n_41.92rpm_25.0W_fric_0.2_1.2973' # _backing_up_4th_order

    file = 'friction_2.18s_75n_41.92rpm_20.0W_fric_0.01_1.0378'  # only quadriceps



    friction_coef_mu = 0.2
    friction_coef_b = float(file[-6:])
    model = brbd.Model(model_path)

    # plot_data()
    plot_power(dump_path, model, file, friction_coef_mu, friction_coef_b, set_gravity_to_zero=False) # plot power
