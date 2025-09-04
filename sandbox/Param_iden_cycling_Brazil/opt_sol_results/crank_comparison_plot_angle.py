import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
matplotlib.use('Qt5Agg')  # Use 'Qt5Agg' for PyQt5 compatibility, 'Qt6Agg' if using PyQt6
import biorbd as brbd


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

    activity = {
        # 'leg_model_cyclic_muscle_inversed_root_3': 'fw_rotation_0.7250000000000001_25_41.92_f35_m6_dist95cm',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'fw_rotation_3.0450000000000004_105_41.92_f35_m6',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'fw_rotation_1.0150000000000001_35_41.92_f35_m6',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'fw_rotation_2.4650000000000003_85_41.92_f35_m6',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'fw_rotation_3.0450000000000004_105_41.92_f35_m6',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'fw_rotation_15W_105_41.92_f35_m6',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'n105_6.0W_41.92_f35_m6',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'n105_10.0W_41.92_f35_m6',
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_n35_15.0W_42_f35_m6'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_n100_10.0W_42_f35_m6'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_0.73s_25n_42rpm_10.0W_fric_0.0_0.5169'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_1.45s_50n_42rpm_10.0W_fric_0.0_0.5169'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_3.05s_105n_42rpm_10.0W_fric_0.0_0.5169'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_0.87s_30n_41.92rpm_10.0W_fric_0.0_0.5189'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_1.02s_35n_41.92rpm_10.0W_fric_0.0_0.5189'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_0.58s_20n_41.92rpm_6.0W_fric_0.0_0.3114'
        # 'leg_model_cyclic_muscle_inversed_root_3': 'friction_2.03s_70n_41.92rpm_10.0W_fric_0.0_0.5189'
        'leg_model_cyclic_muscle_inversed_root_3': 'friction_2.32s_80n_41.92rpm_10.0W_fric_0.0_0.353'
    }

    f_data_aux, q_data_aux, dq_data_aux, pw_data_aux, tau_res_data_aux = [], [], [], [], []
    names_crank = ['convetional']
    for n, (model, file) in enumerate(activity.items()):
        num_model = brbd.Model(f"../models/{model}.bioMod")
        n_mus = num_model.nbMuscles()
        n_q = num_model.nbQ()
        muscle_names = tuple(s.to_string() for s in num_model.muscleNames())
        file_name = f"fes_leg_pedalling/{file}.pkl"
        data = get_data(file_name)
        print(f"{file_name}, iterations:{data["n_iterations"]}, time:{round(data["real_time"], 2)}")
        q_sol, qdot_sol = data["q"], data["qdot"]
        f_sol, pw_sol, time_sol = data["f"], data["pw"], data["time"]
        tau_res_sol = data["tau_resistive"] if data["tau_resistive"] else None

        time = data["time"]
        t_array = np.concatenate(
            [np.array(t.full()).flatten()[:-1] if i < len(time) - 1 else np.array(t.full()).flatten() for i, t in
             enumerate(time)])

        f_data_aux.append(organize_data(f_sol, n_mus))
        q_data_aux.append(organize_data(q_sol, n_q))
        dq_data_aux.append(organize_data(qdot_sol, n_q))
        pw_data_aux.append(np.array(pw_sol))
        tau_res_data_aux.append(np.array(tau_res_sol))

    activity = {'leg_model_cyclic_muscle_inversed_root_3': 'eval_pwr_res_constant_n35_10.0W_41.92_f35_m6'}
    tau_gear = []
    names_crank = ['convetional']
    for n, (model, file) in enumerate(activity.items()):
        num_model = brbd.Model(f"../models/{model}.bioMod")
        n_mus = num_model.nbMuscles()
        n_q = num_model.nbQ()
        muscle_names = tuple(s.to_string() for s in num_model.muscleNames())
        file_name = f"fes_leg_pedalling/{file}.pkl"
        data = get_data(file_name)
        print(f"{file_name}, iterations:{data["n_iterations"]}, time:{round(data["real_time"], 2)}")
        q_sol, qdot_sol = data["q"], data["qdot"]
        # f_sol, pw_sol, time_sol = data["f"], data["pw"], data["time"]
        tau_control = data["tau_joints"] if data["tau_joints"] else None

        time = data["time"]
        t_array = np.concatenate(
            [np.array(t.full()).flatten()[:-1] if i < len(time) - 1 else np.array(t.full()).flatten() for i, t in
             enumerate(time)])

        # tau_gear.append(organize_data(tau_control, n_q))
        # tau_gear.append(np.array(tau_control))
        aux = [float(tau[0, 0] * -1) for tau in tau_control]
        tau_gear.append(aux)

    q_data_aux[0][0] = q_data_aux[0][0] * -1
    dq_data_aux[0][0] = dq_data_aux[0][0] * -1


    fig, axes = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(16, 6))

    radial_offsets = np.linspace(10.5, 20.5, 8)
    q_radians = q_data_aux[0][0] % (2*np.pi) # * -1 cause pedal right in the biomechanical model has -pi offset
    q_size = len(q_radians)
    # Plot the activation for each index in pw_data_aux
    for idx in range(n_mus):
        pw_data = np.array(pw_data_aux[0][:, idx]).flatten()
        pw_size = len(pw_data)
        repeat_factor = q_size // pw_size
        pw_data_stretched = np.array([np.repeat(pw, repeat_factor)[:q_size] for pw in pw_data]).flatten()
        # Find the indices where pw_data is non-zero (activation points)
        activated_points = np.where(pw_data_stretched > 120e-6)[0]
        if len(activated_points) > 0:
            axes[0].plot(q_radians[activated_points], np.full_like(activated_points, radial_offsets[idx]), 'p',
                    label=f"{muscle_names[idx]}")

    axes[0].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes[0].set_title("Stimulation Pattern", fontsize=14)
    axes[0].set_xticklabels(['0°', '90°', '180°', '270°'])
    axes[0].set_yticks([])  # Remove radial ticks
    axes[0].set_yticklabels([])  # Remove radial tick labels
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 0), fontsize='small', frameon=True, ncol=1)
    axes[0].legend(loc='center', fontsize='medium', frameon=False, ncol=1)

    q_radians = q_data_aux[0][0]
    qdot_radians = dq_data_aux[0][0]
    axes[1].plot(q_radians, qdot_radians, label="Velocity [rad/s]", color="tab:blue")
    axes[1].set_title("Crank Angular Velocity", fontsize=14)
    axes[1].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axes[1].set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
    axes[1].tick_params(labelsize=10)
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].legend(loc='lower center', fontsize='medium', frameon=False, ncol=1)

    axes[1].plot(q_radians[0], qdot_radians[0], 'ro')  # Red dot at start
    axes[1].text(q_radians[0], qdot_radians[0] + 0.5, 'Start', color='red', fontsize=10, ha='left')

    axes[1].plot(q_radians[-1], qdot_radians[-1], 'ro')  # Red dot at start
    axes[1].text(q_radians[-1], qdot_radians[-1] + 0.5, 'End', color='red', fontsize=10, ha='right')

    # print mean values
    mean_rpm = np.mean(qdot_radians) * (60 / (2 * np.pi))
    std_rpm = np.std(qdot_radians) * (60 / (2 * np.pi))
    print(f'Mean velocity: {mean_rpm:.2f} ± {std_rpm:.2f} RPM')
    # plt.legend(fontsize='small', frameon=False, ncol=4)

    # power
    # tau_res_data = np.array(tau_res_data_aux[0][:,]).flatten()
    # q_size = len(q_data_aux[0][0])
    # tau_res_size = len(tau_res_data)
    # repeat_factor = q_size // tau_res_size
    # tau_res_data_stretched = np.array([np.repeat(pw, repeat_factor)[:q_size] for pw in tau_res_data]).flatten()

    tau_gear_size = len(tau_gear[0])
    repeat_factor = q_size // tau_gear_size
    tau_gear_data_stretched = np.array([np.repeat(pw, repeat_factor)[:q_size] for pw in tau_gear]).flatten()

    min_len = min(len(q_data_aux[0][0]), len(tau_gear_data_stretched))
    q_radians = q_data_aux[0][0][:min_len-20]
    pwr_radians = dq_data_aux[0][0][:min_len-20] * tau_gear_data_stretched[:min_len-20]
    # pwr_radians = dq_data_aux[0][0][:-21] * tau_gear_data_stretched[:-20]
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

    axes[2].plot(q_radians[-1],pwr_radians[-1], 'ro')  # Red dot at start
    axes[2].text(q_radians[-1],pwr_radians[-1] + 0.5, 'End', color='red', fontsize=10, ha='right')

    # Smart zoom on power
    pwr_min = np.min(pwr_radians)
    pwr_max = np.max(pwr_radians)
    margin = 0.1 * (pwr_max - pwr_min)
    axes[2].set_ylim(pwr_min - margin, pwr_max + margin)
    # Optional: add line at 0
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)

    # print mean values
    mean_pwr = np.mean(pwr_radians)
    std_pwr = np.std(pwr_radians)
    print(f'Mean power: {mean_pwr:.2f} ± {std_pwr:.2f} W')


    plt.savefig('images/stim_pattern.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()