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
        # 6: 'n105_6.0W_41.92_f35_m6',
        # 10: 'n105_10.0W_41.92_f35_m6',
        # 15: 'n105_15.0W_41.92_f35_m6',
        20: 'friction_3.05s_105n_41.92rpm_20.0W_fric_0.2_1.0378'
    }
    size_font_subtitle = 10

    names_crank = ['convetional']
    for n, (model, file) in enumerate(activity.items()):
        f_data_aux, q_data_aux, dq_data_aux, pw_data_aux, tau_res_data_aux = [], [], [], [], []
        num_model = brbd.Model(f"../models/leg_model_cyclic_muscle_inversed_root_3.bioMod")
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

        tau_gear = []
        names_crank = ['convetional']
        # num_model = brbd.Model(f"../models/{model}.bioMod")
        # n_mus = num_model.nbMuscles()
        # n_q = num_model.nbQ()
        # muscle_names = tuple(s.to_string() for s in num_model.muscleNames())
        file_name = f"fes_leg_pedalling/eval_pwr_{file}.pkl"
        # file_name = 'fes_leg_pedalling/eval_pwr_res_constant_n35_10.0W_41.92_f35_m6.pkl'
        data = get_data(file_name)
        print(f"{file_name}, iterations:{data["n_iterations"]}, time:{round(data["real_time"], 2)}")
        # q_sol, qdot_sol = data["q"], data["qdot"]
        # f_sol, pw_sol, time_sol = data["f"], data["pw"], data["time"]
        tau_control = data["tau_joints"] if data["tau_joints"] else None

        time = data["time"]
        t_array = np.concatenate(
            [np.array(t.full()).flatten()[:-1] if i < len(time) - 1 else np.array(t.full()).flatten() for i, t in
             enumerate(time)])

        # tau_gear.append(organize_data(tau_control, n_q))
        # tau_gear.append(np.array(tau_control))
        aux = [float(tau[0, 0]) for tau in tau_control]
        tau_gear.append(aux)

        q_data_aux[0][0] = q_data_aux[0][0] * -1
        dq_data_aux[0][0] = dq_data_aux[0][0] * -1

        radial_offsets = np.linspace(10.5, 20.5, 8)
        q_radians = q_data_aux[0][0] % (2*np.pi)
        num_cycles = int((q_data_aux[0][0][-1] - q_data_aux[0][0][0]) / (2*np.pi)) + 1
        fig, axes = plt.subplots(num_cycles, 3, subplot_kw={'polar': True}, figsize=(16, 8))
        x = 0
        indices = [0]
        for cycle in range(num_cycles):
            target_value = float(q_data_aux[0][0][0] + (2 * np.pi) * (cycle + 1))
            x = int(np.argmin(np.abs(target_value - q_data_aux[0][0])))     # Find the index of the closest value
            print(x)
            indices.append(x)

            q_size = len(q_radians)
            # Plot the activation for each index in pw_data_aux
            for idx in range(n_mus):
                pw_data = np.array(pw_data_aux[0][:, idx]).flatten()
                pw_size = len(pw_data)
                repeat_factor = q_size // pw_size
                pw_data_stretched = np.array([np.repeat(pw, repeat_factor)[:q_size] for pw in pw_data]).flatten()
                # Find the indices where pw_data is non-zero (activation points)
                pw_data_stretched_seg = pw_data_stretched[indices[-2]: indices[-1]]
                q_radians_aux = q_radians[indices[-2]: indices[-1]]
                activated_points = np.where(pw_data_stretched_seg > 120e-6)[0]
                if len(activated_points) > 0:
                    axes[cycle, 0].plot(q_radians_aux[activated_points], np.full_like(activated_points, radial_offsets[idx]), 'p',
                            label=f"{muscle_names[idx]}")

            axes[cycle, 0].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
            if cycle == 0:
                axes[cycle, 0].set_title("Stimulation Pattern", fontsize=size_font_subtitle)
            else:
                axes[cycle, 0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='small', frameon=False, ncol=3)
            axes[cycle, 0].set_xticklabels(['0°', '90°', '180°', '270°'])
            axes[cycle, 0].tick_params(labelsize=8)
            axes[cycle, 0].set_yticks([])  # Remove radial ticks
            axes[cycle, 0].set_yticklabels([])  # Remove radial tick labels


            q_radians_vel = q_data_aux[0][0][indices[-2]: indices[-1]]
            qdot_radians_vel = dq_data_aux[0][0][indices[-2]: indices[-1]]
            axes[cycle, 1].plot(q_radians_vel, qdot_radians_vel, label="Velocity [rad/s]", color="tab:blue")
            if cycle == 0:
                axes[cycle, 1].set_title("Crank Angular Velocity", fontsize=size_font_subtitle)
            else:
                axes[cycle, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='small', frameon=False, ncol=1)
            axes[cycle, 1].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
            axes[cycle, 1].set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
            axes[cycle, 1].tick_params(labelsize=8)
            axes[cycle, 1].grid(True, linestyle='--', linewidth=0.5)
            axes[cycle, 1].plot(q_radians_vel[0], qdot_radians_vel[0], 'ro')  # Red dot at start
            axes[cycle, 1].text(q_radians_vel[0], qdot_radians_vel[0] + 0.5, 'Start', color='red', fontsize=10, ha='left')
            axes[cycle, 1].plot(q_radians_vel[-1], qdot_radians_vel[-1], 'ro')  # Red dot at start
            axes[cycle, 1].text(q_radians_vel[-1], qdot_radians_vel[-1] + 0.5, 'End', color='red', fontsize=10, ha='right')

            # print mean values
            mean_rpm = np.mean(qdot_radians_vel) * (60 / (2 * np.pi))
            std_rpm = np.std(qdot_radians_vel) * (60 / (2 * np.pi))
            print(f'Mean velocity: {mean_rpm:.2f} ± {std_rpm:.2f} RPM')


            # power
            tau_res_data = np.array(tau_res_data_aux[0][:,]).flatten()
            q_size = len(q_data_aux[0][0])
            tau_res_size = len(tau_res_data)
            repeat_factor = q_size // tau_res_size
            tau_res_data_stretched = np.array([np.repeat(pw, repeat_factor)[:q_size] for pw in tau_res_data]).flatten()

            tau_gear_size = len(tau_gear[0])
            repeat_factor = q_size // tau_gear_size
            tau_gear_data_stretched = np.array([np.repeat(pw, repeat_factor)[:q_size] for pw in tau_gear]).flatten()

            q_radians_pwr = q_data_aux[0][0][indices[-2]: indices[-1]]
            pwr_radians = dq_data_aux[0][0][indices[-2]: indices[-1]] * tau_gear_data_stretched[indices[-2]: indices[-1]]
            pwr_radians_res = dq_data_aux[0][0][indices[-2]: indices[-1]] * tau_res_data_stretched[indices[-2]: indices[-1]]
            axes[cycle, 2].plot(q_radians_pwr, pwr_radians, label="Power [W]", color="tab:green")
            # axes[2].plot(q_radians, pwr_radians_res, label="Res Power", color="tab:blue")
            axes[cycle, 2].fill_between(q_radians_pwr, 0, pwr_radians, alpha=0.1, color="tab:green")
            # axes[2].fill_between(q_radians, 0, pwr_radians_res, alpha=0.1, color="tab:blue")
            if cycle == 0:
                axes[cycle, 2].set_title("Power Output", fontsize=size_font_subtitle)
            else:
                axes[cycle, 2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='small', frameon=False, ncol=1)
            axes[cycle, 2].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
            axes[cycle, 2].set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=12)
            axes[cycle, 2].grid(True, linestyle='--', linewidth=0.5)
            axes[cycle, 2].tick_params(labelsize=8)
            axes[cycle, 2].plot(q_radians_pwr[0], pwr_radians[0], 'ro')  # Red dot at start
            axes[cycle, 2].text(q_radians_pwr[0], pwr_radians[0] + 0.5, 'Start', color='red', fontsize=10, ha='left')
            axes[cycle, 2].plot(q_radians_pwr[-1],pwr_radians[-1], 'ro')  # Red dot at start
            axes[cycle, 2].text(q_radians_pwr[-1],pwr_radians[-1] + 0.5, 'End', color='red', fontsize=10, ha='right')

            # Smart zoom on power
            pwr_min = np.min(pwr_radians)
            pwr_max = np.max(pwr_radians)
            margin = 0.1 * (pwr_max - pwr_min)
            axes[cycle, 2].set_ylim(pwr_min - margin, pwr_max + margin)
            # Optional: add line at 0
            axes[cycle, 2].axhline(0, color='gray', linestyle='--', linewidth=1)

            # print mean values
            mean_pwr = np.mean(pwr_radians)
            std_pwr = np.std(pwr_radians)
            print(f'Mean power: {mean_pwr:.2f} ± {std_pwr:.2f} W')


            fig.suptitle(f"Pedalling evaluation at {file[5:7]} W", fontsize=12)
            fig.savefig(f'images/stim_pattern{file[5:7]}_W.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.show()