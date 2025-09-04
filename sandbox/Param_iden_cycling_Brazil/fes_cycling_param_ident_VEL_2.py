import os, pickle, numpy as np, pandas as pd
from bioptim import BiorbdModel
from casadi import interp1d
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
# import biorbd as brbd

from Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import unpack_sol, find_nearest, find_nearest2
from Param_iden_cycling_Brazil.ocp_param_indent.common.torque_evaluation import msr, remove_torque_offset, \
    data_for_ocp, intervals
from Param_iden_cycling_Brazil.ocp_param_indent.tau1_tau2_km import tau1_tau2_km_ident
from Param_iden_cycling_Brazil.ocp_param_indent.a_relation import a_relation_ident
from Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import read_params, init_model_parameters, \
    copy_params_right_to_left, load_pickle
from Param_iden_cycling_Brazil.ocp_param_indent.v1_v2 import v1_v2_ident
from Param_identification.statistical_analysis.averagevalues import knee_angles

# constant parameters
stim_freq = 35  # stimulation frequency in Hertz
t_duration = round(1 / stim_freq, 6)  # duration of a stimulation pulse
n_window = 6
window_size = 10  # window sized used for filtering torque data
offset_interval = 15  # torque data that will be used prior to the start of the contraction
only_parameter_a = False
with_residual_torque = False


def main():
    # PROBLEM PARAMETERS TO BE SET EACH TIME
    data_information = [
        # [model, muscle, [node_start, node_end]] <- node_start and end corresponds to a position within the crank cycle
        ["models/legcyc_rectfem_r.bioMod", 'rect_fem_r_vel', [0, -1]],
        # ["models/legcyc_vasmed_r.bioMod", 'vasmed_r', [0, -1]],
        # ["models/legcyc_hams_r.bioMod", 'bflh140_r', [0, -1]]
        # ["models/legcyc_rectfem_r.bioMod", 'rect_fem_l', [0, -1]],
        # ["models/legcyc_vasmed_r.bioMod", 'vasmed_l', [0, -1]],
        # ["models/legcyc_hams_r.bioMod", 'bflh140_l', [20, -1]]

    ]
    muscle_params_file = f'opt_sol_results/pickles/cycling_muscles_params.pkl'
    with_friction = False
    with_resistive_torque = True
    # samp_step = 2               # ATTENTION! this will impact the number of nodes in the problem
    # the higher this number, the lower the number of shooting nodes in the problem
    save_results = False  # ATTENTION! Use TRUE only if you want to save the identified parameters on the file.
    print_flag = True  # if one wants to have an additional visualization of the torque information
    plot_graphs = True  # for an additional visualization of the OCP resulting plots
    show_animate = True  # for an additional visualization of the OCP resulting animation

    print(f'\nInitial parameters are: {load_pickle(muscle_params_file)}')
    for d, dt in enumerate(data_information):
        # reading data file
        model_path, muscle, qoffset = dt
        init_model_parameters(model_path, muscle_params_file)
        data_file = msr[muscle]
        df = pd.read_csv(f'data_exp/{data_file}', delimiter='\t')
        # torque = df['Torque (Nm)'].values.flatten()     # remember to update file and remove RIGHT or LEFT
        # crank_angles = df['Angle(deg)'].values.flatten()

        data = df[['Angle(deg)', 'Torque (Nm)']]
        offset = data[intervals[muscle]['passive_1'][0]:intervals[muscle]['passive_1'][1]]
        cadence = df['Cadence(rpm)']

        data_angles = data["Angle(deg)"].values.reshape(-1, 1)
        offset_angles = offset["Angle(deg)"].values.reshape(-1, 1)
        crank_angle_offset = 90  # right leg
        crank_angles = [
            angle + crank_angle_offset if angle + crank_angle_offset < 360 else angle + crank_angle_offset - 360 for
            angle in data_angles]

        offset_angles = [
            angle + crank_angle_offset if angle + crank_angle_offset < 360 else angle + crank_angle_offset - 360 for
            angle in offset_angles]
        offset_torques = offset["Torque (Nm)"].values

        tree = cKDTree(offset_angles)  # Build a KDTree for efficient nearest neighbor search
        distances, indices = tree.query(crank_angles)  # Find nearest angles in offset for each angle in data
        nearest_offset_torques = offset_torques[indices]  # Get corresponding offset torque values
        data["Corrected_Torque"] = data["Torque (Nm)"] - nearest_offset_torques
        torque_data = np.array((data["Torque (Nm)"] - nearest_offset_torques)).flatten()
        crank_angles = np.array(crank_angles).flatten()
        stim_angles = np.zeros_like(crank_angles)
        start, end = intervals[muscle]['active']
        for i, angle in enumerate(crank_angles):
            if (angle >= 335 or angle <= 180) and (start <= i <= end):
                stim_angles[i] = 1
        # stim_angles = np.logical_or(crank_angles >= 335, crank_angles <= 180).astype(int)

        # figure 1
        fig, ax1 = plt.subplots()
        # torque
        color = 'tab:blue'
        ax1.set_xlabel('samples')
        ax1.set_ylabel('Torque (Nm)', color=color)
        ax1.plot(torque_data, color=color, label='Corrected Torque')
        ax1.plot(data["Torque (Nm)"], color='tab:red', label='Raw Torque')
        ax1.plot(stim_angles, color='b', label='Stim Angles')
        ax1.tick_params(axis='y', labelcolor=color)
        # anle
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Angle (°)', color=color)
        ax2.plot(crank_angles, color=color, label='Angle')
        ax2.tick_params(axis='y', labelcolor=color)
        # Title
        plt.title('Torque and Angle Over Time')
        ax1.grid(True)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        # creating data for OCP
        freq_sampling = 50  # according to data collected from rotor 2INPOWER (FAST DATA 20 ms)
        tau_offset = 0.5
        flag_activated = False
        interval_cnt, aux = 0, 0
        ft_mp, angle, stim_dt, time = [], [], [], []

        # adjusting sampled torque data to the stimulation frequency
        T = len(torque_data) / freq_sampling
        t_data = np.linspace(0, T, len(torque_data))
        t_ocp = np.linspace(0, T, int(T * stim_freq))
        interp_func = interp1d(t_data, torque_data, kind='linear')
        torque_data = interp_func(t_ocp)

        # t_data = np.linspace(0, T, len(crank_angles))
        interp_func = interp1d(t_data, crank_angles, kind='linear')
        angle_data = interp_func(t_ocp)
        interp_func = interp1d(t_data, stim_angles, kind='linear')
        stim_data = interp_func(t_ocp)

        # figure 2
        # Data adjusted to the stimulation frequency
        fig, ax1 = plt.subplots()
        # torque
        color = 'tab:blue'
        ax1.set_xlabel('samples')
        ax1.set_ylabel('Torque (Nm)', color=color)
        ax1.plot(torque_data, color=color, label='Corrected Torque')
        ax1.plot(stim_data, color='tab:red', label='Stim Data')
        ax1.tick_params(axis='y', labelcolor=color)
        # anle
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Angle (°)', color=color)
        ax2.plot(angle_data, color=color, label='Angle')
        ax2.tick_params(axis='y', labelcolor=color)
        # Title
        plt.title('Torque and Angle Over Time')
        ax1.grid(True)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        mdl_param = read_params(model_path, muscle_params_file)[muscle[:-4]]
        start, end = intervals[muscle]['identif']
        # preciso de posicao para track da posicao, torque para track e preciso da velocidade inicial
        tau_target = torque_data[start:end]
        q_data = angle_data[start:end]
        stim_flag = stim_data[start:end]

        # compute crank angle to be tracked
        for i in range(1, len(q_data)):
            if (q_data[i] < q_data[i-1] - 100) and (q_data[i] > 250):
                q_data[i] = q_data[i - 1]

        # # compute crank angle to be tracked
        # for i in range(1, len(q_data)):
        #     if q_data[i] < q_data[i-1] - 100:
        #         for j in range(i, len(q_data)):
        #             q_data[j] = q_data[j] + 360

        duration, nb_shooting, target_vel, target_power = 6.00, 300, 42, 15
        dump_path = f'opt_sol_results/kinematics/fw_rotation_{duration}_{nb_shooting}_{target_vel}'  # cycling kinematics
        from Param_iden_cycling_Brazil.leg_pedalling_torque_driven import main as main_torque
        if not os.path.isfile(dump_path):
            print(f"Intializing kinematics...")
            target_v = (-(target_vel * 2 * np.pi) / 60)  # rpm to rad/s
            torque_crank = target_power / target_v
            friction_coef_b = round(torque_crank / target_v, 4)
            friction_coef_mu = 0.0
            model_pedalling = 'models/leg_model_cyclic_muscle_inversed_root_3.bioMod'
            main_torque(model_pedalling, dump_path, duration, nb_shooting, target_v, target_power, False, True,
                        friction_coef_mu, friction_coef_b)
        sol_dict = pickle.load(open(dump_path, 'rb'))
        qvec = np.array(unpack_sol(sol_dict["q"])).T
        dqvec = np.array(unpack_sol(sol_dict["qdot"])).T

        # knee angles° and crank angles
        bio_model = BiorbdModel(model_path)
        idx = bio_model.name_dof.index('Gear_RotZ')
        for i in range(1, len(q_data)):
            if (q_data[i] + 250 < (q_data[i-1])):
                for j in range(i, len(q_data)):
                    q_data[j] = q_data[j] + 360

        for i in range(1, len(q_data)):
            if (q_data[i] < (q_data[i-1])):
                q_data[i] = q_data[i-1]

        q_crank_target = []
        for i, angle in enumerate(q_data):
            # x = find_nearest(qvec[[0], q_interval_init:q_interval_end],
            #              value=-((angle / 180) * np.pi),
            #              idx_dof=idx)
            x = find_nearest2(qvec, value=-((angle / 180) * np.pi) - 2*np.pi, idx_dof=idx)
            q_crank_target.append(x[0])
            print(-((angle / 180) * np.pi)- 2*np.pi)
            print(x[0])

        # range to evaluate the knee angles in the lower part of the pedaling cycle
        # q_interval_init, q_interval_end = qoffset
        q_init_posis = find_nearest2(qvec, value=-((angle / 180) * np.pi), idx_dof=idx)
        kni_pos_value_at = None
        for i in range(qvec.shape[1]):
            if np.allclose(qvec[:, i], q_init_posis.flatten(), atol=1e-8):
                kni_pos_value_at = i
                break
        print(f"Índice encontrado: {kni_pos_value_at}")
        kni_pos_value_at = kni_pos_value_at - 1

        # compute crank angle to be tracked
        # for i in range(1, len(q_crank_target)):
        #     if (q_crank_target[i] -2 < q_crank_target[i-1]) and (abs(q_crank_target[i-1] - q_crank_target[i]) > 1):
        #         q_crank_target[i] = q_crank_target[i - 1]

        # from Param_iden_cycling.ocp_param_indent.common.params_handler import read_params
        # muscle_params_file = f'opt_sol_results/pickles/cycling_muscles_params.pkl'
        # mdl_param = read_params(model_path, muscle_params_file)[muscle]

        ocp_parameters = {
            'biorbd_model_path': model_path,
            'q_init': qvec,
            'dq_init': dqvec,
            'kni_pos_value_at': kni_pos_value_at,
            'with_friction': with_friction,
            'with_resistive_torque': with_resistive_torque,
            'with_residual_torque': with_residual_torque,
            'pulse_period': t_duration,  # Attention: duration of each pulse
            'phase_period': t_duration * len(tau_target),  # Attention: duration of each phase of the problem
            'n_shooting': len(tau_target),
            'n_pulse_trunc': n_window,
            'target_torque': tau_target,
            'stim_flag': stim_flag,
            'target_crank_position': q_crank_target,
            'n_phases': 1,
            # 'joint_angle': crank_ang,
            # 'params': mdl_param,
            # 'only_parameter_a': only_parameter_a,
            'muscle': muscle,
            'stim_freq': stim_freq,
        }

        # PROBLEM PARAMETERS
        # data = load_pickle('opt_sol_results/pickles/pd0ptt.pkl')
        # print(data[muscle])
        # a = data[muscle][0]
        # pd0 = data[muscle][1]
        # pdt = data[muscle][2]

        # need to be reviewed
        a = 1
        pd0 = 0.00011785  # average value
        pdt = 0.00086029  # average value
        print(f'using pd0 and pdt average values. pd0 = {pd0} and pdt = {pdt}')

        fes = (pd0, 1600e-6, pd0, 800e-6)
        fes = (pd0, 1600e-6, pd0, 700e-6) if muscle == 'bflh140_r' or muscle == 'bflh140_l' else fes
        problem_params = {
            "f": (0, 1000, 0),  # f_min, f_max, f_init
            "cn": (0.0, 2.0, 0.0),  # cn_min, cn_max, cn_init
            "tau": (0.0, 0.0, 0.0),  # tau_min, tau_max, tau_init
            "fes": fes,  # fes_min, fes_max (800+800us), fes_init, fes_sc (300+300us)
            "v1_param": (0.0007, 0.003, 0.0007),  # v1_min, v1_max, v1_init
            "v2_param": (0.002, 0.065, 0.002),  # v2_min, v2_max, v2_init
            "alpha": (-1.0, 2.0, 0.0),  # alpha_min, alpha_max, alpha_init
            "pd0pdt": (a, pd0, pdt),  #
        }

        # --- Prepare the optimal control program for identification of tau1, tau2 and km parameters --- #
        v1_v2_ident(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, print_flag,
                    show_animate)
        # copy_params_right_to_left(model_path, muscle_params_file, option=1) #option 1 right to left. option 2 left to right
    print(f'\nFinal parameters are: {load_pickle(muscle_params_file)}')


if __name__ == "__main__":
    main()
