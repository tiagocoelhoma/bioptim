import os, pickle, numpy as np, pandas as pd
from bioptim import BiorbdModel
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

from Param_iden_cycling.ocp_param_indent.common.com_functions import unpack_sol, find_nearest
from Param_iden_cycling.ocp_param_indent.common.torque_evaluation import msr, remove_torque_offset, data_for_ocp, \
    intervals_vel
from Param_iden_cycling.ocp_param_indent.tau1_tau2_km import tau1_tau2_km_ident
from Param_iden_cycling.ocp_param_indent.a_relation import a_relation_ident
from Param_iden_cycling.ocp_param_indent.common.params_handler import read_params, init_model_parameters

# PROBLEM PARAMETERS
problem_params = {
    "f": (0, 2500, 0),  # f_min, f_max, f_init
    "cn": (0.0, 2.0, 0.0),  # cn_min, cn_max, cn_init
    "tau": (0.0, 0.0, 0.0),  # tau_min, tau_max, tau_init
    "fes": (86e-6, 800e-6, 86e-6, 486e-6),  # fes_min, fes_max, fes_init, fes_sc
    "ascale": (100.0, 50e3, 100.0),  # ascale_min, ascale_max, ascale_init
    "km": (0.020, 0.800, 0.30130779),  # km_param_min, km_param_max, km_param_init
    "tau_1": (0.010, 0.1, 0.06637256),  # tau_1_param_min, tau_1_param_max, tau_1_param_init
    "tau_2": (0.001, 0.2, 0.001),  # tau_2_param_min, tau_2_param_max, tau_2_param_init
    "alpha": (-1.0, 2.0, 0.0),  # alpha_min, alpha_max, alpha_init
    "pd_offset": 86e-6,  # pd_offset
}

# parameters for retrieving experimental data
# muscle = 'STATIC_2_RF'  # define the muscle {STATIC_2_VM, STATIC_2_RF, STATIC_2_HM}
# muscle = 'STATIC_2_RF_TEST'  # knee angles at 70 and 80
# muscle = 'STATIC_2_RF_TEST_1_pulse_70'  # knee angles at 70
# muscle = 'STATIC_2_RF_TEST_1_pulse_90'  # knee angles at 90
# muscle = 'STATIC_3_RF_TEST'  # knee angles at 110 and 120

# constant parameters
stim_freq = 35                          # stimulation frequency in Hertz
t_duration = round(1 / stim_freq, 6)    # duration of a stimulation pulse
n_window = 6
window_size = 10                        # window sized used for filtering torque data
offset_interval = 100                   # torque data that will be used prior to the start of the contraction
only_parameter_a = False
with_residual_torque = False

def main():
    # PROBLEM PARAMETERS TO BE SET EACH TIME
    data_information = [
        # ["models/legcyc_rectfem_r.bioMod", 'rect_fem_r', [0, 50]],
        # ["models/legcyc_hams_r.bioMod", 'bflh140_r', [45, -1]],
        # ["models/legcyc_rectfem_r.bioMod", 'DYN_RF_350', [0, 50]],
        # ["models/legcyc_rectfem_r.bioMod", 'DYN_RF_400', [0, 50]],
        # ["models/legcyc_rectfem_r.bioMod", 'DYN_VM_350', [0, 50]],
        ["models/legcyc_rectfem_r.bioMod", 'DYN_VM_400', [0, 50]],
    ]

    # 'DYN_RF_350': "files/0167_Datafile.txt",
    # 'DYN_RF_400': "files/0175_Datafile.txt",
    # 'DYN_VM_350': "files/0181_Datafile.txt",
    # 'DYN_VM_400': "files/0185_Datafile.txt",
    # 'DYN_HM_350': "files/0177_Datafile.txt",
    # 'DYN_HM_400': "files/0179_Datafile.txt",
    muscle_params_file = f'../opt_sol_results/pickles/cycling_muscles_params.pkl'
    with_friction = True
    samp_step = 1               # ATTENTION! this will impact the number of nodes in the problem
                                # the higher this number, the lower the number of shooting nodes in the problem
    save_results = False        # ATTENTION! Use TRUE onlu if you want to save the identified parameters on the file.
    print_flag = True           # if one wants to have an additional visualization of the torque information
    plot_graphs = False         # for an additional visualization of the OCP resulting plots
    show_animate = False        # for an additional visualization of the OCP resulting animation

    for d, dt in enumerate(data_information):
        # reading data file
        model_path, muscle, qoffset = dt
        init_model_parameters(model_path, muscle_params_file)
        data_file = msr[muscle]
        df = pd.read_csv(f'data_exp/{data_file}', delimiter='\t')
        data = df[['Crank_angle_wired[deg] ', 'Powermeter_Torque_right[Nm] ']]
        offset = data[intervals_vel[dt[1]][0]:intervals_vel[dt[1]][1]]
        cadence = df['Cadence[RPM] ']

        data_angles = data["Crank_angle_wired[deg] "].values.reshape(-1, 1)
        offset_angles = offset["Crank_angle_wired[deg] "].values.reshape(-1, 1)
        offset_torques = offset["Powermeter_Torque_right[Nm] "].values

        tree = cKDTree(offset_angles)                       # Build a KDTree for efficient nearest neighbor search
        distances, indices = tree.query(data_angles)        # Find nearest angles in offset for each angle in data
        nearest_offset_torques = offset_torques[indices]    # Get corresponding offset torque values
        data["Corrected_Torque"] = data["Powermeter_Torque_right[Nm] "] - nearest_offset_torques


        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('samples')
        # ax1.set_ylabel(label_1, color=color)
        ax1.plot(data["Corrected_Torque"], color=color)
        ax1.plot(data["Powermeter_Torque_right[Nm] "], color='tab:red')
        ax1.plot(cadence, color='tab:orange')
        cadence
        ax1.tick_params(axis='y', labelcolor=color)

        # ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        # color = 'tab:red'
        # # ax2.set_ylabel(label_2, color=color)  # we already handled the x-label with ax1
        # ax2.plot(data_2, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        # torque_dt = remove_torque_offset(torque, muscle, window_size, offset_interval, print_flag)
        fxd_range = True if muscle == 'bflh140_r' else False
        tau_target, stim_flag, knee_ang, nodes, times = data_for_ocp(torque,
                                                                     muscle,
                                                                     offset_interval,
                                                                     samp_step,
                                                                     print_flag,
                                                                     contraction=2,
                                                                     fxd_range=fxd_range)
        mdl_param = read_params(model_path, muscle_params_file)[muscle]

        n_phases = len(tau_target)
        duration, nb_shooting, target_vel = 1.75, 100, 35
        dump_path = f'opt_sol_results/kinematics/fw_rotation_{duration}_{nb_shooting}_{target_vel}'  # cycling kinematics
        from examples.leg_cyclic_torque_driven import main as main_torque
        if not os.path.isfile(dump_path):
            print(f"Intializing kinematics...")
            main_torque(duration, nb_shooting, target_vel, crank_opt_flag=False, prin_res=True)
        sol_dict = pickle.load(open(dump_path, 'rb'))
        qvec = np.array(unpack_sol(sol_dict["q"])).T

        # knee angles° and crank angles
        bio_model = BiorbdModel(model_path)
        idx = bio_model.name_dof.index('knee_r_RotZ')
        q_init_list = []
        # range to evaluate the knee angles in the lower part of the pedaling cycle
        q_interval_init, q_interval_end = qoffset
        for i, angle in enumerate(knee_ang):
            value = find_nearest(qvec[:, q_interval_init:q_interval_end], value=(np.pi - ((angle / 180) * np.pi)),
                                 idx_dof=idx)
            q_init_list.append(value)

        ocp_parameters = {
            'biorbd_model_path': model_path,
            'q_init': q_init_list,
            'with_friction': with_friction,
            'with_residual_torque': with_residual_torque,
            'pulse_period': t_duration,  # Attention: duration of each pulse
            'phase_period': times,  # Attention: duration of each phase of the problem
            'n_shooting': nodes,
            'n_pulse_trunc': n_window,
            'target': tau_target,
            'stim_flag': stim_flag,
            'n_phases': n_phases,
            'joint_angle': knee_ang,
            'params': mdl_param,
            'only_parameter_a': only_parameter_a,
            'muscle': muscle,
            'stim_freq': stim_freq,
        }

        # --- Prepare the optimal control program for identification of tau1, tau2 and km parameters --- #
        # tau1_tau2_km_ident(ocp_parameters, problem_params, save_results, plot_graphs, show_animate)
        a_relation_ident(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, show_animate)


if __name__ == "__main__":
    main()
