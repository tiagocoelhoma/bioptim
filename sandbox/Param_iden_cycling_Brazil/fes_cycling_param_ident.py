import os, pickle, numpy as np, pandas as pd
from bioptim import BiorbdModel
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.a_relation import a_relation_ident
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import unpack_sol, find_nearest
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import (
    read_params, load_pickle, init_model_parameters)
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.torque_evaluation import msr, data_for_ocp
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.prediction import prediction
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.tau1_tau2_km import tau1_tau2_km_ident


def get_initial_position():
    # dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/fw_rotation_{duration}_{nb_shooting}_{target_vel}'  # cycling kinematics
    dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/cycling'
    from sandbox.cycling_torque import main as main_torque
    if not os.path.isfile(dump_path):
        print(f"Initializing kinematics. Remember to update this file when the biomechanical model changes!")
        biorbd_model_path = 'models/legcyc.bioMod'
        q_init = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])
        main_torque(biorbd_model_path, dump_path, q_init)
    sol_dict = pickle.load(open(dump_path, 'rb'))
    qvec = np.array(unpack_sol(sol_dict["q"])).T

    return qvec


# constant parameters
stim_freq = 35                          # stimulation frequency in Hertz
t_duration = round(1 / stim_freq, 6)    # duration of a stimulation pulse
n_window = 5
window_size = 10                        # window sized used for filtering torque data
offset_interval = 10                    # torque data that will be used prior to the start of the contraction
with_residual_torque = False
with_friction = False
with_resistive_torque = True

def main():
    # PROBLEM PARAMETERS TO BE SET EACH TIME
    data_information = [
        # [model, muscle, [node_start, node_end]] <- node_start and end corresponds to a position within the crank cycle
        # ["models/legcyc_rectfem_r.bioMod", 'rect_fem_r', [0, 25]],
        # ["models/legcyc_vasmed_r.bioMod", 'vasmed_r', [0, 25]],
        # ["models/legcyc_hams_r.bioMod", 'bflh140_r', [0, -1]]
        # ["models/legcyc_rectfem_r.bioMod", 'rect_fem_l', [0, -1]],
        ["models/legcyc_vasmed_r.bioMod", 'vasmed_l', [0, -1]],
        # ["models/legcyc_hams_r.bioMod", 'bflh140_l', [0, -1]]
    ]
    muscle_params_file = f'Param_iden_cycling_Brazil/opt_sol_results/pickles/cycling_muscles_params.pkl'

    save_results = True         # ATTENTION! Use TRUE onlu if you want to save the identified parameters on the file.
    print_flag = False           # if one wants to have an additional visualization of the torque information
    plot_graphs = False         # for an additional visualization of the OCP resulting plots
    show_animate = False        # for an additional visualization of the OCP resulting animation

    qvec = get_initial_position()

    print(f'\nInitial parameters are: {load_pickle(muscle_params_file)}')
    for d, dt in enumerate(data_information):
        # reading data file
        model_path, muscle, qoffset = dt
        init_model_parameters(model_path, muscle_params_file)
        data_file = msr[muscle]
        df = pd.read_csv(f'Param_iden_cycling_Brazil/data_exp/{data_file}', delimiter='\t')
        torque = df['Torque (Nm)'].values.flatten()
        crank_angles = df['Angle(deg)'].values.flatten()

        # angle measured in the ROTOR IN2POWER but corresponding to the 0 degress in simulation
        crank_angle_offset = 90  # right leg
        crank_angle_offset = 270 if muscle[-1] == 'l' else crank_angle_offset
        crank_angles = [
            angle + crank_angle_offset if angle + crank_angle_offset < 360 else angle + crank_angle_offset - 360 for angle
            in crank_angles]
        # obtaining the stimulation intervals automatically
        tau_target, stim_flag, crank_ang, nodes, times = data_for_ocp(torque,
                                                                     crank_angles,
                                                                     muscle,
                                                                     offset_interval,
                                                                     stim_freq,
                                                                     window_size,
                                                                     print_flag,
                                                                     contraction=1)
        mdl_param = read_params(model_path, muscle_params_file)[muscle]

        # knee angles° and crank angles
        bio_model = BiorbdModel(model_path)
        idx = bio_model.name_dof.index('Gear_RotZ')
        q_init_list = []
        # range to evaluate the knee angles in the lower part of the pedaling cycle
        q_interval_init, q_interval_end = qoffset
        for i, angle in enumerate(crank_ang):
            value = find_nearest(qvec[:, q_interval_init:q_interval_end],
                                 value=-((angle / 180) * np.pi),
                                 idx_dof=idx)
            q_init_list.append(value)
            print(f'crank_ang: {crank_ang}, converted: {value}')

        ocp_parameters = {
            'biorbd_model_path': model_path,
            'q_init': q_init_list,
            'with_friction': with_friction,
            'with_resistive_torque': with_resistive_torque,
            'with_residual_torque': with_residual_torque,
            'pulse_period': t_duration,  # Attention: duration of each pulse
            'phase_period': times,  # Attention: duration of each phase of the problem
            'n_shooting': nodes,
            'n_pulse_trunc': n_window,
            'target': tau_target,
            'stim_flag': stim_flag,
            'n_phases': len(tau_target),
            'joint_angle': crank_ang,
            'params': mdl_param,
            'muscle': muscle,
            'stim_freq': stim_freq,
        }

        # PROBLEM PARAMETERS
        # data = load_pickle('opt_sol_results/pickles/pd0ptt.pkl')
        # print(data[muscle])
        # a = data[muscle][0]
        # pd0 = data[muscle][1]
        # pdt = data[muscle][2]

        # Fitted Parameters:
        a = 1.05493875e+00
        pd0 = 1.11878174e-04
        pdt = 5.69997838e-04
        print(f'using pd0 and pdt average values. pd0 = {pd0} and pdt = {pdt}')

        fes = {'fes_min': pd0, 'fes_max': 1600e-6, 'fes_init': 800e-6, 'fes_sc': 800e-6}
        fes = {'fes_min': pd0, 'fes_max': 1600e-6, 'fes_init': 800e-6, 'fes_sc': 800e-6} if muscle == 'bflh140_r' or muscle == 'bflh140_l' else fes
        problem_params = {
            "f": (0, 5000, 0),  # f_min, f_max, f_init
            "cn": (0.0, 2.0, 0.0),  # cn_min, cn_max, cn_init
            "tau": (0.0, 0.0, 0.0),  # tau_min, tau_max, tau_init
            "fes": fes, # fes_min, fes_max (800+800us), fes_init, fes_sc (300+300us)
            "ascale": (0, 50e3, 5e3),  # ascale_min, ascale_max, ascale_init
            "km": (0.020, 0.800, 0.30130779),  # km_param_min, km_param_max, km_param_init
            "tau_1": (0.010, 0.1, 0.06637256),  # tau_1_param_min, tau_1_param_max, tau_1_param_init
            "tau_2": (0.001, 0.2, 0.01),  # tau_2_param_min, tau_2_param_max, tau_2_param_init
            "alpha": (-1.0, 2.0, 0.0),  # alpha_min, alpha_max, alpha_init
            "pd0pdt": (a, pd0, pdt),  #
        }

        # --- Prepare the optimal control program for identification of tau1, tau2 and km parameters --- #
        tau1_tau2_km_ident(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, show_animate)
        a_relation_ident(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, show_animate, print_flag)
        # copy_params_right_to_left(model_path, muscle_params_file, option=1) #option 1 right to left. option 2 left to right
        prediction(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, show_animate)
        # statistic_plots(muscle)
    print(f'\nFinal parameters are: {load_pickle(muscle_params_file)}')


if __name__ == "__main__":
    main()
