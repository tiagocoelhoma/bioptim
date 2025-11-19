import os, pickle, numpy as np, pandas as pd
from bioptim import BiorbdModel
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.a_relation import a_relation_ident
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import unpack_sol, find_nearest
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import (
    read_params, load_pickle, init_model_parameters)
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.torque_evaluation import msr, data_for_ocp
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.prediction import prediction
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.tau1 import tau1_ident
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.tau2_km import tau2_km_ident




def get_initial_position():
    # dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/fw_rotation_{duration}_{nb_shooting}_{target_vel}'  # cycling kinematics
    # the kinematics for only one cycle is important for the identification process to get one position of the crank
    dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/kinematics_one_cycle_only'
    # dump_path = f'Param_iden_cycling_Brazil/opt_sol_results/kinematics/Fes_leg_pedalling'
    from sandbox.cycling_torque import main as main_torque
    if not os.path.isfile(dump_path):
        print(f"Initializing kinematics. If you have changed the biomechanical model, remember to update q_init !")
        biorbd_model_path = 'models/legcyc.bioMod'
        # q_init = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])
        # q_init = np.array([[-6.3003], [4.9974], [1.9865], [0.1753], [5.5781], [1.0031], [0.5679]])
        # q_init = np.array([[-6.3003], [4.956], [2.1215], [0.3629], [5.4744], [1.242], [0.7569]])  # good convergence for trque
        # q_init = np.array([[-6.3003], [4.9352], [2.1458], [0.3629], [5.4536], [1.2742], [0.7569]])
        # q_init = np.array([[-6.3003], [4.873], [2.1091], [0.508], [5.4329], [1.2191], [0.8191]])
        # q_init = np.array([[-6.3003], [4.8938], [2.1183], [0.591], [5.4536], [1.2053], [0.7983]])
        q_init = np.array([[-6.3003], [4.9352], [2.1183], [0.6117], [5.4744], [1.2053], [0.9228]])
        q_init = np.array([[-0.1055], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]]) # negative torque first stim 06/11/2025
        q_init = np.array([[-0.1055], [4.9974], [2.1304], [0.3421], [5.5159], [1.2261], [0.6947]])
        q_init = np.array([[-0.1055], [5.0389], [2.1304], [0.3421], [5.5366], [1.2261], [0.6947]])
        q_init = np.array([[-0.0104], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])# negative torque first stim 06/11/2025
        # q_init = np.array([[-0.0422], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]]) # negative torque first stim 06/11/2025
        # q_init = np.array([[-0.1476], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]]) # negative torque first stim 06/11/2025
        q_init = np.array([[-0.0104], [4.9974], [2.137], [0.3421], [5.5159], [1.2327], [0.6947]])
        q_init = np.array([[-0.0104], [4.8938], [2.137], [0.3421], [5.4329], [1.2525], [0.6947]])
        q_init = np.array([[-0.0104], [4.956], [2.1172], [0.3629], [5.4951], [1.2063], [0.6947]])
        q_init = np.array([[-0.0104], [4.9663], [2.1172], [0.3629], [5.4951], [1.2129], [0.7154]])
        q_init = np.array([[-0.1476], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])
        # q_init = np.array([[-0.1037], [4.9663], [2.1172], [0.3629], [5.4951], [1.2129], [0.7154]])  # tested 33c6bbc1
        q_init = np.array([[-0.0104], [4.9663], [2.1172], [0.3629], [5.4951], [1.2129], [0.7154]])
        q_init = np.array([[-0.0104], [4.9663], [2.0974], [0.3836], [5.5054], [1.1799], [0.7361]])

        q_init = np.array([[-0.0104], [5.0182], [2.104], [0.3421], [5.5366], [1.1865], [0.7361]])
        q_init = np.array([[-0.0104], [5.0596], [2.104], [0.3421], [5.5677], [1.1865], [0.7154]])
        # q_init = np.array([[-0.0104], [4.9663], [2.0182], [0.3421], [5.5469], [1.0411], [0.7154]])

        main_torque(biorbd_model_path, dump_path, q_init, n_shooting=64, final_time=1.5)
    sol_dict = pickle.load(open(dump_path, 'rb'))
    qvec = np.array(unpack_sol(sol_dict["q"])).T

    return qvec


# constant parameters
stim_freq = 35                          # stimulation frequency in Hertz
t_duration = round(1 / stim_freq, 6)    # duration of a stimulation pulse
n_window = 6
window_size = 10                        # window sized used for filtering torque data
offset_interval = 10                    # torque data that will be used prior to the start of the contraction
with_residual_torque = False
with_friction = False
with_resistive_torque = True

def main():
    # PROBLEM PARAMETERS TO BE SET EACH TIME
    data_information = [
        # [model, muscle, [node_start, node_end]] <- node_start and end corresponds to a position within the crank cycle
        ["models/legcyc_rectfem_r.bioMod", 'rect_fem_r', [1, -1]],
        # ["models/legcyc_vasmed_r.bioMod", 'vasmed_r', [1, -1]],
        # ["models/legcyc_hams_r.bioMod", 'bflh140_r', [1, -1]],
        ["models/legcyc_rectfem_r.bioMod", 'rect_fem_l', [1, -1]],
        # ["models/legcyc_vasmed_r.bioMod", 'vasmed_l', [1, -1]],
        # ["models/legcyc_hams_r.bioMod", 'bflh140_l', [1, -1]]
    ]
    muscle_params_file = f'Param_iden_cycling_Brazil/opt_sol_results/pickles/cycling_muscles_params.pkl'

    save_results = True         # ATTENTION! Use TRUE onlu if you want to save the identified parameters on the file.
    print_flag = True           # if one wants to have an additional visualization of the torque information
    plot_graphs = False         # for an additional visualization of the OCP resulting plots
    show_animate = False        # for an additional visualization of the OCP resulting animation

    qvec = get_initial_position()
    # from sandbox.cycling_torque import main as get_pd0pdt
    # from FES_leg_param_ident_pdpdt_plot import main as get_pd0pdt
    # a, pd0, pdt = get_pd0pdt(print_flag=print_flag)
    # print(f'using pd0 and pdt average values. pd0 = {pd0} and pdt = {pdt}')

    # todo criar funcao para executar? e ler os parametros de pd0 pdt
    filename = 'Param_iden_cycling_Brazil/opt_sol_results/pickles/pd0pdt_v2.pkl'
    pd0_pdt_data = load_pickle(filename)


    print(f'\nInitial parameters are: {load_pickle(muscle_params_file)}')
    for d, dt in enumerate(data_information):
        # reading data file
        model_path, muscle, qoffset = dt
        init_model_parameters(model_path, muscle_params_file)
        data_file = msr[muscle]
        df = pd.read_csv(f'Param_iden_cycling_Brazil/data_exp/{data_file}', delimiter='\t')
        torque = df['Torque (Nm)'].values.flatten()
        crank_angles = df['Angle(deg)'].values.flatten()

        # retrieve pd0pdt parameters
        a, pd0, pdt = pd0_pdt_data[muscle]
        print(f"Parameters pd0 and pdt for muscle {muscle} {pd0_pdt_data[muscle]}")

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
            print(f'crank_ang: {angle}, converted: {value}')

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
        fes = {'fes_min': pd0, 'fes_max': 1600e-6, 'fes_init': 800e-6, 'fes_sc': 800e-6}
        fes = {'fes_min': pd0, 'fes_max': 1600e-6, 'fes_init': 800e-6, 'fes_sc': 800e-6} if muscle == 'bflh140_r' or muscle == 'bflh140_l' else fes
        problem_params = {
            "f": (0, 10000, 0),  # f_min, f_max, f_init
            "cn": (0.0, 2.0, 0.0),  # cn_min, cn_max, cn_init
            "tau": (0.0, 0.0, 0.0),  # tau_min, tau_max, tau_init
            "fes": fes, # fes_min, fes_max (800+800us), fes_init, fes_sc (300+300us)
            "ascale": (0, 100e3, 5e3),  # ascale_min, ascale_max, ascale_init
            "km": (0.020, 0.800, 0.30130779),  # km_param_min, km_param_max, km_param_init
            "tau_1": (0.010, 0.2, 0.044),  # tau_1_param_min, tau_1_param_max, tau_1_param_init
            "tau_2": (0.001, 0.25, 0.018),  # tau_2_param_min, tau_2_param_max, tau_2_param_init
            "alpha": (-1.0, 2.0, 0.0),  # alpha_min, alpha_max, alpha_init
            "pd0pdt": (a, pd0, pdt),  #
        }

        # --- Prepare the optimal control program for identification of tau1, tau2 and km parameters --- #
        # tau1_ident(ocp_parameters, problem_params, muscle_params_file, save_results, print_flag, show_animate)
        # tau2_km_ident(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, show_animate)
        # a_relation_ident(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, show_animate, print_flag)
        # copy_params_right_to_left(model_path, muscle_params_file, option=1) #option 1 right to left. option 2 left to right
        prediction(ocp_parameters, problem_params, muscle_params_file, save_results, plot_graphs, show_animate)
        # statistic_plots(muscle)
    print(f'\nFinal parameters are: {load_pickle(muscle_params_file)}')


if __name__ == "__main__":
    main()
