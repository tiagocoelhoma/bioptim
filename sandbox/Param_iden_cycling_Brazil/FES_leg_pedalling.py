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

from Param_iden_cycling_Brazil.FES_leg_pedalling_calculate_power import calculate_power, plot_power
from custom_package.fes_configure_probem import FesDynamicsFcn
from custom_package.fes_objectives import FesObjective
import pickle, os
from custom_package.fes_plotting import unpack_sol
from leg_pedalling_torque_driven import main as main_torque


def get_joint_indexes(bio_model):
    # [a.to_string() for a in bio_model.model.nameDof()]
    # ['Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ']
    # nb_mus = bio_model.model.nbMuscles()
    muscles = [a.to_string() for a in bio_model.model.muscleNames()]
    right_muscles = [m for m in muscles if m.endswith('_r')]
    left_muscles = [m for m in muscles if m.endswith('_l')]
    knee_r_idx = tuple(s.to_string() for s in bio_model.model.nameDof()).index('knee_r_RotZ')
    knee_l_idx = tuple(s.to_string() for s in bio_model.model.nameDof()).index('knee_l_RotZ')

    return [knee_r_idx] * len(right_muscles) + [knee_l_idx] * len(left_muscles)


def prepare_ocp(
        biorbd_model_path: str,
        q_init: np.array,
        dq_init: np.array,
        phase_period: float = 0,
        n_shooting: int = 0,
        with_friction: bool = False,
        with_torque: bool = False,
        max_joint_torque: int = 0,
        n_pulse_trunc: int = 0,
        target_power: int = 0,
        kni_pos_value_at: int = 0,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver: OdeSolverBase = OdeSolver.RK2(),
        muscle_params: dict = {},
        muscle_params_pd0_pdt: dict = {},
        target_velocity: float = 0,
        friction_coef_mu: float = 0,
        friction_coef_b: float = 0,
) -> OptimalControlProgram:

    # Problem parameters and BiorbdModel
    bio_model = BiorbdModel(biorbd_model_path)
    final_time = phase_period * n_shooting
    numerical_model = brbd.Model(biorbd_model_path)
    joint_idx = [s.to_string() for s in numerical_model.nameDof()]
    idx_crank = joint_idx.index('Gear_RotZ')
    joint_idx = get_joint_indexes(bio_model)

    muscle_names = [s.to_string() for s in numerical_model.muscleNames()]
    pd0pdt_param, fes_min_lst = [], []
    pd0, pdt = 86e-6, 0.000138
    # need to be reviewed
    a = 1
    pd0 = 0.00011785  # average value
    pdt = 0.00086029  # average value
    fes_min = pd0
    print(f'using pd0 and pdt average values. pd0 = {pd0} and pdt = {pdt}')
    # for m, mus_name in enumerate(muscle_names):
    #     # pd0pdt_param.append([muscle_params_pd0_pdt[mus_name][1], muscle_params_pd0_pdt[mus_name][2]])
    #     # fes_min_lst.append(muscle_params_pd0_pdt[mus_name][1])
    #     pd0 = pd0 + muscle_params_pd0_pdt[mus_name][1]
    #     pdt = pdt + muscle_params_pd0_pdt[mus_name][2]
    # pd0 = pd0 / numerical_model.nbMuscles()
    # pdt = pdt / numerical_model.nbMuscles()

    # Dynamics
    dynamics = DynamicsList()
    extra_parameters = {'t_phase': phase_period,
                        'n_pulse_trunc': n_pulse_trunc,
                        'pd_offset': pd_offset,
                        'joint_index': joint_idx,
                        'muscle_params': muscle_params,
                        # 'muscle_params_pd0_pdt': muscle_params_pd0_pdt,
                        'muscle_params_pd0_pdt': [pd0, pdt],
                        'friction_coef_mu': friction_coef_mu,
                        'friction_coef_b': friction_coef_b,
                        'idx_crank': idx_crank}
    dynamics.add(FesDynamicsFcn.FES_CYCLING,
                 expand=True,
                 with_torque=with_torque,
                 with_contact=True,
                 fes_driven_task_flag=True,
                 with_friction=with_friction,
                 phase_dynamics=phase_dynamics,
                 **extra_parameters)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="pw", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_CONTROL, node=Node.ALL, key="pw", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="cn", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="f", weight=1e0)
    # if with_friction:
    #     objective_functions.add(
    #         # FesObjective.custom_func_maximize_crank_torque,
    #         FesObjective.custom_func_maximize_crank_power,
    #         custom_type=ObjectiveFcn.Mayer,
    #         node=Node.ALL,
    #         index=idx_crank,
    #         target=target_power,
    #         quadratic=True,
    #         weight=1e2,
    #         **extra_parameters,
    #     )

    muscle_names = [s.to_string() for s in numerical_model.muscleNames()]
    if 'rect_fem_r' in muscle_names:
        objective_functions.add(
            FesObjective.custom_func_minimize_cocontraction,
            custom_type=ObjectiveFcn.Lagrange,
            # custom_type=ObjectiveFcn.Mayer,
            # node=Node.ALL,
            target=0,
            mus_num_1=muscle_names.index('rect_fem_r'),
            mus_num_2=muscle_names.index('rect_fem_l'),
            quadratic=True,
            weight=1e2,
        )
    if 'vasmed_r' in muscle_names:
        objective_functions.add(
            FesObjective.custom_func_minimize_cocontraction,
            custom_type=ObjectiveFcn.Lagrange,
            # custom_type=ObjectiveFcn.Mayer,
            # node=Node.ALL,
            target=0,
            mus_num_1=muscle_names.index('vasmed_r'),
            mus_num_2=muscle_names.index('vasmed_l'),
            quadratic=True,
            weight=1e2,
        )
    if 'bflh140_r' in muscle_names:
        objective_functions.add(
            FesObjective.custom_func_minimize_cocontraction,
            custom_type=ObjectiveFcn.Lagrange,
            # custom_type=ObjectiveFcn.Mayer,
            # node=Node.ALL,
            target=0,
            mus_num_1=muscle_names.index('bflh140_r'),
            mus_num_2=muscle_names.index('bflh140_l'),
            quadratic=True,
            weight=1e2,
        )
    # todo: verify if Lagrange objective is better while trying to achieve lower co-contraction
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        # ObjectiveFcn.Mayer.TRACK_STATE,
        # node=Node.ALL,
        key="qdot",
        index=idx_crank,
        target= target_velocity,
        weight=1e3,
    )

    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"))
    x_bounds["qdot"].min[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    x_bounds["qdot"].max[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    # setting the initial position for the legs and gear (starting from a position different from 0 or pi)
    x_bounds["q"].min[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    x_bounds["q"].max[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    x_bounds.add("cn",
                 min_bound=[cn_min] * bio_model.nb_muscles,
                 max_bound=[cn_max] * bio_model.nb_muscles)
    x_bounds.add("f",
                 min_bound=[f_min] * bio_model.nb_muscles,
                 max_bound=[f_max] * bio_model.nb_muscles)
    for m in range(bio_model.nb_muscles):
        x_bounds.add(f"alpha_{m}",
                     min_bound=[alpha_min] * n_pulse_trunc,
                     max_bound=[alpha_max] * n_pulse_trunc)
        x_bounds[f"alpha_{m}"][:, [0]] = alpha_init
    # set to a steady position in the beginning with no forces (starting node @ phase 0)
    x_bounds["f"][:, [0]] = 0
    x_bounds["cn"][:, [0]] = 0

    u_bounds = BoundsList()
    # gear, pedal_r, pedal_l, knee_r, knee_l, hip_r, hip_l
    # negative torque and positive torque would correspond to the flexor and extensor muscle activation
    if with_torque:
        u_bounds["tau"] = (
            [-max_joint_torque, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0])
        u_bounds["tau"].min[:, :] = 0
        u_bounds["tau"].max[:, :] = 0
        u_bounds["tau"].min[[0], [0]] = -max_joint_torque

    u_bounds.add("pw",
                 min_bound=[fes_min] * bio_model.nb_muscles,
                 # min_bound=[pd0] * bio_model.nb_muscles,
                 # min_bound = fes_min_lst,
                 max_bound=[fes_max] * bio_model.nb_muscles)
    u_bounds["pw"].max[:,[0]] = fes_min
    # u_bounds["pw"].max[:, [0]] = pd0
    x_init = InitialGuessList()
    x_init.add('q', q_init, InterpolationType.EACH_FRAME)
    x_init.add('qdot', dq_init, InterpolationType.EACH_FRAME)
    for m in range(bio_model.nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc

    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    u_scaling["pw"] = [fes_max] * bio_model.nb_muscles

    ocp = OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        u_scaling=u_scaling,
        ode_solver=ode_solver,
        n_threads=16,
    )
    return ocp


# PROBLEM PARAMETERS
f_min, f_max, f_init = 0, 5000, 0
cn_min, cn_max, cn_init = 0.0, 1.5, 0.0
tau_min, tau_max, tau_init = -0.00, 0.00, 0.0
fes_min, fes_max, fes_init = 86e-6, 1600e-6, 86e-6
alpha_min, alpha_max, alpha_init = -1.0, 2.0, 0.0
pd_offset = fes_min


def main():

    folder_path="opt_sol_results/fes_leg_pedalling/"
    n_window=6      # leave at 6 otherwise change dynamics (it was done to remove the loop in the dynamics)
    n_nodes=75
    stim_freq=35
    target_velocity_rpm=41.92
    target_power=25.0
    max_joint_torque=50.0
    with_torque=True
    with_friction=True
    crank_opt_flag=False
    kni_pos_value_at = 10
    kni_pos_value_at = 4 if crank_opt_flag else kni_pos_value_at
    friction_coef_b=0.78
    friction_coef_mu=0.05

    target_v = (-(target_velocity_rpm * 2 * np.pi) / 60)  # rpm to rad/s
    torque_crank = target_power / target_v
    friction_coef_b = round(torque_crank / target_v, 4)

    model_path = "../models/leg_model_cyclic_muscle_inversed_root_3.bioMod"
    model_path = "models/leg_model_cyclic_muscle_inversed_root_opt_only_seg.bioMod" if crank_opt_flag else model_path

    # --- Prepare the optimal control program --- #
    phase_period = round(1 / stim_freq, 3)
    duration = phase_period * n_nodes
    # get cycling kinematic from torque_driven example to start at a velocity different from 0 rpm
    if not crank_opt_flag:
        dump_name = \
            f"ocp_{round(duration, 2)}s_{n_nodes}n_{target_velocity_rpm}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}"
    else:
        dump_name = \
            f"ocp_{round(duration, 2)}s_{n_nodes}n_{target_velocity_rpm}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}_opt_crank"
    from Param_iden_cycling.ocp_param_indent.common.params_handler import read_params
    num_model = brbd.Model(model_path)
    filename = f'opt_sol_results/pickles/cycling_muscles_params.pkl'
    muscle_params = read_params(model_path, filename)

    for m, muscle_name in enumerate(muscle_params):
        vec = muscle_params[muscle_name][0]
        if len(vec) < 6:
            padded_vec = np.pad(vec, (6 - len(vec), 0), mode='constant', constant_values=0)
            muscle_params[muscle_name][0] = padded_vec.tolist()

    filename = f'opt_sol_results/pickles/pd0ptt.pkl'
    muscle_params_pd0_pdt = read_params(model_path, filename)

    # Intializing kinematics
    dump_path = 'opt_sol_results/kinematics/' + dump_name
    n_mus = num_model.nbMuscles()
    if not os.path.isfile(dump_path):
        print(f"Intializing kinematics...")
        main_torque(model_path, dump_path, duration, n_nodes, target_v, target_power, crank_opt_flag, False,
             friction_coef_mu, friction_coef_b)
    sol_dict = pickle.load(open(dump_path, 'rb'))
    qvec = np.array(unpack_sol(sol_dict["q"])).T
    dqvec = np.array(unpack_sol(sol_dict["qdot"])).T


    if not crank_opt_flag:
        filename = f'friction_{round(duration, 2)}s_{n_nodes}n_{target_velocity_rpm}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}'
    else:
        filename = f'friction_{round(duration, 2)}s_{n_nodes}n_{target_velocity_rpm}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}_opt_crank'
    to_be_solved = True
    if to_be_solved:
        print(f'Running fes_pedalling, f{stim_freq}_n{n_nodes}_a{n_window}_m{n_mus}')
        ocp = prepare_ocp(biorbd_model_path=model_path,
                          q_init=qvec,
                          dq_init=dqvec,
                          max_joint_torque=max_joint_torque,
                          with_torque=with_torque,
                          with_friction=with_friction,
                          phase_period=phase_period,
                          n_shooting=n_nodes,
                          n_pulse_trunc=n_window,
                          target_power=target_power,
                          kni_pos_value_at=kni_pos_value_at,
                          muscle_params=muscle_params,
                          muscle_params_pd0_pdt=muscle_params_pd0_pdt,
                          target_velocity=target_v,
                          friction_coef_b=friction_coef_b,
                          friction_coef_mu=friction_coef_mu)

        # --- Solve the program --- #
        sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=5500, _tol=1e-3))
        # sol.graphs()
        # sol.animate()

        # save data
        states = sol.stepwise_states()
        controls = sol.stepwise_controls()
        q_sol, qdot_sol, cn_sol, f_sol = states["q"], states["qdot"], states["cn"], states["f"]
        pw_sol = controls["pw"]
        tau_joints = controls["tau"] if "tau" in controls else None
        tau_res_sol = controls["resistive_torque"] if "resistive_torque" in controls else None
        data = {
            "q": q_sol, "qdot": qdot_sol, "cn": cn_sol, "f": f_sol,
            "pw": pw_sol, "tau_resistive": tau_res_sol, "tau_joints": tau_joints,
            "time": sol.stepwise_time(), "n_iterations": sol.iterations, "final_time": ocp.phase_time,
            "solver_time": sol.solver_time_to_optimize, "real_time": sol.real_time_to_optimize,
            "n_node": n_nodes, "freq": stim_freq, "n_trunc": n_window,
            "target_vel": target_velocity_rpm, 'target_pwr': target_power,
        }
        with open(f'{folder_path}'+filename+'.pkl', "wb") as file:
            del sol.ocp
            pickle.dump(data, file)

    # calculate power with a torque driven example (inverse dynamics with contact points is not implemented yet)
    dump_path = f'opt_sol_results/fes_leg_pedalling/'
    print(f'Calculating power {dump_path}{filename}')
    to_be_calculated = True
    if to_be_calculated:
        calculate_power(dump_path,
                        model_path,
                        # 'models/leg_model_cyclic_muscle_inversed_root_3_gravity_0.bioMod',
                        filename,
                        stim_freq=35,
                        friction_coef_mu=friction_coef_mu,
                        friction_coef_b=friction_coef_b,
                        print_res=True,
                        set_gravity_to_zero=True)
    plot_power(dump_path,
               num_model,
               filename,
               friction_coef_mu=friction_coef_mu,
               friction_coef_b=friction_coef_b,
               set_gravity_to_zero=False)     # plot power


if __name__ == "__main__":
    main()

    # solver.set_tol(1e-5)
    # solver.set_dual_inf_tol(1e-4)
    # solver.set_constr_viol_tol(1e-4)
    # solver.set_compl_inf_tol(1e-4)
    # solver.set_maximum_iterations(2000)
    # solver.set_hessian_approximation("limited-memory")
    # solver.set_bound_frac(1e-8)
    # solver.set_bound_push(1e-8)
    # solver.set_nlp_scaling_method("none")
    # solver.set_initialization_options(1e-4)
