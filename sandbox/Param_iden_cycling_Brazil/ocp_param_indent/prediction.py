"""

"""
import numpy as np
import biorbd as brbd
from casadi import MX
import matplotlib.pyplot as plt

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    InitialGuessList,
    InterpolationType, PhaseDynamics, VariableScalingList, Node, ConstraintFcn
)

from Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import custom_u_bounds
from custom_package.fes_configure_probem import FesDynamicsFcn
from Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import update_parameters_value


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
        q_init: list = [],
        phase_period: tuple = 0,
        n_shooting: list = [],
        with_friction: bool = False,
        with_resistive_torque: bool = False,
        with_residual_torque: bool = False,
        pulse_period: float = 0,
        n_pulse_trunc: int = 0,
        target: list = [],
        stim_flag: list = [],
        params: list = [],
        problem_parameters: dict = {},
        ode_solver: OdeSolverBase = OdeSolver.RK2(),
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:

    f_min, f_max, f_init = problem_parameters["f"]
    cn_min, cn_max, cn_init = problem_parameters["cn"]
    tau_min, tau_max, tau_init = problem_parameters["tau"]
    fes_min, fes_max, fes_init, fes_sc = problem_parameters["fes"]
    alpha_min, alpha_max, alpha_init = problem_parameters["alpha"]
    pd0 = problem_parameters["pd0pdt"][1]
    pdt = problem_parameters["pd0pdt"][2]

    # Problem parameters and BiorbdModel
    bio_model = BiorbdModel(biorbd_model_path)
    # the different crank positions give different torque offsets given the existence of gravity.
    # In order to track the full torque target but ignoring the offset values, the gravity was removed
    bio_model.set_gravity(MX([0, 0, 0]))

    numerical_model = brbd.Model(biorbd_model_path)
    joint_idx = [s.to_string() for s in numerical_model.nameDof()]
    idx_crank = joint_idx.index('Gear_RotZ')
    joint_idx = get_joint_indexes(bio_model)

    # Dynamics
    dynamics = DynamicsList()
    extra_parameters = {'t_phase': pulse_period,
                        'n_pulse_trunc': n_pulse_trunc,
                        'pd_offset': pd0,
                        'pd0': pd0,
                        'pdt': pdt,
                        'joint_index': joint_idx,
                        'muscle_params': params,
                        # 'muscle_params_pd0_pdt': [pd0, pdt],
                        'friction_coef_mu': 0,
                        'knee_angle': q_init[2],
                        'friction_coef_b': 0,
                        'idx_crank': idx_crank}
    dynamics.add(FesDynamicsFcn.FES_PARAM_PREDICTION,
    # dynamics.add(FesDynamicsFcn.FES_CYCLING,
                 expand=True,
                 with_torque=with_residual_torque,
                 with_contact=True,
                 with_friction=with_friction,
                 with_resistive_torque=with_resistive_torque,
                 phase_dynamics=phase_dynamics,
                 **extra_parameters)

    # Add constraints and objectives
    constraints = ConstraintList()
    objective_functions = ObjectiveList()

    tau_target = np.array(target) * np.array(stim_flag)
    max_torque_index = np.argmax(target * stim_flag)
    print(f'Max Torque with stim on: {target[max_torque_index]}')

    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0)
    # for node, node_target in enumerate(tau_target):
    #     weight = 1e3
    #     weight = 1e6 if stim_flag[node] == 1 else weight
    #     objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e2)
    #     # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e4)
    #     # bio_model.name_dof -> ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
    #     # if stim_flag[node] == 1:
    #     # objective_functions.add(
    #     #     FesObjective.custom_func_maximize_crank_torque,
    #     #     # FesObjective.custom_func_track_torque,
    #     #     custom_type=ObjectiveFcn.Mayer,
    #     #     node=node,
    #     #     index=0,
    #     #     target=node_target,
    #     #     quadratic=True,
    #     #     weight=1e4,
    #     # )
    #     # objective_functions.add(
    #     #     ObjectiveFcn.Mayer.TRACK_STATE,
    #     #     key="qdot",
    #     #     index=0,  # must be the gear (attention)
    #     #     node=node,
    #     #     target=0,
    #     #     # target velocity at gear crank. The index in the objective func should be set accordingly
    #     #     weight=1e4,
    #     # )
    constraints.add(ConstraintFcn.TRACK_STATE,
                    key="qdot",
                    node=Node.ALL,
                    index=0,
                    target=0,
                    weight=1e4,
                    )

    # Path constraint
    nb_tau = bio_model.nb_tau
    nb_muscles = bio_model.nb_muscles
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds["q"].min[:, [0]] = q_init
    x_bounds["q"].max[:, [0]] = q_init
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"))
    x_bounds.add("cn", min_bound=[cn_min] * nb_muscles, max_bound=[cn_max] * nb_muscles)
    x_bounds.add("f", min_bound=[f_min] * nb_muscles, max_bound=[f_max] * nb_muscles)
    for m in range(bio_model.nb_muscles):
        x_bounds.add(f"alpha_{m}", min_bound=[alpha_min] * n_pulse_trunc, max_bound=[alpha_max] * n_pulse_trunc)
    for m in range(bio_model.nb_muscles):
        x_bounds[f"alpha_{m}"][:, [0]] = alpha_init
    x_bounds["cn"][:, [0]] = 0
    x_bounds["f"][:, [0]] = 0
    x_bounds["qdot"][:, [0]] = 0.0  # no velocities at the start
    # x_bounds["qdot"].min[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still
    # x_bounds["qdot"].max[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still

    x_init = InitialGuessList()
    for m in range(nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc
    x_init["q"] = q_init

    # Define control path constraint
    u_bounds = BoundsList()
    # gear, pedal_r, pedal_l, knee_r, knee_l, hip_r, hip_l
    # negative torque and positive torque would correspond to the flexor and extensor muscle activation
    if with_resistive_torque:
        u_bounds.add("resistive_torque", min_bound=[-50], max_bound=[50]) # torque positive at the gear
    if with_residual_torque:
        u_bounds.add("tau", min_bound=[tau_min] * nb_tau, max_bound=[tau_max] * nb_tau)
        u_bounds["tau"][:, [0]] = 0
    u_bounds["resistive_torque"][:, [0]] = 0

    extra_params_u = {"n_channels": 1, "stim_flag": stim_flag, "fes_sc": fes_sc, "fes_min": fes_min}
    u_bounds.add(
        "pw",
        min_bound=custom_u_bounds,
        max_bound=custom_u_bounds,
        interpolation=InterpolationType.CUSTOM,
        **extra_params_u,
    )

    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    u_scaling["pw"] = [fes_max]

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_period,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        u_scaling=u_scaling,
        ode_solver=ode_solver,
        n_threads=16,
    )
    return ocp


def prediction(ocp_parameters,
                     problem_parameters,
                     filename,
                     save_results=False,
                     plot_graphs=False,
                     print_flag=False,
                     show_animate=False):

    model_path = ocp_parameters['biorbd_model_path']
    q_init_list = ocp_parameters['q_init']
    with_resistive_torque = ocp_parameters['with_resistive_torque']
    with_friction = ocp_parameters['with_friction']
    with_residual_torque = ocp_parameters['with_residual_torque']
    t_duration = ocp_parameters['pulse_period']  # Attention: duration of each pulse
    times = ocp_parameters['phase_period']  # Attention: duration of each phase of the problem
    nodes = ocp_parameters['n_shooting']
    n_window = ocp_parameters['n_pulse_trunc']
    tau_target = ocp_parameters['target']
    stim_flag = ocp_parameters['stim_flag']
    n_phases = ocp_parameters['n_phases']
    crank_ang = ocp_parameters['joint_angle']
    mdl_param = ocp_parameters['params']
    muscle = ocp_parameters['muscle']
    stim_freq = ocp_parameters['stim_freq']


    # --- Prepare the optimal control program --- #
    from Param_iden_cycling.ocp_param_indent.common.params_handler import read_params
    mdl_param = read_params(model_path, filename)
    predicted_torques = []
    knee_angle, crank_angle = [], []

    print(f'\nRunning identification of the A function for {muscle} using {model_path} with parameters {read_params(model_path, filename)}')
    for i in range(n_phases):
        to_be_solved = True
        if to_be_solved:
            ocp = prepare_ocp(biorbd_model_path=model_path,
                              q_init=q_init_list[i],
                              with_friction=with_friction,
                              with_resistive_torque=with_resistive_torque,
                              with_residual_torque=with_residual_torque,
                              pulse_period=t_duration,     # Attention: duration of each pulse
                              phase_period=times[i],       # Attention: duration of each phase of the problem
                              n_shooting=nodes[i],
                              n_pulse_trunc=n_window,
                              target=tau_target[i],
                              stim_flag=stim_flag[i],
                              params=mdl_param,
                              problem_parameters=problem_parameters)

            # --- Solve the program --- #
            sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000, _tol=1e-4))

            # --- Show results --- #
            # sol.print_cost()
            if plot_graphs:
                sol.graphs()
            if show_animate:
                sol.animate()
            print('test')
            # model = brbd.Model(model_path)
            # states = sol.stepwise_states()
            # time = sol.stepwise_time(to_merge=SolutionMerge.ALL)
            # nb_nodes = len(states["q"])
            # nb_int = np.array(states["q"][0]).shape[1] - 1
            # predicted_torque = np.zeros(((nb_nodes - 1) * nb_int))
            # time_flat = np.zeros(((nb_nodes - 1) * nb_int))
            predicted_torques.append(np.concatenate(sol.stepwise_controls()["resistive_torque"]))
        knee_angle.append(float(q_init_list[i][2]))
    #     crank_angle.append(float(q_init_list[i][0]))
    # ocp_parameters['crank_angle'] = crank_angle
    ocp_parameters['predicted_torques'] = predicted_torques

    # save data
    update_parameters_value('opt_sol_results/pickles/prediction.pkl', muscle, ocp_parameters)

    if print_flag:
        plt.figure(figsize=(10, 5))

        for i, torque in enumerate(tau_target):
            center = crank_ang[i]
            n = len(torque)
            x = np.arange(n) - n // 2 + center  # center the curve at the knee angle
            plt.plot(x, torque, label=f'knee = {center}°')
            plt.plot(x, predicted_torques[i], label=f'knee = {center}°', linestyle='dotted')

        plt.xlabel("Knee angle (°)")
        plt.ylabel("Torque (Nm)")
        plt.title("Torque curves centered at their respective knee angles")
        plt.legend()
        plt.grid(True)
        plt.xticks(crank_ang)  # show only the knee angles if you want it cleaner
        plt.tight_layout()
        plt.show()

