"""

"""
import numpy as np
from casadi import MX
import biorbd as brbd

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
    InterpolationType, ParameterList, ParameterObjectiveList, PhaseDynamics, VariableScaling, VariableScalingList,
    PhaseTransitionList, PhaseTransitionFcn, Node, ConstraintFcn,
)

from Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import set_muscle_ascale, set_muscle_tau1_param, \
    set_muscle_tau2_param, set_muscle_km_param, custom_u_bounds
from custom_package.fes_configure_probem import FesDynamicsFcn
from custom_package.fes_objectives import FesObjective
from custom_package.fes_plotting import dump_sol2


def prepare_ocp(
        biorbd_model_path: str,
        q_init: list = [],
        ode_solver: OdeSolverBase = OdeSolver.RK2(),
        phase_period: tuple = 0,
        n_shooting: list = [],
        with_friction: bool = False,
        with_resistive_torque: bool = False,
        with_residual_torque: bool = False,
        pulse_period: float = 0,
        n_pulse_trunc: int = 0,
        target: list = [],
        stim_flag: list = [],
        n_phases: int = 1,
        joint_angle: list = [],
        params: list = [],
        only_parameter_a: bool = False,
        use_sx: bool = False,
        problem_parameters: dict = {},
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    f_min, f_max, f_init = problem_parameters["f"]
    cn_min, cn_max, cn_init = problem_parameters["cn"]
    tau_min, tau_max, tau_init = problem_parameters["tau"]
    fes_min, fes_max, fes_init, fes_sc = problem_parameters["fes"]
    ascale_min, ascale_max, ascale_init = problem_parameters["ascale"]
    km_param_min, km_param_max, km_param_init = problem_parameters["km"]
    tau_1_param_min, tau_1_param_max, tau_1_param_init = problem_parameters["tau_1"]
    tau_2_param_min, tau_2_param_max, tau_2_param_init = problem_parameters["tau_2"]
    alpha_min, alpha_max, alpha_init = problem_parameters["alpha"]
    pd0 = problem_parameters["pd0pdt"][1]
    pdt = problem_parameters["pd0pdt"][2]

    # Define the parameter to optimize as required in the actual version of bioptim
    parameters = ParameterList(use_sx=use_sx)
    parameter_objectives = ParameterObjectiveList()
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    a_scaling = VariableScaling("a_scale_param", np.array([1]))
    parameters.add(
        "a_scale_param",  # The name of the parameter
        set_muscle_ascale,  # The function that modifies the biorbd model
        size=1,  # The number of elements this particular parameter vector has
        scaling=a_scaling,
    )
    parameter_bounds.add("a_scale_param",
                         min_bound=[ascale_min],
                         max_bound=[ascale_max],
                         interpolation=InterpolationType.CONSTANT)
    parameter_init["a_scale_param"] = ascale_init

    if only_parameter_a is False:
        tau_1_scaling = VariableScaling("tau_1_param", np.array([1]))
        parameters.add(
            "tau_1_param",  # The name of the parameter
            set_muscle_tau1_param,  # The function that modifies the biorbd model
            size=1,  # The number of elements this particular parameter vector has
            scaling=tau_1_scaling,
        )
        parameter_bounds.add("tau_1_param",
                             min_bound=[tau_1_param_min],
                             max_bound=[tau_1_param_max],
                             interpolation=InterpolationType.CONSTANT)
        parameter_init["tau_1_param"] = tau_1_param_init

        tau_2_scaling = VariableScaling("tau_2_param", np.array([1]))
        parameters.add(
            "tau_2_param",  # The name of the parameter
            set_muscle_tau2_param,  # The function that modifies the biorbd model
            size=1,  # The number of elements this particular parameter vector has
            scaling=tau_2_scaling,
        )
        parameter_bounds.add("tau_2_param",
                             min_bound=[tau_2_param_min],
                             max_bound=[tau_2_param_max],
                             interpolation=InterpolationType.CONSTANT)
        parameter_init["tau_2_param"] = tau_2_param_init

        km_scaling = VariableScaling("km_param", np.array([1]))
        parameters.add(
            "km_param",  # The name of the parameter
            set_muscle_km_param,  # The function that modifies the biorbd model
            size=1,  # The number of elements this particular parameter vector has
            scaling=km_scaling,
        )
        parameter_bounds.add("km_param",
                             min_bound=[km_param_min],
                             max_bound=[km_param_max],
                             interpolation=InterpolationType.CONSTANT)
        parameter_init["km_param"] = km_param_init

    # Problem parameters and BiorbdModel
    bio_model = tuple(BiorbdModel(biorbd_model_path, parameters=parameters) for _ in range(n_phases))

    # set_muscle_ascale(bio_model[0], 24)
    # set_muscle_tau1_param(bio_model[0], 25)
    # set_muscle_tau2_param(bio_model[0], 26)
    # set_muscle_km_param(bio_model[0], 27)
    # a_scale = bio_model[0].model.muscle(0).characteristics().muscleAscale().to_mx()
    # tau1 = bio_model[0].model.muscle(0).characteristics().muscleDingTau1Param().to_mx()
    # tau2 = bio_model[0].model.muscle(0).characteristics().muscleDingTau2Param().to_mx()
    # km = bio_model[0].model.muscle(0).characteristics().muscleDingKmParam().to_mx()

    # init parameters
    for phase_index in range(n_phases):
        # the different crank positions give different torque offsets given the existence of gravity.
        # In order to track the full torque target but ignoring the offset values, the gravity was removed
        bio_model[phase_index].set_gravity(MX([0, 0, 0]))
        # bio_model[phase_index].model.getGravity().to_mx()
        # set_muscle_ascale(bio_model[phase_index], ascale_min)
        # set_muscle_tau1_param(bio_model[phase_index], params[1])
        # set_muscle_tau2_param(bio_model[phase_index], params[2])
        # set_muscle_km_param(bio_model[phase_index], params[3])
    # as a_scale parameter depends on the knee extension angle, we should get this index for later use in the dynamics
    # bio_model.name_dof -> ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
    joint_idx = tuple(s.to_string() for s in bio_model[0].model.nameDof()).index('knee_r_RotZ')

    # Dynamics
    dynamics = DynamicsList()
    for phase_index in range(n_phases):
        extra_parameters = {'t_phase': pulse_period,
                            'n_pulse_trunc': n_pulse_trunc,
                            'pd0': pd0,
                            'pdt': pdt,
                            'fes_min': fes_min,
                            'muscle_params': params,
                            'joint_index': joint_idx,
                            'validation_flag': False}
        dynamics.add(FesDynamicsFcn.FES_PARAM_IDENT,
                     expand=True,
                     with_residual_torque=with_residual_torque,
                     with_contact=True,
                     with_friction=with_friction,
                     with_resistive_torque=with_resistive_torque,
                     only_parameter_a=only_parameter_a,
                     identification_flag=True,
                     phase_dynamics=phase_dynamics,
                     **extra_parameters)

    # Add constraints and objectives
    constraints = ConstraintList()
    objective_functions = ObjectiveList()

    for i in range(n_phases):
        tau_target = np.array(target[i]) * np.array(stim_flag[i])
        max_torque = np.max(tau_target)
        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e1, phase=i,
        #                         quadratic=True)
        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e1, phase=i,
        #                         quadratic=True)
        print(
            f'Max torque with stim on phase {i}, with nodes={n_shooting[i]} (crank angle: {joint_angle[i]}): {max_torque} Nm')

        mask = stim_flag[i] == 1
        max_value = np.max(tau_target[mask])

        for node, node_target in enumerate(tau_target):
            weight = 1e2
            weight = 1e5 if stim_flag[i][node] == 1 else weight
            # weight = 1e6 if node_target == max_value else weight
            # weight = 1e4
            objective_functions.add(
                FesObjective.custom_func_maximize_crank_torque,
                custom_type=ObjectiveFcn.Mayer,
                node=node,
                index=0,
                target=node_target,
                quadratic=True,
                weight=weight,
                phase=i,
            )
    constraints.add(ConstraintFcn.TRACK_STATE,
                    key="qdot",
                    node=Node.ALL,
                    index=0,
                    target=0,
                    # weight=weight / 10,
                    weight=1e3,
                    phase=i,
                    )

    # Path constraint
    nb_tau = bio_model[0].nb_tau
    nb_muscles = bio_model[0].nb_muscles
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    for phase_index in range(n_phases):
        x_bounds.add("q", bounds=bio_model[phase_index].bounds_from_ranges("q"), phase=phase_index)
        x_bounds[phase_index]["q"].min[:, [0]] = q_init[phase_index]
        x_bounds[phase_index]["q"].max[:, [0]] = q_init[phase_index]
        x_bounds.add("qdot", bounds=bio_model[phase_index].bounds_from_ranges("qdot"), phase=phase_index)
        x_bounds.add("cn",
                     min_bound=[cn_min] * nb_muscles,
                     max_bound=[cn_max] * nb_muscles,
                     phase=phase_index)
        x_bounds.add("f",
                     min_bound=[f_min] * nb_muscles,
                     max_bound=[f_max] * nb_muscles,
                     phase=phase_index)
        for m in range(bio_model[phase_index].nb_muscles):
            x_bounds.add(f"alpha_{m}",
                         min_bound=[alpha_min] * n_pulse_trunc,
                         max_bound=[alpha_max] * n_pulse_trunc,
                         phase=phase_index)
        for m in range(bio_model[phase_index].nb_muscles):
            x_bounds[phase_index][f"alpha_{m}"][:, [0]] = alpha_init
        x_bounds[phase_index]["cn"][:, [0]] = 0
        x_bounds[phase_index]["f"][:, [0]] = 0
        x_bounds[phase_index]["qdot"][:, [0]] = 0.0  # no velocities at the start


    for m in range(nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc
    x_init["q"] = q_init[phase_index]

    # Define control path constraint
    u_bounds = BoundsList()
    # gear, pedal_r, pedal_l, knee_r, knee_l, hip_r, hip_l
    for i in range(n_phases):
        # negative torque and positive torque would correspond to the flexor and extensor muscle activation
        if with_resistive_torque:
            u_bounds.add("resistive_torque", min_bound=[-50], max_bound=[50], phase=i)  # torque positive at the gear
            # u_bounds[i]["resistive_torque"][:, [0]] = 0  # remove, otherwise, due to constraints, will enter in rest. mode
        # u_bounds[i]["resistive_torque"][:, [0]] = 0
        if with_residual_torque:
            u_bounds.add("tau", min_bound=[tau_min] * nb_tau, max_bound=[tau_max] * nb_tau, phase=i)
            u_bounds[i]["tau"][:, [0]] = 0
        extra_params_u = {"n_channels": 1, "stim_flag": stim_flag[i], "fes_sc": fes_sc, "fes_min": fes_min}
        u_bounds.add(
            "pw",
            min_bound=custom_u_bounds,
            max_bound=custom_u_bounds,
            interpolation=InterpolationType.CUSTOM,
            **extra_params_u,
            phase=i,
        )

    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    u_scaling["pw"] = [fes_max]

    phase_transitions = PhaseTransitionList()
    for i in range(n_phases - 1):
        phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

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
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        n_threads=16,
        phase_transitions=phase_transitions,
    )

    # import biorbd as brbd
    # numerical_model = brbd.Model(biorbd_model_path)
    # ocp.add_plot("Joint Torques",
    #              lambda t0, phases_dt, node_idx, x, u, p, a, d: fes_plot_tau(x, u, numerical_model),
    #              plot_type=PlotType.PLOT)

    return ocp


def tau1_tau2_km_ident(ocp_parameters,
                       problem_parameters,
                       filename,
                       save_results=False,
                       plot_graphs=False,
                       show_animate=False):

    model_path = ocp_parameters['biorbd_model_path']
    q_init_list = ocp_parameters['q_init']
    with_friction = ocp_parameters['with_friction']
    with_resistive_torque = ocp_parameters['with_resistive_torque']
    with_residual_torque = ocp_parameters['with_residual_torque']
    t_duration = ocp_parameters['pulse_period']  # Attention: duration of each pulse
    times = ocp_parameters['phase_period']  # Attention: duration of each phase of the problem
    nodes = ocp_parameters['n_shooting']
    n_window = ocp_parameters['n_pulse_trunc']
    tau_target = ocp_parameters['target']
    stim_flag = ocp_parameters['stim_flag']
    n_phases = ocp_parameters['n_phases']
    knee_ang = ocp_parameters['joint_angle']
    mdl_param = ocp_parameters['params']
    only_parameter_a = ocp_parameters['only_parameter_a']
    muscle = ocp_parameters['muscle']
    stim_freq = ocp_parameters['stim_freq']

    # --- Prepare the optimal control program --- #
    print(f'\nRunning identification tau1, tau2, km for {muscle} using {model_path}...')
    ocp = prepare_ocp(biorbd_model_path=model_path,
                      q_init=q_init_list,
                      with_friction=with_friction,
                      with_residual_torque=with_residual_torque,
                      with_resistive_torque=with_resistive_torque,
                      pulse_period=t_duration,  # Attention: duration of each pulse
                      phase_period=times,  # Attention: duration of each phase of the problem
                      n_shooting=nodes,
                      n_pulse_trunc=n_window,
                      target=tau_target,
                      stim_flag=stim_flag,
                      n_phases=n_phases,
                      joint_angle=knee_ang,
                      params=mdl_param,
                      only_parameter_a=only_parameter_a,
                      problem_parameters=problem_parameters)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000, _tol=1e-4))

    a_scale_param = float(sol.parameters["a_scale_param"][0])
    print(f"Optimized a_scale_param: {a_scale_param}")
    parameters = [a_scale_param]

    if only_parameter_a is False:
        tau_1_param = float(sol.parameters["tau_1_param"][0])
        print(f"Optimized tau_1_param: {tau_1_param}")
        tau_2_param = float(sol.parameters["tau_2_param"][0])
        print(f"Optimized tau_2_param: {tau_2_param}")
        km_param = float(sol.parameters["km_param"][0])
        print(f"Optimized km_param: {km_param}")
        parameters = [a_scale_param, tau_1_param, tau_2_param, km_param]
    print(f'Identified Parameters = {parameters}')

    if save_results:
        from Param_iden_cycling.ocp_param_indent.common.params_handler import read_params, update_parameters_value
        num_model = brbd.Model(model_path)
        mus_names = num_model.muscleNames()
        muscle_params = read_params(model_path, filename)
        # for idx, mn in enumerate(mus_names):
        #     m = mn.to_string()
        print(f'Before we had the following parameters for muscle {muscle}: {muscle_params[muscle]}')
        muscle_params[muscle][1] = tau_1_param
        muscle_params[muscle][2] = tau_2_param
        muscle_params[muscle][3] = km_param
        update_parameters_value(filename, muscle, muscle_params[muscle])
        muscle_params = read_params(model_path, filename)
        print(f'Now we have the following parameters for muscle {muscle}: {muscle_params[muscle]}')
    # if save_results:
    #     num_model = brbd.Model(model_path)
    #     filename = f'opt_sol_results/pickles/{muscle}_results.pkl'
    #     dump_sol2(num_model, ocp, sol, filename, len(nodes), parameters, n_phases, stim_freq)

    # --- Show results --- #
    # sol.print_cost()
    if plot_graphs:
        sol.graphs()
    if show_animate:
        sol.animate()

#
# if __name__ == "__main__":
#     main()
