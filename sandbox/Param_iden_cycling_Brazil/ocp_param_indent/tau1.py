"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""

import numpy as np
from casadi import MX
import biorbd as brbd

from bioptim import (
    OptimalControlProgram,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    DynamicsOptionsList,
    PhaseDynamics,
    DynamicsOptions, ConstraintList, ObjectiveList, VariableScalingList, Node, ContactType, ParameterList,
    ParameterObjectiveList, VariableScaling, InterpolationType, ConstraintFcn, PlotType,
    BiorbdModel, PhaseTransitionList, PhaseTransitionFcn,
)
from sandbox.FES_leg_superimpose_marker import n_shooting

# import the custom model
from sandbox.custom_package.fes_dynamics import ParamIdentDynamics, set_muscle_ascale, set_muscle_tau2_param, \
    set_muscle_km_param, set_muscle_tau1_param, ParamTau1IdentDynamics
from sandbox.custom_package.fes_objectives import FesObjective

from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import read_params, \
    update_parameters_value, get_coefficients_sqrd

from casadi import MX
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import custom_u_bounds



def prepare_ocp(
    biorbd_model_path: str,
    q_init: list = [],
    phase_period: tuple = 0,
    n_shooting: list = [],
    with_residual_torque: bool = False,
    pulse_period: float = 0,
    n_pulse_trunc: int = 0,
    target: list = [],
    stim_flag: list = [],
    n_phases: int = 1,
    joint_angle: list = [],
    params: list = [],
    with_torque: bool = False,
    with_friction: bool = True,
    with_resistive_torque: bool = True,
    only_parameter_a: bool = False,
    problem_parameters: dict = {},
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    n_threads: int = 16,
    expand_dynamics: bool = True,
    use_sx: bool = False,
    **extra_params,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    f_min, f_max, f_init = problem_parameters["f"]
    cn_min, cn_max, cn_init = problem_parameters["cn"]
    tau_min, tau_max, tau_init = problem_parameters["tau"]
    # fes_min, fes_max, fes_init, fes_sc = problem_parameters["fes"]
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

    # Add constraints and objectives (to be included in the following loop)
    objective_functions = ObjectiveList()
    constraints = ConstraintList()
    dynamics = DynamicsOptionsList()

    model = ([
        ParamTau1IdentDynamics(biorbd_model_path,
                           parameters=parameters,
                           with_contact=True,
                           with_residual_torque=False,
                           with_resistive_torque=with_resistive_torque,
                           **extra_params)
    ] * n_phases)

    # This should be a multiphase problem
    for i in range(n_phases):
        model[i].set_gravity(MX([0, 0, 0]))
        # bio_model[phase_index].model.getGravity().to_mx()
        # set_muscle_ascale(model, ascale_max)
        # set_muscle_tau1_param(model, params[1])
        # set_muscle_tau2_param(model, params[2])
        # set_muscle_km_param(model, params[3])
        # get_muscle_ascale(model)
        # model.model.muscle(0).characteristics().muscleDingTau1Param().to_mx()

        # as a_scale parameter depends on the knee extension angle, we should get this index for later use in the dynamics
        # bio_model.name_dof -> ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
        idx_crank = model[i].name_dof.index('Gear_RotZ')
        muscles = model[i].muscle_names
        right_muscles = [m for m in muscles if m.endswith('_r')]
        left_muscles = [m for m in muscles if m.endswith('_l')]
        joint_idx = [model[i].name_dof.index('knee_r_RotZ')] * len(right_muscles) + [
            model[i].name_dof.index('knee_l_RotZ')] * len(left_muscles)
        extra_params.update({'joint_idx': joint_idx})
        extra_params.update({'idx_crank': idx_crank})

        # DynamicsOptions
        dynamics.add(
            DynamicsOptions(
                ode_solver=ode_solver,
                expand_dynamics=expand_dynamics,
                phase_dynamics=phase_dynamics,
                with_contact=True,
                # with_resistive_torque=with_resistive_torque,
                only_parameter_a=only_parameter_a,
                # with_residual_torque=True,
                n_pulse_trunc=n_pulse_trunc,
                with_torque=with_torque,
                with_friction=with_friction,
                **extra_params,
            )
        )

        for node, node_target in enumerate(target[i]):
            objective_functions.add(
                FesObjective.custom_func_maximize_crank_torque,
                custom_type=ObjectiveFcn.Mayer,
                node=node,
                index=0,
                target=node_target,
                quadratic=True,
                weight=1e4,
                phase=i,
            )

        constraints.add(ConstraintFcn.TRACK_STATE,
                        key="qdot",
                        node=Node.ALL,
                        index=0,
                        target=0,
                        # weight=1e3,
                        phase=i,
                        )

    # Path constraint
    nb_tau = model[0].nb_tau
    nb_muscles = model[0].nb_muscles
    x_bounds = BoundsList()

    for phase_index in range(n_phases):
        x_bounds.add("q", bounds=model[phase_index].bounds_from_ranges("q"), phase=phase_index)
        x_bounds[phase_index]["q"].min[:, [0]] = q_init[phase_index]
        x_bounds[phase_index]["q"].max[:, [0]] = q_init[phase_index]
        x_bounds.add("qdot", bounds=model[phase_index].bounds_from_ranges("qdot"), phase=phase_index)
        x_bounds.add("cn",
                     min_bound=[cn_min] * nb_muscles,
                     max_bound=[cn_max] * nb_muscles,
                     phase=phase_index)
        x_bounds.add("f",
                     min_bound=[f_min] * nb_muscles,
                     max_bound=[f_max] * nb_muscles,
                     phase=phase_index)
        for m in range(model[phase_index].nb_muscles):
            x_bounds.add(f"alpha_{m}",
                         min_bound=[alpha_min] * n_pulse_trunc,
                         max_bound=[alpha_max] * n_pulse_trunc,
                         phase=phase_index)
        for m in range(model[phase_index].nb_muscles):
            x_bounds[phase_index][f"alpha_{m}"][:, [0]] = alpha_init
        # x_bounds[phase_index]["cn"][:, [0]] = 0
        # x_bounds[phase_index]["f"][:, [0]] = 0
        x_bounds[phase_index]["qdot"][:, [0]] = 0.0  # no velocities at the start

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

        # extra_params_u = {"n_channels": 1, "stim_flag": stim_flag[i], "fes_params": problem_parameters["fes"]}
        u_bounds.add("pw", min_bound=problem_parameters["fes"]['fes_min'], max_bound=problem_parameters["fes"]['fes_min'], phase=i)

    x_init = InitialGuessList()
    for m in range(nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc
    x_init["q"] = q_init[phase_index]

    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    u_scaling["pw"] = [problem_parameters["fes"]['fes_max']]     # 1channel only

    phase_transitions = PhaseTransitionList()
    for i in range(n_phases - 1):
        phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

    ocp = OptimalControlProgram(
        model,
        n_shooting,
        phase_period,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        constraints=constraints,
        objective_functions=objective_functions,
        u_scaling=u_scaling,
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        use_sx=False,
        n_threads=n_threads,
        phase_transitions=phase_transitions,
    )

    return ocp


def tau1_ident(ocp_parameters,
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
    n_pulse_trunc = ocp_parameters['n_pulse_trunc']
    tau_target = ocp_parameters['target']
    stim_flag = ocp_parameters['stim_flag']
    n_phases = ocp_parameters['n_phases']
    knee_ang = ocp_parameters['joint_angle']
    mdl_param = ocp_parameters['params']
    muscle = ocp_parameters['muscle']
    a, pd0, pdt = problem_parameters['pd0pdt']

    # --- Prepare the optimal control program --- #
    print(f'\nRunning identification tau1, tau2, km for {muscle} using {model_path}...')

    import numpy as np
    import matplotlib.pyplot as plt

    tau_targets, n_shootings, phase_period = [], [], []
    for i, (tau, stim) in enumerate(zip(tau_target, stim_flag)):
        # 1️Detectar onde o estímulo é desligado (1 → 0)
        stim_off_indices = np.where((stim[:-1] == 1) & (stim[1:] == 0))[0]

        for off_idx in stim_off_indices:
            # 2️Procurar o máximo após o desligamento
            tau_segment = tau[off_idx:]
            if len(tau_segment) < 5:
                continue

            tau_max = np.max(tau_segment)
            tau_50 = 0.5 * tau_max
            tau_25 = 0.25 * tau_max

            # Índices relativos onde o torque cruza 50% e 25%
            idx_50 = np.where(tau_segment <= tau_50)[0]
            if len(idx_50) == 0:
                continue
            start_idx = idx_50[0]

            idx_25 = np.where(tau_segment <= tau_25)[0]
            if len(idx_25) == 0:
                continue
            end_idx = idx_25[0]

            # 3️Extrair o trecho correspondente
            end_idx = end_idx +1
            decay_segment = tau_segment[start_idx:end_idx]

            # Visualizar
            if plot_graphs:
                plt.figure()
                plt.plot(tau, label='Torque total')
                plt.axvline(off_idx, color='r', linestyle='--', label='stim off')
                plt.axhline(tau_50, color='orange', linestyle=':', label='50%')
                plt.axhline(tau_25, color='green', linestyle=':', label='25%')
                plt.plot(np.arange(off_idx + start_idx, off_idx + end_idx), decay_segment, 'k', linewidth=2,
                         label='Trecho 50–25%')
                plt.title(f"Curva {i} - Decaimento torque")
                plt.legend()
                plt.show()

            tau_targets.append(decay_segment)
            n_shootings.append(len(decay_segment))
            phase_period.append(t_duration * len(decay_segment))


    extra_params = {'t_phase': t_duration,
                    # 'n_pulse_trunc': n_pulse_trunc,
                    'pd0': pd0,
                    'pdt': pdt,
                    'muscle_params': mdl_param,
                    }

    ocp = prepare_ocp(biorbd_model_path=model_path,
                      q_init=q_init_list,
                      with_friction=with_friction,
                      with_residual_torque=with_residual_torque,
                      with_resistive_torque=with_resistive_torque,
                      pulse_period=t_duration,  # Attention: duration of each pulse
                      phase_period=phase_period,  # Attention: duration of each phase of the problem
                      n_shooting=n_shootings,
                      n_pulse_trunc=n_pulse_trunc,
                      target=tau_targets,
                      n_phases=n_phases,
                      joint_angle=knee_ang,
                      params=mdl_param,
                      only_parameter_a=False,
                      problem_parameters=problem_parameters,
                      **extra_params)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000, _tol=1e-4))

    # a_scale_param = float(sol.parameters["a_scale_param"][0])
    # print(f"Optimized a_scale_param: {a_scale_param}")
    # parameters = [a_scale_param]

    tau_1_param = float(sol.parameters["tau_1_param"][0])
    print(f"Optimized tau_1_param: {tau_1_param}")
    # tau_2_param = float(sol.parameters["tau_2_param"][0])
    # print(f"Optimized tau_2_param: {tau_2_param}")
    # km_param = float(sol.parameters["km_param"][0])
    # print(f"Optimized km_param: {km_param}")
    # parameters = [a_scale_param, tau_1_param, tau_2_param, km_param]
    # print(f'Identified Parameters = {parameters}')

    if save_results:
        num_model = brbd.Model(model_path)
        muscle_params = read_params(model_path, filename)
        print(f'Before we had the following parameters for muscle {muscle}: {muscle_params[muscle]}')
        muscle_params[muscle][1] = tau_1_param
        update_parameters_value(filename, muscle, muscle_params[muscle])
        muscle_params = read_params(model_path, filename)
        print(f'Now we have the following parameters for muscle {muscle}: {muscle_params[muscle]}')

    # --- Show results --- #
    # sol.print_cost()
    if plot_graphs:
        sol.graphs()
    if show_animate:
        sol.animate()

#
# if __name__ == "__main__":
#     main()
