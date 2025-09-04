"""

"""
import numpy as np
import biorbd as brbd
from casadi import MX

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
    PlotType, Node, ConstraintFcn,
)

from Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import set_muscle_ascale, set_muscle_tau1_param, \
    set_muscle_tau2_param, set_muscle_km_param, custom_u_bounds, fes_plot_tau, set_muscle_v2_param, set_muscle_v1_param
from Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import copy_params_right_to_left, get_coefficients_sqrd
from custom_package.fes_configure_probem import FesDynamicsFcn
from custom_package.fes_objectives import FesObjective
from custom_package.fes_plotting import fes_plot_ddq, \
    custom_plot_callback_param_id
# from sandbox.Param_iden_cycling.data_exp.data_evaluation import evaluation, legs, get_exp_data_mp, mode

def prepare_ocp(
        biorbd_model_path: str,
        q_init: list = [],
        dq_init: list = [],
        kni_pos_value_at: int = 0,
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
        q_crank_target: list = [],
        params: list = [],
        use_sx: bool = False,
        problem_parameters: dict = {},
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:

    f_min, f_max, f_init = problem_parameters["f"]
    cn_min, cn_max, cn_init = problem_parameters["cn"]
    tau_min, tau_max, tau_init = problem_parameters["tau"]
    fes_min, fes_max, fes_init, fes_sc = problem_parameters["fes"]
    v1_param_min, v1_param_max, v1_param_init = problem_parameters["v1_param"]
    v2_param_min, v2_param_max, v2_param_init = problem_parameters["v2_param"]
    alpha_min, alpha_max, alpha_init = problem_parameters["alpha"]
    pd0 = problem_parameters["pd0pdt"][1]
    pdt = problem_parameters["pd0pdt"][2]

    # Define the parameter to optimize as required in the actual version of bioptim
    parameters = ParameterList(use_sx=use_sx)
    parameter_objectives = ParameterObjectiveList()
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    # bio_model.model.muscle(i).characteristics().setMuscleDingV1Param(value)
    # muscleDingV1Param

    v1_scaling = VariableScaling("v1_scale_param", np.array([1]))
    parameters.add(
        "v1_param",  # The name of the parameter
        set_muscle_v1_param,  # The function that modifies the biorbd model
        size=1,  # The number of elements this particular parameter vector has
        scaling=v1_scaling,
    )
    parameter_bounds.add("v1_param",
                         min_bound=[v1_param_min],
                         max_bound=[v1_param_max],
                         interpolation=InterpolationType.CONSTANT)
    parameter_init["v1_param"] = v1_param_init

    v2_scaling = VariableScaling("v2_scale_param", np.array([1]))
    parameters.add(
        "v2_param",  # The name of the parameter
        set_muscle_v2_param,  # The function that modifies the biorbd model
        size=1,  # The number of elements this particular parameter vector has
        scaling=v2_scaling,
    )
    parameter_bounds.add("v2_param",
                         min_bound=[v2_param_min],
                         max_bound=[v2_param_max],
                         interpolation=InterpolationType.CONSTANT)
    parameter_init["v2_param"] = v2_param_init

    # Problem parameters and BiorbdModel
    bio_model = BiorbdModel(biorbd_model_path, parameters=parameters)

    # init parameters
    # the different crank positions give different torque offsets given the existence of gravity.
    # In order to track the full torque target but ignoring the offset values, the gravity was removed
    bio_model.set_gravity(MX([0, 0, 0]))
    set_muscle_ascale(bio_model, 2000)
    set_muscle_tau1_param(bio_model, params[1])
    set_muscle_tau2_param(bio_model, params[2])
    set_muscle_km_param(bio_model, params[3])
    # as a_scale parameter depends on the knee extension angle, we should get this index for later use in the dynamics
    # bio_model.name_dof -> ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
    joint_idx = tuple(s.to_string() for s in bio_model.model.nameDof()).index('knee_r_RotZ')
    # Dynamics
    dynamics = DynamicsList()
    extra_parameters = {'t_phase': pulse_period,
                        'n_pulse_trunc': n_pulse_trunc,
                        'muscle_params': params,
                        'pd0': pd0,
                        'pdt': pdt,
                        'fes_min': fes_min,
                        'joint_index': joint_idx,
                        'validation_flag': False}
    dynamics.add(FesDynamicsFcn.FES_PARAM_IDENT_VEL,
                 expand=True,
                 with_residual_torque=with_residual_torque,
                 with_contact=True,
                 with_friction=with_friction,
                 phase_dynamics=phase_dynamics,
                 with_resistive_torque=with_resistive_torque,
                 **extra_parameters)

    # Add constraints and objectives
    constraints = ConstraintList()
    objective_functions = ObjectiveList()

    tau_target = np.array(target) * np.array(stim_flag)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e1, quadratic=True)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e3, quadratic=True)
    for node, node_target in enumerate(tau_target):
        weight = 1e1
        weight = 1e3 if stim_flag[node] > 0 else weight

        torque_at_knee_flag = False         # todo: update this flag
        # bio_model.name_dof -> ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
        # if stim_flag[node] == 1:
        # objective_functions.add(
        #     FesObjective.custom_func_maximize_crank_torque,
        #     custom_type=ObjectiveFcn.Mayer,
        #     node=node,
        #     index=0,
        #     target=node_target if node_target > 0 else 0,
        #     quadratic=True,
        #     weight=weight,
        # )
        objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            key="q",
            node=node,
            index=0,
            target=q_crank_target[node],
            quadratic=True,
            weight=weight,
        )

    # Path constraint
    nb_tau = bio_model.nb_tau
    nb_muscles = bio_model.nb_muscles
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"))
    x_bounds["qdot"].min[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    x_bounds["qdot"].max[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    # setting the initial position for the legs and gear (starting from a position different from 0 or pi)
    x_bounds["q"].min[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    x_bounds["q"].max[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    x_bounds.add("cn", min_bound=[cn_min] * nb_muscles, max_bound=[cn_max] * nb_muscles)
    x_bounds.add("f", min_bound=[f_min] * nb_muscles, max_bound=[f_max] * nb_muscles)
    for m in range(bio_model.nb_muscles):
        x_bounds.add(f"alpha_{m}", min_bound=[alpha_min] * n_pulse_trunc, max_bound=[alpha_max] * n_pulse_trunc)
    for m in range(bio_model.nb_muscles):
        x_bounds[f"alpha_{m}"][:, [0]] = alpha_init
    x_bounds["cn"][:, [0]] = 0
    x_bounds["f"][:, [0]] = 0

    x_init = InitialGuessList()
    for m in range(nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc
    # x_init.add('q', q_init, InterpolationType.EACH_FRAME)
    # x_init.add('qdot', dq_init, InterpolationType.EACH_FRAME)

    # Define control path constraint
    u_bounds = BoundsList()
    # gear, pedal_r, pedal_l, knee_r, knee_l, hip_r, hip_l
    # negative torque and positive torque would correspond to the flexor and extensor muscle activation
    if with_resistive_torque:
        u_bounds.add("resistive_torque", min_bound=[-50], max_bound=[50]) # torque positive at the gear
    if with_residual_torque:
        u_bounds.add("tau", min_bound=[tau_min] * nb_tau, max_bound=[tau_max] * nb_tau)
        u_bounds["tau"][:, [0]] = 0

    pw_fixed_flag = True
    if pw_fixed_flag is False:
        u_bounds.add("pw", min_bound=[fes_min], max_bound=[fes_max])
    else:
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
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        n_threads=16,
    )
    return ocp


def v1_v2_ident(ocp_parameters,
                     problem_parameters,
                     filename,
                     save_results=False,
                     plot_graphs=False,
                     print_flag=False,
                     show_animate=False):

    model_path = ocp_parameters['biorbd_model_path']
    q_init = ocp_parameters['q_init']
    dq_init = ocp_parameters['dq_init']
    with_resistive_torque = ocp_parameters['with_resistive_torque']
    with_friction = ocp_parameters['with_friction']
    with_residual_torque = ocp_parameters['with_residual_torque']
    t_duration = ocp_parameters['pulse_period']  # Attention: duration of each pulse
    times = ocp_parameters['phase_period']  # Attention: duration of each phase of the problem
    nodes = ocp_parameters['n_shooting']
    n_window = ocp_parameters['n_pulse_trunc']
    tau_target = ocp_parameters['target_torque']
    stim_flag = ocp_parameters['stim_flag']
    q_crank_target = ocp_parameters['target_crank_position']
    muscle = ocp_parameters['muscle']
    kni_pos_value_at = ocp_parameters['kni_pos_value_at']

    # --- Prepare the optimal control program --- #
    from Param_iden_cycling.ocp_param_indent.common.params_handler import read_params
    mdl_param = read_params(model_path, filename)[muscle[:-4]]

    print(f'\nRunning identification of the A function for {muscle} using {model_path} with parameters {read_params(model_path, filename)}')
    to_be_solved = True
    if to_be_solved:
        ocp = prepare_ocp(biorbd_model_path=model_path,
                          q_init=q_init,
                          dq_init=dq_init,
                          kni_pos_value_at=kni_pos_value_at,
                          with_friction=with_friction,
                          with_resistive_torque=with_resistive_torque,
                          with_residual_torque=with_residual_torque,
                          pulse_period=t_duration,     # Attention: duration of each pulse
                          phase_period=times,       # Attention: duration of each phase of the problem
                          n_shooting=nodes,
                          n_pulse_trunc=n_window,
                          target=tau_target,
                          stim_flag=stim_flag,
                          q_crank_target=q_crank_target,
                          params=mdl_param,
                          problem_parameters=problem_parameters)

        # --- Solve the program --- #
        sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000, _tol=1e-4))

        print(f"Optimized v1_param: {sol.parameters["v1_param"]}")
        print(f"Optimized v2_param: {sol.parameters["v2_param"]}")
        # parameters.append(float(a_scale_param[0]))

        # --- Show results --- #
        # sol.print_cost()
        if plot_graphs:
            sol.graphs()
        if show_animate:
            sol.animate()
    # knee_angle.append(float(q_init[2]))

    # if you want to insert the parameters manually, comment the call for the problem above and uncomment the following
    # parameters:
    # the values obtained for 'rect_fem_r'
    # parameters = [7387.902043965209, 1409.1857974363413, 953.4974326531005, 987.8884427809242, 754.8649695914827,
    #                   364.6133808065618]
    # parameters = [7387.902043965209, 1409.1857974363413, 953.4974326531005, 987.8884427809242]
    # parameters = [1953.310464892863, 2442.790848558021, 1956.9204634043087, 2426.1405966408934, 2758.2329005931497, 18030.110761063843]
    # copy_params_right_to_left(model_path, filename)

    '''
    VASMED_L a_scale_values = [100.00002126985724, 100.00001745886775, 50000.000087574954, 50000.00009450911, 20861.31725721443, 13309.650884782723, 13776.155102681132, 10062.113650822828, 7577.588537366644, 6306.08973609324, 5238.234633140508, 4586.461067980735, 4364.515137295022, 4138.057549112105, 3469.1937969848154, 2787.1303842772063, 2113.600448274472, 1075.0645254509272]
    Coefficients vasmed_l: [-6380.2555, 44299.6650, -114648.1031, 131182.9561, -55963.6337]
    '''
    # parameters = [100.00002126985724, 100.00001745886775, 50000.000087574954, 50000.00009450911, 20861.31725721443, 13309.650884782723, 13776.155102681132, 10062.113650822828, 7577.588537366644, 6306.08973609324, 5238.234633140508, 4586.461067980735, 4364.515137295022, 4138.057549112105, 3469.1937969848154, 2787.1303842772063, 2113.600448274472, 1075.0645254509272]
                    # if  not to_be_solved:
                    #     if muscle == 'rect_fem_r':
                    #         parameters = [99.9999994416452, 99.99999941319524, 99.99999919494361, 99.99999924473568, 38581.17385456332, 43699.43282551227,
                    #          11845.965583801366, 6101.718982647732, 5551.820967445735, 3674.848492859111, 2909.4523338345825, 2887.34774126013,
                    #          2408.4029252329715, 1933.661181795956, 1774.8921029359387, 859.2503187138742]
                    #     if muscle == 'vasmed_l':
                    #         parameters = [100.00002126985724, 100.00001745886775, 50000.000087574954, 50000.00009450911, 20861.31725721443, 13309.650884782723, 13776.155102681132, 10062.113650822828, 7577.588537366644, 6306.08973609324, 5238.234633140508, 4586.461067980735, 4364.515137295022, 4138.057549112105, 3469.1937969848154, 2787.1303842772063, 2113.600448274472, 1075.0645254509272]
                    #
                    # # saving identified parameters
                    # print(f'a_scale_values = {parameters}')
                    # from Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import read_params, update_parameters_value, get_coefficients
                    # a_params = [round(a, 4) for a in parameters]
                    # muscle_params = read_params(model_path, filename)
                    # muscle_params[muscle][0] = get_coefficients_sqrd(muscle, a_params, knee_angle, print_flag)
                    # if save_results:
                    #     update_parameters_value(filename, muscle, muscle_params[muscle])
