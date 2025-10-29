"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""

import numpy as np
from casadi import MX
import biorbd as brbd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
    BiorbdModel, PhaseTransitionList, PhaseTransitionFcn, SolutionMerge,
)
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.com_functions import custom_u_bounds
from sandbox.Param_iden_cycling_Brazil.ocp_param_indent.common.params_handler import read_params, \
    update_parameters_value, get_coefficients_sqrd
from sandbox.Param_iden_cycling_Brazil.opt_sol_results.save_torques import salvar_torques

# import the custom model
from sandbox.custom_package.fes_dynamics import ParamIdentDynamics, set_muscle_ascale, set_muscle_tau2_param, \
    set_muscle_km_param, set_muscle_tau1_param, ParamPredictionDynamics
from sandbox.custom_package.fes_objectives import FesObjective

from casadi import MX


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
    alpha_min, alpha_max, alpha_init = problem_parameters["alpha"]

    # Add constraints and objectives (to be included in the following loop)
    objective_functions = ObjectiveList()
    constraints = ConstraintList()
    dynamics = DynamicsOptionsList()

    model = ParamPredictionDynamics(biorbd_model_path,
                                    with_contact=True,
                                    with_residual_torque=False,
                                    with_resistive_torque=with_resistive_torque,
                                    **extra_params)

    # This should be a multiphase problem

    model.set_gravity(MX([0, 0, 0]))
    # bio_model[phase_index].model.getGravity().to_mx()
    # set_muscle_ascale(model, ascale_max)
    # set_muscle_tau1_param(model, params[1])
    # set_muscle_tau2_param(model, params[2])
    # set_muscle_km_param(model, params[3])
    # get_muscle_ascale(model)
    # model.model.muscle(0).characteristics().muscleDingTau1Param().to_mx()

    # as a_scale parameter depends on the knee extension angle, we should get this index for later use in the dynamics
    # bio_model.name_dof -> ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
    idx_crank = model.name_dof.index('Gear_RotZ')
    muscles = model.muscle_names
    right_muscles = [m for m in muscles if m.endswith('_r')]
    left_muscles = [m for m in muscles if m.endswith('_l')]
    joint_idx = [model.name_dof.index('knee_r_RotZ')] * len(right_muscles) + [
        model.name_dof.index('knee_l_RotZ')] * len(left_muscles)
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

    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        key="qdot",
        index=0,
        node=Node.ALL,
        target=0,
        weight=1e8,
    )

    # constraints.add(ConstraintFcn.TRACK_STATE,
    #                 key="qdot",
    #                 node=Node.ALL,
    #                 index=0,
    #                 target=0,
    #                 # weight=1e3,
    #                 )

    # Path constraint
    nb_tau = model.nb_tau
    nb_muscles = model.nb_muscles

    tau_target = np.array(target) * np.array(stim_flag)
    max_torque_index = np.argmax(target * stim_flag)
    print(f'Max Torque with stim on: {target[max_torque_index]}')

    x_bounds = BoundsList()
    x_bounds.add("q", bounds=model.bounds_from_ranges("q"))
    x_bounds["q"].min[:, [0]] = q_init
    x_bounds["q"].max[:, [0]] = q_init
    x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"))
    x_bounds.add("cn",
                 min_bound=[cn_min] * nb_muscles,
                 max_bound=[cn_max] * nb_muscles)
    x_bounds.add("f",
                 min_bound=[f_min] * nb_muscles,
                 max_bound=[f_max] * nb_muscles)
    for m in range(model.nb_muscles):
        x_bounds.add(f"alpha_{m}",
                     min_bound=[alpha_min] * n_pulse_trunc,
                     max_bound=[alpha_max] * n_pulse_trunc)
    for m in range(model.nb_muscles):
        x_bounds[f"alpha_{m}"][:, [0]] = alpha_init
    x_bounds["cn"][:, [0]] = 0
    x_bounds["f"][:, [0]] = 0
    x_bounds["qdot"][:, [0]] = 0.0  # no velocities at the start

    # Define control path constraint
    u_bounds = BoundsList()
    # gear, pedal_r, pedal_l, knee_r, knee_l, hip_r, hip_l
    # negative torque and positive torque would correspond to the flexor and extensor muscle activation
    if with_resistive_torque:
        u_bounds.add("resistive_torque", min_bound=[-50], max_bound=[50])  # torque positive at the gear
        # u_bounds[i]["resistive_torque"][:, [0]] = 0  # remove, otherwise, due to constraints, will enter in rest. mode
    # u_bounds[i]["resistive_torque"][:, [0]] = 0
    if with_residual_torque:
        u_bounds.add("tau", min_bound=[tau_min] * nb_tau, max_bound=[tau_max] * nb_tau)
        u_bounds["tau"][:, [0]] = 0

    extra_params_u = {"n_channels": 1, "stim_flag": stim_flag, "fes_params": problem_parameters["fes"]}
    u_bounds.add(
        "pw",
        min_bound=custom_u_bounds,
        max_bound=custom_u_bounds,
        interpolation=InterpolationType.CUSTOM,
        **extra_params_u,
    )

    x_init = InitialGuessList()
    for m in range(nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc
    x_init["q"] = q_init

    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    u_scaling["pw"] = [problem_parameters["fes"]['fes_max']]  # 1channel only

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
        use_sx=False,
        n_threads=n_threads,
    )

    # import biorbd as brbd
    # numerical_model = brbd.Model(biorbd_model_path)
    # ocp.add_plot("Joint Torques",
    #              lambda t0, phases_dt, node_idx, x, u, p, a, d: fes_plot_tau(x, u, numerical_model),
    #              plot_type=PlotType.PLOT)

    return ocp


def prediction(ocp_parameters,
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
    n_pulse_trunc = ocp_parameters['n_pulse_trunc']
    tau_target = ocp_parameters['target']
    stim_flag = ocp_parameters['stim_flag']
    n_phases = ocp_parameters['n_phases']
    crank_ang = ocp_parameters['joint_angle']
    mdl_param = ocp_parameters['params']
    # only_parameter_a = ocp_parameters['only_parameter_a']
    muscle = ocp_parameters['muscle']
    stim_freq = ocp_parameters['stim_freq']

    a, pd0, pdt = problem_parameters['pd0pdt']
    mdl_param = read_params(model_path, filename)[muscle]
    predicted_torques = []
    knee_angle, crank_angle = [], []

    # --- Prepare the optimal control program --- #
    print(
        f'\nRunning prediction {muscle} using {model_path} with parameters {read_params(model_path, filename)}')
    for i in range(n_phases):
        extra_params = {'t_phase': t_duration,
                        # 'n_pulse_trunc': n_pulse_trunc,
                        'knee_angle': q_init_list[i][2],
                        'pd0pdt': problem_parameters['pd0pdt'],
                        # 'pd0': pd0,
                        # 'pdt': pdt,
                        'muscle_params': mdl_param,
                        }
        ocp = prepare_ocp(biorbd_model_path=model_path,
                          q_init=q_init_list[i],
                          with_friction=with_friction,
                          with_residual_torque=with_residual_torque,
                          with_resistive_torque=with_resistive_torque,
                          pulse_period=t_duration,  # Attention: duration of each pulse
                          phase_period=times[i],  # Attention: duration of each phase of the problem
                          n_shooting=nodes[i],
                          n_pulse_trunc=n_pulse_trunc,
                          target=tau_target[i],
                          stim_flag=stim_flag[i],
                          n_phases=n_phases,
                          # joint_angle=knee_ang[i],
                          params=mdl_param,
                          only_parameter_a=True,
                          problem_parameters=problem_parameters,
                          **extra_params)

        # --- Solve the program --- #
        sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=500, _tol=1e-4))

        # --- Show results --- #
        # sol.print_cost()
        if plot_graphs:
            sol.graphs()
        if show_animate:
            sol.animate()
        # model = brbd.Model(model_path)
        # states = sol.stepwise_states()
        # time = sol.stepwise_time(to_merge=SolutionMerge.ALL)
        # nb_nodes = len(states["q"])
        # nb_int = np.array(states["q"][0]).shape[1] - 1
        # predicted_torque = np.zeros(((nb_nodes - 1) * nb_int))
        # time_flat = np.zeros(((nb_nodes - 1) * nb_int))

        predcted_tau = np.concatenate(sol.stepwise_controls()["resistive_torque"]).flatten()
        predicted_torques.append(predcted_tau)
        time = sol.stepwise_time(to_merge=SolutionMerge.ALL)

        # states = sol.stepwise_states()
        # nb_nodes = len(states["q"])
        # nb_int = np.array(states["q"][0]).shape[1] - 1
        # predicted_torque = np.zeros(((nb_nodes - 1) * nb_int))
        # time_flat = np.zeros(((nb_nodes - 1) * nb_int))
        # time = np.linspace(0, 1, len(tau_target_interpolated))

        if save_results:
            arquivo = "Param_iden_cycling_Brazil/opt_sol_results/pickles/torques_crankset.json"
            salvar_torques(
                arquivo, muscle, crank_ang[i],
                torque_predito=predcted_tau,
                torque_medido=tau_target[i]
            )

    knee_angle.append(float(q_init_list[i][2]))
    #     crank_angle.append(float(q_init_list[i][0]))
    # ocp_parameters['crank_angle'] = crank_angle
    ocp_parameters['predicted_torques'] = predicted_torques

    # save data
    update_parameters_value('Param_iden_cycling_Brazil/opt_sol_results/pickles/prediction.pkl', muscle, ocp_parameters)

    rmse_values = []
    r2_values = []

    plt.figure(figsize=(10, 5))
    for i, torque in enumerate(tau_target):
        center = crank_ang[i]
        n = len(torque)
        x = np.arange(n) - n // 2 + center  # center the curve at the knee angle
        plt.plot(x, torque)
        plt.plot(x, predicted_torques[i], linestyle='dotted')

        y_true = np.array(torque)
        y_pred = np.array(predicted_torques[i])

        # === Cálculo das métricas ===
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        rmse_values.append(rmse)
        r2_values.append(r2)

    # === Estatísticas globais ===
    mean_rmse = np.mean(rmse_values)
    mean_r2 = np.mean(r2_values)

    # === Anotação na figura ===
    textstr = '\n'.join((
        r'$\mathrm{RMSE_{mean}}=%.3f$' % (mean_rmse,),
        r'$\mathrm{R^2_{mean}}=%.3f$' % (mean_r2,)
    ))

    plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # MAS queremos uma legenda genérica, não repetida por cor
    from matplotlib.lines import Line2D

    # Cria "linhas falsas" só para legenda
    custom_lines = [
        Line2D([0], [0], color='k', linestyle='-'),  # medida
        Line2D([0], [0], color='k', linestyle='--')  # predita
    ]

    plt.legend(custom_lines, ['measured', 'predicted'])

    plt.xlabel("Crank angle (°)")
    plt.ylabel("Torque [Nm]")
    plt.title("Torque curves centered at their respective crank angles")

    plt.grid(True)
    plt.xticks(crank_ang)  # show only the knee angles if you want it cleaner
    plt.tight_layout()

    plt.savefig(f'Param_iden_cycling_Brazil/opt_sol_results/images/prediction_{muscle}.pdf', format='pdf', dpi=300,
                bbox_inches='tight')
    plt.show()
#
# if __name__ == "__main__":
#     main()
