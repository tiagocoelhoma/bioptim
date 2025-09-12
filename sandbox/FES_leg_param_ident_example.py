"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""

import numpy as np

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
    BiorbdModel,
)

# import the custom model
from sandbox.custom_package.fes_dynamics import ParamIdentDynamics, set_muscle_ascale, set_muscle_tau2_param, \
    set_muscle_km_param, set_muscle_tau1_param
from sandbox.custom_package.fes_objectives import FesObjective

from casadi import MX

def custom_u_bounds(current_shooting_point: int, n_channels: int, stim_flag: list, fes_params: list) -> np.ndarray:
    if stim_flag[current_shooting_point] == 0:
        return np.array([fes_params[0]] * n_channels)
    else:
        return np.array([fes_params[1]] * n_channels)


def get_muscle_ascale(bio_model: BiorbdModel):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        print(bio_model.model.muscle(i).characteristics().muscleAscale().to_mx())



def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    with_torque: bool = False,
    with_friction: bool = True,
    max_joint_torque: float = 0.0,
    with_resistive_torque: bool = True,
    only_parameter_a: bool = False,
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

    parameters = ParameterList(use_sx=use_sx)
    parameter_objectives = ParameterObjectiveList()
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    a_scaling = VariableScaling("a_scale_param", np.array([0.1]))
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

    if not only_parameter_a:
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


    model = ParamIdentDynamics(biorbd_model_path,
                               parameters=parameters,
                               with_contact=True,
                               with_residual_torque=False,
                               with_resistive_torque=with_resistive_torque,
                               **extra_params)

    model.set_gravity(MX([0, 0, 0]))
    # bio_model[phase_index].model.getGravity().to_mx()
    # set_muscle_ascale(model, ascale_max)
    # set_muscle_tau1_param(model, params[1])
    # set_muscle_tau2_param(model, params[2])
    # set_muscle_km_param(model, params[3])
    # get_muscle_ascale(model)
    # model.model.muscle(0).characteristics().muscleDingTau1Param().to_mx()

    idx_crank = model.name_dof.index('Gear_RotZ')
    muscles = model.muscle_names
    right_muscles = [m for m in muscles if m.endswith('_r')]
    left_muscles = [m for m in muscles if m.endswith('_l')]
    joint_idx = [model.name_dof.index('knee_r_RotZ')] * len(right_muscles) + [model.name_dof.index('knee_l_RotZ')] * len(left_muscles)
    extra_params.update({'joint_idx': joint_idx})
    extra_params.update({'idx_crank': idx_crank})

    # DynamicsOptions
    dynamics = DynamicsOptionsList()
    dynamics.add(
        DynamicsOptions(
            ode_solver=ode_solver,
            expand_dynamics=expand_dynamics,
            phase_dynamics=phase_dynamics,
            with_contact=True,
            # with_resistive_torque=with_resistive_torque,
            only_parameter_a=only_parameter_a,
            # with_residual_torque=True,
            with_torque=with_torque,
            with_friction=with_friction,
            **extra_params,
        )
    )
    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="pw", weight=1e-1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="cn", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="f", weight=1e0)
    stim_flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tau_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 5, 3, 2, 0, 0, 0]
    # tau_target = stim_flag
    for node, node_target in enumerate(tau_target):
        # test
        # if stim_flag[node] == 1:
        #     objective_functions.add(
        #         ObjectiveFcn.Mayer.TRACK_STATE,
        #         key="qdot",
        #         index=idx_crank,
        #         node=node,
        #         target=-0.25,
        #         # target= dq_init[0, kni_pos_value_at],
        #         weight=1e6,
        #     )
        weight = 10
        weight = 1e4 if stim_flag[node] == 1 else weight
        # bio_model.name_dof -> ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
        objective_functions.add(
            FesObjective.custom_func_maximize_crank_torque,
            custom_type=ObjectiveFcn.Mayer,
            node=node,
            index=0,
            target=node_target,
            quadratic=True,
            weight=weight,
        )

    # # Add constraint functions
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_STATE,
                    key="qdot",
                    node=Node.ALL,
                    index=0,
                    target=0,
                    # weight=1e3,
                    )

    q_init = np.array([[-6.2831], [5.0917], [1.9391], [0.2198], [5.659], [0.9176], [0.6453]])
    q_init = np.array([[-7.512679802050503],
                       [6.582997425587252],
                       [1.8543485578570926],
                       [0.2360338186026125],
                       [6.3928748389662164],
                       [1.492368547948381],
                       [0.7494364151304703]])

    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"].min[:, [0]] = q_init
    x_bounds["q"].max[:, [0]] = q_init
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0]] = 0.0  # no velocities at the start

    x_bounds.add("cn",
                 min_bound=[cn_min] * model.nb_muscles,
                 max_bound=[cn_max] * model.nb_muscles)
    x_bounds.add("f",
                 min_bound=[f_min] * model.nb_muscles,
                 max_bound=[f_max] * model.nb_muscles)
    for m in range(model.nb_muscles):
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
    if with_friction:
        u_bounds["resistive_torque"] = ([-50], [50])    # torque positive at the gear
    # if with_torque:
    #     u_bounds["tau"] = (
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [max_joint_torque, 0, 0, 0, 0, 0, 0])

    extra_params_u = {"n_channels": 1, "stim_flag": stim_flag, "fes_params": [fes_min, fes_max, fes_init]}
    u_bounds.add(
        "pw",
        min_bound=custom_u_bounds,
        max_bound=custom_u_bounds,
        interpolation=InterpolationType.CUSTOM,
        **extra_params_u,
    )
    x_init = InitialGuessList()
    # x_init.add('q', q_init, InterpolationType.EACH_FRAME)
    # x_init.add('qdot', dq_init, InterpolationType.EACH_FRAME)
    for m in range(model.nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc

    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    u_scaling["pw"] = [1e-3] * 1    # 1channel only

    ocp = OptimalControlProgram(
        model,
        n_shooting,
        final_time,
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
    )

    import biorbd as brbd
    numerical_model = brbd.Model(biorbd_model_path)
    ocp.add_plot("Joint Torques",
                 lambda t0, phases_dt, node_idx, x, u, p, a, d: fes_plot_tau(x, u, numerical_model),
                 plot_type=PlotType.PLOT)

    return ocp


def fes_plot_tau(x: MX, u: MX, model) -> MX:
    nb_q = model.nbDof()
    nb_m = model.nbMuscles()
    q = x[: nb_q, :]
    dq = x[nb_q: nb_q * 2, :]
    cn = x[nb_q * 2:nb_q * 2 + nb_m, :]
    f = x[nb_q * 2 + nb_m:nb_q * 2 + nb_m * 2, :]

    model.setGravity(np.array([0, 0, 0]))

    mjt = np.zeros((nb_q, q.shape[1]))
    for i in range(q.shape[1]):
        if f.size != 0:
            mjt[:, i] = model.muscularJointTorque(f.flatten(), q[:, i].flatten(), dq[:, i].flatten()).to_array()

    ddq = np.zeros((nb_q, q.shape[1]))
    for i in range(q.shape[1]):
        ddq[:, i] = model.ForwardDynamicsConstraintsDirect(q[:, i].flatten(), dq[:, i].flatten(),
                                                           mjt.flatten()).to_array()

    tau = np.zeros((nb_q, q.shape[1]))
    for i in range(q.shape[1]):
        tau[:, i] = model.InverseDynamics(q[:, i].flatten(), dq[:, i].flatten(), ddq.flatten()).to_array()

    return tau


# PROBLEM PARAMETERS
f_min, f_max, f_init = 0, 2000, 0
cn_min, cn_max, cn_init = 0.0, 1.5, 0.0
tau_min, tau_max, tau_init = 0.00, 0.00, 0.0
fes_min, fes_max, fes_init = 86e-6, 1600e-6, 86e-6  # todo: verificar scala?
alpha_min, alpha_max, alpha_init = -1.0, 2.0, 0.0
pd_offset = fes_min
stim_freq = 35  # stimulation frequency in Hertz
n_pulse_trunc = 5
phase_period = round(1 / stim_freq, 3)
n_shooting = 30

ascale_min, ascale_max, ascale_init = 0.0, 50e3, 100.0
km_param_min, km_param_max, km_param_init = 0.020, 0.800, 0.30130779
tau_1_param_min, tau_1_param_max, tau_1_param_init = 0.010, 0.1, 0.06637256
tau_2_param_min, tau_2_param_max, tau_2_param_init = 0.001, 0.2, 0.001

# km = 0.180
# tau1 = 0.044099
# tau2 = 0.018522

def main():
    """
        Basic FES-driven example to superimpose markers
    """
    biorbd_model_path = "models/legcyc_rectfem_r.bioMod"
    # biorbd_model_path = "models/leg4.bioMod"
    # --- Prepare the ocp --- #
    muscle_params = [2500, 0.044099, 0.018522, 0.180]
    pd0 = 0.000086  # average value
    pdt = 0.000138  # average value

    extra_params = {'t_phase': phase_period,
                    'n_pulse_trunc': n_pulse_trunc,
                    'pd0': pd0,
                    'pdt': pdt,
                    'muscle_params': muscle_params,
                    }
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        # with_contact=True,
        # with_residual_torque=False,
        # with_resistive_torque=True,
        final_time=phase_period * n_shooting,
        n_shooting=n_shooting,
        only_parameter_a=False,
        **extra_params,
    )

    # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    # ocp.print(to_console=True, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=True)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver=solver)

    a_scale_param = sol.parameters["a_scale_param"]
    tau_1_param = sol.parameters["tau_1_param"]
    tau_2_param = sol.parameters["tau_2_param"]
    km_param = sol.parameters["km_param"]
    print(f"Optimized a_scale_param: {a_scale_param}")
    print(f"Optimized a_scale_param: {tau_1_param}")
    print(f"Optimized a_scale_param: {tau_2_param}")
    print(f"Optimized a_scale_param: {km_param}")
    # parameters.append(a_scale_param[0])

    sol.graphs()
    # --- Show results --- #
    # sol.graphs(show_bounds=True)
    sol.print_cost()

    # --- Animation --- #
    # not implemented yet
    sol.animate()


if __name__ == "__main__":
    main()
