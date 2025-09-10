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
    DynamicsOptions, ConstraintList, ObjectiveList, VariableScalingList, Node, ContactType,
)

# import the custom model
from sandbox.custom_package.fes_dynamics import CustomFesDynamics


def prepare_ocp(
    model: CustomFesDynamics,
    final_time: float,
    n_shooting: int,
    with_torque: bool = False,
    with_friction: bool = False,
    max_joint_torque: float = 0.0,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    n_threads: int = 16,
    expand_dynamics: bool = True,
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
            fes_driven_task_flag=False, # todo to be changed
            with_torque=with_torque,
            with_friction=with_friction,
            **extra_params,
        )
    )
    # Add constraint functions
    constraints = ConstraintList()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="pw", weight=1e-1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="cn", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="f", weight=1e0)
    # if with_friction:
    #     objective_functions.add(
    #         FesObjective.custom_func_maximize_crank_power,
    #         custom_type=ObjectiveFcn.Mayer,
    #         node=Node.ALL_SHOOTING,
    #         index=idx_crank,
    #         target=-target_power,   # should be negative here
    #         quadratic=True,
    #         weight=1e3,
    #     )

    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        key="qdot",
        index=idx_crank,
        node=Node.ALL,
        target=-4.4,
        # target= dq_init[0, kni_pos_value_at],
        weight=1e3,
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
    # x_bounds.add("q", bounds=model.bounds_from_ranges("q"))
    # x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"))
    # x_bounds["qdot"].min[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    # x_bounds["qdot"].max[:, [0]] = dq_init[:, kni_pos_value_at, np.newaxis]  # so the contact points would be still
    # setting the initial position for the legs and gear (starting from a position different from 0 or pi)
    # x_bounds["q"].min[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    # x_bounds["q"].max[:, [0]] = q_init[:, kni_pos_value_at, np.newaxis]
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"].min[:, [0]] = q_init
    x_bounds["q"].max[:, [0]] = q_init
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"].max[0, :] = [0]  # gear moving backwards only (negative velocity)
    x_bounds["qdot"].min[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still
    x_bounds["qdot"].max[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still

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
    # if with_friction:
    #     u_bounds["resistive_torque"] = ([0], [10])    # torque positive at the gear
    # if with_torque:
    #     u_bounds["tau"] = (
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [max_joint_torque, 0, 0, 0, 0, 0, 0])
    u_bounds.add("pw",
                 min_bound=[fes_min] * model.nb_muscles,
                 max_bound=[fes_max] * model.nb_muscles)
    x_init = InitialGuessList()
    # x_init.add('q', q_init, InterpolationType.EACH_FRAME)
    # x_init.add('qdot', dq_init, InterpolationType.EACH_FRAME)
    for m in range(model.nb_muscles):
        x_init[f"alpha_{m}"] = [0] * n_pulse_trunc

    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    u_scaling["pw"] = [1e-3] * model.nb_muscles

    return OptimalControlProgram(
        model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        use_sx=False,
        n_threads=n_threads,
    )


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
n_shooting = 20

def main():
    """
        Basic FES-driven example to superimpose markers
    """
    biorbd_model_path = "models/legcyc.bioMod"
    # biorbd_model_path = "models/leg4.bioMod"
    # --- Prepare the ocp --- #
    joint_idx = 0
    extra_params = {'t_phase': phase_period,
                    'n_pulse_trunc': n_pulse_trunc,
                    'pd_offset': pd_offset,
                    'joint_index': joint_idx,
                    # 'validation_flag': False,
                    }
    ocp = prepare_ocp(
        # model=CustomFesDynamics(biorbd_model_path, contact_types=[ContactType.RIGID_EXPLICIT], with_contact=True, with_residual_torque=True, **extra_params),
        model=CustomFesDynamics(biorbd_model_path, with_contact=True, with_residual_torque=False, **extra_params),
        final_time=phase_period * n_shooting,
        n_shooting=n_shooting,
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
    sol.graphs()
    # --- Show results --- #
    # sol.graphs(show_bounds=True)
    sol.print_cost()

    # --- Animation --- #
    # not implemented yet
    sol.animate()


if __name__ == "__main__":
    main()
