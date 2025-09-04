import pickle
from typing import Any

import numpy as np
import biorbd as brbd
from casadi import MX, casadi

from bioptim.limits.path_conditions import InitialGuessList

from bioptim import (
    BiorbdModel,
    Dynamics,
    DynamicsFcn,
    ObjectiveFcn,
    BoundsList,
    Solver,
    Node,
    Axis,
    PhaseDynamics,
    ObjectiveList,
    OptimalControlProgram,
    OdeSolverBase,
    OdeSolver, PlotType, NonLinearProgram, DynamicsFunctions, DynamicsEvaluation, PenaltyController,
)
from custom_package.fes_plotting import dump_sol
from bioptim.dynamics.configure_problem import _check_contacts_in_biorbd_model, ConfigureProblem
from casadi import MX, SX, horzcat
from custom_package.fes_objectives import FesObjective


def custom_track_power(controller: PenaltyController) -> MX:
    tau = controller.controls["tau"].cx
    q = controller.states["q"].cx
    qdot = controller.states["qdot"].cx

    # Compute the generalized accelerations constrained to the contact points
    ddq = controller.model.model.ForwardDynamics(q, qdot, tau)
    # ddq = controller.model.model.ForwardDynamicsConstraintsDirect(q,qdot,tau)
    joint_torques = controller.model.model.InverseDynamics(q, qdot, ddq).to_mx()

    return qdot * joint_torques


def custom_track_power_by_friction(controller: PenaltyController, **extra_arguments: Any) -> MX:
    friction_coef_b = extra_arguments.get('friction_coef_b')
    friction_coef_mu = extra_arguments.get('friction_coef_mu')
    crank_vel = controller.states["qdot"].cx  # select only actuacted states

    # even if torque is 0, we can still have velocity. Although power is 0, it is correct because it does not need to
    # produce torque in specific moments of the pedalling cycle, only if the power needed to be achieved is high
    # this justifies why we have different stimulation patterns, for example, for 6W, the muscles are not always activated
    # we can try to work on average power over the cycling
    return casadi.power(crank_vel, 2) * friction_coef_b # * 2


def calculate_friction_torque(angular_velocity, mu, B):
    """
    Calculate the frictional torque including Coulomb and viscous friction effects.

    Parameters:
    - angular_velocity: Current angular velocity of the crankset (rad/s)
    - mu: Coulomb friction coefficient
    - B: Viscous friction coefficient

    Returns:
    - friction_torque: Total frictional torque (N*m)
    """
    coulomb_friction_torque = -mu * casadi.sign(angular_velocity)  # Coulomb friction torque
    viscous_friction_torque = -B * angular_velocity  # Viscous friction torque
    friction_torque = coulomb_friction_torque + viscous_friction_torque  # Total frictional torque

    return friction_torque


def custom_torque_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_contact: bool = False,
        with_residual_torque: bool = False,
        with_friction: bool = False,
        fes_driven_task_flag: bool = False,
        prediction_flag: bool = False,
        only_parameter_a: bool = False,
        identification_flag: bool = False,
        **extra_parameters: list[MX] | list[SX]
) -> DynamicsEvaluation:

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    # remember to check the inertia at the gear in the biomechanical model (bioMod)
    friction_coef_mu = extra_parameters['friction_coef_mu']
    friction_coef_b = extra_parameters['friction_coef_b']
    idx_crank = extra_parameters['idx_crank']
    friction_torque = calculate_friction_torque(qdot[idx_crank], friction_coef_mu,
                                                friction_coef_b) if with_friction else 0
    friction_vector = casadi.MX.zeros(tau.shape)
    friction_vector[idx_crank] = friction_torque
    tau = tau + friction_vector
    # tau[0, :] = tau[0, :] + friction_torque

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces=None)
    dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
    dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
    dxdt[nlp.states["qdot"].index, :] = ddq

    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def custom_forces_from_torque_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        **extra_parameters: list[MX] | list[SX],
) -> MX | SX:
    q = nlp.get_var_from_states_or_controls("q", states, controls)
    qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
    tau = nlp.get_var_from_states_or_controls("tau", states, controls)
    tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
    tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau

    friction_coef_b = extra_parameters.get('friction_coef_b')
    friction_coef_mu = extra_parameters.get('friction_coef_mu')
    idx_crank = nlp.model.name_dof.index('Gear_RotZ')
    friction_torque = calculate_friction_torque(qdot[idx_crank], mu=friction_coef_mu,
                                                B=friction_coef_b)  # if with_friction else 0
    friction_vector = casadi.MX.zeros(tau.shape)
    friction_vector[idx_crank] = friction_torque
    tau = tau + friction_vector
    # tau[0, :] = tau[0, :] + friction_torque

    external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
    return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)


def configure_custom_torque_driven(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        with_contact: bool = False,
        with_residual_torque: bool = False,
        with_friction: bool = False,
        **extra_parameters):
    _check_contacts_in_biorbd_model(with_contact, nlp.model.nb_contacts, nlp.phase_idx)

    # DECLARE STATES
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    # DECLARE CONTROLS
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        custom_torque_driven,
        with_contact=with_contact,
        # with_passive_torque=with_passive_torque,
        # with_ligament=with_ligament,
        with_friction=with_friction,
        **extra_parameters,
    )

    # Configure the contact forces
    if with_contact:
        ConfigureProblem.configure_contact_function(ocp, nlp, custom_forces_from_torque_driven, **extra_parameters)


def prepare_ocp(
        model_path,
        cycle_shooting,
        cycle_duration,
        target_vel,
        target_power,
        max_joint_torque,
        q_init,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=True,
        with_friction=False,
        friction_coef_mu: float = 0,
        friction_coef_b: float = 0,
        ode_solver: OdeSolverBase = OdeSolver.RK4(),
):
    model = BiorbdModel(model_path)
    # dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN,
    numerical_model = brbd.Model(model_path)
    joint_idx = [s.to_string() for s in numerical_model.nameDof()]
    idx_crank = joint_idx.index('Gear_RotZ')
    extra_parameters = {'friction_coef_mu': friction_coef_mu,
                        'friction_coef_b': friction_coef_b,
                        'idx_crank': idx_crank}
    dynamics = Dynamics(configure_custom_torque_driven,
                        expand_dynamics=expand_dynamics,
                        phase_dynamics=phase_dynamics,
                        with_contact=True,
                        with_friction=with_friction,
                        **extra_parameters,
                        )

    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1e0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1e1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1e0)
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        key="qdot",
        index=0,            # must be the gear (attention)
        node=Node.ALL,
        target= target_vel, # target velocity at gear crank. The index in the objective func should be set accordingly
        weight=1e3,
    )
    objective_functions.add(
        custom_track_power_by_friction,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.ALL,
        index=idx_crank,
        target=target_power,
        quadratic=True,
        weight=1e2,
        **extra_parameters,
    )
    # objective_functions.add(
    #     custom_track_power,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.ALL,
    #     index=idx_crank,
    #     target=target_power,
    #     quadratic=True,
    #     weight=1e4,
    # )

    x_bound = BoundsList()
    x_bound["q"] = model.bounds_from_ranges("q")
    x_bound["q"].min[:, [0]] = q_init
    x_bound["q"].max[:, [0]] = q_init
    x_bound["qdot"] = model.bounds_from_ranges("qdot")
    x_bound["qdot"].max[0, :] = [0]  # gear moving backwards only (negative velocity)
    x_bound["qdot"].min[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still
    x_bound["qdot"].max[:, [0]] = 0  # Initial velocity = 0 so the contact points would be still

    u_bound = BoundsList()
    # ('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
    u_bound["tau"] = (
        [-max_joint_torque, 0, 0, 0, 0, 0, 0],
        [max_joint_torque, 0, 0, 0, 0, 0, 0])
    u_bound["tau"].max[[3], [0]] = 0  # starting with negative torque right leg (so, removing the positive torque)
    u_bound["tau"].min[[4], [0]] = 0  # starting with positive torque left leg (so, removing the negative torque)

    x_init = InitialGuessList()
    x_init["q"] = q_init

    u_init = InitialGuessList()

    ocp = OptimalControlProgram(
        model,
        dynamics,
        cycle_shooting,
        cycle_duration,
        objective_functions=objective_functions,
        x_bounds=x_bound,
        u_bounds=u_bound,
        x_init=x_init,
        u_init=u_init,
        ode_solver=ode_solver,
        n_threads=16,
    )

    return ocp


def main(model_path, dump_path, duration, nb_shooting, target_v, target_power, crank_opt_flag, prin_res=True,
         friction_coef_mu=0, friction_coef_b=0):


    q_init = np.array([[-6.3003], [4.9974], [2.1215], [0.3421], [5.5159], [1.2117], [0.6947]])
    q_init = [[-1.5495], [4.5664], [1.6352], [0.1985], [1.1329], [1.5297], [0.5722]] if crank_opt_flag else q_init

    max_joint_torque = 50
    ocp = prepare_ocp(model_path,
                      cycle_shooting=nb_shooting,
                      cycle_duration=duration,
                      target_vel=target_v,
                      target_power=target_power,
                      max_joint_torque=max_joint_torque,
                      with_friction=True,
                      q_init=q_init,
                      friction_coef_mu=friction_coef_mu,
                      friction_coef_b=friction_coef_b)

    # Solve the program
    sol = ocp.solve(solver=Solver.IPOPT(show_online_optim=False, _max_iter=500, _tol=1e-4))
    if prin_res:
        sol.graphs()
        # sol.print_cost()
        # sol.animate(n_frames=50)
        sol.animate()
    print("Initialization failure" if sol.status else "Initialization success")

    states = sol.stepwise_states()
    controls = sol.stepwise_controls()
    solver_time = sol.solver_time_to_optimize
    real_time = sol.real_time_to_optimize
    n_iterations = sol.iterations
    time = sol.stepwise_time()
    final_time = ocp.phase_time

    q_sol = states["q"]
    qdot_sol = states["qdot"]
    tau_joints = controls["tau"]

    data = {
        "q": q_sol,
        "qdot": qdot_sol,
        "tau_joints": tau_joints,
        "time": time,
        "solver_time": solver_time,
        "real_time": real_time,
        "n_iterations": n_iterations,
        "final_time": final_time,
    }
    with open(dump_path, "wb") as file:
        del sol.ocp
        pickle.dump(data, file)
        print(f"Solution dumped to {dump_path}")


if __name__ == "__main__":

    crank_opt_flag = False
    duration = 1.0
    nb_shooting = 40
    target_vel = 42
    target_power = 15
    prin_res = True
    friction_coef_mu = 0.01

    target_v = (-(target_vel * 2 * np.pi) / 60)  # rpm to rad/s
    torque_crank = target_power / target_v
    friction_coef_b = round(torque_crank / target_v, 4)

    model_path = "../models/leg_model_cyclic_muscle_inversed_root_3.bioMod"
    dump_path = 'opt_sol_results/kinematics/' + \
                f"ocp_{round(duration, 2)}s_{nb_shooting}n_{target_vel}rpm_{target_power}W_fric_{friction_coef_mu}_{friction_coef_b}"
    dump_path = dump_path + "_opt" if crank_opt_flag else dump_path

    main(model_path, dump_path, duration, nb_shooting, target_v, target_power, crank_opt_flag, prin_res, friction_coef_mu, friction_coef_b)
    from opt_sol_results.result_analysis_initial_vel import main as main_plot

    main_plot(model_path, dump_path)

