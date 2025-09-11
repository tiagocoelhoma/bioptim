from typing import Any

import numpy as np
from biorbd import GeneralizedVelocity, GeneralizedCoordinates

from bioptim.limits import penalty
from casadi import MX, vertcat
from bioptim import PenaltyController
import casadi as ca


class FesObjective():

    @staticmethod
    def custom_func_track_torque(controller: PenaltyController) -> MX:
        muscle_joint_torque = controller.model.model.muscularJointTorque(controller.states["f"].cx,
                                                                         controller.states["q"].cx,
                                                                         controller.states["qdot"].cx).to_mx()
        return muscle_joint_torque


    @staticmethod
    def custom_func_track_pedal_torque(controller: PenaltyController) -> MX:
        # Get the muscular joint torque from muscle forces
        muscle_joint_torque = controller.model.muscle_joint_torque_from_muscle_forces(controller.states["f"].mx,
                                                                                      controller.states["q"].mx,
                                                                                      controller.states["qdot"].mx)

        # Compute the generalized accelerations constrained to the contact points
        # controller.model.name_dof
        #('Gear_RotZ', 'pedal_r_RotZ', 'knee_r_RotZ', 'hip_r_RotZ', 'pedal_l_RotZ', 'knee_l_RotZ', 'hip_l_RotZ')
        ddq = controller.model.constrained_forward_dynamics(controller.states["q"].mx,
                                                            controller.states["qdot"].mx,
                                                            muscle_joint_torque)

        joint_torques = controller.model.inverse_dynamics(controller.states["q"].mx,
                                                          controller.states["qdot"].mx,
                                                          ddq)

        return controller.mx_to_cx(
            "joint_torques",
            joint_torques,
            controller.states["f"],
            controller.states["q"],
            controller.states["qdot"],
        )


    def custom_func_track_crank_torque(controller: PenaltyController) -> MX:
        # controller is a high level interface to access to many feature of the optimal control problem
        # everything is hidden in it,
        # we had to make that kind of interface to support multinode constraints and objective in several projects
        # It is a bit more complicated to understand at first
        # but it is more flexible and more powerful at the end of the day

        muscle_joint_torque = controller.model.muscle_joint_torque(controller.controls["muscles"].mx,
                                                                   controller.states["q"].mx,
                                                                   controller.states["qdot"].mx)
        # Don't put your target here. All objective function essentially support a target value, it is automatically
        # handled later on. So here we just want to return the expression that will be used to compute the value to be
        # tracked.

        # muscle_joint_torque is MX graph but as we need to handle MX and SX,
        # we need automatic conversion that is done by mx_to_cx, that turn it into a casadi function automatically
        # this is now a standard way to send the expression to bioptim,
        return controller.mx_to_cx(
            "muscle_joint_torque",
            muscle_joint_torque,
            controller.controls["muscles"],
            controller.states["q"],
            controller.states["qdot"],
        )


    def custom_func_track_crank_power(controller: PenaltyController) -> MX:

        muscle_joint_torque = controller.model.muscle_joint_torque(controller.controls["muscles"].mx,
                                                                   controller.states["q"].mx,
                                                                   controller.states["qdot"].mx)
        crank_vel = controller.states["qdot"].cx_start

        return crank_vel * muscle_joint_torque


    def custom_func_maximize_crank_power(controller: PenaltyController, **extra_arguments: Any) -> MX:

        '''
        FES_leg_pedalling.py problem starts with a constant velocity from which q and qdot are obtained from
         leg_cyclic_torque_driven.py. The velocity value is defined in this file and it is the one defined in the crank
         optimization problem (Sabrina's work) and compatible with experimental data.
        In this sense, in the problem defined in FES_leg_pedalling.py, an opposite torque is created at the crank axis
        in order to achieve a zero acceleration. The objective function is defined here.
        In this sense, the velocity will be constant, and it will be possible to calculate the power produced. Without
        this workaround, it was hard to have the inverse_dynamics and constrained_forward_dynamics in the objective function.

        todo: verify if the residual torque function is a better idea. In this case, one should increase the
             resistance for achieving the same power as in Sabrina`s work
        '''
        friction_coef_b = extra_arguments.get('friction_coef_b')
        friction_coef_mu = extra_arguments.get('friction_coef_mu')
        crank_vel = controller.states["qdot"].cx  # select only actuacted states
        # velocities = controller.controls["resistive_torque"].cx_start  # select only actuacted states
        # resistive_torque = extra_arguments['res_tau']
        return ca.power(crank_vel, 2) * friction_coef_b


    def custom_func_minimize_cocontraction(controller: PenaltyController, mus_num_1: int, mus_num_2: int) -> MX:
        activations = controller.controls["pw"].cx_start
        return activations[mus_num_1] * activations[mus_num_2]


    def custom_track_power_by_friction(controller: PenaltyController, **extra_arguments: Any) -> MX:
        friction_coef_b = extra_arguments.get('friction_coef_b')
        friction_coef_mu = extra_arguments.get('friction_coef_mu')
        crank_vel = controller.states["qdot"].cx  # select only actuacted states

        # even if torque is 0, we can still have velocity. Although power is 0, it is correct because it does not need to
        # produce torque in specific moments of the pedalling cycle, only if the power needed to be achieved is high
        # this justifies why we have different stimulation patterns, for example, for 6W, the muscles are not always activated
        # we can try to work on average power over the cycling
        return ca.power(crank_vel, 2) * friction_coef_b  # * 2


    def custom_func_maximize_crank_torque(controller: PenaltyController) -> MX:
        # crank_vel = controller.states["qdot"].cx_start  # select only actuacted states
        return controller.controls["resistive_torque"].cx_start  # select only actuacted states

    def custom_func_res_ddq(controller: PenaltyController) -> MX:
        muscle_joint_torque = controller.model.muscle_joint_torque_from_muscle_forces(controller.states["f"].mx,
                                                                                      controller.states["q"].mx,
                                                                                      controller.states["qdot"].mx)

        tau_total = muscle_joint_torque - controller.controls["resistive_torque"].mx
        # tau_total = muscle_joint_torque[0,:] - controller.controls["resistive_torque"].mx
        # tau_total = controller.controls["resistive_torque"].mx - muscle_joint_torque[[0],:]

        # Compute the generalized accelerations constrained to the contact points
        res_ddq = controller.model.constrained_forward_dynamics(controller.states["q"].mx,
                                                                controller.states["qdot"].mx,
                                                                tau_total)

        return controller.mx_to_cx(
            "res_ddq",
            res_ddq,
            controller.controls["resistive_torque"],
            controller.states["f"],
            controller.states["q"],
            controller.states["qdot"],
        )