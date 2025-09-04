"""

"""
import numpy as np
from casadi import MX

from bioptim import (
    BiorbdModel,
    PenaltyController, BiMapping, )


def unpack_sol(list):
    out = []
    for item in list:
        out.append([it[0] for it in item])
    return out


def find_nearest(q_pos, value, idx_dof=0):
    array = np.asarray(q_pos[idx_dof,:] % (-2 * np.pi))
    array = np.asarray(q_pos[idx_dof,:] % (2 * np.pi)) if q_pos[idx_dof,0] > 0 else array
    idx = (np.abs(array - value)).argmin()
    return q_pos[:,idx, np.newaxis]

def find_nearest2(q_pos, value, idx_dof=0):
    array = np.asarray(q_pos[idx_dof, :])
    idx = (np.abs(array - value)).argmin()
    return q_pos[:, idx, np.newaxis]


def set_muscle_ascale(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleAscale(value)


def set_muscle_km_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingKmParam(value)


def set_muscle_tau1_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingTau1Param(value)


def set_muscle_tau2_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingTau2Param(value)


def set_muscle_v1_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingV1Param(value)


def set_muscle_v2_param(bio_model: BiorbdModel, value: MX):
    nb_mus = bio_model.model.nbMuscles()
    for i in range(nb_mus):
        bio_model.model.muscle(i).characteristics().setMuscleDingV2Param(value)


def custom_u_bounds(current_shooting_point: int, n_channels: int, stim_flag: list, fes_sc: float, fes_min: float) -> np.ndarray:
    if stim_flag[current_shooting_point] == 0:
        return np.array([fes_min] * n_channels)
    else:
        return np.array([fes_sc] * n_channels)


def custom_phase_transition(
    controllers: list[PenaltyController, PenaltyController], coef: float, states_mapping: BiMapping = None
) -> MX:

    if states_mapping is None:
        states_mapping = BiMapping(range(controllers[0].states.cx.shape[0]), range(controllers[1].states.cx.shape[0]))

    states_pre = states_mapping.to_second.map(controllers[0].states.cx)
    states_post = states_mapping.to_first.map(controllers[1].states.cx)

    return states_pre * coef - states_post


def fes_plot_tau(x: MX, u: MX, model) -> MX:
    nb_q = model.nbDof()
    nb_m = model.nbMuscles()
    q = x[: nb_q, :]
    dq = x[nb_q: nb_q * 2, :]
    cn = x[nb_q * 2:nb_q * 2 + nb_m, :]
    f = x[nb_q * 2 + nb_m:nb_q * 2 + nb_m * 2, :]

    mjt = np.zeros((nb_q, q.shape[1]))
    for i in range(q.shape[1]):
        if f.size != 0:
            mjt[:, i] = model.muscularJointTorque(f.flatten(), q[:, i].flatten(), dq[:, i].flatten()).to_array()

    ddq = np.zeros((nb_q, q.shape[1]))
    for i in range(q.shape[1]):
        ddq[:, i] = model.ForwardDynamicsConstraintsDirect(q[:, i].flatten(), dq[:, i].flatten(), mjt.flatten()).to_array()

    tau = np.zeros((nb_q, q.shape[1]))
    for i in range(q.shape[1]):
        tau[:, i] = model.InverseDynamics(q[:, i].flatten(), dq[:, i].flatten(), ddq.flatten()).to_array()

    return tau